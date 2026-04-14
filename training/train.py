from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from dataclasses import asdict
from types import SimpleNamespace
import argparse
import json

import torch
from torch.utils.data import DataLoader
import yaml

from training.dataset_loader import UAV123SiameseDataset
from models.tracker.siamese_tracker import SiameseTracker
from training.losses import SiameseLoss
from training.trainer import SiameseTrainer


def load_config(config_path: str | Path) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Config file must contain a YAML mapping/object.")

    return config


def collate_siamese_samples(batch):
    if not batch:
        raise ValueError("Received empty batch in collate_siamese_samples.")

    template = torch.stack([sample.template for sample in batch], dim=0)
    search = torch.stack([sample.search for sample in batch], dim=0)
    search_bbox = torch.stack([sample.search_bbox for sample in batch], dim=0)
    template_bbox = torch.stack([sample.template_bbox for sample in batch], dim=0)

    sequence_name = [sample.sequence_name for sample in batch]
    template_frame_index = torch.tensor(
        [sample.template_frame_index for sample in batch],
        dtype=torch.long,
    )
    search_frame_index = torch.tensor(
        [sample.search_frame_index for sample in batch],
        dtype=torch.long,
    )
    difficulty_tags = [sample.difficulty_tags for sample in batch]

    return SimpleNamespace(
        template=template,
        search=search,
        search_bbox=search_bbox,
        template_bbox=template_bbox,
        sequence_name=sequence_name,
        template_frame_index=template_frame_index,
        search_frame_index=search_frame_index,
        difficulty_tags=difficulty_tags,
    )


def build_dataloader(
    dataset: UAV123SiameseDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_siamese_samples,
    )


def resolve_device(device_name: str | None) -> torch.device:
    if device_name is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device_name = device_name.lower()
    if device_name == "cuda" and not torch.cuda.is_available():
        print("Requested CUDA but it is not available. Falling back to CPU.")
        return torch.device("cpu")

    return torch.device(device_name)


def maybe_make_dir(path_value: str | Path) -> Path:
    path = Path(path_value)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_optimizer(
    model: torch.nn.Module,
    learning_rate: float,
    backbone_learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    backbone_param_ids = {id(p) for p in model.backbone.parameters() if p.requires_grad}
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    other_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in backbone_param_ids
    ]

    param_groups = []
    if other_params:
        param_groups.append(
            {
                "params": other_params,
                "lr": learning_rate,
                "weight_decay": weight_decay,
            }
        )
    if backbone_params:
        param_groups.append(
            {
                "params": backbone_params,
                "lr": backbone_learning_rate,
                "weight_decay": weight_decay,
            }
        )

    if not param_groups:
        raise ValueError("No trainable parameters found when building optimizer.")

    return torch.optim.Adam(param_groups)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Siamese UAV person tracker")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to YAML training config",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    dataset_root = config["dataset_root"]
    processed_root = config["processed_root"]

    template_size = int(config.get("template_size", 127))
    search_size = int(config.get("search_size", 255))

    batch_size = int(config.get("batch_size", 8))
    epochs = int(config.get("epochs", 1))
    learning_rate = float(config.get("learning_rate", 1e-4))
    backbone_learning_rate = float(config.get("backbone_learning_rate", learning_rate * 0.1))
    weight_decay = float(config.get("weight_decay", 1e-4))

    freeze_backbone = bool(config.get("freeze_backbone", True))
    unfreeze_last_n_backbone_blocks = int(config.get("unfreeze_last_n_backbone_blocks", 0))
    device = resolve_device(config.get("device"))

    checkpoint_dir = maybe_make_dir(config.get("checkpoint_dir", "checkpoints"))
    log_dir = maybe_make_dir(config.get("log_dir", "logs"))

    backbone_variant = config.get("backbone_variant", "small")
    feature_channels = int(config.get("feature_channels", 96))
    pretrained_backbone = bool(config.get("pretrained_backbone", True))
    normalize_input = bool(config.get("normalize_input", True))

    cls_weight = float(config.get("cls_weight", 1.0))
    reg_weight = float(config.get("reg_weight", 1.0))
    smooth_l1_beta = float(config.get("smooth_l1_beta", 1.0))

    num_workers = int(config.get("num_workers", 0))
    pin_memory = bool(config.get("pin_memory", device.type == "cuda"))

    train_samples_per_epoch = config.get("train_samples_per_epoch")
    val_samples_per_epoch = config.get("val_samples_per_epoch")

    train_dataset = UAV123SiameseDataset(
        dataset_root=dataset_root,
        processed_root=processed_root,
        split="train",
        template_size=template_size,
        search_size=search_size,
        samples_per_epoch=train_samples_per_epoch,
    )

    val_dataset = UAV123SiameseDataset(
        dataset_root=dataset_root,
        processed_root=processed_root,
        split="val",
        template_size=template_size,
        search_size=search_size,
        samples_per_epoch=val_samples_per_epoch,
    )

    train_loader = build_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = build_dataloader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = SiameseTracker(
        backbone_variant=backbone_variant,
        pretrained_backbone=pretrained_backbone,
        feature_channels=feature_channels,
        freeze_backbone=freeze_backbone,
        normalize_input=normalize_input,
    ).to(device)

    if freeze_backbone and unfreeze_last_n_backbone_blocks > 0:
        model.backbone.unfreeze_last_n_feature_blocks(unfreeze_last_n_backbone_blocks)
        print(f"Unfroze last {unfreeze_last_n_backbone_blocks} backbone feature block(s).")

    criterion = SiameseLoss(
        cls_weight=cls_weight,
        reg_weight=reg_weight,
        search_size=search_size,
        smooth_l1_beta=smooth_l1_beta,
    ).to(device)

    optimizer = build_optimizer(
        model=model,
        learning_rate=learning_rate,
        backbone_learning_rate=backbone_learning_rate,
        weight_decay=weight_decay,
    )

    trainer = SiameseTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=checkpoint_dir,
        search_size=search_size,
    )

    print("Starting training...")
    print(f"Device: {device}")
    print(f"Train samples per epoch: {len(train_dataset)}")
    print(f"Val samples per epoch: {len(val_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Backbone learning rate: {backbone_learning_rate}")
    print(f"Freeze backbone: {freeze_backbone}")
    print(f"Unfreeze last backbone blocks: {unfreeze_last_n_backbone_blocks}")

    best_val_loss = float("inf")
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        train_stats = trainer.train_one_epoch(train_loader)
        val_stats = trainer.validate_one_epoch(val_loader)

        print(f"\nEpoch {epoch}/{epochs}")
        print(trainer.format_stats("train", train_stats))
        print(trainer.format_stats("val  ", val_stats))

        checkpoint_path = trainer.save_checkpoint(
            epoch=epoch,
            train_stats=train_stats,
            val_stats=val_stats,
            filename=f"epoch_{epoch:03d}.pth",
        )
        print(f"Saved checkpoint: {checkpoint_path}")

        if val_stats.total_loss < best_val_loss:
            best_val_loss = val_stats.total_loss
            best_path = trainer.save_checkpoint(
                epoch=epoch,
                train_stats=train_stats,
                val_stats=val_stats,
                filename="best.pth",
            )
            print(f"Updated best checkpoint: {best_path}")

        history.append(
            {
                "epoch": epoch,
                "train": asdict(train_stats),
                "val": asdict(val_stats),
            }
        )

    history_path = log_dir / "train_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining finished. History saved to: {history_path}")


if __name__ == "__main__":
    main()
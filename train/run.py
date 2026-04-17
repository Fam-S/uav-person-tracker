from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import argparse
import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from data.dataset import UAV123SiameseDataset
from models.siamese import SiameseTracker
from models.losses import SiameseLoss
from train.metrics import compute_batch_metrics


@dataclass(frozen=True)
class EpochStats:
    total_loss: float
    cls_loss: float
    reg_loss: float
    mean_iou: float
    mean_score: float
    num_batches: int


class SiameseTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        checkpoint_dir: str | Path = "checkpoints",
        search_size: int = 255,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.search_size = int(search_size)

    def _move_batch_to_device(self, batch: Any) -> dict[str, Any]:
        return {
            "template": batch.template.to(self.device, non_blocking=True),
            "search": batch.search.to(self.device, non_blocking=True),
            "search_bbox": batch.search_bbox.to(self.device, non_blocking=True),
            "template_bbox": batch.template_bbox.to(self.device, non_blocking=True),
            "sequence_name": batch.sequence_name,
            "template_frame_index": batch.template_frame_index,
            "search_frame_index": batch.search_frame_index,
            "difficulty_tags": batch.difficulty_tags,
        }

    def train_one_epoch(self, dataloader: DataLoader) -> EpochStats:
        self.model.train()

        total_loss_sum = 0.0
        cls_loss_sum = 0.0
        reg_loss_sum = 0.0
        mean_iou_sum = 0.0
        mean_score_sum = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc="train", leave=False, unit="batch")
        for batch in pbar:
            batch_data = self._move_batch_to_device(batch)

            self.optimizer.zero_grad(set_to_none=True)

            outputs = self.model(
                template=batch_data["template"],
                search=batch_data["search"],
            )

            loss_out = self.criterion(
                cls_logits=outputs["cls_logits"],
                bbox_pred=outputs["bbox_pred"],
                search_bbox=batch_data["search_bbox"],
            )

            loss_out.total_loss.backward()
            self.optimizer.step()

            metrics = compute_batch_metrics(
                cls_logits=outputs["cls_logits"].detach(),
                bbox_pred=outputs["bbox_pred"].detach(),
                search_bbox=batch_data["search_bbox"].detach(),
                search_size=self.search_size,
            )

            total_loss_sum += float(loss_out.total_loss.item())
            cls_loss_sum += float(loss_out.cls_loss.item())
            reg_loss_sum += float(loss_out.reg_loss.item())
            mean_iou_sum += float(metrics["mean_iou"].item())
            mean_score_sum += float(metrics["mean_score"].item())
            num_batches += 1

            pbar.set_postfix(loss=f"{loss_out.total_loss.item():.4f}", iou=f"{metrics['mean_iou'].item():.4f}")

        if num_batches == 0:
            raise ValueError("Training dataloader produced zero batches.")

        return EpochStats(
            total_loss=total_loss_sum / num_batches,
            cls_loss=cls_loss_sum / num_batches,
            reg_loss=reg_loss_sum / num_batches,
            mean_iou=mean_iou_sum / num_batches,
            mean_score=mean_score_sum / num_batches,
            num_batches=num_batches,
        )

    @torch.no_grad()
    def validate_one_epoch(self, dataloader: DataLoader) -> EpochStats:
        self.model.eval()

        total_loss_sum = 0.0
        cls_loss_sum = 0.0
        reg_loss_sum = 0.0
        mean_iou_sum = 0.0
        mean_score_sum = 0.0
        num_batches = 0

        for batch in tqdm(dataloader, desc="val  ", leave=False, unit="batch"):
            batch_data = self._move_batch_to_device(batch)

            outputs = self.model(
                template=batch_data["template"],
                search=batch_data["search"],
            )

            loss_out = self.criterion(
                cls_logits=outputs["cls_logits"],
                bbox_pred=outputs["bbox_pred"],
                search_bbox=batch_data["search_bbox"],
            )

            metrics = compute_batch_metrics(
                cls_logits=outputs["cls_logits"],
                bbox_pred=outputs["bbox_pred"],
                search_bbox=batch_data["search_bbox"],
                search_size=self.search_size,
            )

            total_loss_sum += float(loss_out.total_loss.item())
            cls_loss_sum += float(loss_out.cls_loss.item())
            reg_loss_sum += float(loss_out.reg_loss.item())
            mean_iou_sum += float(metrics["mean_iou"].item())
            mean_score_sum += float(metrics["mean_score"].item())
            num_batches += 1

        if num_batches == 0:
            raise ValueError("Validation dataloader produced zero batches.")

        return EpochStats(
            total_loss=total_loss_sum / num_batches,
            cls_loss=cls_loss_sum / num_batches,
            reg_loss=reg_loss_sum / num_batches,
            mean_iou=mean_iou_sum / num_batches,
            mean_score=mean_score_sum / num_batches,
            num_batches=num_batches,
        )

    def save_checkpoint(
        self,
        epoch: int,
        train_stats: EpochStats,
        val_stats: EpochStats | None = None,
        filename: str | None = None,
    ) -> Path:
        checkpoint_path = self.checkpoint_dir / (filename or f"epoch_{epoch:03d}.pth")

        payload = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_stats": train_stats.__dict__,
            "val_stats": val_stats.__dict__ if val_stats is not None else None,
        }

        torch.save(payload, checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str | Path) -> int:
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return int(checkpoint["epoch"])

    @staticmethod
    def format_stats(prefix: str, stats: EpochStats) -> str:
        return (
            f"{prefix} | "
            f"loss={stats.total_loss:.4f} | "
            f"cls={stats.cls_loss:.4f} | "
            f"reg={stats.reg_loss:.4f} | "
            f"iou={stats.mean_iou:.4f} | "
            f"score={stats.mean_score:.4f} | "
            f"batches={stats.num_batches}"
        )


def validate_config(config: dict) -> None:
    cfg_train = config.get("train", {})
    missing = [k for k in ("dataset_root", "processed_root") if k not in cfg_train]
    if missing:
        raise ValueError(f"Missing required keys under 'train:' in config: {missing}")


def load_config(config_path: str | Path) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Config file must contain a YAML mapping/object.")

    validate_config(config)
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
        default="config.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (e.g. checkpoints/epoch_005.pth)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    cfg_model = config.get("model", {})
    cfg_train = config.get("train", {})

    dataset_root = cfg_train["dataset_root"]
    processed_root = cfg_train["processed_root"]

    template_size = int(cfg_model.get("template_size", 127))
    search_size = int(cfg_model.get("search_size", 255))

    batch_size = int(cfg_train.get("batch_size", 8))
    epochs = int(cfg_train.get("epochs", 1))
    learning_rate = float(cfg_train.get("learning_rate", 1e-4))
    backbone_learning_rate = float(cfg_train.get("backbone_learning_rate", learning_rate * 0.1))
    weight_decay = float(cfg_train.get("weight_decay", 1e-4))

    freeze_backbone = bool(cfg_train.get("freeze_backbone", True))
    unfreeze_last_n_backbone_blocks = int(cfg_train.get("unfreeze_last_n_backbone_blocks", 0))
    device = resolve_device(cfg_train.get("device"))

    checkpoint_dir = maybe_make_dir(cfg_train.get("checkpoint_dir", "checkpoints"))
    log_dir = maybe_make_dir(cfg_train.get("log_dir", "logs"))

    backbone_variant = cfg_model.get("backbone", "small")
    feature_channels = int(cfg_model.get("feature_channels", 96))
    pretrained_backbone = bool(cfg_model.get("pretrained", True))
    normalize_input = bool(cfg_model.get("normalize_input", True))

    cls_weight = float(cfg_train.get("cls_weight", 1.0))
    reg_weight = float(cfg_train.get("reg_weight", 1.0))
    smooth_l1_beta = float(cfg_train.get("smooth_l1_beta", 1.0))

    num_workers = int(cfg_train.get("num_workers", 0))
    pin_memory = bool(cfg_train.get("pin_memory", device.type == "cuda"))

    train_samples_per_epoch = cfg_train.get("train_samples_per_epoch")
    val_samples_per_epoch = cfg_train.get("val_samples_per_epoch")

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

    start_epoch = 1
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume) + 1
        print(f"Resumed from {args.resume}, starting at epoch {start_epoch}")

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

    for epoch in tqdm(range(start_epoch, epochs + 1), desc="epochs", unit="epoch"):
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

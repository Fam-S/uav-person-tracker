from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import ProjectConfig, load_config
from data import CompetitionSiameseDataset
from models import SiamAPNppMobileOne
from models.losses import SiamAPNLoss


@dataclass(slots=True)
class EpochStats:
    epoch: int
    mean_total_loss: float
    mean_cls_loss: float
    mean_reg_loss: float
    num_batches: int


class SiameseTrainer:
    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        # Use the configured device when possible, but fall back to CPU if CUDA was requested
        # on a machine that does not have a GPU.
        self.device = torch.device(
            config.train.device if torch.cuda.is_available() or config.train.device == "cpu" else "cpu"
        )
        # The model predicts the target box inside the search crop.
        self.model = SiamAPNppMobileOne(
            feature_channels=config.model.feature_channels,
            pretrained_path=config.model.pretrained_path if config.model.pretrained else None,
            normalize_input=config.model.normalize_input,
        ).to(self.device)
        # The loss compares predicted search-crop boxes against dataset targets.
        self.criterion = SiamAPNLoss(
            search_size=config.model.search_size,
            smooth_l1_beta=config.train.smooth_l1_beta,
        ).to(self.device)
        # Adam updates the model weights after every batch.
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )
        self.checkpoint_dir = Path(config.train.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_loss = float("inf")

    def build_dataloader(self) -> DataLoader:
        # The dataset generates template/search pairs on demand instead of storing
        # every training pair on disk ahead of time.
        dataset = CompetitionSiameseDataset(
            raw_root=self.config.train.dataset_root,
            template_size=self.config.model.template_size,
            search_size=self.config.model.search_size,
            context_amount=self.config.model.context_amount,
            samples_per_epoch=self.config.train.train_samples_per_epoch,
            translation_jitter=self.config.train.translation_jitter,
            scale_jitter=self.config.train.scale_jitter,
            seed=0,
        )
        return DataLoader(
            dataset,
            batch_size=self.config.train.batch_size,
            shuffle=False,
            num_workers=self.config.train.num_workers,
            pin_memory=self.config.train.pin_memory and self.device.type == "cuda",
        )

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> EpochStats:
        # train() enables training-time behavior such as BatchNorm updates.
        self.model.train()
        total_loss_sum = 0.0
        reg_loss_sum = 0.0
        num_batches = 0

        progress = tqdm(dataloader, desc=f"epoch {epoch}", unit="batch")
        for batch in progress:
            # Move one batch of tensors from CPU memory into the training device.
            template = batch["template"].to(self.device)
            search = batch["search"].to(self.device)
            search_bbox = batch["search_bbox_xywh"].to(self.device)

            # Forward pass: the model predicts one box for each template/search pair.
            outputs = self.model(template, search)
            loss_out = self.criterion(bbox_pred=outputs["bbox_pred"], search_bbox=search_bbox)

            # Backward pass: clear old gradients, compute new gradients, then update weights.
            self.optimizer.zero_grad(set_to_none=True)
            loss_out.total_loss.backward()
            self.optimizer.step()

            # Keep running averages so the progress bar shows how the epoch is going.
            total_loss_sum += float(loss_out.total_loss.detach().cpu())
            reg_loss_sum += float(loss_out.reg_loss.detach().cpu())
            num_batches += 1
            progress.set_postfix(
                loss=f"{total_loss_sum / num_batches:.4f}",
                reg=f"{reg_loss_sum / num_batches:.4f}",
            )

        if num_batches == 0:
            raise ValueError("Training dataloader produced zero batches.")

        return EpochStats(
            epoch=epoch,
            mean_total_loss=total_loss_sum / num_batches,
            mean_cls_loss=0.0,
            mean_reg_loss=reg_loss_sum / num_batches,
            num_batches=num_batches,
        )

    def save_checkpoint(self, epoch: int, stats: EpochStats, is_best: bool = False) -> Path:
        checkpoint_path = self.checkpoint_dir / f"epoch_{epoch:03d}.pth"
        # Save enough information to reload the model and optimizer later.
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "stats": asdict(stats),
                "config_path": str(self.config.config_path),
            },
            checkpoint_path,
        )
        if is_best:
            # Keep a second copy with a stable filename for easy evaluation/inference.
            best_path = self.checkpoint_dir / "best.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "stats": asdict(stats),
                    "config_path": str(self.config.config_path),
                },
                best_path,
            )
        return checkpoint_path


def run_training(config: ProjectConfig) -> list[EpochStats]:
    # Build everything once, then reuse the same dataloader across epochs.
    trainer = SiameseTrainer(config)
    dataloader = trainer.build_dataloader()
    history: list[EpochStats] = []

    print(f"device={trainer.device} batches_per_epoch={len(dataloader)}")
    for epoch in range(1, config.train.epochs + 1):
        # One epoch means iterating through all sampled batches once.
        stats = trainer.train_epoch(dataloader, epoch)
        is_best = stats.mean_total_loss < trainer.best_loss
        if is_best:
            trainer.best_loss = stats.mean_total_loss
        trainer.save_checkpoint(epoch, stats, is_best=is_best)
        history.append(stats)
        tag = " *" if is_best else ""
        print(
            f"epoch={stats.epoch} batches={stats.num_batches} "
            f"loss={stats.mean_total_loss:.4f} "
            f"reg={stats.mean_reg_loss:.4f}{tag}"
        )

    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the minimal SiamAPN++ + MobileOne-S2 model.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                        help="Override config value, e.g. --override train.dataset_root=/kaggle/input/data")
    args = parser.parse_args()

    overrides = {}
    for item in args.override:
        key, _, value = item.partition("=")
        if not key or not value:
            parser.error(f"Invalid --override format: '{item}'. Use KEY=VALUE.")
        overrides[key] = value

    config = load_config(args.config, overrides=overrides or None)
    run_training(config)


if __name__ == "__main__":
    main()

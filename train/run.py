from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import ProjectConfig, load_config
from data import CompetitionSiameseDataset
from models import SiamAPNppMobileOne


@dataclass(slots=True)
class EpochStats:
    epoch: int
    mean_total_loss: float
    mean_cls_loss: float
    mean_reg_loss: float
    num_batches: int


class SiameseTrainer:
    def __init__(self, config: ProjectConfig, resume_checkpoint: str | Path | None = None) -> None:
        self.config = config
        # Use the configured device when possible, but fall back to CPU if CUDA was requested
        # on a machine that does not have a GPU.
        self.device = torch.device(
            config.train.device if torch.cuda.is_available() or config.train.device == "cpu" else "cpu"
        )
        self.model = SiamAPNppMobileOne(
            feature_channels=config.model.feature_channels,
            pretrained_path=config.model.pretrained_path if config.model.pretrained else None,
            normalize_input=config.model.normalize_input,
            search_size=config.model.search_size,
            output_size=config.model.output_size,
            anchor_stride=config.model.anchor_stride,
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
        self.start_epoch = 1
        if resume_checkpoint is not None:
            self._load_checkpoint(self._resolve_resume_checkpoint(resume_checkpoint))

    def _resolve_resume_checkpoint(self, resume_checkpoint: str | Path) -> Path:
        checkpoint_str = str(resume_checkpoint).strip()
        if not checkpoint_str:
            latest = self._find_latest_epoch_checkpoint()
            if latest is None:
                raise FileNotFoundError(
                    f"No epoch_*.pth checkpoint found in {self.checkpoint_dir} to resume from"
                )
            return latest

        checkpoint_path = Path(checkpoint_str)
        if not checkpoint_path.is_absolute():
            checkpoint_path = (self.checkpoint_dir / checkpoint_path).resolve() if checkpoint_path.parts == (checkpoint_path.name,) else checkpoint_path
        return checkpoint_path

    def _find_latest_epoch_checkpoint(self) -> Path | None:
        pattern = re.compile(r"^epoch_(\d+)\.pth$")
        candidates: list[tuple[int, Path]] = []
        for path in self.checkpoint_dir.glob("epoch_*.pth"):
            match = pattern.match(path.name)
            if match:
                candidates.append((int(match.group(1)), path))
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]

    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        if "model_state_dict" not in ckpt:
            raise ValueError(f"Checkpoint missing model_state_dict: {checkpoint_path}")
        if "optimizer_state_dict" not in ckpt:
            raise ValueError(f"Checkpoint missing optimizer_state_dict: {checkpoint_path}")

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        resumed_epoch = int(ckpt.get("epoch", 0))
        self.start_epoch = resumed_epoch + 1

        best_loss = ckpt.get("best_loss")
        if best_loss is None:
            stats = ckpt.get("stats", {})
            if isinstance(stats, dict):
                best_loss = stats.get("mean_total_loss")
        if best_loss is not None:
            self.best_loss = float(best_loss)

    def build_dataloader(self) -> DataLoader:
        # The dataset generates template/search pairs on demand instead of storing
        # every training pair on disk ahead of time.
        dataset = CompetitionSiameseDataset(
            raw_root=self.config.train.dataset_root,
            template_size=self.config.model.template_size,
            search_size=self.config.model.search_size,
            output_size=self.config.model.output_size,
            context_amount=self.config.model.context_amount,
            samples_per_epoch=self.config.train.train_samples_per_epoch,
            translation_jitter=self.config.train.translation_jitter,
            scale_jitter=self.config.train.scale_jitter,
            seed=0,
        )
        dataloader_kwargs: dict[str, object] = {}
        if self.config.train.num_workers > 0:
            # Keep worker prefetch shallow so random-access video decoding does not
            # queue too many decoded samples in RAM at once.
            dataloader_kwargs["prefetch_factor"] = 1
        return DataLoader(
            dataset,
            batch_size=self.config.train.batch_size,
            shuffle=False,
            num_workers=self.config.train.num_workers,
            pin_memory=self.config.train.pin_memory and self.device.type == "cuda",
            **dataloader_kwargs,
        )

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> EpochStats:
        # train() enables training-time behavior such as BatchNorm updates.
        self.model.train()
        total_loss_sum = 0.0
        cls_loss_sum = 0.0
        reg_loss_sum = 0.0
        num_batches = 0
        report_every = 20
        total_batches = len(dataloader)

        for batch in dataloader:
            # Move one batch of tensors from CPU memory into the training device.
            model_batch = {
                key: value.to(self.device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }

            outputs = self.model(model_batch)

            # Backward pass: clear old gradients, compute new gradients, then update weights.
            self.optimizer.zero_grad(set_to_none=True)
            outputs["total_loss"].backward()
            self.optimizer.step()

            # Keep running averages so we can report clean epoch-level averages at the end.
            total_loss_sum += float(outputs["total_loss"].detach().cpu())
            cls_loss_sum += float(outputs["cls_loss"].detach().cpu())
            reg_loss_sum += float(outputs["loc_loss"].detach().cpu())
            num_batches += 1

            if num_batches % report_every == 0 or num_batches == total_batches:
                print(
                    f"epoch={epoch} batch={num_batches}/{total_batches} "
                    f"loss={total_loss_sum / num_batches:.4f} "
                    f"cls={cls_loss_sum / num_batches:.4f} "
                    f"reg={reg_loss_sum / num_batches:.4f}",
                    flush=True,
                )

        if num_batches == 0:
            raise ValueError("Training dataloader produced zero batches.")

        return EpochStats(
            epoch=epoch,
            mean_total_loss=total_loss_sum / num_batches,
            mean_cls_loss=cls_loss_sum / num_batches,
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
                "best_loss": self.best_loss,
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
                    "best_loss": self.best_loss,
                    "config_path": str(self.config.config_path),
                },
                best_path,
            )
        return checkpoint_path


def run_training(config: ProjectConfig, resume_checkpoint: str | Path | None = None) -> list[EpochStats]:
    # Build everything once, then reuse the same dataloader across epochs.
    print("initializing trainer...", flush=True)
    trainer = SiameseTrainer(config, resume_checkpoint=resume_checkpoint)
    print(f"device={trainer.device} building dataloader...", flush=True)
    dataloader = trainer.build_dataloader()
    history: list[EpochStats] = []

    start_epoch = trainer.start_epoch
    end_epoch = config.train.epochs if resume_checkpoint is None else start_epoch + config.train.epochs - 1

    print(f"device={trainer.device} batches_per_epoch={len(dataloader)}", flush=True)
    for epoch in range(start_epoch, end_epoch + 1):
        # One epoch means iterating through all sampled batches once.
        dataloader.dataset.set_epoch(epoch)
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
            f"cls={stats.mean_cls_loss:.4f} "
            f"reg={stats.mean_reg_loss:.4f}{tag}",
            flush=True,
        )

    return history


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train the minimal SiamAPN++ + MobileOne-S2 model.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "-r",
        "--resume",
        nargs="?",
        const="",
        default=None,
        metavar="CHECKPOINT",
        help="Resume from a checkpoint path, or from the latest epoch_*.pth if no path is provided.",
    )
    parser.add_argument("-o", "--override", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    config = load_config(args.config, overrides=args.override)
    run_training(config, resume_checkpoint=args.resume)


if __name__ == "__main__":
    main()

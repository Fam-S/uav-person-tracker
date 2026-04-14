from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from training.metrics import compute_batch_metrics


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

        for batch in dataloader:
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

        for batch in dataloader:
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
from __future__ import annotations

import argparse
from pathlib import Path
import torch

from data.dataset import UAV123SiameseDataset
from models.siamese import SiameseTracker
from models.losses import SiameseLoss
from train.metrics import compute_batch_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect tracker predictions on real samples")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pth")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--dataset-root", type=str, default="data/raw/UAV123")
    parser.add_argument("--processed-root", type=str, default="data/processed")
    parser.add_argument("--split", type=str, default="val")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = UAV123SiameseDataset(
        dataset_root=args.dataset_root,
        processed_root=args.processed_root,
        split=args.split,
        template_size=127,
        search_size=255,
        seed=7,
    )

    model = SiameseTracker(
        backbone_variant="small",
        pretrained_backbone=True,
        feature_channels=96,
        freeze_backbone=True,
        normalize_input=True,
    ).to(device)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = SiameseLoss(
        cls_weight=1.0,
        reg_weight=1.0,
        search_size=255,
    ).to(device)

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Using device: {device}")
    print(f"Inspecting split: {args.split}")
    print("-" * 80)

    with torch.no_grad():
        num_samples = min(args.num_samples, len(dataset))

        for idx in range(num_samples):
            sample = dataset[idx]

            template = sample.template.unsqueeze(0).to(device)
            search = sample.search.unsqueeze(0).to(device)
            search_bbox = sample.search_bbox.unsqueeze(0).to(device)

            outputs = model(template, search)

            loss_out = criterion(
                cls_logits=outputs["cls_logits"],
                bbox_pred=outputs["bbox_pred"],
                search_bbox=search_bbox,
            )

            metrics = compute_batch_metrics(
                cls_logits=outputs["cls_logits"],
                bbox_pred=outputs["bbox_pred"],
                search_bbox=search_bbox,
            )

            pred_bbox = metrics["pred_bbox"][0].detach().cpu()
            pred_score = float(metrics["pred_scores"][0].detach().cpu())
            pred_indices = metrics["pred_indices"][0].detach().cpu()
            iou = float(metrics["ious"][0].detach().cpu())

            target_bbox = search_bbox[0].detach().cpu()
            positive_indices = loss_out.positive_indices[0].detach().cpu()

            cls_map = torch.sigmoid(outputs["cls_logits"][0, 0]).detach().cpu()

            print(f"Sample {idx + 1}/{num_samples}")
            print(f"sequence_name      : {sample.sequence_name}")
            print(f"template_frame     : {sample.template_frame_index}")
            print(f"search_frame       : {sample.search_frame_index}")
            print(f"pred_indices       : (gy={int(pred_indices[0])}, gx={int(pred_indices[1])})")
            print(f"target_indices     : (gy={int(positive_indices[0])}, gx={int(positive_indices[1])})")
            print(f"pred_score         : {pred_score:.4f}")
            print(f"pred_bbox_xywh     : {[round(float(v), 3) for v in pred_bbox.tolist()]}")
            print(f"target_bbox_xywh   : {[round(float(v), 3) for v in target_bbox.tolist()]}")
            print(f"IoU                : {iou:.6f}")
            print(f"cls_loss           : {float(loss_out.cls_loss):.6f}")
            print(f"reg_loss           : {float(loss_out.reg_loss):.6f}")
            print("cls_map (sigmoid):")
            print(cls_map)
            print("-" * 80)


if __name__ == "__main__":
    main()

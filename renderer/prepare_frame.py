"""Package RGB, camera intrinsics and Mask Transfiner detections for inference."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from network.workflow import save_bundle


def detect(rgb, config, weights, threshold, device):
    # Run with the official Transfiner fork of Detectron2 on PYTHONPATH.
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    cfg = get_cfg()
    cfg.merge_from_file(config)
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.DEVICE = device
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    predictor = DefaultPredictor(cfg)
    instances = predictor(np.ascontiguousarray(rgb[:, :, ::-1]))["instances"].to("cpu")
    return instances.pred_boxes.tensor, instances.pred_masks, instances.pred_classes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--intrinsics", type=float, nargs=4, required=True,
                        metavar=("FX", "FY", "CX", "CY"))
    parser.add_argument("--camera-to-world", help="Optional JSON 4x4 rigid transform")
    parser.add_argument("--category-map", required=True, help="JSON detector ID to release ID mapping")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--detections", help="NPZ containing boxes, masks and categories")
    modes.add_argument("--transfiner-config")
    parser.add_argument("--transfiner-weights")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rgb = np.asarray(Image.open(args.image).convert("RGB")).copy()
    if args.detections:
        with np.load(args.detections, allow_pickle=False) as detections:
            boxes, masks, categories = (torch.from_numpy(detections[key])
                                        for key in ("boxes", "masks", "categories"))
    else:
        if not args.transfiner_weights:
            parser.error("--transfiner-config requires --transfiner-weights")
        boxes, masks, categories = detect(rgb, args.transfiner_config, args.transfiner_weights,
                                          args.threshold, args.device)
    height, width = rgb.shape[:2]
    if boxes.shape != (len(categories), 4) or masks.shape != (len(categories), height, width):
        parser.error("Detections must contain Q boxes and Q masks at the full image resolution")
    mapping = json.loads(Path(args.category_map).read_text(encoding="utf-8"))
    keep = torch.tensor([str(i) in mapping for i in categories.tolist()], dtype=torch.bool)
    labels = [mapping[str(i)] for i in categories[keep].tolist()]
    fx, fy, cx, cy = args.intrinsics
    if fx <= 0 or fy <= 0:
        parser.error("Camera focal lengths must be positive")
    intrinsics = torch.tensor([[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]])
    frame = dict(rgb=torch.from_numpy(rgb).permute(2, 0, 1)[None].float() / 255,
                 intrinsics=intrinsics, boxes=boxes[keep].float(), masks=masks[keep].float(),
                 categories=torch.tensor(labels, dtype=torch.long),
                 image_indices=torch.zeros(len(labels), dtype=torch.long))
    if args.camera_to_world:
        frame["camera_to_world"] = torch.tensor(json.loads(
            Path(args.camera_to_world).read_text(encoding="utf-8")), dtype=torch.float32)
    save_bundle(args.output, **frame)


if __name__ == "__main__":
    main()

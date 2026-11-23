"""Evaluate aligned instances or weak CAD retrieval with prepared validation pairs."""

import argparse
import json

import torch

from .alignment import alignment_success
from .workflow import (cad_embeddings, features_for, load_bundle, load_models,
                       load_weak_encoder, write_json)


def summarize(aligned, retrieved, categories):
    def report(correct):
        return {
            "instance_accuracy": correct.float().mean().item(),
            "class_accuracy": torch.stack([
                correct[categories == category].float().mean() for category in categories.unique()
            ]).mean().item(),
        }
    return {"alignment": report(aligned), "retrieval_aware_alignment": report(retrieved)}


def evaluate_alignment(args, data, catalog):
    model, frontend = load_models(args.checkpoint, args.device)
    voxels = catalog["voxels"].float()
    embeddings = cad_embeddings(model.retrieval, voxels)
    cad_categories = catalog["categories"].to(args.device)
    aligned, retrieved = [], []
    inputs = ("categories", "roi", "mask", "depth_xyz", "rgb", "boxes", "masks",
              "intrinsics", "measured_depth")
    for indices in torch.arange(len(data["categories"])).split(args.batch_size):
        batch = {key: data[key][indices].to(args.device) for key in inputs if key in data}
        roi, mask, xyz, _ = features_for(batch, frontend, args.rgbd)
        result = model(roi, mask, xyz, batch["categories"], voxels, cad_categories,
                       cad_embeddings=embeddings)
        symmetry = data.get("symmetry_order")
        category_ok = batch["categories"] == data.get("target_categories", data["categories"])[indices].to(args.device)
        pose_ok, retrieval_ok = alignment_success(
            result["rotation"], result["translation"], result["scale"],
            data["rotation"][indices].to(args.device), data["translation"][indices].to(args.device),
            data["scale"][indices].to(args.device), category_ok,
            result["cad_indices"] == data["cad_indices"][indices].to(args.device),
            None if symmetry is None else symmetry[indices].to(args.device),
        )
        aligned.append(pose_ok.cpu())
        retrieved.append(retrieval_ok.cpu())
    categories = data.get("target_categories", data["categories"])
    return summarize(torch.cat(aligned), torch.cat(retrieved), categories)


def evaluate_weak(args, data, catalog):
    encoder = load_weak_encoder(args.checkpoint, args.device)
    voxels = catalog["voxels"]
    k = min(5, len(voxels))
    embeddings = cad_embeddings(encoder, voxels.float(), args.batch_size)
    results = []
    for rows in torch.arange(len(data["voxels"])).split(args.batch_size):
        query = encoder(data["voxels"][rows].to(args.device).float())
        predicted = (query @ embeddings.T).topk(k, dim=1).indices.cpu()
        target = data["cad_indices"][rows]
        matched = predicted == target[:, None]
        estimates = voxels[predicted] > 0.5
        truth = voxels[target][:, None] > 0.5
        dims = (2, 3, 4, 5)
        iou = (estimates & truth).sum(dims) / (estimates | truth).sum(dims).clamp_min(1)
        results.append(torch.stack((matched[:, 0], matched.any(1), iou[:, 0], iou.mean(1)), dim=1))
    scores = torch.cat(results).mean(0).tolist()
    return dict(zip(("Accu1", f"Accu{k}", "IoU1", f"IoU{k}"), scores))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("alignment", "weak"))
    for field in ("samples", "catalog", "checkpoint"):
        parser.add_argument(f"--{field}", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rgbd", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    data, catalog = load_bundle(args.samples), load_bundle(args.catalog)
    evaluate = evaluate_alignment if args.mode == "alignment" else evaluate_weak
    result = evaluate(args, data, catalog)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

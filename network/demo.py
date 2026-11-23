"""Retrieve and align CAD models for one or more prepared RGB frames."""

import argparse

import torch

from .workflow import (cad_embeddings, features_for, load_bundle, load_models,
                       load_weak_encoder, scene_states, write_json)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", nargs="+", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--weak-checkpoint")
    parser.add_argument("--rgbd", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    model, frontend = load_models(args.checkpoint, args.device)
    weak = load_weak_encoder(args.weak_checkpoint, args.device)
    catalog = load_bundle(args.catalog)
    voxels = catalog["voxels"].float()
    classes = catalog["categories"].to(args.device)
    embeddings = cad_embeddings(model.retrieval if weak is None else weak, voxels)
    frames = []
    for path in args.frames:
        frame = load_bundle(path, args.device)
        categories = frame["categories"]
        states = []
        if len(categories):
            roi, mask, xyz, _ = features_for(frame, frontend, args.rgbd)
            result = model(roi, mask, xyz, categories, voxels, classes,
                           weak_retrieval=weak, cad_embeddings=embeddings)
            states = scene_states(result, categories, catalog, frame.get("camera_to_world"),
                                  frame.get("track_ids"))
        frames.append({"source": path, "coordinate_frame": "world" if "camera_to_world" in frame
                       else "camera", "objects": states})
    write_json(args.output, {"frames": frames})


if __name__ == "__main__":
    main()

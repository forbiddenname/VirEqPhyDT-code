"""Export occupancy queries and supervised geometric weak-label descriptors."""

import argparse

import torch

from network.workflow import cad_embeddings, features_for, load_bundle, load_models, save_bundle


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("samples", "catalog", "checkpoint", "visual", "queries", "descriptors"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    model, frontend = load_models(args.checkpoint, args.device)
    samples, catalog, visual = (load_bundle(path) for path in (args.samples, args.catalog, args.visual))
    if visual["cad_ids"] != catalog["ids"] or visual["query_ids"] != samples["query_ids"]:
        parser.error("Visual features must use the same ordered query and CAD IDs")
    names = ("roi", "mask", "depth_xyz", "rgb", "boxes", "masks", "intrinsics")
    voxels, geometry = [], []
    for rows in torch.arange(len(samples["query_ids"])).split(args.batch_size):
        batch = {key: samples[key][rows].to(args.device) for key in names if key in samples}
        roi, mask, _, _ = features_for(batch, frontend)
        occupancy = model.voxels(roi, mask)[0].sigmoid()
        voxels.append(occupancy.cpu())
        geometry.append(model.retrieval(occupancy).cpu())
    save_bundle(args.queries, query_ids=samples["query_ids"], voxels=torch.cat(voxels))
    cad_geometry = cad_embeddings(model.retrieval, catalog["voxels"].float(), args.batch_size).cpu()
    save_bundle(args.descriptors, **visual, object_geometry=torch.cat(geometry), cad_geometry=cad_geometry)


if __name__ == "__main__":
    main()

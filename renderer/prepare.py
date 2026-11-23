"""Prepare CAD occupancy catalogs or weak labels from aligned descriptors."""

import argparse
import json
from pathlib import Path

import torch

from network.weak import weak_similarities
from network.workflow import load_bundle, save_bundle


def voxelize(mesh, size):
    """Voxelize a mesh centered and scaled isotropically by its longest extent."""
    import numpy as np

    extent = mesh.extents.max()
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError("CAD mesh must have a finite, nonzero spatial extent")
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    mesh.apply_scale((size - 1) / (size * extent))
    # Align the library's pitch lattice to the release's half-cell centers.
    mesh.apply_translation([-0.5 / size] * 3)
    points = mesh.voxelized(1 / size).fill().points + 0.5 / size
    indices = np.floor((points + 0.5) * size).astype(int).clip(0, size - 1)
    x, y, z = indices.T
    grid = torch.zeros((1, size, size, size), dtype=torch.bool)
    grid[0, z, y, x] = True
    return grid


def catalog_command(args):
    import trimesh

    path = Path(args.manifest)
    items = json.loads(path.read_text(encoding="utf-8"))
    ids = [item["id"] for item in items]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("A CAD manifest must contain unique IDs and at least one mesh")
    grids = []
    for item in items:
        mesh = trimesh.load_mesh(path.parent / item["mesh"])
        grids.append(voxelize(mesh, args.size))
    save_bundle(args.output, ids=ids, voxels=torch.stack(grids),
                categories=torch.tensor([item["category"] for item in items]))


def label_command(args):
    data = load_bundle(args.descriptors, args.device)
    names = ("object_appearance", "cad_appearance", "object_semantic", "cad_semantic",
             "object_geometry", "cad_geometry")
    descriptor = {name: data[name].float() for name in names}
    pieces = []
    for rows in torch.arange(len(descriptor["object_geometry"]), device=args.device).split(args.batch_size):
        pieces.append(weak_similarities(**{
            name: value[rows] if name.startswith("object_") else value
            for name, value in descriptor.items()
        }))
    labels = {key: torch.cat([piece[key].cpu() for piece in pieces]) for key in pieces[0]}
    for name in ("query_ids", "cad_ids"):
        if name in data:
            labels[name] = data[name]
    save_bundle(args.output, **labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="task", required=True)
    catalog = commands.add_parser("catalog")
    catalog.add_argument("--manifest", required=True)
    catalog.add_argument("--size", type=int, choices=(16, 32, 64), default=32)
    catalog.add_argument("--output", required=True)
    labels = commands.add_parser("labels")
    labels.add_argument("--descriptors", required=True)
    labels.add_argument("--batch-size", type=int, default=8)
    labels.add_argument("--device", default="cpu")
    labels.add_argument("--output", required=True)
    args = parser.parse_args()
    (catalog_command if args.task == "catalog" else label_command)(args)


if __name__ == "__main__":
    main()

"""Generate deterministic synthetic inputs for the CPU quick start."""

import argparse
from pathlib import Path

import torch

from network.alignment import voxel_centers
from network.workflow import save_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/example")
    args = parser.parse_args()
    root = Path(args.output)
    rng = torch.Generator().manual_seed(21)
    size, count, cad_count = 16, 16, 20
    grid = voxel_centers(size)
    scale = torch.rand(cad_count, 3, generator=rng) * 0.24 + 0.13
    voxels = (grid.abs()[None] <= scale[:, None, None, None]).all(-1)[:, None]
    cad_ids = [f"box-{i}" for i in range(cad_count)]
    categories = torch.arange(cad_count) % 2
    query_ids = [f"query-{i}" for i in range(count)]
    target = torch.arange(count)
    translation = torch.tensor([0.0, 0.0, 1.2]).repeat(count, 1)
    theta = torch.linspace(-0.8, 0.8, count)
    rotation = torch.eye(3).repeat(count, 1, 1)
    rotation[:, 0, 0] = rotation[:, 2, 2] = theta.cos()
    rotation[:, 0, 2], rotation[:, 2, 0] = theta.sin(), -theta.sin()
    roi = torch.randn(count, 256, 4, 4, generator=rng)
    mask = torch.ones(count, 1, 4, 4)
    xyz = translation[:, :, None, None].expand(-1, -1, 4, 4).clone()
    save_bundle(root / "catalog.pt", voxels=voxels, categories=categories, ids=cad_ids)
    save_bundle(root / "supervised.pt", roi=roi, mask=mask, depth_xyz=xyz,
                rotation=rotation, scale=torch.ones(count, 3), translation=translation,
                categories=categories[target], cad_indices=target, query_ids=query_ids)
    save_bundle(root / "weak_validation.pt", voxels=voxels[target].float(), cad_indices=target)
    extrinsic = torch.eye(4)
    extrinsic[0, 3] = 2
    save_bundle(root / "frame.pt", roi=roi[:2], mask=mask[:2], depth_xyz=xyz[:2],
                categories=categories[:2], camera_to_world=extrinsic)
    save_bundle(
        root / "visual.pt", cad_ids=cad_ids, query_ids=query_ids,
        object_appearance=torch.randn(count, 256, generator=rng),
        cad_appearance=torch.randn(cad_count, 3, 256, generator=rng),
        object_semantic=torch.randn(count, 512, generator=rng),
        cad_semantic=torch.randn(cad_count, 3, 512, generator=rng),
    )


if __name__ == "__main__":
    main()

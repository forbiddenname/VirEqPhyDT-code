"""Tensor bundle I/O and feature-to-scene inference."""

import json
from pathlib import Path

import torch

from .features import RGBFeatures
from .model import VirEqPhy
from .weak import DenseVoxelEncoder


def load_bundle(path, device="cpu"):
    return torch.load(path, map_location=device, weights_only=True)


def save_bundle(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(values, path)


def write_json(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_models(checkpoint, device):
    state = load_bundle(checkpoint, device)
    model = VirEqPhy(state["num_categories"], state["voxel_size"]).to(device)
    model.load_state_dict(state["model"])
    frontend = None
    if "frontend" in state:
        frontend = RGBFeatures().to(device)
        frontend.load_state_dict(state["frontend"])
        frontend.eval()
    return model.eval(), frontend


def features_for(batch, frontend, rgbd=False):
    if "roi" in batch:
        if rgbd:
            raise ValueError("The RGB-D ablation requires RGB tensors and measured_depth")
        return batch["roi"], batch["mask"], batch["depth_xyz"], None
    if frontend is None:
        raise ValueError("RGB inputs require a checkpoint trained with the RGB feature branch")
    return frontend(
        batch["rgb"], batch["boxes"], batch["masks"], batch["intrinsics"],
        batch.get("image_indices"), batch["measured_depth"] if rgbd else None,
    )


@torch.no_grad()
def cad_embeddings(encoder, voxels, batch_size=32):
    device = next(encoder.parameters()).device
    return torch.cat([encoder(chunk.to(device)) for chunk in voxels.split(batch_size)])


def load_weak_encoder(checkpoint, device):
    if checkpoint is None:
        return None
    encoder = DenseVoxelEncoder().to(device)
    encoder.load_state_dict(load_bundle(checkpoint, device)["model"])
    return encoder.eval()


def scene_states(output, categories, catalog, camera_to_world=None, track_ids=None):
    """One scene snapshot. An optional extrinsic maps poses to world coordinates."""
    with torch.no_grad():
        scale = output["scale"]
        rotation = output["rotation"]
        translation = output["translation"]
        if camera_to_world is not None:
            world_rotation = camera_to_world[..., :3, :3]
            translation = (world_rotation @ translation[..., None]).squeeze(-1)
            translation = translation + camera_to_world[..., :3, 3]
            rotation = world_rotation @ rotation
        transforms = torch.eye(4, device=rotation.device).repeat(len(rotation), 1, 1)
        transforms[:, :3, :3] = rotation * scale[:, None, :]
        transforms[:, :3, 3] = translation
    return [
        {
            "instance_id": i if track_ids is None else track_ids[i],
            "category": category,
            "cad_id": catalog["ids"][cad_index],
            "translation": translation[i].tolist(),
            "scale": scale[i].tolist(),
            "rotation": rotation[i].tolist(),
            "transform": transforms[i].tolist(),
        }
        for i, (category, cad_index) in enumerate(zip(categories.tolist(), output["cad_indices"].tolist()))
    ]

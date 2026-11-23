"""Category orientation priors and voxel-weighted pose refinement."""

import math

import torch
from torch import nn


def voxel_centers(size, *, device=None, dtype=None):
    axis = (torch.arange(size, device=device, dtype=dtype) + 0.5) / size - 0.5
    z, y, x = torch.meshgrid(axis, axis, axis, indexing="ij")
    return torch.stack((x, y, z), dim=-1)


def rotation_distance(a, b):
    relative = a @ b.transpose(-1, -2)
    cosine = (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    sine = torch.linalg.vector_norm(
        torch.stack(
            (relative[..., 2, 1] - relative[..., 1, 2],
             relative[..., 0, 2] - relative[..., 2, 0],
             relative[..., 1, 0] - relative[..., 0, 1]), dim=-1
        ), dim=-1
    ) / 2
    return torch.atan2(sine, cosine.clamp(-1, 1))


@torch.no_grad()
def fit_orientation_priors(rotations, categories, num_categories, k=8, max_steps=30, seed=0):
    """Fit k-medoids to training rotations using SO(3) angular distance."""
    generator = torch.Generator().manual_seed(seed)
    result = []
    for category in range(num_categories):
        samples = rotations[categories == category]
        if len(samples) < k:
            raise ValueError(f"Category {category} needs at least {k} training rotations")
        # Bound temporary matrices while computing each cluster's pairwise costs.
        medoids = torch.randperm(len(samples), generator=generator)[:k].to(rotations.device)
        for _ in range(max_steps):
            assignments = rotation_distance(samples[:, None], samples[medoids]).argmin(1)
            assignments[medoids] = torch.arange(k, device=medoids.device)
            updated = medoids.clone()
            for index in range(k):
                members = torch.where(assignments == index)[0]
                group = samples[members]
                costs = torch.cat([
                    rotation_distance(chunk[:, None], group).sum(1)
                    for chunk in group.split(128)
                ])
                updated[index] = members[costs.argmin()]
            if torch.equal(updated, medoids):
                break
            medoids = updated
        result.append(samples[medoids])
    return torch.stack(result)


def weighted_procrustes(source, target, weights):
    """Fit an origin-centered rotation to matched points using Eq. (2)."""
    covariance = source.transpose(-1, -2) @ (target * weights[..., None])
    # The rotation is unique for noncollinear support. Avoid unstable SVD
    # gradients for repeated singular values by differentiating its optimum.
    with torch.no_grad():
        u, values, vh = torch.linalg.svd(covariance)
        sign = torch.linalg.det(vh.transpose(-1, -2) @ u.transpose(-1, -2))
        correction = torch.ones_like(values)
        correction[..., -1] = sign
        rotation = (vh.transpose(-1, -2) * correction[..., None, :]) @ u.transpose(-1, -2)
        signed = values * correction
        tol = values[..., 0] * 1e-6
        valid = (values[..., 1] > tol) & (signed[..., 1] + signed[..., 2] > tol)
    eye = torch.eye(3, device=source.device, dtype=source.dtype)
    if torch.is_grad_enabled():
        fixed = covariance.detach() @ rotation
        system = fixed.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None] * eye - fixed
        system = torch.where(valid[..., None, None], system, eye)
        delta = (covariance - covariance.detach()) @ rotation
        skew = delta.transpose(-1, -2) - delta
        rhs = torch.stack((skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]), dim=-1)
        x, y, z = torch.linalg.solve(system, rhs.unsqueeze(-1)).squeeze(-1).unbind(-1)
        zero = torch.zeros_like(x)
        tangent = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1)
        rotation = rotation @ (eye + tangent.reshape(*x.shape, 3, 3))
    return torch.where(valid[..., None, None], rotation, eye)


def refine_rotation(coarse, scales, occupancy, coordinates, cad_occupancy, threshold=0.5):
    """Match voxel coordinates at indices shared by the object and CAD."""
    size = occupancy.shape[-1]
    grid = voxel_centers(size, device=occupancy.device, dtype=occupancy.dtype).reshape(-1, 3)
    source = (grid @ coarse.transpose(-1, -2)) * scales[:, None, :]
    target = coordinates.flatten(2).transpose(1, 2)
    selected = (occupancy >= threshold) & (cad_occupancy > 0.5)
    weights = (occupancy * selected).flatten(1)
    return weighted_procrustes(source, target, weights) @ coarse


class PoseHead(nn.Module):
    def __init__(self, num_categories, voxel_size):
        super().__init__()
        self.register_buffer("priors", torch.eye(3).repeat(num_categories, 8, 1, 1))
        self.downsample = nn.Conv3d(3, 9, kernel_size=3, stride=2, padding=1)
        self.scale = nn.Sequential(
            nn.Linear(9 * (voxel_size // 2) ** 3, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_categories * 3)
        )
        self.orientation = nn.Linear(9, num_categories * 8)
        self.depth_center = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
        nn.init.dirac_(self.depth_center.weight)
        self.translation = nn.Sequential(
            nn.Linear(256 + 3, 128), nn.ReLU(), nn.Linear(128, num_categories * 3)
        )
        nn.init.zeros_(self.translation[-1].weight)
        nn.init.zeros_(self.translation[-1].bias)

    def forward(self, coordinates, depth_xyz, mask, roi, categories):
        rows = torch.arange(len(categories), device=categories.device)
        features = self.downsample(coordinates).relu()
        scale = self.scale(features.flatten(1)).unflatten(1, (-1, 3))[rows, categories]
        logits = self.orientation(features.mean((2, 3, 4))).unflatten(1, (-1, 8))
        logits = logits[rows, categories]
        center = (self.depth_center(depth_xyz * mask) * mask).sum((2, 3))
        center = center / mask.sum((2, 3)).clamp_min(1)
        roi_mean = (roi * mask).sum((2, 3)) / mask.sum((2, 3)).clamp_min(1)
        offset = self.translation(torch.cat((roi_mean, center), 1)).unflatten(1, (-1, 3))
        return scale.clamp(-8, 8).exp(), center + offset[rows, categories], logits

    def coarse_rotation(self, logits, categories):
        return self.priors[categories, logits.argmax(1)]


def alignment_success(rotation, translation, scale, target_rotation, target_translation,
                      target_scale, category_match, cad_match, symmetry_order=None):
    """Scan2CAD pose thresholds, with optional canonical-y symmetry (0 is continuous)."""
    angle = rotation_distance(rotation, target_rotation)
    if symmetry_order is not None:
        for order in (2, 4):
            selected = symmetry_order == order
            for index in range(1, order):
                theta = math.tau * index / order
                c, s = math.cos(theta), math.sin(theta)
                yaw = rotation.new_tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]])
                candidate = rotation_distance(rotation, target_rotation @ yaw)
                angle = torch.where(selected, torch.minimum(angle, candidate), angle)
        axis_cosine = (rotation[..., :, 1] * target_rotation[..., :, 1]).sum(-1)
        angle = torch.where(symmetry_order == 0, axis_cosine.clamp(-1, 1).acos(), angle)
    aligned = (
        category_match & (angle <= math.radians(20))
        & ((translation - target_translation).norm(dim=-1) <= 0.2)
        & ((scale / target_scale).mean(-1).sub(1).abs() <= 0.2)
    )
    return aligned, aligned & cad_match

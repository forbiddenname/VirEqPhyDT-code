"""Shared voxel occupancy, coordinate prediction and supervised retrieval."""

import torch
from torch import nn
from torch.nn import functional as F

from .alignment import PoseHead, refine_rotation, rotation_distance, voxel_centers


class VoxelDecoder(nn.Module):
    def __init__(self, size=32):
        super().__init__()
        if size not in (16, 32, 64):
            raise ValueError("Voxel size must be 16, 32 or 64")
        self.seed_size = size // 16
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.encode = nn.Conv2d(256, 32, kernel_size=3, padding=1)
        self.project = nn.Linear(128, 64 * self.seed_size ** 3)
        # Pix2Vox-style four shared upsampling stages and two output heads.
        channels = (64, 64, 32, 16, 8)
        self.decode = nn.Sequential(*[
            nn.Sequential(nn.ConvTranspose3d(before, after, 4, stride=2, padding=1),
                          nn.GroupNorm(4, after), nn.ReLU())
            for before, after in zip(channels, channels[1:])
        ])
        self.occupancy = nn.ConvTranspose3d(8, 1, kernel_size=1)
        self.coordinates = nn.ConvTranspose3d(8, 3, kernel_size=1)

    def forward(self, roi, mask):
        encoded = self.pool(self.encode(roi * mask).relu()).flatten(1)
        seed = self.project(encoded).reshape(len(roi), 64, *((self.seed_size,) * 3))
        shared = self.decode(seed)
        return self.occupancy(shared), self.coordinates(shared)


class ResidualBlock(nn.Module):
    """3D residual block adapted from the workspace's ROCA retrieval encoder."""

    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ReLU(), nn.Conv3d(channels, channels, 3, padding=1),
            nn.ReLU(), nn.Conv3d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.layers(x)


class VoxelEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        channels = (1, 16, 32, 64, 128)
        self.blocks = nn.Sequential(*[
            nn.Sequential(nn.Conv3d(before, after, 3, stride=2, padding=1),
                          ResidualBlock(after))
            for before, after in zip(channels, channels[1:])
        ])
        self.embed = nn.Linear(128, 256)

    def forward(self, voxels):
        return F.normalize(self.embed(self.blocks(voxels).mean((2, 3, 4))), dim=1)


def occlude_voxels(voxels):
    cube = voxels.shape[-1]
    extent = cube // 4
    positions = torch.randint(cube - extent + 1, (len(voxels), 3, 1), device=voxels.device)
    coordinates = torch.arange(cube, device=voxels.device) - positions
    z, y, x = ((coordinates >= 0) & (coordinates < extent)).unbind(dim=1)
    erased = z[:, :, None, None] & y[:, None, :, None] & x[:, None, None, :]
    return voxels.masked_fill(erased[:, None], 0)


def best_cad(scores, categories, cad_categories, class_matched=True):
    if class_matched:
        candidates = categories[:, None] == cad_categories[None]
        if not candidates.any(1).all():
            raise ValueError("Every query category must have a CAD candidate")
        scores = scores.masked_fill(~candidates, -torch.inf)
    return scores.argmax(dim=1)


def hard_negative_nce(scores, cad_indices, categories, cad_categories, count=16, factor=1.0):
    rows = torch.arange(len(scores), device=scores.device)
    positives = scores[rows, cad_indices]
    candidates = cad_categories[None] == categories[:, None]
    candidates[rows, cad_indices] = False
    negatives = scores.masked_fill(~candidates, -torch.inf)
    hardest = negatives.topk(min(count, scores.shape[1]), dim=1).values
    # logsumexp with an always-finite positive term handles singleton classes.
    return (torch.cat((positives[:, None], hardest + scores.new_tensor(factor).log()), dim=1)
            .logsumexp(1) - positives).mean()


class VirEqPhy(nn.Module):
    def __init__(self, num_categories, voxel_size=32):
        super().__init__()
        self.voxel_size = voxel_size
        self.voxels = VoxelDecoder(voxel_size)
        self.retrieval = VoxelEncoder()
        self.pose = PoseHead(num_categories, voxel_size)

    def forward(self, roi, mask, depth_xyz, categories, cad_voxels, cad_categories,
                *, weak_retrieval=None, cad_embeddings=None):
        occupancy_logits, coordinates = self.voxels(roi, mask)
        occupancy = occupancy_logits.sigmoid()
        encoder = self.retrieval if weak_retrieval is None else weak_retrieval
        query_voxels = occlude_voxels(occupancy) if self.training else occupancy
        if cad_embeddings is None:
            cad_embeddings = encoder(cad_voxels)
        scores = encoder(query_voxels) @ cad_embeddings.T
        cad_indices = best_cad(scores, categories, cad_categories, weak_retrieval is None)
        scale, translation, logits = self.pose(coordinates, depth_xyz, mask, roi, categories)
        coarse = self.pose.coarse_rotation(logits, categories)
        retrieved = cad_voxels[cad_indices.to(cad_voxels.device)].to(occupancy)
        rotation = refine_rotation(coarse, scale, occupancy, coordinates, retrieved)
        return dict(occupancy_logits=occupancy_logits, coordinates=coordinates, scores=scores,
                    cad_indices=cad_indices, scale=scale, translation=translation,
                    coarse=coarse, orientation_logits=logits, rotation=rotation)


def geometric_losses(output, target, categories, cad_voxels, cad_categories, priors):
    """MV, MR and MA losses in Sections IV-B to IV-E."""
    cad_indices = target["cad_indices"]
    target_voxels = cad_voxels[cad_indices]
    size = target_voxels.shape[-1]
    grid = voxel_centers(size, dtype=target_voxels.dtype, device=target_voxels.device)
    scaled = grid.reshape(1, -1, 3) * target["scale"][:, None]
    coordinate_target = scaled @ target["rotation"].transpose(-1, -2)
    coordinate_target = coordinate_target.transpose(1, 2).reshape(-1, 3, size, size, size)
    coordinate_loss = (output["coordinates"] - coordinate_target).square().sum(1).mean()
    voxel_loss = F.binary_cross_entropy_with_logits(output["occupancy_logits"], target_voxels)
    retrieval_loss = hard_negative_nce(output["scores"], cad_indices, categories, cad_categories)
    pose_bin = rotation_distance(target["rotation"][:, None], priors[categories]).argmin(1)
    # Squared matrix error stays differentiable at zero and pi.
    rotation_loss = F.cross_entropy(output["orientation_logits"], pose_bin)
    rotation_loss = rotation_loss + F.mse_loss(output["rotation"], target["rotation"])
    pose_loss = (
        F.l1_loss(output["scale"], target["scale"])
        + 0.5 * F.l1_loss(output["translation"], target["translation"])
        + 1.5 * rotation_loss
    )
    return voxel_loss + coordinate_loss, retrieval_loss, pose_loss

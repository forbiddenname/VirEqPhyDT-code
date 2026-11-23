"""RGB feature and estimated-depth branches for supplied refined instances."""

import torch
from torch import nn
from torch.nn import functional as F


def berhu(predicted, target):
    valid = torch.isfinite(target) & (target > 0)
    residual = (predicted - torch.where(valid, target, predicted)).abs()
    cutoff = (0.2 * residual.detach().amax()).clamp_min(1e-8)
    loss = torch.where(residual <= cutoff, residual, (residual.square() + cutoff.square()) / (2 * cutoff))
    return (loss * valid).sum() / valid.sum().clamp_min(1)


class RGBFeatures(nn.Module):
    def __init__(self, roi_size=16, imagenet=False):
        super().__init__()
        from torchvision.models import ResNet50_Weights
        from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

        self.backbone = resnet_fpn_backbone(
            backbone_name="resnet50", weights=ResNet50_Weights.DEFAULT if imagenet else None,
            trainable_layers=5,
        )
        self.depth_head = nn.Sequential(nn.Conv2d(4 * 256, 128, 3, padding=1), nn.ReLU(),
                                        nn.Conv2d(128, 1, 1))
        self.roi_size = roi_size
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def forward(self, rgb, boxes, masks, intrinsics, image_indices=None, measured_depth=None):
        from torchvision.ops import MultiScaleRoIAlign, roi_align

        batch, _, height, width = rgb.shape
        if image_indices is None:
            image_indices = torch.arange(len(boxes), device=boxes.device)
        features = self.backbone((rgb - self.mean) / self.std)
        # Reuse the four ResNet FPN levels for both available output branches.
        layers = [features[str(level)] for level in range(4)]
        fused = torch.cat([
            F.interpolate(level, size=layers[0].shape[-2:], mode="bilinear", align_corners=False)
            for level in layers
        ], dim=1)
        depth = F.interpolate(F.softplus(self.depth_head(fused)), size=(height, width),
                              mode="bilinear", align_corners=False)
        used_depth = depth if measured_depth is None else measured_depth
        y, x = torch.meshgrid(
            torch.arange(height, dtype=rgb.dtype, device=rgb.device),
            torch.arange(width, dtype=rgb.dtype, device=rgb.device), indexing="ij"
        )
        pixels = torch.stack((x, y, torch.ones_like(x))).flatten(1)
        rays = torch.linalg.solve(intrinsics, pixels).reshape(batch, 3, height, width)
        image_rois = torch.cat((image_indices[:, None].to(boxes), boxes), dim=1)
        mask_rois = torch.cat((torch.arange(len(boxes), device=boxes.device)[:, None].to(boxes),
                               boxes), dim=1)
        xyz = roi_align(rays * used_depth, image_rois, self.roi_size, 1.0, aligned=True)
        mask = roi_align(masks[:, None].to(rgb), mask_rois, self.roi_size, 1.0, aligned=True)
        # The pooler groups ROIs by image. Restore the caller's original object order.
        order = image_indices.argsort(stable=True)
        per_image = [boxes[image_indices == i] for i in range(batch)]
        pooler = MultiScaleRoIAlign(["0", "1", "2", "3"], self.roi_size, sampling_ratio=2)
        roi = pooler(features, per_image, [(height, width)] * batch)[order.argsort()]
        return roi, mask, xyz, depth

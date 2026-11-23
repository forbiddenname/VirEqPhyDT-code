"""Weak association labels and cumulative perturbed Top-k in Section V."""

import torch
from torch import nn
from torch.nn import functional as F


def normalize_rows(scores):
    low = scores.amin(dim=1, keepdim=True)
    span = scores.amax(dim=1, keepdim=True) - low
    return (scores - low) / span.clamp_min(1e-8)


@torch.no_grad()
def weak_similarities(object_appearance, cad_appearance, object_semantic, cad_semantic,
                      object_geometry, cad_geometry):
    """Inputs are QxD object and CxRxD rendered descriptors, plus QxD/CxD geometry."""
    def best_views(objects, views):
        matches = torch.einsum("qd,crd->qcr", F.normalize(objects), F.normalize(views, dim=-1))
        scores, indices = matches.max(dim=-1)
        candidates = torch.arange(len(views), device=views.device)
        return scores, views[candidates, indices]

    appearance, best_appearance = best_views(object_appearance, cad_appearance)
    semantic, best_semantic = best_views(object_semantic, cad_semantic)
    objects = torch.cat((object_appearance, object_semantic), dim=-1)
    rendered = torch.cat((best_appearance, best_semantic), dim=-1)
    appearance_semantic = F.cosine_similarity(objects[:, None], rendered, dim=-1)
    geometry = F.normalize(object_geometry) @ F.normalize(cad_geometry).T
    return {
        "appearance": normalize_rows(appearance),
        "semantic": normalize_rows(semantic),
        "appearance_semantic": normalize_rows(appearance_semantic),
        "geometry": normalize_rows(geometry),
        "combined": normalize_rows(0.4 * appearance_semantic + 0.6 * geometry),
    }


class DenseVoxelEncoder(nn.Module):
    def __init__(self, growth=12):
        super().__init__()
        self.stem = nn.Conv3d(1, 24, 3, stride=2, padding=1)
        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        width = 24
        for index, count in enumerate((2, 3, 3, 2)):
            self.blocks.append(nn.ModuleList([
                nn.Conv3d(width + growth * layer, growth, 3, padding=1)
                for layer in range(count)
            ]))
            width += growth * count
            if index < 3:
                self.transitions.append(nn.Conv3d(width, width // 2, 1, stride=2))
                width //= 2
        self.embedding = nn.Linear(width, 256)

    def forward(self, voxels):
        x = self.stem(voxels)
        for index, block in enumerate(self.blocks):
            for layer in block:
                x = torch.cat((x, layer(x.relu())), dim=1)
            if index < len(self.transitions):
                x = self.transitions[index](x.relu())
        return F.normalize(self.embedding(x.mean((2, 3, 4))), dim=-1)


class PerturbedTopK(torch.autograd.Function):
    """Return rank marginals. Cumulative sums give the Eq. (7) indicators."""

    @staticmethod
    def forward(ctx, scores, k, samples, sigma):
        batch, candidates = scores.shape
        noise = scores.new_empty((batch, samples, candidates)).normal_()
        order = (scores[:, None] + sigma * noise).topk(k, dim=-1).indices
        indices = order.transpose(1, 2)
        counts = scores.new_zeros((batch, k, candidates))
        counts.scatter_add_(2, indices, torch.ones_like(indices, dtype=scores.dtype))
        ctx.save_for_backward(order, noise)
        ctx.sigma = sigma
        return counts / samples

    @staticmethod
    def backward(ctx, gradient):
        order, noise = ctx.saved_tensors
        # Apply the Monte Carlo Jacobian without building a C-by-C matrix.
        upstream = gradient.gather(2, order.transpose(1, 2)).sum(1)
        grad_scores = (upstream[:, :, None] * noise).mean(1) / ctx.sigma
        return grad_scores, None, None, None


def candidate_pool(combined, k, negative_ratio, positive_pool):
    if k < 1 or negative_ratio not in (0, 1, 2):
        raise ValueError("Use K>=1 and negative ratio 0/1/2")
    if combined.shape[1] < 3 * k:
        raise ValueError("The CAD pool must have at least 3K candidates")
    if positive_pool and positive_pool < k:
        raise ValueError("A bounded positive candidate pool must contain at least K entries")
    # Stable sorting makes candidate partitions disjoint even for tied scores.
    ordering = combined.argsort(dim=1, descending=True, stable=True)
    positives = ordering[:, :-2 * k]
    if positive_pool:
        positives = positives[:, :positive_pool]
    negative_count = negative_ratio * k
    bottom = ordering[:, -2 * k:]
    sample_order = torch.rand(bottom.shape, device=combined.device).argsort(dim=1)
    negatives = bottom.gather(1, sample_order[:, :negative_count])
    return torch.cat((positives, negatives), dim=1)


def weak_pool_loss(scores, labels, k, negative_count, samples, sigma):
    if samples < 1 or sigma <= 0:
        raise ValueError("Use a positive perturbation count and sigma")
    normalized = (scores.clamp(-1, 1) + 1) / 2
    ranking = PerturbedTopK.apply(normalized, k, samples, sigma)
    cumulative = ranking.cumsum(1)
    positive = -(cumulative * labels["combined"][:, None]).sum(-1).mean()
    negative = scores.new_zeros(())
    if negative_count:
        penalty = 1 - 0.5 * (
            labels["appearance_semantic"][:, -negative_count:]
            + labels["geometry"][:, -negative_count:]
        )
        negative = (cumulative[..., -negative_count:] * penalty[:, None]).sum(-1).mean()
    return positive + negative


def weak_topk_loss(scores, labels, k=5, negative_ratio=1, samples=64, sigma=0.05,
                   positive_pool=0):
    candidates = candidate_pool(labels["combined"], k, negative_ratio, positive_pool)
    pool_labels = {name: labels[name].gather(1, candidates)
                   for name in ("combined", "appearance_semantic", "geometry")}
    return weak_pool_loss(scores.gather(1, candidates), pool_labels,
                          k, negative_ratio * k, samples, sigma)

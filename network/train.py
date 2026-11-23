"""Train VirEqPhy or its optional weak retrieval encoder from tensor bundles."""

import argparse
from collections.abc import Iterator

import torch
from torch import nn

from .alignment import fit_orientation_priors
from .features import RGBFeatures, berhu
from .model import VirEqPhy, geometric_losses
from .weak import DenseVoxelEncoder, candidate_pool, weak_pool_loss
from .workflow import features_for, load_bundle, save_bundle


class DynamicTaskWeights:
    """DWA using losses from the previous two iterations."""

    def __init__(self, start=10_000, temperature=2.0):
        self.start = start
        self.temperature = temperature
        self.history = []

    def __call__(self, losses, initial, step):
        values = torch.stack(losses)
        weights = values.new_tensor(initial)
        if step >= self.start and len(self.history) == 2:
            old, recent = self.history
            weights = (recent / old.clamp_min(1e-8) / self.temperature).softmax(0) * len(losses)
        self.history = [*self.history[-1:], values.detach()]
        return (weights * values).sum()


def batches(count, size) -> Iterator[torch.Tensor]:
    if count == 0 or size < 1:
        raise ValueError("Training requires a nonempty dataset and a positive batch size")
    while True:
        yield from torch.randperm(count).split(size)


def supervised_step(core, frontend, batch, catalog, candidate_count, rgbd):
    categories = batch["categories"]
    cad_categories = catalog["categories"]
    required = batch["cad_indices"].cpu()
    candidates = [required]
    for category in categories.unique().tolist():
        matching = torch.where(cad_categories == category)[0]
        available = matching[~torch.isin(matching, required)]
        candidates.append(available[torch.randperm(len(available))[:candidate_count]])
    selected = torch.cat(candidates).unique(sorted=True)
    batch["cad_indices"] = torch.searchsorted(selected, required).to(categories.device)
    voxels = catalog["voxels"][selected].to(categories.device).float()
    cad_categories = cad_categories[selected].to(categories.device)
    roi, mask, xyz, depth = features_for(batch, frontend, rgbd)
    output = core(roi, mask, xyz, categories, voxels, cad_categories)
    geometry = geometric_losses(output, batch, categories, voxels, cad_categories, core.pose.priors)
    if depth is not None:
        return (berhu(depth, batch["depth_target"]), *geometry), (0.5, 1, 1, 2)
    return geometry, (1, 1, 2)


def train_supervised(args, data, catalog):
    num_categories = int(data["categories"].max()) + 1
    core = VirEqPhy(num_categories, catalog["voxels"].shape[-1]).to(args.device)
    priors = fit_orientation_priors(data["rotation"], data["categories"], num_categories, seed=args.seed)
    core.pose.priors.copy_(priors)
    frontend = RGBFeatures(imagenet=args.imagenet).to(args.device) if "rgb" in data else None
    params = list(core.parameters())
    if frontend is not None:
        params.extend(frontend.parameters())
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9)
    dynamic_weights = DynamicTaskWeights(args.dwa_start)
    fields = ("categories", "cad_indices", "rotation", "translation", "scale",
              "rgb", "boxes", "masks", "intrinsics", "depth_target", "measured_depth",
              "roi", "mask", "depth_xyz")
    for step, indices in enumerate(batches(len(data["categories"]), args.batch_size), 1):
        batch = {name: data[name][indices].to(args.device) for name in fields if name in data}
        optimizer.zero_grad(set_to_none=True)
        losses, initial = supervised_step(core, frontend, batch, catalog, args.cad_candidates, args.rgbd)
        loss = dynamic_weights(losses, initial, step)
        loss.backward()
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(f"Step {step} | loss {loss.item():.6f}", flush=True)
        if step == args.steps:
            break
    checkpoint = dict(model=core.state_dict(), num_categories=num_categories, voxel_size=core.voxel_size)
    if frontend is not None:
        checkpoint["frontend"] = frontend.state_dict()
    save_bundle(args.output, **checkpoint)


def train_weak(args, data, catalog):
    labels = load_bundle(args.labels)
    if "cad_ids" in labels and labels["cad_ids"] != catalog["ids"]:
        raise ValueError("Weak-label columns do not match this CAD catalog")
    if "query_ids" in labels and labels["query_ids"] != data.get("query_ids"):
        raise ValueError("Weak-label rows do not match this query bundle")
    expected = len(data["voxels"]), len(catalog["voxels"])
    names = ("combined", "appearance_semantic", "geometry")
    if any(labels[name].shape != expected for name in names):
        raise ValueError("Weak labels must have one row per query and one column per catalog CAD")
    encoder = DenseVoxelEncoder().to(args.device)
    optimizer = torch.optim.SGD(encoder.parameters(), lr=args.lr, momentum=0.9)
    for step, rows in enumerate(batches(expected[0], args.batch_size), 1):
        batch_labels = {name: labels[name][rows].to(args.device) for name in names}
        pool = candidate_pool(batch_labels["combined"], args.k, args.negative_ratio, args.positive_pool)
        unique, inverse = torch.unique(pool, return_inverse=True)
        query = encoder(data["voxels"][rows].to(args.device).float())
        candidates = encoder(catalog["voxels"][unique.cpu()].to(args.device).float())[inverse]
        scores = (query[:, None] * candidates).sum(-1)
        pool_labels = {name: batch_labels[name].gather(1, pool) for name in names}
        loss = weak_pool_loss(scores, pool_labels, args.k, args.k * args.negative_ratio,
                              args.perturbations, args.sigma)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(f"Step {step} | loss {loss.item():.6f}", flush=True)
        if step == args.steps:
            break
    save_bundle(args.output, model=encoder.state_dict())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="method", required=True)
    for method in ("supervised", "weak"):
        command = commands.add_parser(method)
        for argument in ("samples", "catalog", "output"):
            command.add_argument(f"--{argument}", required=True)
        command.add_argument("--device", default="cpu")
        command.add_argument("--steps", type=int, default=90_000)
        command.add_argument("--batch-size", type=int, default=32)
        command.add_argument("--lr", type=float, default=0.001)
        command.add_argument("--log-every", type=int, default=100)
        command.add_argument("--seed", type=int, default=0)
        if method == "supervised":
            command.add_argument("--dwa-start", type=int, default=10_000)
            command.add_argument("--cad-candidates", type=int, default=16)
            command.add_argument("--imagenet", action="store_true")
            command.add_argument("--rgbd", action="store_true")
        else:
            command.add_argument("--labels", required=True)
            command.add_argument("--k", type=int, default=5)
            command.add_argument("--negative-ratio", type=int, choices=(0, 1, 2), default=1)
            command.add_argument("--positive-pool", type=int, default=32)
            command.add_argument("--perturbations", type=int, default=64)
            command.add_argument("--sigma", type=float, default=0.05)
    args = parser.parse_args()
    if min(args.steps, args.log_every, args.batch_size) < 1 or args.lr <= 0:
        parser.error("Steps, logging period, batch size and learning rate must be positive")
    torch.manual_seed(args.seed)
    data, catalog = load_bundle(args.samples), load_bundle(args.catalog)
    train = train_supervised if args.method == "supervised" else train_weak
    train(args, data, catalog)


if __name__ == "__main__":
    main()

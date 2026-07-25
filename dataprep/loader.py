from __future__ import annotations

import glob
import os
import random
from functools import partial

import torch
import torchvision.transforms as transforms
import webdataset as wds

import moco.loader


def identity(value):
    return value


def custom_collate(batch, feature_names):
    """Collate image views, per-teacher features, and WSI metadata."""
    q1_list, q2_list, wsi_list = [], [], []
    feature_batches = {name: [] for name in feature_names}

    for sample in batch:
        (q1, q2) = sample[0]
        metadata = sample[-1]
        if "wsi" not in metadata:
            raise KeyError("Each sample JSON must contain a 'wsi' field.")

        q1_list.append(q1)
        q2_list.append(q2)
        wsi_list.append(metadata["wsi"])

        for index, name in enumerate(feature_names):
            feature_batches[name].append(sample[index + 1])

    features = {
        name: torch.stack(feature_batches[name])
        for name in feature_names
    }

    return (
        torch.stack(q1_list),
        torch.stack(q2_list),
        features,
        {"wsis": wsi_list},
    )


def get_moco_transforms(crop_min):
    """Build the asymmetric MoCo v3 two-crop augmentation pipeline."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    augmentation1 = [
        transforms.RandomResizedCrop(224, scale=(crop_min, 1.0)),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)],
            p=0.8,
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([moco.loader.GaussianBlur((0.1, 2.0))], p=1.0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]

    augmentation2 = [
        transforms.RandomResizedCrop(224, scale=(crop_min, 1.0)),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)],
            p=0.8,
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([moco.loader.GaussianBlur((0.1, 2.0))], p=0.1),
        transforms.RandomApply([moco.loader.Solarize()], p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]

    return moco.loader.TwoCropsTransform(
        transforms.Compose(augmentation1),
        transforms.Compose(augmentation2),
    )


def get_wds_loader(args):
    """Create the WebDataset loader used for MuCoDi pretraining."""
    feature_names = sorted(
        pair.split(":", maxsplit=1)[0].strip()
        for pair in args.teacher_dims.split(",")
        if pair.strip()
    )
    if not feature_names:
        raise ValueError("No teacher features are configured.")

    shard_dir = getattr(args, "shard_dir", None)
    if not shard_dir:
        from dataprep.config import load_profile

        shard_dir = load_profile(getattr(args, "dataset", "default"))["shard_dir"]

    all_tars = sorted(glob.glob(os.path.join(shard_dir, "train-*.tar")))
    if not all_tars:
        raise RuntimeError(f"No train-*.tar shards found in: {shard_dir}")

    # Every DDP rank must see the same ordered shard list before split_by_node.
    # Use a fixed base seed here; per-rank randomness is applied later by workers.
    base_seed = args.seed if args.seed is not None else 0
    shard_rng = random.Random(base_seed)
    shard_rng.shuffle(all_tars)

    total_tars = len(all_tars)
    shard_count = getattr(args, "shard_count", None)
    if shard_count is not None:
        if shard_count < 1:
            raise ValueError("--shard-count must be positive.")
        all_tars = all_tars[: min(shard_count, total_tars)]

    if args.rank == 0:
        print(
            f"\n[WebDataset] loaded {len(all_tars)}/{total_tars} shards "
            f"with teacher features {feature_names}\n"
        )

    feature_extensions = [f"{name}.npy" for name in feature_names]
    extract_fields = ["png", *feature_extensions, "json"]
    collate_fn = partial(custom_collate, feature_names=feature_names)

    dataset = wds.DataPipeline(
        wds.SimpleShardList(all_tars),
        wds.split_by_node,
        wds.split_by_worker,
        wds.tarfile_to_samples(),
        wds.shuffle(20000),
        wds.decode("pil"),
        wds.to_tuple(*extract_fields),
        wds.map_tuple(
            get_moco_transforms(args.crop_min),
            *([torch.from_numpy] * len(feature_names)),
            identity,
        ),
        wds.batched(
            args.batch_size,
            partial=False,
            collation_fn=collate_fn,
        ),
    )

    return wds.WebLoader(
        dataset,
        batch_size=None,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

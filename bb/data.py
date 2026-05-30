from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

Task = Literal["classification", "generation"]

IMAGENET_STATS = {
    "classes": 1000,
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
}

CIFAR_STATS = {
    "cifar10": {
        "classes": 10,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
    },
    "cifar100": {
        "classes": 100,
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
    },
}


@dataclass
class LoaderBundle:
    loader: DataLoader
    sampler: DistributedSampler | None
    num_classes: int


class SyntheticCIFAR(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        size: int,
        num_classes: int,
        task: Task,
        seed: int,
        image_size: int = 32,
    ) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.labels = torch.randint(0, num_classes, (size,), generator=generator)
        images = torch.rand(size, 3, image_size, image_size, generator=generator)
        if task == "generation":
            images = images * 2 - 1
        self.images = images

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.images[index], int(self.labels[index])


def dataset_num_classes(name: str) -> int:
    key = name.lower()
    if key == "imagenet":
        return int(IMAGENET_STATS["classes"])
    if key not in CIFAR_STATS:
        raise ValueError(
            f"Unsupported dataset {name!r}; use 'cifar10', 'cifar100', or 'imagenet'."
        )
    return int(CIFAR_STATS[key]["classes"])


def build_transform(
    dataset_name: str,
    task: Task,
    train: bool,
    augment: bool,
    image_size: int,
    resize_size: int | None = None,
) -> transforms.Compose:
    if dataset_name == "imagenet":
        if task != "classification":
            raise ValueError("ImageNet is only wired for classification in this repo.")
        interpolation = transforms.InterpolationMode.BICUBIC
        if train and augment:
            ops: list[object] = [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.08, 1.0),
                    ratio=(3.0 / 4.0, 4.0 / 3.0),
                    interpolation=interpolation,
                ),
                transforms.RandomHorizontalFlip(),
            ]
        else:
            crop_pct = 0.875
            default_resize = int(round(image_size / crop_pct))
            ops = [
                transforms.Resize(resize_size or default_resize, interpolation=interpolation),
                transforms.CenterCrop(image_size),
            ]
        ops.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_STATS["mean"], IMAGENET_STATS["std"]),
            ]
        )
        return transforms.Compose(ops)

    if task == "classification":
        ops: list[object] = []
        if train and augment:
            ops.extend(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                ]
            )
        stats = CIFAR_STATS[dataset_name]
        ops.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(stats["mean"], stats["std"]),
            ]
        )
        return transforms.Compose(ops)

    ops = []
    if train and augment:
        ops.append(transforms.RandomHorizontalFlip())
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return transforms.Compose(ops)


def build_dataset(config: dict, task: Task, train: bool, seed: int) -> Dataset:
    data_cfg = config["data"]
    name = str(data_cfg.get("dataset", "cifar10")).lower()
    num_classes = dataset_num_classes(name)
    image_size = int(data_cfg.get("image_size", 32 if name.startswith("cifar") else 224))

    if bool(data_cfg.get("synthetic", False)):
        split_key = "synthetic_train_samples" if train else "synthetic_val_samples"
        split_size = int(data_cfg.get(split_key, 64))
        return SyntheticCIFAR(
            split_size,
            num_classes,
            task,
            seed=seed + (0 if train else 10000),
            image_size=image_size,
        )

    transform = build_transform(
        dataset_name=name,
        task=task,
        train=train,
        augment=bool(data_cfg.get("augment", True)),
        image_size=image_size,
        resize_size=data_cfg.get("resize_size"),
    )
    root = str(data_cfg.get("root", "data"))
    download = bool(data_cfg.get("download", True))
    if name == "cifar10":
        return datasets.CIFAR10(root=root, train=train, download=download, transform=transform)
    if name == "cifar100":
        return datasets.CIFAR100(root=root, train=train, download=download, transform=transform)

    split_key = "train_dir" if train else "val_dir"
    split_dir = data_cfg.get(split_key)
    split_root = str(split_dir) if split_dir else f"{root}/{'train' if train else 'val'}"
    return datasets.ImageFolder(root=split_root, transform=transform)


def build_dataloader(
    config: dict,
    task: Task,
    train: bool,
    seed: int,
    rank: int = 0,
    world_size: int = 1,
) -> LoaderBundle:
    dataset = build_dataset(config, task=task, train=train, seed=seed)
    data_cfg = config["data"]
    batch_key = "batch_size" if train else "eval_batch_size"
    batch_size = int(data_cfg.get(batch_key, data_cfg["batch_size"]))
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=train,
            seed=seed,
            drop_last=train,
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train and sampler is None,
        sampler=sampler,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
        persistent_workers=int(data_cfg.get("num_workers", 4)) > 0,
    )
    dataset_name = str(data_cfg.get("dataset", "cifar10"))
    return LoaderBundle(
        loader=loader,
        sampler=sampler,
        num_classes=dataset_num_classes(dataset_name),
    )

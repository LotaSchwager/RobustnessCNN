# core/data.py
from dataclasses import dataclass
from typing import Literal, Tuple
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

DatasetName = Literal["cifar10", "cifar100", "mnist", "fashionmnist", "svhn"]

@dataclass
class DataConfig:
    name: DatasetName
    root: str = "./data"
    batch_size: int = 128
    test_batch_size: int = 256
    num_workers: int = 4
    use_cuda: bool = True
    download: bool = True

def dataset_stats(name: str) -> Tuple[Tuple[float,...], Tuple[float,...], int]:
    """Devuelve (mean, std, num_classes). Valores típicos de literatura."""
    if name == "cifar10":
        return (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616), 10
    if name == "cifar100":
        # valores comunes para CIFAR-100 (aprox, estándar en implementaciones)
        return (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761), 100
    if name in ("mnist", "fashionmnist"):
        return (0.1307,), (0.3081,), 10
    if name == "svhn":
        # muchos usan estos (SVHN es distinto); también puedes calcularlos después si quieres
        return (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970), 10
    raise ValueError(f"Dataset no soportado: {name}")

def make_transforms(name: str):
    """Transforms sin Normalize (Normalize va en NormalizeLayer)."""
    if name in ("cifar10", "cifar100"):
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
        ])
        return train_tf, test_tf

    if name in ("mnist", "fashionmnist"):
        train_tf = transforms.Compose([transforms.ToTensor()])
        test_tf  = transforms.Compose([transforms.ToTensor()])
        return train_tf, test_tf

    if name == "svhn":
        train_tf = transforms.Compose([transforms.ToTensor()])
        test_tf  = transforms.Compose([transforms.ToTensor()])
        return train_tf, test_tf

    raise ValueError(f"Transforms no definidos para: {name}")

def make_datasets(cfg: DataConfig):
    train_tf, test_tf = make_transforms(cfg.name)

    if cfg.name == "cifar10":
        train_ds = torchvision.datasets.CIFAR10(cfg.root, train=True, download=cfg.download, transform=train_tf)
        test_ds  = torchvision.datasets.CIFAR10(cfg.root, train=False, download=cfg.download, transform=test_tf)

    elif cfg.name == "cifar100":
        train_ds = torchvision.datasets.CIFAR100(cfg.root, train=True, download=cfg.download, transform=train_tf)
        test_ds  = torchvision.datasets.CIFAR100(cfg.root, train=False, download=cfg.download, transform=test_tf)

    elif cfg.name == "mnist":
        train_ds = torchvision.datasets.MNIST(cfg.root, train=True, download=cfg.download, transform=train_tf)
        test_ds  = torchvision.datasets.MNIST(cfg.root, train=False, download=cfg.download, transform=test_tf)

    elif cfg.name == "fashionmnist":
        train_ds = torchvision.datasets.FashionMNIST(cfg.root, train=True, download=cfg.download, transform=train_tf)
        test_ds  = torchvision.datasets.FashionMNIST(cfg.root, train=False, download=cfg.download, transform=test_tf)

    elif cfg.name == "svhn":
        train_ds = torchvision.datasets.SVHN(cfg.root, split="train", download=cfg.download, transform=train_tf)
        test_ds  = torchvision.datasets.SVHN(cfg.root, split="test", download=cfg.download, transform=test_tf)

    else:
        raise ValueError(f"Dataset no soportado: {cfg.name}")

    mean, std, num_classes = dataset_stats(cfg.name)
    return train_ds, test_ds, mean, std, num_classes

def make_loaders(cfg: DataConfig):
    train_ds, test_ds, mean, std, num_classes = make_datasets(cfg)
    kwargs = {"num_workers": cfg.num_workers, "pin_memory": cfg.use_cuda} if cfg.use_cuda else {}

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, **kwargs)
    test_loader  = DataLoader(test_ds, batch_size=cfg.test_batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader, mean, std, num_classes
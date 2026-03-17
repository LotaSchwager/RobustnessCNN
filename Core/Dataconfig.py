# Core/Dataconfig.py
import os
from dataclasses import dataclass
from typing import Literal, Tuple

import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

DatasetName = Literal["cifar10", "cifar100", "mnist", "fashionmnist", "svhn"]

@dataclass
class DataConfig:
    """
    Configuración del dataset.

    Estructura de carpetas esperada bajo `root` (por defecto "./data"):
        data/
          cifar-10/
            cifar-10-batches-py/   ← extraído de torchvision
          cifar-100/
            meta, test, train      ← archivos del dataset CIFAR-100

    Los datasets deben estar en local (download=False por defecto).
    """
    name:            DatasetName
    root:            str  = "./data"   # Directorio BASE (no la carpeta del dataset)
    batch_size:      int  = 128
    test_batch_size: int  = 256
    num_workers:     int  = 4
    use_cuda:        bool = True
    download:        bool = False      # False: solo local; no requiere internet


# =============================================================================
# Helpers internos
# =============================================================================

def dataset_stats(name: str) -> Tuple[Tuple[float, ...], Tuple[float, ...], int]:
    """Devuelve (mean, std, num_classes). Valores estándar de la literatura."""
    if name == "cifar10":
        return (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616), 10
    if name == "cifar100":
        return (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761), 100
    if name in ("mnist", "fashionmnist"):
        return (0.1307,), (0.3081,), 10
    if name == "svhn":
        return (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970), 10
    raise ValueError(f"Dataset no soportado: {name}")


def _resolve_dataset_root(base_root: str, name: str) -> str:
    """
    Devuelve el directorio `root` que torchvision debe recibir para cada
    dataset, teniendo en cuenta la estructura de carpetas local.

    CIFAR-10:
        torchvision espera: {root}/cifar-10-batches-py/
        nuestros datos   : {base}/cifar-10/cifar-10-batches-py/
        → root = {base}/cifar-10

    CIFAR-100:
        torchvision espera: {root}/cifar-100-python/{meta,test,train}
        nuestros datos   : {base}/cifar-100/{meta,test,train}   (sin subfolder)
        → se crea un symlink {base}/cifar-100-python → cifar-100   (una sola vez)
        → root = {base}

    Resto de datasets: se asume que ya están bajo {base} en la forma
    estándar de torchvision.
    """
    if name == "cifar10":
        candidate = os.path.join(base_root, "cifar-10")
        if os.path.isdir(candidate):
            return candidate          # → {base}/cifar-10/cifar-10-batches-py/ ✓
        # Fallback: el usuario puede haber dejado los datos directamente en base
        return base_root

    if name == "cifar100":
        # Destino que torchvision buscará: {base}/cifar-100-python/
        target_link = os.path.join(base_root, "cifar-100-python")
        source_dir  = os.path.join(base_root, "cifar-100")

        if not os.path.exists(target_link) and os.path.isdir(source_dir):
            # Crea un symlink relativo: data/cifar-100-python -> cifar-100
            # (relativo para que funcione aunque se mueva el proyecto)
            rel_source = os.path.relpath(source_dir, base_root)
            os.symlink(rel_source, target_link)
            print(
                f"[DATA] Symlink creado: {target_link} → {rel_source}  "
                f"(requerido por torchvision CIFAR-100)"
            )
        return base_root              # → {base}/cifar-100-python/ ✓ (via symlink)

    # mnist, fashionmnist, svhn, etc.: estructura estándar de torchvision
    return base_root


def make_transforms(name: str):
    """Transforms sin Normalize (Normalize se aplica dentro de NormalizeLayer)."""
    if name in ("cifar10", "cifar100"):
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_tf = transforms.Compose([transforms.ToTensor()])
        return train_tf, test_tf

    if name in ("mnist", "fashionmnist"):
        tf = transforms.Compose([transforms.ToTensor()])
        return tf, tf

    if name == "svhn":
        tf = transforms.Compose([transforms.ToTensor()])
        return tf, tf

    raise ValueError(f"Transforms no definidos para: {name}")


def make_datasets(cfg: DataConfig):
    train_tf, test_tf = make_transforms(cfg.name)
    root = _resolve_dataset_root(cfg.root, cfg.name)

    if cfg.name == "cifar10":
        train_ds = torchvision.datasets.CIFAR10(
            root, train=True,  download=cfg.download, transform=train_tf)
        test_ds  = torchvision.datasets.CIFAR10(
            root, train=False, download=cfg.download, transform=test_tf)

    elif cfg.name == "cifar100":
        train_ds = torchvision.datasets.CIFAR100(
            root, train=True,  download=cfg.download, transform=train_tf)
        test_ds  = torchvision.datasets.CIFAR100(
            root, train=False, download=cfg.download, transform=test_tf)

    elif cfg.name == "mnist":
        train_ds = torchvision.datasets.MNIST(
            root, train=True,  download=cfg.download, transform=train_tf)
        test_ds  = torchvision.datasets.MNIST(
            root, train=False, download=cfg.download, transform=test_tf)

    elif cfg.name == "fashionmnist":
        train_ds = torchvision.datasets.FashionMNIST(
            root, train=True,  download=cfg.download, transform=train_tf)
        test_ds  = torchvision.datasets.FashionMNIST(
            root, train=False, download=cfg.download, transform=test_tf)

    elif cfg.name == "svhn":
        train_ds = torchvision.datasets.SVHN(
            root, split="train", download=cfg.download, transform=train_tf)
        test_ds  = torchvision.datasets.SVHN(
            root, split="test",  download=cfg.download, transform=test_tf)

    else:
        raise ValueError(f"Dataset no soportado: {cfg.name}")

    mean, std, num_classes = dataset_stats(cfg.name)
    return train_ds, test_ds, mean, std, num_classes


def make_loaders(cfg: DataConfig):
    train_ds, test_ds, mean, std, num_classes = make_datasets(cfg)

    kwargs = (
        {"num_workers": cfg.num_workers, "pin_memory": True}
        if cfg.use_cuda
        else {}
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, **kwargs)
    test_loader  = DataLoader(
        test_ds,  batch_size=cfg.test_batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader, mean, std, num_classes
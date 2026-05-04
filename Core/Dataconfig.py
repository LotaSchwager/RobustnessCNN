# Core/Dataconfig.py
import os
from dataclasses import dataclass
from typing import Literal, Tuple

import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset

DatasetName = Literal["cifar10", "cifar100", "mnist", "fashionmnist", "svhn", "riawelc"]

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
          RIAWELC/
            Dataset_partitioned/
              DB - Copy/
                training/   ← Difetto1, Difetto2, Difetto4, NoDifetto
                testing/    ← idem
                validation/ ← idem

    Los datasets deben estar en local (download=False por defecto).

    use_randaugment: Si True, añade RandAugment al pipeline de entrenamiento.
        RandAugment (Cubuk et al., 2020) aplica N transformaciones aleatorias
        de una lista predefinida (rotación, brillo, contraste, etc.) con
        magnitud M. En defensa adversarial, la literatura muestra que el
        augmentation agresivo mejora tanto la acc natural como la robustez
        (Rebuffi et al., 2021; Gowal et al., 2021). Se activa por defecto.
        Para desactivarlo, pasar use_randaugment=False en DataConfig.

    randaugment_n: Número de transformaciones a aplicar por imagen (default 2).
    randaugment_m: Magnitud de cada transformación, rango [0, 30] (default 9).
        Valores más altos = augmentation más agresivo. 9 es un valor moderado
        que da beneficio sin degradar la acc natural en CIFAR-10.

    riawelc_img_size: Tamaño de imagen para RIAWELC (default 224). Las
        imágenes originales son 227×227 grayscale; se redimensionan a 224
        para compatibilidad con arquitecturas estándar.
    """
    name:             DatasetName
    root:             str  = "./data"
    batch_size:       int  = 128
    test_batch_size:  int  = 256
    num_workers:      int  = 10
    use_cuda:         bool = True
    download:         bool = False
    use_randaugment:  bool = True    # activo por defecto
    randaugment_n:    int  = 2       # número de transformaciones
    randaugment_m:    int  = 9       # magnitud de cada transformación
    riawelc_img_size: int  = 224     # tamaño final de imagen para RIAWELC


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
    if name == "riawelc":
        # RIAWELC: 24,407 imágenes radiográficas de soldaduras, 227×227, grayscale.
        # 4 clases: Difetto1 (LP), Difetto2 (PO), Difetto4 (CR), NoDifetto (ND).
        # Como las imágenes se convierten a 3 canales (Grayscale→RGB), los 3
        # canales son idénticos. Valores aproximados para imágenes radiográficas
        # 8-bit; se recomienda recalcular con compute_dataset_stats() si se
        # requiere precisión fina.
        return (0.5, 0.5, 0.5), (0.25, 0.25, 0.25), 4
    raise ValueError(f"Dataset no soportado: {name}")


def make_transforms(cfg: DataConfig):
    """
    Construye los transforms de entrenamiento y test.

    La normalización NO se incluye aquí — se aplica dentro de NormalizeLayer
    encapsulada en el modelo, de modo que las perturbaciones adversariales
    también pasan por ella correctamente.

    Para CIFAR-10/100 el pipeline de entrenamiento es:
        RandomCrop(32, padding=4)    → variación espacial básica
        RandomHorizontalFlip()       → invarianza a espejo horizontal
        RandAugment(n, m)            → augmentation adicional si está activo
        ToTensor()                   → convierte a [0,1]

    Para RIAWELC (227×227 grayscale) el pipeline es:
        Grayscale(3)                 → convierte 1 canal a 3 canales (RGB)
        Resize(256)                  → escalado previo
        RandomCrop(224)              → recorte aleatorio (train)
        CenterCrop(224)              → recorte centrado (test)
        RandomHorizontalFlip()       → augmentation geométrico (train)
        RandomVerticalFlip()         → idem (train): radioía sin orientación canónica
        RandomRotation(10)           → ±10° para variabilidad de posición del defecto
        RandAugment(n, m)            → augmentation adicional si activo (train)
        ToTensor()                   → convierte a [0,1]

    RandAugment se coloca después de las transformaciones geométricas básicas
    y antes de ToTensor, que es el orden estándar en la literatura.
    """
    name = cfg.name

    if name in ("cifar10", "cifar100"):
        train_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        if cfg.use_randaugment:
            # RandAugment disponible desde torchvision 0.11
            train_list.append(
                transforms.RandAugment(
                    num_ops    = cfg.randaugment_n,
                    magnitude  = cfg.randaugment_m,
                )
            )
        train_list.append(transforms.ToTensor())
        train_tf = transforms.Compose(train_list)
        test_tf  = transforms.Compose([transforms.ToTensor()])
        return train_tf, test_tf

    if name in ("mnist", "fashionmnist"):
        tf = transforms.Compose([transforms.ToTensor()])
        return tf, tf

    if name == "svhn":
        tf = transforms.Compose([transforms.ToTensor()])
        return tf, tf

    if name == "riawelc":
        img_size = cfg.riawelc_img_size  # 224 por defecto

        # Transformaciones base comunes: grayscale→3ch + resize
        base_list = [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(img_size + 32),    # 256 para RandomCrop(224)
        ]

        # Train: augmentation estándar estilo ImageNet + variabilidad específica
        # de radioía de soldaduras (sin orientación canónica vertical/horizontal)
        train_list = base_list + [
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=10),
        ]
        if cfg.use_randaugment:
            train_list.append(
                transforms.RandAugment(
                    num_ops    = cfg.randaugment_n,
                    magnitude  = cfg.randaugment_m,
                )
            )
        train_list.append(transforms.ToTensor())
        train_tf = transforms.Compose(train_list)

        # Test: resize + center crop determinista
        test_tf = transforms.Compose(
            base_list + [
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
            ]
        )
        return train_tf, test_tf

    raise ValueError(f"Transforms no definidos para: {name}")


def make_datasets(cfg: DataConfig):
    train_tf, test_tf = make_transforms(cfg)
    root = cfg.root

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

    elif cfg.name == "riawelc":
        # RIAWELC usa ImageFolder con la estructura extraída del RAR:
        #   data/RIAWELC/Dataset_partitioned/DB - Copy/{training,testing,validation}/
        #     {Difetto1, Difetto2, Difetto4, NoDifetto}/
        #
        # Se combinan training + validation para maximizar datos de entrenamiento,
        # y se usa testing como test set (consistente con el resto del proyecto
        # que solo maneja train/test).
        riawelc_root = os.path.join(root, "RIAWELC", "Dataset_partitioned", "DB - Copy")

        train_dir = os.path.join(riawelc_root, "training")
        test_dir  = os.path.join(riawelc_root, "testing")
        val_dir   = os.path.join(riawelc_root, "validation")

        if not os.path.isdir(train_dir):
            raise FileNotFoundError(
                f"[RIAWELC] No se encontró el directorio de entrenamiento:\n"
                f"  {train_dir}\n"
                f"Asegúrate de haber extraído el RAR en "
                f"data/RIAWELC/Dataset_partitioned/"
            )

        train_ds_main = torchvision.datasets.ImageFolder(train_dir, transform=train_tf)
        test_ds       = torchvision.datasets.ImageFolder(test_dir,  transform=test_tf)

        # Combinar training + validation si existe
        if os.path.isdir(val_dir):
            val_ds   = torchvision.datasets.ImageFolder(val_dir, transform=train_tf)
            train_ds = ConcatDataset([train_ds_main, val_ds])
            print(f"[RIAWELC] training ({len(train_ds_main)}) + "
                  f"validation ({len(val_ds)}) = {len(train_ds)} imágenes de entrenamiento")
        else:
            train_ds = train_ds_main
            print(f"[RIAWELC] {len(train_ds)} imágenes de entrenamiento")

        print(f"[RIAWELC] {len(test_ds)} imágenes de test")
        print(f"[RIAWELC] Clases: {train_ds_main.classes}")

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
        train_ds, batch_size=cfg.batch_size, shuffle=True,  **kwargs)
    test_loader  = DataLoader(
        test_ds,  batch_size=cfg.test_batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader, mean, std, num_classes

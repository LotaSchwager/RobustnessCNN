from .resnet import ResNet18, ResNet50
from .vgg import VGG
from .wideresnet import WideResNet, WideResNet16_8, WideResNet28_10, WideResNet34_10


def get_model(name: str, num_classes: int = 10, img_size: int = 32,
              dropout_rate: float = 0.0):
    """
    Fábrica de modelos centralizada.

    Parámetros
    ----------
    name         : nombre del modelo (ver abajo).
    num_classes  : número de clases de salida.
    img_size     : tamaño de la imagen de entrada (lado en px).
                   Controla el stem del modelo:
                   • ≤32 → stem CIFAR (conv 3×3/stride-1)
                   • >32 → stem ImageNet (conv 7×7/stride-2 + MaxPool)
                   Para RIAWELC usar img_size=224.
    dropout_rate : tasa de dropout para WideResNet (0.0 = sin dropout).
                   Se recomienda 0.3 para RIAWELC.

    Modelos soportados:
      • resnet18, resnet50
      • vgg10, vgg11, vgg13, vgg16, vgg19
      • wideresnet / wrn                    → WRN-28-10 (default)
      • wideresnet-D-K / wrn-D-K            → WRN-D-K  (ej: wrn-34-10)
        - separadores admitidos: '-' o '_'
        - alias directos: wrn28_10, wrn34_10, wrn16_8
    """
    name_lower = name.lower()

    # --- ResNet ---
    if name_lower == 'resnet18':
        return ResNet18(num_classes=num_classes, img_size=img_size)
    if name_lower == 'resnet50':
        return ResNet50(num_classes=num_classes, img_size=img_size)

    # --- VGG ---
    if name_lower.startswith('vgg'):
        vgg_type = name.upper()
        return VGG(vgg_type, num_classes=num_classes)

    # --- WideResNet ---
    if name_lower in ('wideresnet', 'wrn'):
        return WideResNet28_10(num_classes=num_classes, img_size=img_size,
                               dropout_rate=dropout_rate)
    if name_lower.startswith('wideresnet') or name_lower.startswith('wrn'):
        parts = name_lower.replace('-', '_').split('_')
        nums  = [p for p in parts if p.isdigit()]
        if len(nums) == 2:
            depth, widen = int(nums[0]), int(nums[1])
            return WideResNet(depth=depth, widen_factor=widen,
                              num_classes=num_classes, img_size=img_size,
                              dropout_rate=dropout_rate)
        if len(nums) == 1:
            return WideResNet(depth=int(nums[0]), widen_factor=10,
                              num_classes=num_classes, img_size=img_size,
                              dropout_rate=dropout_rate)
        return WideResNet28_10(num_classes=num_classes, img_size=img_size,
                               dropout_rate=dropout_rate)

    raise ValueError(f"Modelo '{name}' no reconocido en la fábrica de Models. "
                     f"Asegúrate de registrarlo en Models/__init__.py")

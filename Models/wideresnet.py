"""
WideResNet (Zagoruyko & Komodakis, 2016).

Es la arquitectura de referencia usada por Madry et al. (2018) y Wang et al.
(MART, 2020) para entrenamiento adversarial.

Variantes provistas:
  • WRN-16-8   →  ~11M parámetros
  • WRN-28-10  →  ~36M parámetros, estándar para CIFAR-10/100
  • WRN-34-10  →  ~46M parámetros, estándar para entrenamiento adversarial

Soporte de tamaño de imagen (parámetro img_size):
  • img_size <= 32 (CIFAR):   stem 3×3/stride-1, sin MaxPool.
      32 → 32 (block1) → 16 (block2) → 8 (block3) → 1 (AdaptivePool)
  • img_size > 32 (RIAWELC):  stem 7×7/stride-2 + MaxPool 3×3/stride-2.
      224 → 112 → 56 (block1) → 28 (block2) → 14 (block3) → 1 (AdaptivePool)
      Esto replica el stem ImageNet estándar y evita que block1 opere
      sobre mapas de 224×224, reduciendo memoria y tiempo de cómputo ~16×.

Dropout (parámetro dropout_rate):
  • 0.0 (default) para CIFAR. Para RIAWELC se recomienda 0.3 (más
    complejo, mayor riesgo de sobreajuste).

Construcción genérica con WideResNet(depth=d, widen_factor=k, ...) requiere
que (d - 4) sea múltiplo de 6.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _BasicBlock(nn.Module):
    """Bloque preactivado BN-ReLU-Conv-BN-ReLU-Conv con shortcut."""

    def __init__(self, in_planes, out_planes, stride, dropout_rate=0.0):
        super().__init__()
        self.bn1   = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None

        self.equal_in_out = (in_planes == out_planes and stride == 1)
        if not self.equal_in_out:
            self.conv_shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                           stride=stride, padding=0, bias=False)
        else:
            self.conv_shortcut = None

    def forward(self, x):
        if not self.equal_in_out:
            x   = F.relu(self.bn1(x), inplace=True)
            out = self.conv1(x)
        else:
            out = F.relu(self.bn1(x), inplace=True)
            out = self.conv1(out)

        out = F.relu(self.bn2(out), inplace=True)
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.conv2(out)

        shortcut = x if self.equal_in_out else self.conv_shortcut(x)
        return out + shortcut


class _NetworkBlock(nn.Module):
    """Pila de N bloques residuales del mismo ancho."""

    def __init__(self, num_layers, in_planes, out_planes, stride, dropout_rate=0.0):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(
                _BasicBlock(
                    in_planes  = in_planes if i == 0 else out_planes,
                    out_planes = out_planes,
                    stride     = stride if i == 0 else 1,
                    dropout_rate = dropout_rate,
                )
            )
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    """
    WideResNet genérico con soporte para CIFAR (32×32) y datasets grandes
    como RIAWELC (224×224).

    Notación WRN-d-k:
      depth        = d
      widen_factor = k

    Restricción: (depth - 4) % 6 == 0

    Para img_size > 32 se usa un stem tipo ImageNet (7×7/stride-2 + MaxPool)
    que evita que block1 procese feature maps del tamaño de la imagen original.
    """

    def __init__(self, depth=28, widen_factor=10, dropout_rate=0.0,
                 num_classes=10, img_size=32):
        super().__init__()
        assert (depth - 4) % 6 == 0, \
            f"WideResNet: (depth - 4) debe ser múltiplo de 6. Se recibió depth={depth}."

        n = (depth - 4) // 6
        k = widen_factor
        n_channels = [16, 16 * k, 32 * k, 64 * k]

        # ── Stem: diseño dependiente del tamaño de entrada ─────────────────────
        # CIFAR (≤32px): conv 3×3/stride-1 — preserva la resolución pequeña.
        # ImageNet/RIAWELC (>32px): conv 7×7/stride-2 + MaxPool 3×3/stride-2
        #   para reducir agresivamente la resolución antes de los bloques
        #   anchos, replicando el stem estándar de ImageNet.
        if img_size > 32:
            self.conv1 = nn.Conv2d(3, n_channels[0], kernel_size=7,
                                   stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv2d(3, n_channels[0], kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.maxpool = None

        self.block1 = _NetworkBlock(n, n_channels[0], n_channels[1], stride=1,
                                    dropout_rate=dropout_rate)
        self.block2 = _NetworkBlock(n, n_channels[1], n_channels[2], stride=2,
                                    dropout_rate=dropout_rate)
        self.block3 = _NetworkBlock(n, n_channels[2], n_channels[3], stride=2,
                                    dropout_rate=dropout_rate)

        self.bn1 = nn.BatchNorm2d(n_channels[3])
        self.fc  = nn.Linear(n_channels[3], num_classes)

        self.n_channels = n_channels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.conv1(x)
        if self.maxpool is not None:
            out = self.maxpool(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = F.relu(self.bn1(out), inplace=True)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.n_channels)
        return self.fc(out)


def WideResNet16_8(num_classes=10, dropout_rate=0.0, img_size=32):
    return WideResNet(depth=16, widen_factor=8, dropout_rate=dropout_rate,
                      num_classes=num_classes, img_size=img_size)

def WideResNet28_10(num_classes=10, dropout_rate=0.0, img_size=32):
    return WideResNet(depth=28, widen_factor=10, dropout_rate=dropout_rate,
                      num_classes=num_classes, img_size=img_size)

def WideResNet34_10(num_classes=10, dropout_rate=0.0, img_size=32):
    return WideResNet(depth=34, widen_factor=10, dropout_rate=dropout_rate,
                      num_classes=num_classes, img_size=img_size)


def test():
    for builder, name in [
        (WideResNet16_8,  "WRN-16-8"),
        (WideResNet28_10, "WRN-28-10"),
        (WideResNet34_10, "WRN-34-10"),
    ]:
        # Test CIFAR (32×32)
        net = builder(num_classes=10, img_size=32)
        y   = net(torch.randn(2, 3, 32, 32))
        n_p = sum(p.numel() for p in net.parameters())
        print(f"{name:>10} [32×32]   out={tuple(y.shape)}  params={n_p/1e6:.2f}M")

        # Test RIAWELC (224×224)
        net = builder(num_classes=4, img_size=224, dropout_rate=0.3)
        y   = net(torch.randn(2, 3, 224, 224))
        print(f"{name:>10} [224×224] out={tuple(y.shape)}")

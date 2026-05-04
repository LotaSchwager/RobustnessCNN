import torch
import torch.nn as nn
import torch.nn.functional as F

# Clase BasicBlock
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# Clase Bottleneck
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# Clase ResNet
class ResNet(nn.Module):
    """
    ResNet con soporte para CIFAR (32×32) y datasets grandes como RIAWELC (224×224).

    Stem según img_size:
      • img_size <= 32 (CIFAR):    conv 3×3/stride-1 (preserva resolución pequeña)
          32 → 32 (layer1) → 16 (layer2) → 8 (layer3) → 4 (layer4) → 1 (pool)
      • img_size >  32 (RIAWELC): conv 7×7/stride-2 + MaxPool 3×3/stride-2
          224 → 112 → 56 (layer1) → 28 (layer2) → 14 (layer3) → 7 (layer4) → 1 (pool)
    """
    def __init__(self, block, num_blocks, num_classes=10, img_size=32):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # ── Stem dependiente del tamaño de entrada ─────────────────────────────
        if img_size > 32:
            # Stem ImageNet: reduce 224→56 antes de los bloques residuales,
            # evitando operar con feature maps enormes en layer1.
            self.conv1   = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            # Stem CIFAR: mantiene la resolución para imágenes pequeñas.
            self.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.maxpool = None

        self.bn1    = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.maxpool is not None:
            out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# Funciones de creación de Redes
def ResNet18(num_classes=10, img_size=32):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, img_size=img_size)

def ResNet50(num_classes=10, img_size=32):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, img_size=img_size)

def test():
    for img_size, nc in [(32, 10), (224, 4)]:
        net = ResNet18(num_classes=nc, img_size=img_size)
        y = net(torch.randn(1, 3, img_size, img_size))
        print(f"ResNet18 [{img_size}×{img_size}]: output={tuple(y.shape)}")
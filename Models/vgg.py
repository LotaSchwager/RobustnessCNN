import torch
import torch.nn as nn

cfg = {
    'VGG10': [32, 'M', 64, 'M', 128, 128, 'M'],  
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

# Clase VGG
class VGG(nn.Module):
    def __init__(self, vgg_name, in_channels=3, num_classes=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name], in_channels)
        self.classifier = nn.Linear(512, num_classes)


    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, in_channels):
        layers = []

        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AdaptiveAvgPool2d((1, 1))]
        return nn.Sequential(*layers)

def vgg10(in_channels=3):
    return VGG('VGG10', in_channels)

def vgg11(in_channels=3):
    return VGG('VGG11', in_channels)

def vgg13(in_channels=3):
    return VGG('VGG13', in_channels)

def vgg16(in_channels=3):
    return VGG('VGG16', in_channels)

def vgg19(in_channels=3):
    return VGG('VGG19', in_channels)

# Función de prueba
def test():
    net = vgg11(in_channels=1)   
    x = torch.randn(2, 1, 32, 32)
    y = net(x)
    print(y.size())  
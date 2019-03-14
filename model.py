import torch.nn as nn


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, stride):
        super(Bottleneck, self).__init__()
        self.use_residual = stride == 1 and in_channels == out_channels

        hidden_channels = expansion * in_channels

        self.conv = nn.Sequential(
            # pointwise-conv
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU6(inplace=True),
            # depthwise-conv
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=stride, padding=1, groups=hidden_channels,
                      bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU6(inplace=True),
            # pointwise-conv linear
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )


    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


def _make_bottlenecks(in_channels, architecture):
    layers = []

    for configs in architecture:
        strides = [configs.stride] + [1] * (configs.num_blocks - 1)

        for stride in strides:
            layers.append(Bottleneck(in_channels, configs.out_channels, configs.expansion, stride))
            in_channels = configs.out_channels

    return nn.Sequential(*layers)


class MobileNetV2(nn.Module):
    def __init__(self, architecture, in_channels, num_classes):
        super(MobileNetV2, self).__init__()

        self.relu = nn.ReLU6(inplace=True)

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.bottlenecks = _make_bottlenecks(32, architecture)

        self.conv2 = nn.Conv2d(architecture[-1].out_channels, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.2, inplace=True)
        self.fc = nn.ModuleList([nn.Linear(1280, c) for c in num_classes])


    def forward(self, x, task=0):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.bottlenecks(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.avg_pool(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)
        x = self.fc[task](x)

        return x


"""
class SmallMobileNetV2(nn.Module):
    def __init__(self, architecture, in_channels, num_classes):
        super(SmallMobileNetV2, self).__init__()

        self.relu = nn.ReLU6(inplace=True)

        self.bottlenecks = _make_bottlenecks(in_channels, architecture)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.2, inplace=True)
        self.fc = nn.ModuleList([nn.Linear(architecture[-1].out_channels, c) for c in num_classes])


    def forward(self, x, task=0):
        x = self.bottlenecks(x)

        x = self.relu(x)

        x = self.avg_pool(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)
        x = self.fc[task](x)

        return x
"""

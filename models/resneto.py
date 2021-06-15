from torch.nn import AvgPool2d, BatchNorm2d, Linear, Module, ReLU, Sequential

# from ._container import Classifier, make_conv_pool_activ
from typing import Callable, Optional

import torch
from torch.nn import Conv2d, MaxPool2d, Module, Sequential, Softmax

ActivT = Optional[Callable[[], Module]]


def make_conv_pool_activ(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    activation: ActivT = None,
    pool_size: Optional[int] = None,
    pool_stride: Optional[int] = None,
    **conv_kwargs
):
    layers = [Conv2d(in_channels, out_channels, kernel_size, **conv_kwargs)]
    if activation:
        layers.append(activation())
    if pool_size is not None:
        layers.append(MaxPool2d(pool_size, stride=pool_stride))
    return layers


class Classifier(Module):
    def __init__(
        self, convs: Sequential, linears: Sequential, use_softmax: bool = True
    ):
        super().__init__()
        self.convs = convs
        self.linears = linears
        self.softmax = Softmax(1) if use_softmax else Sequential()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.convs(inputs)
        return self.softmax(self.linears(outputs.view(outputs.shape[0], -1)))


class BasicBlock(Module):
    def __init__(self, ins, outs, shortcut=False):
        super().__init__()
        stride = 2 if shortcut else 1
        self.mainline = Sequential(
            *make_conv_pool_activ(ins, outs, 3, ReLU, padding=1, stride=stride),
            *make_conv_pool_activ(outs, outs, 3, padding=1)
        )
        self.relu1 = ReLU()
        self.shortcut = (
            Sequential(*make_conv_pool_activ(ins, outs, 1, stride=stride))
            if shortcut
            else Sequential()
        )

    def forward(self, input_):
        return self.relu1(self.mainline(input_) + self.shortcut(input_))


class ResNet18O(Classifier):
    def __init__(self):
        convs = Sequential(
            *make_conv_pool_activ(3, 16, 3, ReLU, padding=1),
            BasicBlock(16, 16),
            BasicBlock(16, 16),
            BasicBlock(16, 16),
            BasicBlock(16, 32, True),
            BasicBlock(32, 32),
            BasicBlock(32, 32),
            BasicBlock(32, 64, True),
            BasicBlock(64, 64),
            BasicBlock(64, 64),
            AvgPool2d(8)
        )
        linears = Sequential(Linear(64, 10))
        super().__init__(convs, linears)


class Bottleneck(Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.mainline = Sequential(
            *make_conv_pool_activ(in_planes, planes, 1, stride=stride),
            BatchNorm2d(planes, eps=0.001),
            ReLU(),
            *make_conv_pool_activ(planes, planes, 3, padding=1),
            BatchNorm2d(planes, eps=0.001),
            ReLU(),
            *make_conv_pool_activ(planes, self.expansion * planes, 1),
            BatchNorm2d(self.expansion * planes, eps=0.001)
        )
        self.relu1 = ReLU()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = Sequential(
                *make_conv_pool_activ(
                    in_planes, self.expansion * planes, 1, stride=stride
                ),
                BatchNorm2d(self.expansion * planes, eps=0.001)
            )
        else:
            self.shortcut = Sequential()

    def forward(self, input_):
        return self.relu1(self.mainline(input_) + self.shortcut(input_))


class ResNet50(Classifier):
    def __init__(self):
        convs = Sequential(
            *make_conv_pool_activ(
                3, 64, 7, ReLU, pool_size=3, pool_stride=2, padding=3, stride=2
            ),
            BatchNorm2d(64, eps=0.001),
            Bottleneck(64, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 128, stride=2),
            Bottleneck(512, 128),
            Bottleneck(512, 128),
            Bottleneck(512, 128),
            Bottleneck(512, 256, stride=2),
            Bottleneck(1024, 256),
            Bottleneck(1024, 256),
            Bottleneck(1024, 256),
            Bottleneck(1024, 256),
            Bottleneck(1024, 256),
            Bottleneck(1024, 512, stride=2),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512),
            AvgPool2d(7)
        )
        linears = Sequential(Linear(2048, 1000))
        super().__init__(convs, linears)

from torch.nn import Linear, ReLU, Sequential, Tanh
from typing import Callable, Optional

import torch
from torch.nn import Conv2d, MaxPool2d, Module, Sequential, Softmax

# from ._container import Classifier, make_conv_pool_activ


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
        self.quant = torch.quantization.QuantStub()
        self.convs = convs
        self.linears = linears
        self.softmax = Softmax(1) if use_softmax else Sequential()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.quant(inputs) 
        outputs = self.convs(inputs)
        
        # print(outputs.shape)

        # outputs = self.softmax(self.linears(outputs.reshape(outputs.shape[0], -1)))
        outputs = self.linears(outputs.reshape(outputs.shape[0], -1))
        # print('After softmax :', outputs.shape)

        outputs = self.dequant(outputs)
        return outputs

class AlexNet(Classifier):
    def __init__(self):
        convs = Sequential(
            *make_conv_pool_activ(3, 64, 11, Tanh, pool_size=2, padding=5),
            *make_conv_pool_activ(64, 192, 5, Tanh, pool_size=2, padding=2),
            *make_conv_pool_activ(192, 384, 3, Tanh, padding=1),
            *make_conv_pool_activ(384, 256, 3, Tanh, padding=1),
            *make_conv_pool_activ(256, 256, 3, Tanh, pool_size=2, padding=1)
        )
        linears = Sequential(Linear(4096, 10))
        super().__init__(convs, linears)


class AlexNet2(Classifier):
    def __init__(self):
        convs = Sequential(
            *make_conv_pool_activ(3, 32, 3, Tanh, padding=1),
            *make_conv_pool_activ(32, 32, 3, Tanh, pool_size=2, padding=1),
            *make_conv_pool_activ(32, 64, 3, Tanh, padding=1),
            *make_conv_pool_activ(64, 64, 3, Tanh, pool_size=2, padding=1),
            *make_conv_pool_activ(64, 128, 3, Tanh, padding=1),
            *make_conv_pool_activ(128, 128, 3, Tanh, pool_size=2, padding=1)
        )
        linears = Sequential(Linear(2048, 10))
        super().__init__(convs, linears)


class AlexNetImageNet(Classifier):
    def __init__(self):
        convs = Sequential(
            *make_conv_pool_activ(
                3, 64, 11, ReLU, padding=2, stride=4, pool_size=3, pool_stride=2
            ),
            *make_conv_pool_activ(
                64, 192, 5, ReLU, padding=2, pool_size=3, pool_stride=2
            ),
            *make_conv_pool_activ(192, 384, 3, ReLU, padding=1),
            *make_conv_pool_activ(384, 256, 3, ReLU, padding=1),
            *make_conv_pool_activ(
                256, 256, 3, ReLU, padding=1, pool_size=3, pool_stride=2
            )
        )
        linears = Sequential(
            Linear(9216, 4096), ReLU(), Linear(4096, 4096), ReLU(), Linear(4096, 1000),
        )
        super().__init__(convs, linears)


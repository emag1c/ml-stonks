import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, TYPE_CHECKING, Union, List, Callable, Sequence
from collections import OrderedDict
from functools import partial

if TYPE_CHECKING:
    from torch import tensor


class Initializer:
    def init_module(self, module: nn.Module):
        pass

    def __call__(self, module: nn.Module):
        self.init_module(self)


class KaimingInitializer(Initializer):

    def __init__(self, a=0, mode='fan_in', nonlinearity='leaky_relu'):
        self._kaiming = partial(nn.init.kaiming_normal_, a=a, mode=mode, nonlinearity=nonlinearity)

    def init_module(self, module: nn.Module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                self._kaiming(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)


class XavierInitializer(Initializer):

    def init_module(self, module: nn.Module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                # use xavier normal
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)


def _make_conv_sequence(input_features: int,
                        output_features: int,
                        kernel_size=1,
                        stride=1,
                        bias=False,
                        batch_norm=True,
                        activation: nn.Module = nn.ReLU) -> nn.Sequential:
    layers = OrderedDict()
    if batch_norm is True:
        layers['bn1'] = nn.BatchNorm2d(input_features)
    layers[activation.__name__ + '1'] = activation()
    layers['conv1'] = nn.Conv2d(input_features, output_features, kernel_size=kernel_size, stride=stride, bias=bias)

    return nn.Sequential(layers)


def _param_at_index(v: Union[Sequence, int, Callable[[int], int]], i: int):
    if callable(v):
        return v(i)
    if hasattr(v, '__getitem__'):
        return v[i]
    return v


class DenseLayer(nn.Module):
    def __init__(self,
                 input_features: int,
                 output_features: int,
                 bn_size: int,
                 drop_rate: float,
                 batch_norm=True,
                 activation: nn.Module = nn.ReLU):
        """
        :param input_features: number of input features
        :param output_features: the growth rate ``k``
        :param bn_size: the bottle neck size
        :param drop_rate: the dropout rate
        """

        super(DenseLayer, self).__init__()

        seq1_output = output_features * bn_size
        # do not use batch norm if we're using SELU activation
        self.seq1 = _make_conv_sequence(input_features, seq1_output,
                                        kernel_size=1, stride=1,
                                        batch_norm=batch_norm, activation=activation)
        self.seq2 = _make_conv_sequence(seq1_output, output_features,
                                        kernel_size=3, stride=1,
                                        batch_norm=batch_norm, activation=activation)
        self.drop_rate = drop_rate

    def forward(self, x: Union[torch.tensor, List[torch.Tensor]]) -> torch.Tensor:
        x = [x] if isinstance(x, torch.Tensor) else x
        out = self.seq2(self.seq1(torch.cat(x, 1)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return out


class DenseBlock(nn.ModuleDict):

    def __init__(self,
                 num_layers: int,
                 input_features: int,
                 bn_size: int,
                 layer_features: int,
                 drop_rate: float,
                 batch_norm: bool = True,
                 activation: nn.Module = nn.ReLU):

        super(DenseBlock, self).__init__()

        # the number of output features is equal to the growth rate * the number of layers
        self.output_features = layer_features * num_layers

        # add all dense layers up to ``num_layers``
        for i in range(num_layers):
            layer = DenseLayer(
                input_features + i * layer_features,
                output_features=layer_features,
                bn_size=bn_size,
                drop_rate=drop_rate,
                batch_norm=batch_norm,
                activation=activation,
            )
            self.add_module(f'denselayer{i + 1}', layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for name, layer in self.items():
            features.append(layer(features))
        return torch.cat(features, 1)


class Transition(nn.Sequential):
    def __init__(self,
                 input_features: int,
                 output_features: int,
                 activation: nn.Module = nn.ReLU):
        super(Transition, self).__init__(OrderedDict([
            ('norm', nn.BatchNorm2d(input_features)),
            (activation.__name__, activation()),
            ('conv', nn.Conv2d(input_features, output_features,
                               kernel_size=1, stride=1, bias=False)),
            ('pool', nn.AvgPool2d(kernel_size=2, stride=2))
        ]))


class DenseNet(nn.Module):
    def __init__(self,
                 growth_rate: Union[Sequence, int, Callable[[int], int]] = 32,
                 block_config=(6, 12, 24, 16),
                 init_features=64,
                 bn_size: Union[Sequence, int, Callable[[int], int]] = 4,
                 drop_rate: Union[Sequence, float, Callable[[int], float]] = 0.,
                 num_classes=1000,
                 batch_norm=True,
                 initializer: Initializer = KaimingInitializer,
                 activation: nn.Module = nn.ReLU):

        super(DenseNet, self).__init__()
        self.activation = activation()

        # first convolution sequence
        self.features = nn.Sequential()
        self.features.add_module('conv0', nn.Conv2d(3, init_features, kernel_size=7, stride=2,
                                                    padding=3, bias=False))
        if batch_norm:
            self.features.add_module('norm0', nn.BatchNorm2d(init_features))
        self.features.add_module(f'{activation.__name__}0', activation())
        self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        # add the dense blocks
        input_features = init_features
        for i, num_layers in enumerate(block_config):

            block = DenseBlock(
                num_layers=num_layers,
                input_features=input_features,
                bn_size=_param_at_index(bn_size, i),
                layer_features=_param_at_index(growth_rate, i),
                drop_rate=_param_at_index(drop_rate, i),
                batch_norm=batch_norm,
                activation=activation
            )

            self.features.add_module(f'denseblock{i + 1}', block)
            input_features += block.output_features
            # all dense blocks, except the last one, get a transition block with output features 1/2 the size of input
            if i != len(block_config) - 1:
                trans = Transition(input_features=input_features,
                                   output_features=input_features // 2,
                                   activation=activation)
                self.features.add_module(f'transition{i + 1}', trans)
                input_features = input_features // 2

        # final batch norm
        if batch_norm:
            self.features.add_module(f'norm{len(block_config) + 1}', nn.BatchNorm2d(input_features))

        # linear layer
        self.classifier = nn.Linear(input_features, num_classes)

        # initialize the model with provided initializer
        initializer(self)

    def forward(self, x: torch.tensor) -> torch.tensor:
        features = self.features(x)
        out = self.activation(features)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def densenet121(**kwargs) -> DenseNet:
    return DenseNet(growth_rate=32,
                    block_config=(6, 12, 24, 17),
                    init_features=64,
                    initializer=KaimingInitializer(),
                    activation=partial(nn.ReLU, inplace=True),
                    **kwargs)


def densenet121Selu(**kwargs) -> DenseNet:
    return DenseNet(growth_rate=32,
                    block_config=(6, 12, 24, 17),
                    init_features=64,
                    initializer=XavierInitializer(),
                    batch_norm=False,
                    activation=partial(nn.SELU, inplace=True),
                    **kwargs)


def densenet121_2(**kwargs) -> DenseNet:
    return DenseNet(growth_rate=(12, 24, 32, 24, 12),
                    block_config=(6, 12, 13, 12, 6),
                    drop_rate=(0., .1, .2, .1, 0.),
                    init_features=64,
                    initializer=KaimingInitializer(),
                    activation=partial(nn.ReLU, inplace=True),
                    **kwargs)



class TwoInputDenseNet(nn.Module):
    def __init__(self,
                 growth_rate: Union[Sequence, int, Callable[[int], int]] = 32,
                 block_config=(6, 12, 24, 16),
                 init_features=64,
                 bn_size: Union[Sequence, int, Callable[[int], int]] = 4,
                 drop_rate: Union[Sequence, float, Callable[[int], float]] = 0.,
                 num_classes=1000):

        super(DenseNet, self).__init__()
        self.activation = activation()

        # first convolution sequence
        self.features = nn.Sequential()
        self.features.add_module('conv0', nn.Conv2d(3, init_features, kernel_size=7, stride=2,
                                                    padding=3, bias=False))
        if batch_norm:
            self.features.add_module('norm0', nn.BatchNorm2d(init_features))
        self.features.add_module(f'{activation.__name__}0', activation())
        self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        # add the dense blocks
        input_features = init_features
        for i, num_layers in enumerate(block_config):

            block = DenseBlock(
                num_layers=num_layers,
                input_features=input_features,
                bn_size=_param_at_index(bn_size, i),
                layer_features=_param_at_index(growth_rate, i),
                drop_rate=_param_at_index(drop_rate, i),
                batch_norm=batch_norm,
                activation=activation
            )

            self.features.add_module(f'denseblock{i + 1}', block)
            input_features += block.output_features
            # all dense blocks, except the last one, get a transition block with output features 1/2 the size of input
            if i != len(block_config) - 1:
                trans = Transition(input_features=input_features,
                                   output_features=input_features // 2,
                                   activation=activation)
                self.features.add_module(f'transition{i + 1}', trans)
                input_features = input_features // 2

        # final batch norm
        if batch_norm:
            self.features.add_module(f'norm{len(block_config) + 1}', nn.BatchNorm2d(input_features))

        # linear layer
        self.classifier = nn.Linear(input_features, num_classes)

        # initialize the model with provided initializer
        initializer(self)

    def forward(self, x: torch.tensor) -> torch.tensor:
        features = self.features(x)
        out = self.activation(features)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
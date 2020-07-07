import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, TYPE_CHECKING, Union, List, Callable, Sequence
from collections import OrderedDict

if TYPE_CHECKING:
    from torch import tensor


def _make_conv_sequence(input_features: int,
                        output_features: int,
                        kernel_size=1,
                        stride=1,
                        bias=False,
                        activator_f=nn.ReLU) -> nn.Sequential:
    return nn.Sequential(OrderedDict([
        ('bn1', nn.BatchNorm2d(input_features)),
        ('act1', activator_f()),
        ('conv1', nn.Conv2d(input_features, output_features, kernel_size=kernel_size, stride=stride, bias=bias)),
    ]))


class DenseLayer(nn.Module):
    def __init__(self,
                 input_features: int,
                 output_features: int,
                 bn_size: int,
                 drop_rate: float,
                 activator_f=nn.ReLU):
        """
        :param input_features: number of input features
        :param output_features: the growth rate ``k``
        :param bn_size: the batch norm size passed to our batch norm 2d module
        :param drop_rate: the dropout rate
        """

        super(DenseLayer, self).__init__()

        seq1_output = output_features * bn_size
        self.seq2 = _make_conv_sequence(input_features, seq1_output,
                                        kernel_size=1, stride=1, activator_f=activator_f)
        self.seq2 = _make_conv_sequence(seq1_output, output_features,
                                        kernel_size=3, stride=1, activator_f=activator_f)
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
                 activation_f=nn.ReLU):

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
                activator_f=activation_f
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
                 activation_f=nn.ReLU):
        super(Transition, self).__init__(OrderedDict([
            ('norm', nn.BatchNorm2d(input_features)),
            ('activator', activation_f()),
            ('conv', nn.Conv2d(input_features, output_features,
                               kernel_size=1, stride=1, bias=False)),
            ('pool', nn.AvgPool2d(kernel_size=2, stride=2))
        ]))


def _param_at_index(v: Union[Sequence, int, Callable[[int], int]], i: int):
    if callable(v):
        return v(i)
    if hasattr(v, '__getitem__'):
        return v[i]
    return v


class DenseNet(nn.Module):

    def __init__(self,
                 growth_rate: Union[Sequence, int, Callable[[int], int]] = 32,
                 block_config=(6, 12, 24, 16),
                 init_features=64,
                 bn_size: Union[Sequence, int, Callable[[int], int]] = 4,
                 drop_rate: Union[Sequence, float, Callable[[int], float]] = 0.,
                 num_classes=1000,
                 activation_f=nn.ReLU):

        super(DenseNet, self).__init__()

        # first convolution sequence
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # add the dense blocks
        input_features = init_features
        for i, num_layers in enumerate(block_config):

            block = DenseBlock(
                num_layers=num_layers,
                input_features=input_features,
                bn_size=_param_at_index(bn_size, i),
                layer_features=_param_at_index(growth_rate, i),
                drop_rate=_param_at_index(drop_rate, i),
                activation_f=activation_f
            )

            self.features.add_module(f'denseblock{i + 1}', block)
            input_features += block.output_features
            # all dense blocks, except the last one, get a transition block with output features 1/2 the size of input
            if i != len(block_config) - 1:
                trans = Transition(input_features=input_features,
                                   output_features=input_features // 2,
                                   activation_f=activation_f)
                self.features.add_module(f'transition{i + 1}', trans)
                input_features = input_features // 2

        # final batch norm
        self.features.add_module(f'norm{len(block_config) + 1}', nn.BatchNorm2d(input_features))

        # linear layer
        self.classifier = nn.Linear(input_features, num_classes)

        # initialize the model
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # todo: replace with better initializer from fastai lecture series!
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.tensor) -> torch.tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def densenet121(**kwargs) -> DenseNet:
    return DenseNet(growth_rate=32,
                    block_config=(6, 12, 24, 17),
                    init_features=64,
                    **kwargs)


def densenet121_2(**kwargs) -> DenseNet:
    return DenseNet(growth_rate=(12, 24, 32, 24, 12),
                    block_config=(6, 12, 13, 12, 6),
                    drop_rate=(0., .1, .2, .1, 0.),
                    init_features=64,
                    **kwargs)

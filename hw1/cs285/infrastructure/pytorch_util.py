from typing import Union

import torch
from torch import nn
from collections import OrderedDict

Activation = Union[str, nn.Module]

_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation: Activation = 'tanh',
        output_activation: Activation = 'identity',
) -> nn.Module:
    """
        Builds a feedforward neural network

        arguments:
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer

            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer

        returns:
            MLP (nn.Module)
    """
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]

    # TODO: return a MLP. This should be an instance of nn.Module
    # Note: nn.Sequential is an instance of nn.Module.
    model_list = []
    for i in range(n_layers):
        # Fully connected layer
        layer = ("FC{}".format(i), nn.Linear(input_size, size))
        model_list.append(layer)
        # Activation function, the default is TanH which varies between[-1, 1] meeting action_space's low and high
        layer = ("Activation{}".format(i), activation)
        model_list.append(layer)
        # This layer's output size
        input_size = size
    # Output fully connected layer
    layer = ("Output layer", nn.Linear(input_size, output_size))
    model_list.append(layer)
    # Output activation function. Note: Identity function is used, but nn.Linear whose weights and bias
    # are initialized by nn.init.kaiming_uniform_() may result that the output is not in [-1,1].
    # However, homework1 utilize the identity function worse than TanH function making sure output is in [-1,1].
    output = ("Output Activation", output_activation)
    model_list.append(output)
    # Create sequential model
    model = nn.Sequential(OrderedDict(model_list))
    return model

    raise NotImplementedError


device = None


def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()

# Author = David Hudakloader
# Login = xhudak03
# Subject = BIN
# Year = 2022/2023
# Short Description = trainers.py file of project to subject Biology Inspired Computers. Calls CGP for networks.

import cgp_interface as CGPI
from network_loader import Network_Loader
import os
import cv2 as cv2
import random as random
import torch

device = torch.device("cuda")


def train_multiple_channels(network, layers=[0, 2, 4], mute=True, metapochs=[3, 3, 3],
                            epochs=[100, 100, 100], path="./nns/lenet/", back_memory=False):
    """Trains all channels of single fully connected layer. More or less legacy code.

    Args:
        network (Network_Loader): Instance of network loader including network to optimize to.
        layers (list, optional): Which layers to learn. Defaults to [0, 2, 4].
        mute (bool, optional): Save files and learn from metapochs. Defaults to True.
        metapochs (list, optional): Metapochs for each layer. Defaults to [3, 3, 3].
        epochs (list, optional): Epochs for each layer. Defaults to [100, 100, 100].
        path (str, optional): Path to save file. Defaults to "./nns/lenet/".
        back_memory (bool, optional): Usage of backward propagation in CGP. Defaults to False.
    """
    if not mute:
        max_success = 0
        best_weights = None
        best_func = None

    for i, metapoch, epoch in zip(layers, metapochs, epochs):
        for _ in range(metapoch):
            seed = int.from_bytes(os.urandom(4), 'big')
            original_weights = network.model.classifier[i].weight
            channels = len(original_weights)
            original_weights = original_weights.cpu()
            interface = CGPI.CGP_interface(real_weights=original_weights, channels=channels,
                                           seed=seed, iterations=epoch, path=path, back_memory=back_memory)
            pop = CGPI.train(interface)
            new_weights = torch.Tensor(
                interface.get_result_weights(pop.champion, save=False))
            new_weights = new_weights.cuda()
            network.model.classifier[i].weight = torch.nn.Parameter(
                new_weights)
            if not mute:
                print(f"For seed {seed} we obtained result:")
                success = network.test_model()
                if success > max_success:
                    max_success = success
                    best_func = pop.champion
                    best_weights = new_weights
        if not mute:
            with torch.no_grad():
                network.model.classifier[i].weight = torch.nn.Parameter(
                    best_weights)
                new_weights = torch.Tensor(
                    interface.get_result_weights(pop.champion, save=True))


def train_single_channel(network: Network_Loader = None, iterations: int = 500, metapochs: int = 2,
                         channel: int = 0, layer: int = 6, mute: bool = True, path: str = None,
                         back_memory: bool = False, mem_divisor: int = 10, basic: bool = False):
    """Trains cgp for weights of single channel of fully connected layer.

    Args:
        network (Network_Loader, optional): Network_Loader instance, should include network model. Defaults to None.
        iterations (int, optional): How many generations for single channel. Defaults to 500.
        metapochs (int, optional): How many runs (creations) of CGP circuit to create. Defaults to 2.
        channel (int, optional): Number of channel. Defaults to 0.
        layer (int, optional): Number of layer. Defaults to 6.
        mute (bool, optional): Determines if the weights should be saved and metapochs used for learning. Defaults to True.
        path (str, optional): Path where to print model stuff. Defaults to None.
        back_memory (bool, optional): Determines whether cgp uses backward propagation. Defaults to False.
        mem_divisor (int, optional): Determines which fraction of memory should be memorized. Defaults to 10.
        basic (bool, optional): Determines whether to use only basic functions or not. Defaults to False.
    """
    original_weights = network.model.classifier[layer].weight[channel]
    original_weights = original_weights.cpu()
    if not mute:
        max_success = 0
        best_weights = None
        best_func = None
    print(f"Started training channel {channel} in layer {layer}")
    for i in range(metapochs):
        seed = int.from_bytes(os.urandom(4), 'big')
        interface = CGPI.CGP_interface(
            original_weights, seed=seed, channels=1, iterations=iterations, path=path,
            back_memory=back_memory, mem_divisor=mem_divisor, basic=basic)
        pop = CGPI.train(interface)
        new_weights = torch.Tensor(
            interface.get_result_weights_single(pop.champion, layer=layer, channel=channel, save=False))
        new_weights = new_weights.cuda()
        with torch.no_grad():
            network.model.classifier[layer].weight[channel] = torch.nn.Parameter(
                new_weights)
        if not mute:
            print(f"For seed {seed} we obtained result:")
            success = network.test_model()
            if success > max_success:
                best_func = pop.champion
                best_weights = new_weights
    if not mute:
        with torch.no_grad():
            network.model.classifier[layer].weight[channel] = torch.nn.Parameter(
                best_weights)
            interface.get_result_weights_single(
                best_func, layer, channel, True)
    print()


def train_layers(network=None, layers: list = [1, 4, 6], iterations=[50, 50, 100],
                 metapochs=[1, 1, 1], mutes=[True, True, False],
                 path: str = None, back_memory: bool = False, mem_divisor: int = 10, basic: bool = False):
    """Trains cgps for weights for fully connected layers.

    Args:
        network (Network_Loader, optional): Network_Loader instance, should include network model. Defaults to None.
        layers (list, optional): Which layers should be used. Defaults to [1, 4, 6].
        iterations (list, optional): How many iterations should each layer use. Defaults to [50, 50, 100].
        metapochs (list, optional): How many runs of cgp for each layer. Defaults to [1, 1, 1].
        mutes (list, optional): Determines which layers should be saved. Defaults to [True, True, False].
        path (str, optional): Determines path to save files. Defaults to None.
        back_memory (bool, optional): Determines whether to use backward propagation or not. Defaults to False.
        mem_divisor (int, optional): Determines which fraction of memory to use. Defaults to 10.
        basic (bool, optional): Determines whether to use only basic functions. Defaults to False.

    Returns:
        list: Returns list of accuracies by each layer.
    """
    accuracies = []
    for i, iteration, metapoch, mute in zip(layers, iterations, metapochs, mutes):
        for channel in range(len(network.model.classifier[i].weight)):
            train_single_channel(network, iterations=iteration,
                                 metapochs=metapoch, channel=channel, layer=i, mute=mute,
                                 back_memory=back_memory, path=path, mem_divisor=mem_divisor,
                                 basic=basic)
            accuracies.append(network.test_model())
    return accuracies


def train_kernels(network: Network_Loader, epochs=100, layer: int = 0, mute: bool = True,
                  path=None, back_memory: bool = False, mem_divisor: int = 10, basic: bool = False,
                  metapochs=3):
    """Trains cgp for weights of single layer of convolutional layer layer.

    Args:
        network (Network_Loader, optional): Network_Loader instance, should include network model. Defaults to None.
        epochs (int, optional): How many generations for single layer. Defaults to 100.
        layer (int, optional): Number of layer. Defaults to 6.
        mute (bool, optional): Determines if the weights should be saved and metapochs used for learning. Defaults to True.
        path (str, optional): Path where to print model stuff. Defaults to None.
        back_memory (bool, optional): Determines whether cgp uses backward propagation. Defaults to False.
        mem_divisor (int, optional): Determines which fraction of memory should be memorized. Defaults to 10.
        basic (bool, optional): Determines whether to use only basic functions or not. Defaults to False.
        metapochs (int, optional): How many runs (creations) of CGP circuit to create. Defaults to 2.
    """
    if not mute:
        max_success = 0
        best_weights = None
        best_func = None
    original_weights = network.model.features[layer].weight
    original_weights = original_weights.cpu()
    for i in range(metapochs):
        seed = int.from_bytes(os.urandom(4), 'big')
        interface = CGPI.CGP_interface(
            original_weights, seed, iterations=epochs, convolution=True,
            path=path, back_memory=back_memory, mem_divisor=mem_divisor, basic=basic)
        pop = CGPI.train(interface)
        new_weights = torch.Tensor(
            interface.conv_get_weights(pop.champion, layer, False))
        new_weights = new_weights.cuda()
        with torch.no_grad():
            network.model.features[layer].weight = torch.nn.Parameter(
                new_weights)
        if not mute:
            print(f"For seed {seed} we obtained result:")
            success = network.test_model()
            if success > max_success:
                max_success = success
                best_weights = new_weights
                best_func = pop.champion
    if not mute:
        with torch.no_grad():
            network.model.features[layer].weight = torch.nn.Parameter(
                best_weights)
            interface.conv_get_weights(best_func, layer, True)


def train_conv_layers(network=None, layers=[0, 3, 6, 8, 10], epochs=[100, 100],
                      mute=True, path=None, back_memory=True, mem_divisor: int = 10, basic: bool = False, metapochs=[3, 3, 3, 3, 3]):
    """Trains cgps for weights for convolutional layers.

    Args:
        network (Network_Loader, optional): Network_Loader instance, should include network model. Defaults to None.
        layers (list, optional): Which layers should be used. Defaults to [1, 4, 6].
        epochs (list, optional): How many iterations should each layer use. Defaults to [100, 100, 100, 100, 100].
        mute (bool, optional): Determines whether to use metapochs and whether to save data. Defaults to True.
        path (str, optional): Determines path to save files. Defaults to None.
        back_memory (bool, optional): _description_. Defaults to True.
        mem_divisor (int, optional): Determines which fraction of memory to use. Defaults to 10.
        basic (bool, optional): Determines whether to use only basic functions. Defaults to False.
        metapochs (list, optional): How many runs of cgp for each layer. Defaults to [3,3,3,3,3].

    Returns:
        list: Returns list of accuracies by each layer.
    """
    accuracies = []
    for layer, epoch, metapoch in zip(layers, epochs, metapochs):
        train_kernels(network, epoch, layer, mute,
                      path, back_memory, mem_divisor=mem_divisor, basic=basic, metapochs=metapoch)
        accuracies.append(network.test_model())
    return accuracies

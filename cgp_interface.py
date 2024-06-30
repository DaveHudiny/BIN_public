# Author = David Hudak
# Login = xhudak03
# Subject = BIN
# Year = 2022/2023
# Short Description = cgp_interface.py file of project to subject Biology Inspired Computers. Implements CGP settings and interface.

import cgp
import cv2 as cv2
import numpy as np
import pickle as pkl
import random as random
import torch
import marshal

from blocks import *


class CGP_interface():
    def __init__(self, real_weights: torch.Tensor, seed=1337, channels: int = 10, iterations: int = 100,
                 convolution: bool = False, mem_divisor: int = 10, path: str = "./nns/alexnet_layers/", back_memory: bool = True, basic: bool = False):
        """Interface for CGP algorithm. Contains objective functions, weight saving, informations about instance etc.

        Args:
            real_weights (torch.Tensor): Real weights of focused layer/channel.
            seed (int, optional): Seed for random algorithm. Defaults to 1337.
            channels (int, optional): How many channels are focused on. Defaults to 10.
            iterations (int, optional): How many iterations of CGP algorithm work with. Defaults to 100.
            convolution (bool, optional): Determines if the layer is convolution or fully connected. Defaults to False.
            mem_divisor (int, optional): Determines how big is memory. For example, if value is 10, you divide size of real memory by 10. Defaults to 10.
            path (str, optional): Path to final results of symbolic regression. Defaults to "./nns/alexnet_layers/".
            back_memory (bool, optional): Determines if the circuit uses old outputs as inputs. Defaults to True.
        """
        self.mem_divisor = mem_divisor
        self.path = path
        self.back_memory = back_memory
        self.channels = channels
        self.convolution = convolution
        with torch.no_grad():
            self._real_weights = real_weights.numpy()
            if convolution:
                self.set_conv_cgp()
            else:
                self.set_fc_cgp()

        self.basic = basic
        if basic:
            primitives = (cgp.Add, cgp.Sub, cgp.Mul,
                          cgp.ConstantFloat, DivProtected, Identity, AbsSub)
        else:
            primitives = (cgp.Add, cgp.Sub, cgp.Mul, cgp.ConstantFloat, DivProtected,
                          cgp.IfElse, Identity, AbsSub, Avg, Sin, Cos, LogAbsProt, Sqrt)
        self.population_params = {"n_parents": 3, "seed": seed}
        self.genome_params = {
            "n_inputs": self.n_inputs,
            "n_outputs": 1,
            "n_columns": 8,
            "n_rows": 8,
            "levels_back": 7,
            "primitives": primitives,
        }
        self.ea_params = {"n_offsprings": 12,
                          "mutation_rate": 0.18, "n_processes": 2}
        self.evolve_params = {"max_generations": iterations}

    def set_conv_cgp(self):
        """Sets all important things for symbolic regression of convolution layers.
        """
        self.averages = np.array([[np.mean(real_line) for real_line in channel_set]
                                  for channel_set in self._real_weights])
        self.input_channels = self._real_weights.shape[0]
        self.output_channels = self._real_weights.shape[1]
        self.kernel_shape = [
            self._real_weights.shape[2], self._real_weights.shape[3]]
        if self.back_memory:
            self.weight_getter = self.get_conv_backlook
            self.n_inputs = 6
        else:
            self.weight_getter = self.get_conv_nobacklook
            self.n_inputs = 5
        self.mask = np.arange(
            self.kernel_shape[0] * self.kernel_shape[1])
        self.inputs_x = self.mask // self.kernel_shape[0]
        self.inputs_y = self.mask % self.kernel_shape[1]
        self.fitness_func = self.conv_objective
        self.mem_size = (self.input_channels * self.output_channels *
                         self.kernel_shape[0] * self.kernel_shape[1]) // self.mem_divisor

    def set_fc_cgp(self):
        """Sets all the important stuff for fully connected layers.
        """
        if self.channels > 1:
            real_numpy = self._real_weights
            self.averages = [np.mean(real_line)
                             for real_line in real_numpy]
            self.variances = [np.var(real_line)
                              for real_line in real_numpy]
            self.maxes = [np.max(real_line)
                          for real_line in real_numpy]
            self.mins = [np.min(real_line) for real_line in real_numpy]
            if self.back_memory:
                self.weight_getter = self.get_multiple_backlook
                self.n_inputs = 8
            else:
                self.weight_getter = self.get_multiple_nobacklook
                self.n_inputs = 7
            self.mask = [(np.arange(real_line.shape[0]) /
                          255.0) - 0.5 for real_line in real_numpy]
            self.mem_size = self._real_weights.shape[0] // self.mem_divisor
            self.fitness_func = self.objective
        elif self.channels == 1:
            self.average = np.mean(self._real_weights)
            self.variance = np.mean(self._real_weights)
            self.max = np.mean(self._real_weights)
            self.min = np.mean(self._real_weights)
            self.mask = (np.arange(self._real_weights.shape[0]) / 255.0) - 0.5
            if self.back_memory:
                self.weight_getter = self.get_single_backlook
                self.n_inputs = 7
            else:
                self.weight_getter = self.get_single_nobacklook
                self.n_inputs = 6
            self.mem_size = self._real_weights.shape[0] // self.mem_divisor
            self.fitness_func = self.objective_one_channel
        else:
            assert ("There has to be at least one channel!!!")

    def save_channel_kernel(self, type_lay: str = "conv", layer: int = 0, indices=None, memory=None,
                            func=None, real_vals=None, additional=None, channel: int = "c"):
        """Saves channel or convolution layer to file.

        Args:
            type_lay (str, optional): Type of layer. Defaults to "conv".
            layer (int, optional): Order of layer. Defaults to 0.
            indices (_type_, optional): Indices to store. Defaults to None.
            memory (_type_, optional): Memory to store. Defaults to None.
            func (_type_, optional): Function to store. Defaults to None.
            real_vals (_type_, optional): Real values to store. Defaults to None.
            additional (_type_, optional): Additional values to store (eg. averages, medians...). Defaults to None.
            channel (int, optional): Which channel is stored. Defaults to "c".
        """
        function_code = marshal.dumps(func.__code__)
        if np.max(indices >= 65535):
            indices = indices.astype(np.uint32)
        else:
            indices = indices.astype(np.uint16)
        for name, data in [("indices", indices), ("memory", memory), ("cgp", function_code), ("real", real_vals), ("additional", additional)]:
            with open(f"{self.path}{type_lay}l{layer}x{channel}_{name}", "wb") as handle:
                pkl.dump(data, handle)

    def get_memorized_res(self, og_res, save=False, layer=None, channel=None, func=None, typ="fc"):
        """Does absolute difference against real values and returns numpy array with memory correction.

        Args:
            og_res (np.ndarray): Original results.
            save (bool, optional): Save or not to save file. Defaults to False.
            layer (int, optional): Which layer we work with (for saving purposes). Defaults to None.
            channel (int, optional): Which channel we work with (for saving purposes). Defaults to None.
            func (function, optional): Individual function (cgp result). Defaults to None.
            typ (str, optional): Type of layer. Defaults to "fc".

        Returns:
            np.ndarray: Corrected output of cgp.
        """
        pom_shape = og_res.shape
        result_flat = og_res.flatten()
        abs_diffs = np.abs(result_flat - self._real_weights.flatten())
        indices = np.argpartition(abs_diffs, -self.mem_size)[-self.mem_size:]
        result_flat[indices] = self._real_weights.flatten()[indices]
        result = result_flat.reshape(pom_shape)
        if self.convolution:
            additional = self.averages
        else:
            additional = None
        if save:
            self.save_channel_kernel(typ, layer, indices, self._real_weights.flatten()[
                                     indices], func, self._real_weights, channel=channel, additional=additional)
        return result

    def get_conv_backlook(self, func):
        """Returns new cgp output with backmemory for convolutional layer.

        Args:
            func (function): Individual function.

        Returns:
            list: List of generated values.
        """
        results = []
        for i in range(self.input_channels):
            sub_results = []
            for j in range(self.output_channels):
                function_res = np.zeros((len(self.mask),))
                function_res[0] = func(
                    i, j, 0, 0, self.averages[i][j], 0.0)
                for k, x in enumerate(self.mask[1:], start=1):
                    function_res[k] = func(i, j, x // self.kernel_shape[1], x %
                                           self.kernel_shape[1], self.averages[i][j], function_res[k-1])
                function_res = function_res.reshape(
                    self.kernel_shape[0], self.kernel_shape[1])
                sub_results.append(function_res)
            results.append(sub_results)
        return results

    def get_conv_nobacklook(self, function):
        """Returns new cgp output without backmemory for convolutional layer.

        Args:
            func (function): Individual function.

        Returns:
            list: List of generated values.
        """
        results = []
        for i in range(self.input_channels):
            sub_results = []
            for j in range(self.output_channels):
                function_res = np.array([function(i, j, x // self.kernel_shape[1], x %
                                         self.kernel_shape[1], self.averages[i][j]) for x in self.mask])
                function_res = function_res.reshape(
                    self.kernel_shape[0], self.kernel_shape[1])
                sub_results.append(function_res)
            results.append(sub_results)
        return results

    def conv_objective(self, individual: cgp.IndividualSingleGenome):
        """Sets fitness of individual by objective function.

        Args:
            individual (cgp.IndividualSingleGenome): Individual generated by cgp.

        Returns:
            cgp.IndividualSingleGenome: New evaluated candidate solution.
        """
        func = individual.to_func()
        diff = 0
        results = self.weight_getter(func)
        results = np.array(results)
        results = self.get_memorized_res(results, False)
        abs_diffs = np.abs(results - self._real_weights)
        diff += np.sum(abs_diffs)
        individual.fitness = -np.float64(diff)
        return individual

    def conv_get_weights(self, individual: cgp.IndividualSingleGenome, layer: int = 0, save: bool = False):
        """Creates and returns weights generated by individual with memory.

        Args:
            individual (cgp.IndividualSingleGenome): Individual to generate new values.
            layer (int, optional): Layer of individual. For saving purposes only. Defaults to 0.
            save (bool, optional): Save weigts or not. Defaults to False.

        Returns:
            list: Values generated by CGP.
        """
        func = individual.to_func()
        in_channel = self.weight_getter(func)
        result_weights = np.array(in_channel)
        result_weights_shaped = self.get_memorized_res(
            result_weights, True, layer, channel="c", func=func, typ="c")
        return result_weights_shaped

    def get_single_backlook(self, func):
        """Returns new cgp output with backmemory for single channel of fc layer.

        Args:
            func (function): Individual function.

        Returns:
            list: List of generated values.
        """
        function_res = np.zeros((len(self.mask),))
        function_res[0] = func(
            0, len(self.mask), self.average, self.variance, self.max, self.min, 0)
        for i, x in enumerate(self.mask[:1], start=1):
            function_res[i] = func(x, len(
                self.mask), self.average, self.variance, self.max, self.min, function_res[i-1])
        return function_res

    def get_single_nobacklook(self, func):
        """Returns new cgp output without backmemory for single channel of fc layer.

        Args:
            func (function): Individual function.

        Returns:
            list: List of generated values.
        """
        function_res = np.array([func(x, len(
            self.mask), self.average, self.variance, self.max, self.min) for x in self.mask])
        return function_res

    def objective_one_channel(self, individual: cgp.IndividualSingleGenome):
        """Objective function of single fc channel cgp model which gives fitness to individuals.

        Args:
            individual (cgp.IndividualSingleGenome): Individual to be fitnessed

        Returns:
            cgp.IndividualSingleGenome: Returns fitnessed individual.
        """
        func = individual.to_func()
        function_res = self.weight_getter(func)
        function_res = self.get_memorized_res(function_res, False)
        abs_diffs = np.abs(function_res - self._real_weights)
        diff = np.sum(abs_diffs)
        individual.fitness = -np.float64(diff)
        return individual

    def get_result_weights_single(self, individual: cgp.IndividualSingleGenome, layer: int = 6, channel: int = 0, save: bool = False):
        """Creates and returns weights generated by individual with memory for single fc layer.

        Args:
            individual (cgp.IndividualSingleGenome): Individual to generate new values.
            layer (int, optional): Layer of individual. For saving purposes only. Defaults to 0.
            channel (int, optional): Channel of individual. For saving purposes only. Defaults to 0.
            save (bool, optional): Save weights or not. Defaults to False.

        Returns:
            list: Values generated by CGP.
        """
        func = individual.to_func()
        function_res = self.weight_getter(func)
        function_res = self.get_memorized_res(
            function_res, save, layer, channel, func, "fc")
        return function_res

    def get_multiple_backlook(self, func):
        """Returns new cgp output with backmemory for multiple channels of fc layer.

        Args:
            func (function): Individual function.

        Returns:
            list: List of generated values.
        """
        results = []
        function_res = np.zeros((len(self.mask[0])))
        for i, (submask, avg, var, max, min) in enumerate(zip(self.mask, self.averages, self.variances, self.maxes, self.mins)):
            size = submask.shape[0]
            function_res[0] = func(0, i, size, avg, var, max, min, 0.0)
            for j, x in enumerate(submask[1:], start=1):
                function_res[j] = func(x, size, len(
                    submask), avg, var, max, min, function_res[j-1])
            results.append(function_res)
        return results

    def get_multiple_nobacklook(self, func):
        """Returns new cgp output with backmemory for multiple channels of fc layer.

        Args:
            func (function): Individual function.

        Returns:
            list: List of generated values.
        """
        results = []
        for i, (submask, avg, var, max, min) in enumerate(zip(self.mask, self.averages, self.variances, self.maxes, self.mins)):
            repeated = np.repeat(i, submask.shape[0], axis=0)
            function_res = np.array([func(
                x, y, len(submask), avg, var, max, min) for x, y in zip(repeated, submask)])
            results.append(function_res)
        return results

    def objective(self, individual: cgp.IndividualSingleGenome):
        """Objective function of single fc channel cgp model which gives fitness to multichannel cgp individuals.

        Args:
            individual (cgp.IndividualSingleGenome): Individual to be fitnessed

        Returns:
            cgp.IndividualSingleGenome: Returns fitnessed individual.
        """
        func = individual.to_func()
        results = self.weight_getter(func)
        results = self.get_memorized_res(np.array(results), False)
        abs_diffs = (results - self._real_weights) ** 2
        diff = np.sum(abs_diffs)
        individual.fitness = -float(diff)
        return individual

    def get_result_weights(self, individual: cgp.IndividualSingleGenome, layer: int = 6, channel: int = 0, save: bool = False):
        """Creates and returns weights generated by individual with memory for multiple channels of fc layer.

        Args:
            individual (cgp.IndividualSingleGenome): Individual to generate new values.
            layer (int, optional): Layer of individual. For saving purposes only. Defaults to 0.
            channel (int, optional): Channel of individual. For saving purposes only. Defaults to 0.
            save (bool, optional): Save weights or not. Defaults to False.

        Returns:
            list: Values generated by CGP.
        """
        func = individual.to_func()
        result = self.weight_getter(func)
        result = np.array(result)
        result = self.get_memorized_res(
            og_res=result, save=save, layer=layer, channel=channel, func=func)
        return result


history = {}
history["champion_fitness"] = []
iteration = 0


def recording_callback(pop):
    """Works as callback for evolution algorithm. Actually not used much.

    Args:
        pop (cgp.IndividualMultipleGenome): Current population of cgp.
    """
    global iteration
    if iteration % 5 == 0:
        history["champion_fitness"].append(pop.champion.fitness)
    iteration += 1


def train(interface: CGP_interface = None):
    """Main train function.

    Args:
        interface (CGP_interface, optional): Instantiated CGP_interface object. Defaults to None.

    Returns:
        population: Evolved population.
    """
    pop = cgp.Population(**interface.population_params,
                         genome_params=interface.genome_params)
    ea = cgp.ea.MuPlusLambda(**interface.ea_params)
    cgp.evolve(pop, interface.fitness_func, ea, **interface.evolve_params,
               print_progress=True, callback=recording_callback)
    return pop

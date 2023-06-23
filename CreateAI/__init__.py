"""
This module is created by R0fael for creating AI
"""
import pickle
try:
    from numpy import array, dot, arange, exp, random
except:
    print("you nead install numpy")

random.seed(1)


def inputs(object: list) -> list:
    """
        This function create training inputs
        Example:
        i = inputs([
            [1,0,1],
            [1,0,0],
            [0,0,0],
            [0,1,0],
        ])
    """
    return array(object)


def outputs(object: list) -> list:
    """
    This function create training outputs
    Example:
    o = outputs([[1,1,0,0]])
    """
    return array(object).T


def weights(inputs: int, outputs: int) -> list:
    """
    This function create random weights
    inputs - count of inputs of Neuron
    outputs - count of outputs of Neuron
    Example:
    w = weights(3,1)
    """
    return 2 * random.random((inputs, outputs)) - 1

# FunktionsActivations class code


class FunktionsActivations():
    """
    This class isn't use for you creating ai, but it's nead for working createai
    """
    def sigmoid(x: float | int) -> float:
        """
        This function nead for calculations
        input x - float or int 
        output y - float of calculations for ai 
        Example: 
        Code.sigmoid(1)
        """

        return 1 / (1 + exp(-x))

# Learn class code


class Learn():
    """
    This class nead to teach AIs
    """

    def process(inputs: list, synaptic_weights: list,
                activation_funk=FunktionsActivations.sigmoid) -> list:
        """
        This funktion nead to test results without learning 
        Example: 
        Learn.process([1,1,1],weights)
        """
        return activation_funk(dot(inputs, synaptic_weights))

    def learn_iteration(inputs: list, training_outputs: list,
                        weights: list, activation_funk) -> list:
        """
        This is learning algoritm funktion
        Example:
        Learn.learn_iteration(learning_inputs,learning_outputs,weights)
        """
        outputs = Learn.process(
            inputs, weights, activation_funk=activation_funk)

        err = training_outputs - outputs

        add = dot(inputs.T, err * (outputs * (1 - outputs)))

        return weights + add

    def learn(inputs: list, training_outputs: list,
              synaptic_weights: list, iterations: int, debug=False, activation_funk=FunktionsActivations.sigmoid) -> list:
        """
        This is learning funktion repeats "iterations" times
        Example:
        Learn.learn(learning_inputs,learning_outputs,my_weights,100_000_000)
        """

        if debug:
            iter_now = 1

        for i in arange(iterations):

            synaptic_weights = Learn.learn_iteration(
                inputs, training_outputs, synaptic_weights, activation_funk)

            if debug and i == 0:
                print(f"iteration {iter_now}/{iterations}")

        return synaptic_weights


# Neuron class code

class Neuron():
    """
    This class nead for easyest creating of neuron
    """

    def __init__(self, weights: weights(3, 1),
                 training_inputs: list, training_outputs: list, save_path=None) -> None:
        """
        Create a new neuron
        Example:
        Neuron(weights(3,1),inputs([[1,1,1],[0,0,0],[1,0,1],[0,1,0]]),outputs([[1,0,1,0]]),save_path="my_save.cns")
        """
        self.training_inputs = training_inputs
        self.training_outputs = training_outputs
        self.save_path = save_path
        if save_path == None:
            self.weights = weights
        else:
            try:
                self.open_saved_data()
            except:
                self.weights = weights

    def learn(self, iterations: int,
              debug=False, aktivation=FunktionsActivations.sigmoid) -> list:
        """
        Train a neuron
        Example:
        Neuron.learn(20_000,debug=false)
        """
        self.weights = Learn.learn(self.training_inputs, self.training_outputs, self.weights,
                                   iterations, debug, activation_funk=aktivation)
        if self.save_path != None:
            with open(self.save_path, "wb") as f:
                pickle.dump([self.weights, self.training_inputs,
                            self.training_outputs], f)
        return self.weights

    def process(self, inputs: list) -> list:
        """
        Process outputs of neuron
        Example:
        Neuron.process([1,1,0]) -> list
        """
        return Learn.process(inputs, self.weights)

    def change_data(self, training_inputs=None,
                    training_outputs=None, save_path=None) -> None:
        """
        This function nead for changing
        Inputs and outputs in neuron
        """
        if training_inputs != None:
            self.training_inputs = training_inputs

        if training_outputs != None:
            self.training_outputs = training_outputs

        if save_path != None:
            self.save_path = save_path

    def open_saved_data(self):
        if self.save_path != None:
            with open(self.save_path, "rb") as f:
                file = pickle.load(f)
                self.weights = file[0]
                self.training_inputs = file[1]
                self.training_outputs = file[2]

# NeuronCreate class code


class NeuronCreate():
    """
    This class nead for easyest creating of neuron
    """

    def __init__(self, weights: weights(3, 1),
                 training_inputs: list, training_outputs: list, save_path=None) -> None:
        """
        Create a new neuron
        Example:
        NeuronCreate(weights(3,1),inputs([[1,1,1],[0,0,0],[1,0,1],[0,1,0]]),outputs([[1,0,1,0]]),save_path="my_save.cns")
        """
        self.training_inputs = training_inputs
        self.training_outputs = training_outputs
        self.save_path = save_path
        self.weights = weights

    def learn(self, iterations: int,
              debug=False, aktivation=FunktionsActivations.sigmoid) -> list:
        """
        Train a neuron
        Example:
        NeuronCreate.learn(20_000,debug=false)
        """
        self.weights = Learn.learn(self.training_inputs, self.training_outputs, self.weights,
                                   iterations, debug, activation_funk=aktivation)
        if self.save_path != None:
            with open(self.save_path, "wb") as f:
                pickle.dump([self.weights, self.training_inputs,
                            self.training_outputs], f)
        return self.weights

    def process(self, inputs: list) -> list:
        """
        Process outputs of neuron
        Example:
        Neuron.process([1,1,0]) -> list
        """
        return Learn.process(inputs, self.weights)

    def change_data(self, training_inputs=None,
                    training_outputs=None, save_path=None) -> None:
        """
        This function nead for changing
        Inputs and outputs in neuron
        """
        if training_inputs != None:
            self.training_inputs = training_inputs

        if training_outputs != None:
            self.training_outputs = training_outputs

        if save_path != None:
            self.save_path = save_path

# NeuronOpen class code


class NeuronOpen():
    """
    This class nead for easyest opening of neuron
    """

    def __init__(self, save_path) -> None:
        """
        Create a new neuron
        Example:
        Neuron(weights(3,1),inputs([[1,1,1],[0,0,0],[1,0,1],[0,1,0]]),outputs([[1,0,1,0]]),save_path="my_save.cns")
        """
        self.save_path = save_path
        self.open_saved_data()

    def learn(self, iterations: int,
              debug=False, aktivation=FunktionsActivations.sigmoid) -> list:
        """
        Train a neuron
        Example:
        Neuron.learn(20_000,debug=false)
        """
        self.weights = Learn.learn(self.training_inputs, self.training_outputs, self.weights,
                                   iterations, debug, activation_funk=aktivation)
        if self.save_path != None:
            with open(self.save_path, "wb") as f:
                pickle.dump([self.weights, self.training_inputs,
                            self.training_outputs], f)
        return self.weights

    def process(self, inputs: list) -> list:
        """
        Process outputs of neuron
        Example:
        Neuron.process([1,1,0]) -> list
        """
        return Learn.process(inputs, self.weights)

    def change_data(self, training_inputs=None,
                    training_outputs=None, save_path=None) -> None:
        """
        This function nead for changing
        Inputs and outputs in neuron
        """
        if training_inputs != None:
            self.training_inputs = training_inputs

        if training_outputs != None:
            self.training_outputs = training_outputs

        if save_path != None:
            self.save_path = save_path

    def open_saved_data(self):
        with open(self.save_path, "rb") as f:
            file = pickle.load(f)
            self.weights = file[0]
            self.training_inputs = file[1]
            self.training_outputs = file[2]

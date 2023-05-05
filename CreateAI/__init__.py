
"""
This module is created by R0fael for creating AI
"""
try:
    from numpy import array, dot, arange, random, exp
except:
    from os import system
    try:
        system("pip install numpy")
    except:
        try:
            system("pip3 install numpy")
        except:
            system("sudo pip install numpy")


def inputs(object: list) -> list:
    """
        This function create training inputs
        Example:
        i = CreateAI.inputs([
        [1,0,1],
        [1,0,0],
        [0,0,0],
        [1,1,1],])
    """
    return array(object)


def outputs(object: list) -> list:
    """
    This function create training outputs
    Example:
    o = CreateAI.outputs([1,1,0,1])
    """

    return array(object).T


def weights(inputs: int, outputs: int) -> list:
    """
    This function create random weights
    inputs - count of inputs of Neuron
    outputs - count of outputs of Neuron
    Example:
    w = CreateAI.weights(3,1)
    """
    random.seed(1)
    return 2 * random.random((inputs, outputs)) - 1


class Code():
    def sigmoid(x: float) -> float:
        """
        This function nead for calculations
        input x - float or int 
        output y - float of calculations for ai 
        Example: 
        EasyAI.Code.sigmoid(1)
        """
        return 1 / (1 + exp(-x))

    def line(x: float) -> float:
        return x

    def minmax(x: float) -> float:
        if x < 0.5:
            return 0
        elif x > 0.5:
            return 1
        else:
            return 0.5


class Learn():
    """
    This class nead to teach AIs
    """

    def process(inputs: list, synaptic_weights: list, activation_funk=Code.sigmoid) -> list:
        """
        This funktion nead to test results without learning 
        Example: 
        EasyAI.Learn.process([1,1,1],weight)
        """
        return activation_funk(dot(inputs, synaptic_weights))

    def learn_iteration(inputs: list, training_outputs: list, weights: list, activation_funk=Code.sigmoid) -> list:
        """
        This is learning algoritm funktion
        Example:
        CreateAI.Learn.learn_iteration(learning_inputs,learning_outputs,weights)
        """
        input_layer = inputs
        outputs = Learn.process(
            inputs, weights, activation_funk=activation_funk)

        err = (training_outputs - outputs)[0]
        add = dot(input_layer.T, err * (outputs * (1 - outputs)))

        return weights + add

    def learn(inputs: list, training_outputs: list, synaptic_weights: list, iterations: int, debug=False, batch=1000, activation_funk=Code.sigmoid) -> list:
        """
        This is learning funktion repeats "iterations" times
        Example:
        CreateAI.Learn.learn(learning_inputs,learning_outputs,my_weights,100_000_000)
        """
        if debug:
            batch_now = 1
        for i in set(arange(iterations)):
            synaptic_weights = Learn.learn_iteration(
                inputs, training_outputs, synaptic_weights, activation_funk=activation_funk)
            if debug and i % batch == 0:
                print(f"batch {batch_now}/{iterations/batch}")

        return synaptic_weights


class Neuron():
    """
    This class nead for easyest creating of neuron
    """

    def __init__(self, weights: list, training_inputs: list, training_outputs: list) -> None:
        """
        Create a new neuron
        Example:
        CreateAI.Neuron(CreateAI.weights(3,1),CreateAI.inputs([[1,1,1],[0,0,0],[1,0,1],[0,1,0]]),CreateAI.outputs([1,0,1,0]))
        """
        self.weights = weights
        self.training_inputs = training_inputs
        self.training_outputs = training_outputs

    def learn(self, iterations: int, debug=False, aktivation=Code.sigmoid) -> list:
        """
        Train a neuron
        Example:
        CreateAI.Neuron.learn(20_000,debug=false)
        """
        self.weights = Learn.learn(
            self.training_inputs, self.training_outputs, self.weights, iterations, debug, activation_funk=aktivation)
        return self.weights

    def process(self, inputs):
        """
        Process outputs of neuron
        Example:
        CreateAI.Neuron.process([0,0,0])
        """
        return Learn.process(inputs, self.weights)

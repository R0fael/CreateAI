
"""
This module is created by R0fael for creating AI
"""
import threading

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
        from numpy import array, dot, arange, random, exp


def inputs(object: list) -> list:
    """
        This function create training inputs
        Example:
        i = CreateAI.inputs([
        [1,0,1],
        [1,0,0],
        [0,0,0],
        [0,1,0],])
    """
    return array(object)


def outputs(object: list) -> list:
    """
    This function create training outputs
    Example:
    o = CreateAI.outputs([[1,1,0,0]])
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
    def sigmoid(x: float | int) -> float:
        """
        This function nead for calculations
        input x - float or int 
        output y - float of calculations for ai 
        Example: 
        EasyAI.Code.sigmoid(1)
        """
        return 1 / (1 + exp(-x))

    def line(x: float | int) -> float | int:
        return x

    def minmax(x: float | int) -> float | int:
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

    def process(inputs: list, synaptic_weights: list,
                activation_funk=Code.sigmoid) -> list:
        """
        This funktion nead to test results without learning 
        Example: 
        EasyAI.Learn.process([1,1,1],weight)
        """
        return activation_funk(dot(inputs, synaptic_weights))

    def learn_iteration(inputs: list, training_outputs: list,
                        weights: list, activation_funk) -> list:
        """
        This is learning algoritm funktion
        Example:
        CreateAI.Learn.learn_iteration(learning_inputs,learning_outputs,weights)
        """
        outputs = Learn.process(
            inputs, weights, activation_funk=activation_funk)

        err = training_outputs - outputs
        add = dot(inputs.T, err * (outputs * (1 - outputs)))

        return weights + add

    def learn(inputs: list, training_outputs: list,
              synaptic_weights: list, iterations: int, debug=False, batch=1000, activation_funk=Code.sigmoid, threading_use=False) -> list:
        """
        This is learning funktion repeats "iterations" times
        Example:
        CreateAI.Learn.learn(learning_inputs,learning_outputs,my_weights,100_000_000)
        """
        # learn_theading = threading.Thread(target=Learn.learn_iteration, args=(inputs, training_outputs, synaptic_weights, activation_funk), name="learn-thr")
        if debug:
            batch_now = 1
        for i in arange(iterations):
            # synaptic_weights = Learn.learn_iteration(inputs, training_outputs, synaptic_weights, activation_funk)
            if threading_use:
                learn_theading = threading.Thread(target=Learn.learn_iteration, args=(
                    inputs, training_outputs, synaptic_weights, activation_funk), name="learn-thr")
                learn_theading.start()
            else:
                synaptic_weights = Learn.learn_iteration(
                    inputs, training_outputs, synaptic_weights, activation_funk)
            if debug and i % batch == 0:
                print(f"batch {batch_now}/{iterations/batch}")

        return synaptic_weights

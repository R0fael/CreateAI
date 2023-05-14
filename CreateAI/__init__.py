
"""
This module is created by R0fael for creating AI
"""


try:
    from numpy import array, dot, arange, exp, random
except:
    print("you nead install numpy")


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

    random.seed(1)
    return 2 * random.random((inputs, outputs)) - 1


class Code():
    def sigmoid(x: float | int) -> float:
        """
        This function nead for calculations
        input x - float or int 
        output y - float of calculations for ai 
        Example: 
        Code.sigmoid(1)
        """

        return 1 / (1 + exp(-x))


class Learn():
    """
    This class nead to teach AIs
    """

    def process(inputs: list, synaptic_weights: list,
                activation_funk=Code.sigmoid) -> list:
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
              synaptic_weights: list, iterations: int, debug=False, activation_funk=Code.sigmoid) -> list:
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

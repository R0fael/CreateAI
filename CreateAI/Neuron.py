import pickle

from CreateAI import weights, Code, Learn


class Neuron():
    """
    This class nead for easyest creating of neuron
    """

    def __init__(self, weights: weights(3, 1),
                 training_inputs: list, training_outputs: list, save_path=None) -> None:
        """
        Create a new neuron
        Example:
        Neuron(weights(3,1),inputs([[1,1,1],[0,0,0],[1,0,1],[0,1,0]]),outputs([[1,0,1,0]]),save_path="my_save.txt")
        """
        self.training_inputs = training_inputs
        self.training_outputs = training_outputs
        self.save_path = save_path
        if save_path == None:
            self.weights = weights
        else:
            try:
                with open(save_path, "rb") as f:
                    self.weights = pickle.load(f)
            except:
                self.weights = weights

    def learn(self, iterations: int,
              debug=False, aktivation=Code.sigmoid, batch=100_000) -> list:
        """
        Train a neuron
        Example:
        CreateAI.Neuron.learn(20_000,debug=false)
        """
        self.weights = Learn.learn(self.training_inputs, self.training_outputs, self.weights,
                                   iterations, debug, activation_funk=aktivation, batch=batch)
        if self.save_path != None:
            with open(self.save_path, "wb") as f:
                pickle.dump(self.weights, f)
        return self.weights

    def process(self, inputs: list) -> list:
        """
        Process outputs of neuron
        Example:
        CreateAI.Neuron.process([1,1,0]) -> list
        """
        return Learn.process(inputs, self.weights)

    def change_data(self, training_inputs: list,
                    training_outputs: list) -> None:
        """
        This funktion nead for changing
        Inputs and outputs in neuron
        """
        self.training_inputs = training_inputs
        self.training_outputs = training_outputs

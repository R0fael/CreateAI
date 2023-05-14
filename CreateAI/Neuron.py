import pickle

from createai import weights, Code, Learn


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
              debug=False, aktivation=Code.sigmoid) -> list:
        """
        Train a neuron
        Example:
        Neuron.learn(20_000,debug=false)
        """
        self.weights = Learn.learn(self.training_inputs, self.training_outputs, self.weights,
                                   iterations, debug, activation_funk=aktivation)
        if self.save_path != None:
            with open(self.save_path, "wb") as f:
                pickle.dump(self.weights, f)
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
            with open(self.save_path, "wb") as f:
                self.weights = pickle.dump(self.weights, f)

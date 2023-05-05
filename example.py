# importing CreateAI
from CreateAI import *

n = Neuron(weights(3, 1), inputs([
    [0, 0, 1],
    [1, 1, 1],
    [1, 0, 1],
    [0, 1, 1]
]), outputs([[0, 1, 1, 0]]))  # Neuron creating

n.learn(20_000)  # Neuron learning

print(n.process([1, 1, 0]))  # Neuron test
print(n.process([0, 1, 0]))  # Neuron test

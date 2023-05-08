# importing CreateAI
from CreateAI import *

w = weights(3, 1)

i = inputs([
    [0, 0, 1],
    [1, 1, 1],
    [1, 0, 1],
    [0, 1, 1]
])

o = outputs([[0, 1, 1, 0]])

n = Neuron(w, i, o)  # Neuron creating

n.learn(20_000, debug=True)  # Neuron learning

print(n.process([1, 1, 0]))  # Neuron test

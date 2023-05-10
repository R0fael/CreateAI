# importing CreateAI
from CreateAI import weights, inputs, outputs
from CreateAI.Neuron import Neuron

w = weights(3, 1)

i = inputs([
    [0, 0, 1],
    [1, 1, 1],
    [1, 0, 1],
    [0, 1, 1]
])

o = outputs([[0, 1, 1, 0]])

# Neuron creating with saving
n = Neuron(w, i, o, save_path="my_example_save.cns")

n.learn(20_000, debug=True)  # Neuron learning

print(n.process([1, 1, 0]))  # Neuron test

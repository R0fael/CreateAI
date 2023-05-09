from CreateAI import *
from CreateAI.changer import *

w = weights(2, 1)

i = inputs([[1, 0], [0, 1]])

o = outputs([[1, 0]])

n = Neuron(w, i, o)

n.learn(2)

print(n.process([1, 0]))

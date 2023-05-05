
# Create AI
It's easy tool to create ai in python easy and fast.

## Dependencies
 - numpy
 - knowledge of python
 - our documentation
 - your motivation
 - your brain

## Authors

- [@R0fael](https://www.github.com/R0fael) - programmer

## Features
 - Custom funktion activations
 - Neuron creating in one line of code
 - absolutly open sourse

## Installation

Install my-project with pip

Windows:
```bash
pip install createai
```

MacOS:
```bash
pip3 install createai
```

Linux:
```bash
sudo pip install createai
```

## Example
```python
from CreateAI import *
"""
inputs - create inputs
outputs - create outputs
"""
n = Neuron(weights(3, 1), inputs([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]]), outputs([[0, 1, 1, 0]])) # neuron creating

n.learn(20_000) # learning

print(n.process([1, 1, 0])) # test
print(n.process([0, 1, 0])) # test
```

## How to create custom funktion activation
```python
from CreateAI import *

def my_funk(x):
    return x # your code here

n = Neuron(weights(3, 1), inputs([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]]), outputs([[0, 1, 1, 0]])) # neuron creating

n.learn(20_000,aktivation=my_funk) # learning

print(n.process([1, 1, 0])) # test
print(n.process([0, 1, 0])) # test
```
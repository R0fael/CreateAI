
# CreateAI v0.9.5
It's easy tool to create ai in python easy and fast.
You can edit source code of CreateAI on [Create AI's github](https://github.com/R0fael/CreateAI)

## Dependencies
 - numpy
 - knowledge of python
 - our documentation
 - your motivation
 - your brain

## Authors

 - [@R0fael](https://www.github.com/R0fael) - programmer

## Features
 - Custom function activations - Errors
 - Neuron creating in one line of code
 - absolutly open sourse

## Installation

Installing project with pip

Windows:
```bash
pip install СreateAI
```

MacOS:
```bash
pip3 install СreateAI
```

Linux:
```bash
sudo pip install СreateAI
```

Or if you don't have pip :
download sourse code and editing examples files

## Example
```python
from CreateAI import *
"""
inputs - create inputs
outputs - create outputs
"""
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

```

## How to create custom funktion activation - HAS ERRORS
```python
from CreateAI import *

def my_funk(x):
    return x # your code here

n = Neuron(weights(3, 1), inputs([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]]), outputs([[0, 1, 1, 0]])) # neuron creating

n.learn(20_000,aktivation=my_funk) # learning

print(n.process([1, 1, 0])) # test
print(n.process([0, 1, 0])) # test
```

## Change log
0.8 - First version
0.8.1 - Little bug fix
0.9.2 - Speed up
0.9.3 - Bug Fix
0.9.4 - Bug Fix of bug fix
0.9.5 - convertors

## Plans
0.10.0 - Saving system
0.10.1 - Fixing custom functions activation
0.10.2 - Multi neurons update
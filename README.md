
# CreateAI v0.1.0
It's easy tool to create ai in python easy and fast.
You can edit source code of CreateAI on [Create AI's github](https://github.com/R0fael/CreateAI)

## Dependencies
 - numpy
 - knowledge of python
 - our documentation
 - your motivation
 - your brain

## Authors

 - [@R0fael](https://www.github.com/R0fael) - main programmer

## Features
 - Custom function activations - Errors
 - Neuron creating in one line of code
 - absolutly open sourse

## Installation

Installing project with pip

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
]), outputs([[0, 1, 1, 0]]),save_path="my_save.nws")  # Neuron creating with saving
#.nws -> file extension -> neuron weight save

n.learn(20_000)  # Neuron learning

print(n.process([1, 1, 0]))  # Neuron test
print(n.process([0, 1, 0]))  # Neuron test
```

## Change log

-0.1.0 - first version

-0.1.1 - speed up and little bug fix

## Plans

-0.1.2 - More metods of learning

-0.1.3 - Multi neurons update
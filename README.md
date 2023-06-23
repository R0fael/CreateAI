
# CreateAI v0.1.2.1
It's easy tool to create ai in python easy and fast.
You can look at source code of CreateAI on [Create AI's github](https://github.com/R0fael/CreateAI)

# WARNING!
On my PC python can't find this module.
I am using source code from github.

## Dependencies
 - numpy
 - knowledge of python
 - our documentation
 - your motivation
 - your brain

## Authors

 - [@R0fael](https://www.github.com/R0fael) - main programmer

## Features
 - Custom function activations
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
# importing CreateAI
from createai import *

n = Neuron(weights(3, 1), inputs([
    [0, 0, 1],
    [1, 1, 1],
    [1, 0, 1],
    [0, 1, 1]
]), outputs([[0, 1, 1, 0]]),save_path="my_save.cns")  # Neuron creating with saving
#.cns -> file extension -> neuron weight save

n.learn(20_000)  # Neuron learning

print(n.process([1, 1, 0]))  # Neuron test
print(n.process([0, 1, 0]))  # Neuron test
```

## Change log

-0.1.0 - first version

-0.1.1 - speed up and little bug fix

-0.1.2 - without numpy - error

-0.1.2.1 - big bugfix

## Plans

-0.1.3 - More metods of learning

-0.1.4 - Multi neurons update
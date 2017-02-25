from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#from nn.module import Module
from module import Module
from linear import Linear
from MSE import MSE
import numpy as np

class Sequential(Module):
    def __init__(self):
        super(Sequential, self).__init__()
        self.layers = []

    def add(self, module):
        if self.layers and self.layers[-1].size != module.prev_size:
            raise ValueError("Incompatible layer size!")
        self.layers.append(module)


    def remove(self, module):
        raise NotImplementedError('implement remove to sequential!')

    def forward(self, inputs, outputs):
        self.outputs = inputs
        self.real = outputs
        for layer in self.layers:
            self.outputs = layer.forward(self.outputs, self.real)
        return self.outputs

    def backward(self, *args, **kwargs):
        self.grad_input = self.outputs
        for layer in self.layers[::-1]:
            self.grad_input = layer.backward(self.grad_input)
        return self.grad_input

if __name__ == "__main__":
    model = Sequential()
    model.add(Linear(3,10))
    model.add(MSE(10))
    X = np.array([[3.,4.,5.]]).T
    y = np.array([1,2,3,4,5,6,7,8,9,10]).T
    for i in xrange(10000):
        model.forward(X, y)
        model.backward()

    print(model.forward(X, y))





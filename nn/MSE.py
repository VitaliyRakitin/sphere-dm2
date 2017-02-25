#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
#
# Техносфера, BD-21
# Нейронная сеть.
# Автор: Ракитин Виталий
#
# Mean squared error.
# MSE = 1/2 sum (y_i - b_i)^2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#from nn.module import Module
from module import Module
import numpy as np

class MSE(Module):
    def __init__(self, prev_size, size = 1):
        super(MSE, self).__init__(prev_size, size)
        self.real = np.zeros(self.size)

    def forward(self, *args, **kwargs):
        self.batch, self.real = args        
        self.batch_size = len(self.batch.T)
        return self.layer_function(self.batch)

    def layer_function(self, X):
        if X.shape != (self.prev_size,):
            res = np.repeat(0.0, X.shape[1])
        else:
            res = 0.0
        for i in xrange(len(X)):
            res += (X[i] - self.real[i]) * (X[i] - self.real[i])
        return 2.0 * res / self.prev_size 

    def analytical_grad(self, X):
        '''
        y = w * x + b
        y'x = w.T
        '''
        #print (X - self.real)
        return (X - self.real) / self.prev_size

    def update_grad_input(self, *args, **kwargs):
        self.grad_input = -self.batch_analytical_grad().T


if __name__ == "__main__":
    model = MSE(3)
    X = np.array([[3.,4.,5.],[6., 8., 10.],[9.,12.,15.],[12.,16.,20.]]).T
    y = np.array([3., 4., 5.])
    print(model.forward(X,y))
    print(model.backward(X,y))
    #print(model.gradient_error())
    #print(model.batch_numerical_grad())
    #print(model.batch_size)

#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
#
# Техносфера, BD-21
# Нейронная сеть.
# Автор: Ракитин Виталий
#
# Функция активации --- сигмоида.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#from nn.module import Module

from module import Module
import numpy as np
import math

class Sigmoid(Module):
    def __init__(self, size):
        super(Sigmoid, self).__init__(size, size)

    def forward(self, *args, **kwargs):
        self.batch, _ = args        
        self.batch_size = len(self.batch.T)
        return self.layer_function(self.batch)

    @staticmethod
    def sigmoid(z):
        ''' sigmoid function '''
        return 1./(1. + math.exp(-z))

    def sigmoid_dev(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def batch_func(self, func, X):
        if X.shape != (self.prev_size,):
            batch = []
            for x in X:
               batch.append(map(func, x))
            return np.array(batch)
        return np.array(map(func, X))

    def layer_function(self, X):
        return self.batch_func(self.sigmoid, X)

    def analytical_grad(self, X):
        return self.batch_func(self.sigmoid_dev, X)

    def update_grad_input(self, *args, **kwargs):
        self.grad_input = self.analytical_grad(self.batch)

if __name__ == "__main__":
    s = Sigmoid(3)
    X = np.array([[3.,4.,5.],[5.,6.,7.],[7.,8.,9.],[10.,11.,12.]]).T
    print(s.forward(X,s))
    #s.gradient_error() --- не работает для данного случая
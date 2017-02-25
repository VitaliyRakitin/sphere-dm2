#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
#
# Техносфера, BD-21
# Нейронная сеть.
# Автор: Ракитин Виталий
#
# Шаблон слоя:
#
# X ---------------->|                       |--> Y = output
#                    |          w            |
#                    | grad_inside = dL/dw = |
#                    | = (dL/dY) x (dY/dw)   |
# grad_input = <-----|                       |<-- grad = dL/dY
#  = dL/dX = 
#  = (dL/dY) * (dY/dX)
#
# updatre_grad_input = { dL/dX, dL/dw }
# count_grad_input = dY/dX
# count_grad_inside = dY/dw
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

class Module(object):
    def __init__(self, prev_size = 0, size = 0, batch = None, batch_size = 0, eps = 1e-5):
        self.size = size 
        self.prev_size = prev_size
        self.output = None 
        self.grad_input = None
        self.grad_inside = None
        self.eps = eps
        self.batch = batch
        self.batch_size = batch_size


    def forward(self, *args, **kwargs):
        raise NotImplementedError('implement forward pass!')

    def backward(self, *args, **kwargs):
        self.update_grad_input(*args, **kwargs)
        self.update_parameters(*args, **kwargs)
        return self.grad_input

    def update_grad_input(self, *args, **kwargs):
        raise NotImplementedError('implement computation of gradient w.r.t. input! df(x)/dx!')

    def update_parameters(self, *args, **kwargs):
        # that's fine not to implement this method
        # module may have not parameters (for example - MSE criterion)
        pass

    def count_grad_input(self, *args, **kwargs):
        raise NotImplementedError('implement computation of dY/dX! (count_grad_input)')

    def count_grad_inside(self, *args, **kwargs):
        raise NotImplementedError('implement computation of dY/dw! (count_grad_inside)')

    def layer_function(self, X):
        raise NotImplementedError('implement computation of layer function')

    def analytical_grad(self, X):
        raise NotImplementedError('implement computation of analytical gradient function')

    def batch_analytical_grad(self):   
        grad = []
        for x in self.batch.T:
            grad.append(self.analytical_grad(x))#/self.batch_size)
        #return np.array(grad).sum(axis = 0)
        return np.array(grad)

    def numerical_grad(self, X):
        '''
        f(x,y,z)
        f'x (x,y,z) = (f(x+self.eps,y,z) - f(x-self.eps,y,z))/(2*self.eps)
        grad(f) = (f'x, f'y, f'z) 
        '''
        grad = []
        for j in xrange(len(X)):
            x1 = X.copy()
            x2 = X.copy()
            x1[j] += self.eps
            x2[j] -= self.eps
            grad.append((self.layer_function(x1) - self.layer_function(x2))/(2 * self.eps))
        return np.array(grad)

    def batch_numerical_grad(self):
        batch_grad = []
        for x in self.batch.T:
            batch_grad.append(self.numerical_grad(x) / self.batch_size)
        return np.array(batch_grad).sum(axis = 0)

    def gradient_error(self):
        '''
        Check the sum of the squared differences on each element in   
        numerical and analytical gradients
        '''
        numerical_grad = self.batch_numerical_grad()
        analytical_grad = self.batch_analytical_grad()
        error = np.square((numerical_grad - analytical_grad)).sum()
        return error

    def parametric_grad(self, X):
        raise NotImplementedError('implement computation of gradient by parameters')
    
    def batch_parametric_grad(self):
        raise NotImplementedError('implement computation of batch gradient by parameters')


#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
#
# Техносфера, BD-21
# Нейронная сеть.
# Автор: Ракитин Виталий
#
# Линейный слой.
# Y = W * X + B
# batch --- значение X в данный момент времени

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#from nn.module import Module
from module import Module
import numpy as np


class Linear(Module):
    def __init__(self, prev_size, size = 50, batch = None, lmbd = 1e-3):
        super(Linear, self).__init__(prev_size, size)
        self.weights, self.bais = self.create_params(prev_size, size)
        self.lmbd = lmbd

    @staticmethod
    def create_params(n, m):
        weights = np.random.normal(scale = 0.01, size = (m, n))
        bais = np.random.normal(scale = 0.01, size = (m))
        return weights, bais
  
    def layer_function(self, X): 
        if X.shape != (self.prev_size,):
            bais = np.repeat(self.bais, X.shape[1])
            bais = np.reshape(bais, (self.size, X.shape[1]))
        else:
            bais = self.bais
        return np.dot(self.weights, X) + bais

    def forward(self, *args, **kwargs):
        self.batch, _ = args        
        self.batch_size = len(self.batch.T)
        return self.layer_function(self.batch)

    def analytical_grad(self, X):
        '''
        y = w * x + b
        y'x = w.T
        '''
        return self.weights.T
    
    def batch_parametric_grad(self):
        batch_weights_grad = []
        batch_bais_grad = []
        for x in self.batch.T:
            w, b = self.parametric_grad(x)
            batch_weights_grad.append(w)# / self.batch_size)
            batch_bais_grad.append(b) #/ self.batch_size)
        #return (np.array(batch_weights_grad).sum(axis = 0),  
        #        np.array(batch_bais_grad).sum(axis = 0))
        return np.array(batch_weights_grad), np.array(batch_bais_grad)

    def parametric_grad(self, X):
        '''
        y = w * x + b

        x = x_1, ..., x_n
        y = y_1, ..., y_m

        w_11, ..., w_1n
        ...        ...
        w_m1, ..., w_mn

        d(w * x) / dw = (d(w * x) / dw_ij) = ...
        i-строка, j-й столбец --- столбец длины m, 
        где на i-й позиции стоит элемент x_j

        y'b = (1,...,1) --- вектор длинны b.size
        '''    
        w_grad = np.zeros((self.size, self.prev_size, self.size))
        for i in xrange(len(w_grad)):
            for j in xrange(len(w_grad[i])):
                w_grad[i][j][i] = X[j]
        return w_grad, np.ones(self.bais.size).T


    def update_grad_input(self, *args, **kwargs):
        grad_next, = args
        self.grad_input = np.dot(self.batch_analytical_grad(), grad_next)
        


    def update_parameters(self, *args, **kwargs):
        grad_next, = args
        weights_grad, bais_grad = self.batch_parametric_grad()
        self.weights += self.lmbd * (np.dot(weights_grad, grad_next).sum(axis = 3).sum(axis = 0))
        self.bais += self.lmbd * np.dot(bais_grad, grad_next).sum(axis = 1).sum(axis = 0)


if __name__ == "__main__":
    l = Linear(3,10)
    X = np.array([[3.,4.,5.],[5.,6.,7.],[7.,8.,9.],[10.,11.,12.]]).T
    y = np.array([1,2,3,4,5,6,7,8,9,10]).T
    print (l.forward(X))
    print (l.backward(y))
    print(l.gradient_error())
    #print (l.count_layer_function(X))
    #X = np.array([[22.,42.,35.]]).T
    #(l.parametric_grad(X,y))
    #print (len(X.T))
    #print(l.batch_numerical_gradient(X))
    #print (1e-3)
    #print (X[0] + 1)
import numpy as np
import matplotlib.pyplot as plt

class cg:

    Q = np.matrix('2 1;1 2')
    b = np.matrix('3;0')

    def __init__(self, x1,x2):
        self.x1 = x1
        self.x2 = x2

    def g(self):
        x = np.matrix([[self.x1],[self.x2]])
        delf = self.Q * x - self.b
        return delf

    def d(self):
        g = self.g()
        d = -g
        return d

    def alpha(self):
        g = self.g()
        d = self.d()
        a =- g.T*d
        b =d.T*self.Q*d
        alpha = a.item(0)/b.item(0)
        return alpha

    def next_x1(self):
        alpha = self.alpha()
        d = self.d()
        x = np.matrix([[self.x1],[self.x2]])
        next_x_ = x+ alpha*d
        return next_x_

    def beta(self,g1):
        d = self.d()
        a = g1.T*self.Q*d
        b =d.T*self.Q*d
        beta = a.item(0)/b.item(0)
        return beta

    def d1(self,g1):
        d = self.d()
        beta = self.beta(g1)
        new_d= -g1+beta*d
        return new_d

    def alpha1(self,d1,g1):
        a =- g1.T*d1
        b =d1.T*self.Q*d1
        alpha = a.item(0)/b.item(0)
        return alpha

    def new_x2(self,alpha1,d1):
        x = np.matrix([[self.x1],[self.x2]])
        next_x_ = x+ alpha1*d1
        return next_x_

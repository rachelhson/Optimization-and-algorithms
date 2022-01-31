import numpy as np
import matplotlib.pyplot as plt

class quadric:
    def __init__(self, matrix):
        self.x1 = matrix[0]
        self.x2 = matrix[1]

    def function(self):
        f = (self.x2-self.x1)**4 +12*self.x1*self.x2-self.x1+self.x2-3
        return f

    def diff(self):
        del_x1 =-4*(self.x2-self.x1)**3+12*self.x2 - 1
        del_x2 = 4*(self.x2-self.x1)**3+12*self.x1 + 1
        norm = np.sqrt(del_x1**2+del_x2**2)
        return del_x1,del_x2,norm

    def norm_delfx(self):
        del_x1,del_x2,norm = self.diff()
        del_fx = np.array ([del_x1,del_x2])
        norm_delfx = np.array ([del_x1/norm,del_x2/norm])
        return norm_delfx



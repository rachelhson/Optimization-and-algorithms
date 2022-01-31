import numpy as np
import matplotlib.pyplot as plt

class quadric:
    def __init__(self, matrix):
        self.x1 = matrix[0]
        self.x2 = matrix[1]

    def function(self):
        f = (self.x2-self.x1)**4 +12*self.x1*self.x2-self.x1+self.x2-3
        return f


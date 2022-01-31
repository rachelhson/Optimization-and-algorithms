import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv

class quadric:
    
    def __init__(self, matrix):
        self.x1 = matrix[0][0] 
        self.x2 = matrix[1][0]  
    
    def function(self):
        f = (self.x2-self.x1)**4 +12*self.x1*self.x2-self.x1+self.x2-3
        return f   

    def diff(self):
        del_x1 =-4*(self.x2-self.x1)**3+12*self.x2 - 1
        del_x2 = 4*(self.x2-self.x1)**3+12*self.x1 + 1
        del_ = np.array([[del_x1],[del_x2]])
        return del_

    def hessen(self):
        hes_x1x1 = 12*(self.x2-self.x1)**2
        hes_x1x2 =-12*(self.x2-self.x1)**2+12
        hes_x2x2 = 12*(self.x2-self.x1)**2
        print(hes_x1x1)
        hes_matrix = np.array([[hes_x1x1,hes_x1x2],[hes_x1x2,hes_x2x2]])
        return hes_matrix

    def hessen_pd(self):
        hes_matrix = self.hessen()
        det = hes_matrix[0][0]*hes_matrix[1][1]-hes_matrix[0][1]*hes_matrix[1][0]
        if det > 0 :
            return "positive definite"
        else:
            return "n.p.d cannot guarantee decent" 
            
"""graph for the function"""
def f(x1, x2):
    return  (x2-x1)**4 +12*x1*x2-x1+x2-3

x1 = np.linspace(-1, 1, 30)
x2 = np.linspace(-1, 1, 30)

X, Y = np.meshgrid(x1, x2)
Z = f(X, Y)
print(f"minimum Z value :{np.min(Z)}")

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');
#plt.show()

"""initial point"""
x1_a0= 0.55
x2_a0= 0.7
a_0 = np.array ([[x1_a0],[x2_a0]])
alpha = 0.02
print(f"initial point: {a_0}")

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

x0 = quadric(a_0)

f0 = x0.function()
del_ = x0.diff()
ax.scatter(x1_a0, x2_a0, f0, color ="blue")
hessen_matrix= x0.hessen()
print(f"hessen_matrix: {hessen_matrix}")
hessen_pd= x0.hessen_pd()
print(f"hessen_p.d: {hessen_pd}")
inv_hessen = inv(hessen_matrix)
print(f"inv_hessen_matrix: {hessen_matrix}")
print(inv_hessen)
print(del_)
change = np.matmul(inv_hessen,del_)
print(change)
x_0 = a_0
print(x_0)
x1 = x_0 - change 
print(x1)
x1_ = quadric(x1)
f1 = x1_.function()
ax.scatter(x1[0],x1[1],f1,color='yellow')


x = [x1_a0,x1[0] ] 
y = [x2_a0,x1[1]]
z = [f0,f1]
ax.plot(x,y,z)
# """initial point"""
# x1_a0= -0.9
# x2_a0= -0.5
# a_0 = np.array ([[x1_a0],[x2_a0]])
# x0 = quadric(a_0)
# x_0 = np.array ([[x0.x1],[x0.x2]])
# del_ = x0.diff()
# f0 = x0.function()
# ax.scatter(x0.x1, x0.x2,f0, color ="purple")

# hessen_matrix= x0.hessen()
# print(is_pos_def(hessen_matrix))
# hessen_pd= x0.hessen_pd()
# print(f"hessen_p.d: {hessen_pd}")
# inv_hessen = inv(hessen_matrix)
# print(f"hessen_matrix:{hessen_matrix}")
# print(inv_hessen)
# print(del_)
# change = np.matmul(inv_hessen,del_)
# print(change)
# print(x_0)
# x1 = x_0 - change 
# print(f"x1: {x1}")
# x1_ = quadric(x1)
# f1 = x1_.function()
# ax.scatter(x1[0],x1[1],f1,color='red')

""" second iteration """
x0 = quadric(x1)
x_0 = x1
x_0 = np.array ([[x0.x1],[x0.x2]])
print(f"x_0:{x_0}")
del_ = x0.diff()
f0 = x0.function()
ax.scatter(x0.x1, x0.x2,f0, color ="green")

hessen_matrix= x0.hessen()
print(f"hessen_matrix: {hessen_matrix}")
inv_hessen = inv(hessen_matrix)
print(hessen_matrix)
print(inv_hessen)
print(del_)
change = np.matmul(inv_hessen,del_)
print(change)
print(x_0)
x1 = x_0 - change 
print(x1)
x1_ = quadric(x1)
f1 = x1_.function()
ax.scatter(x1[0],x1[1],f1,color='yellow')
x = [x0.x1,x1[0] ] 
y = [x0.x2,x1[1]]
z = [f0,f1]
ax.plot(x,y,z)
plt.show()
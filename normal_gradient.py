import matplotlib.pyplot as plt
import numpy as np
import math

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

def next_point (x0,alpha,norm_del_fx0):
    x1 = x0 - alpha*norm_del_fx0
    return x1 

"""initial point"""
x1_a0= 0.55
x2_a0= 0.7
a_0 = np.array ([x1_a0,x2_a0])
alpha = 0.02

x0 = quadric(a_0)
x_0 = np.array ([x0.x1,x0.x2])
del_x0x1,delx0x2,norm= x0.diff()
norm_delfx_0 = x0.norm_delfx()
fx0 = x0.function()
print(fx0)
#print(del_x0x1,delx0x2,norm,norm_delfx_0)
x1 = next_point(x_0,alpha,norm_delfx_0)
#print(x1)

x1 = quadric(x1)
x_1 = np.array([x1.x1,x1.x2])
# del_x0x1,delx0x2,norm= x1.diff()
# norm_delfx_0 = x1.norm_delfx()
# fx1 = x1.function()
# print(f"fx1: {fx1}")
# x2 = next_point(x_1,alpha,norm_delfx_0)
# print(x2)

# x2 = quadric(x2)
# x_2 = np.array ([[x2.x1],[x2.x2]])
# del_x0x1,delx0x2,norm= x2.diff()
# norm_delfx_0 = x2.norm_delfx()
# fx2 = x2.function()
# print(f"fx2: {fx2}")
# x3 = next_point(x_1,alpha,norm_delfx_0)
# #print(x3)

def distance(x_0,x_1):
    del_x = x_1[0]-x_0[0]
    del_y = x_1[1]-x_0[1]
    dist = np.sqrt(del_x**2+del_y**2)
    return dist

x = np.array ([x1_a0,x2_a0])
fx = fx0 
print(x)
ax.scatter(x[0], x[1], fx0, color ="blue")
count = 0 
dist = distance(x_0,x_1)
while  (dist > 0.02 or count < 100):
#for i in range(10):
    oper = quadric(x)
    x = np.array([oper.x1,oper.x2])
    print(f"x_{count} = {x}")
    del_x1,delx2,norm= oper.diff()
    norm_delfx = oper.norm_delfx()
    print(f"norm_delfx:{norm_delfx}")
    fx = oper.function()
    print(f"fx_{count}:{fx:.3}")
    #print(del_x0x1,delx0x2,norm,norm_delfx_0)
    x1 = next_point(x,alpha,norm_delfx)
    #
    print(f"x1: {x1[0]}")
    print(quadric(x1).function())
    gx = [x[0], x1[0]]
    gy = [x[1], x1[1]]
    gz = [fx,quadric(x1).function()]
    ax.plot(gx,gy,gz)
    print(f"x_next:{x1}")
    dist = distance(x,x1)
    print(f"dist:{dist}")
    x = x1 
      
    ax.scatter(x[0], x[1], fx, color ="red")
    count = count +1
    print("========================================")
plt.show()
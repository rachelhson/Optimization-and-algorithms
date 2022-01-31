import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

class quadric:
    
    def __init__(self, matrix):
        self.x1 = matrix[0] 
        self.x2 = matrix[1]  
    
    def function(self):
        f = self.x1**2+self.x1*self.x2+self.x2**2
        return f    

    def diff(self):
        del_x1 = 2*self.x1 + self.x2 
        del_x2 = self.x1 + 2*self.x2
        norm = np.sqrt(del_x1**2+del_x2**2)
        return del_x1,del_x2,norm
    def norm_delfx(self):
        del_x1,del_x2,norm = self.diff()
        del_fx = np.array ([del_x1,del_x2])
        norm_delfx = np.array ([del_x1/norm,del_x2/norm])
        return norm_delfx
   
""" initial point """
x1_a0=0.8
x2_a0=-0.25
a_0 = np.array ([[x1_a0],[x2_a0]])
eps = 0.075

"""graph for the function"""
def f(x1, x2):
    return  x1**2+x1*x2+x2**2

x1 = np.linspace(-1, 1, 30)
x2 = np.linspace(-1, 1, 30)

X, Y = np.meshgrid(x1, x2)
Z = f(X, Y)
print(f"minimum Z value :{np.min(Z)}")
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 20, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');

""" initial point"""
def next_x(x0,eps,norm_grad):
    x1 = x0 - eps*norm_grad 
    return x1 

f0 = 1
f1 = 0
i = 1
count = 1

print(quadric(a_0).function())
while (f0 > f1):
#for count in range(10): 
    x0 = quadric(a_0)
    del_x0x1,delx0x2,norm= x0.diff()
    norm_grad = x0.norm_delfx()
    eps = 0.075 * i 
    x1 = next_x(a_0,eps,norm_grad)
    ax.scatter(quadric(x1).x1[0],quadric(x1).x2[0],quadric(x1).function()[0])
    ax.text(quadric(x1).x1[0],quadric(x1).x2[0],quadric(x1).function(),(count,quadric(x1).function()[0]), color='black')  
    print(f'norm_grad:{norm_grad}|eps:{eps}')
    print(f'x{count-1}: {a_0}')
    print(f'x{count}: {x1}')
    f0 = x0.function()
    print(f'f{count-1}: {f0}')
    f1 = quadric(x1).function()
    print(f'f{count}:{f1}')
    a_0 = x1 
    i = 2*i
    count +=1
    ax.scatter(x0.x1[0],x0.x2[0],f0[0])
    ax.text(x0.x1[0],x0.x2[0],f0[0], (count,f0[0]), color='black')    
    print("==========================")
print(count)
if count < 2 : 
        print("warning")
plt.show()

""" Fibonacci for Quadric Form """
import matplotlib.pyplot as plt
import numpy as np
import math

def distance (delx,dely):
    dist = np.sqrt(delx**2+dely**2)
    return dist 

def distance_array (a,b):
    delx = a[0]-b[0]
    dely = a[1]-b[1]
    dist = np.sqrt(delx**2+dely**2)
    return dist

def fibonacci_array(n): 
    a = 0
    b = 1
    f_array = []        
    for i in range(0, n+1):
        if i == 0: # initial
            f_array.append(0)
        elif i == 1: # initial
            f_array.append(1)
        if i > 1: #starting fibonaci n = 2
            c = a + b
            a = b
            b = c
            f_array.append(b)
    return f_array   

fibonacci_sequence = fibonacci_array(20)
fibonacci_sequence = fibonacci_sequence[2:20]
print(f'fibonacci_sequence: {fibonacci_sequence}')

class quadric:
    
    def __init__(self, matrix):
        self.x1 = matrix[0] 
        self.x2 = matrix[1]  
    
    def function(self):
        f = self.x1**2+self.x1*self.x2+self.x2**2
        return f  

    

delta = 0.01
tol = 0.01/3
"""compute N - the number of iteration"""

x1_a0= 0.278
x2_a0=-0.289
a_0 = np.array ([[x1_a0],[x2_a0]])
print(a_0)

x1_b0= 0.343
x2_b0=-0.949
b_0 = np.array ([[x1_b0],[x2_b0]])
l = distance_array(a_0,b_0)
print(f"distance_(a0-b0): {l}")

f_n_ =  (1+2*tol)/(delta/l)
print(f_n_)
ns = np.where(fibonacci_sequence>f_n_)
print(ns)
n = min(ns[0])
print (n)
print(fibonacci_sequence[n])

"""graph for the function"""

def f(x1, x2):
    return  x1**2+x1*x2+x2**2

x1 = np.linspace(-1, 1, 30)
x2 = np.linspace(-1, 1, 30)

X, Y = np.meshgrid(x1, x2)
Z = f(X, Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
#ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');


"""fibonacci search method """

fs = fibonacci_sequence[0:n+1] # show 0:n array
print(fs)
""" initial """
a = a_0
b = b_0
print(f"a0 : {a}")
print(f"b0 : {b}")
print(f"fa0:{quadric(a).function()}")
print(f"fb0:{quadric(b).function()}")
rho = 1-fs[n-1]/fs[n] # F8/F9
new_a = a_0+rho*(b_0-a_0)
new_b = a_0+(1-rho)*(b_0-a_0)
print(f"rho: {rho},{fs[n-1]},{fs[n]}")
print(f"a1 : {new_a}")
print(f"b1 : {new_b}")
f1 =quadric(new_a).function()
f2 =quadric(new_b).function()
print(f"f1: {f1}")
print(f"f2: {f2}")
l = distance_array(a,new_b)
print(f"distance_(a0-b1):{l}")
x = [a_0[0][0],b_0[0][0]]
y = [a_0[1][0],b_0[1][0]]
z = [ 0, 0]
ax.plot(x,y,z)

count = 2
for i in range(n-2, -1, -1): 
    print(f"i:{i}")
    print(f"========={count}==========")
    rho = 1-fs[i]/fs[i+1]
    print(f"rho:{fs[i],fs[i+1]},{rho}")
    
    if f1<f2:
        b = new_b
        new_b = new_a
        new_a = a + rho*(b-a) 
        f2 = f1 
        f1 = quadric(new_a).function()
        distance = distance_array(a,b)
        print(f" a{count+2} gs:{a}")
        print(f" b{count+2} gs:{b}")
        print(f"new_a:{new_a}, new_b:{new_b}")
        print(f"distance _(a{count}-b{count}):{distance}")
        distance_ = distance_array(a,new_b)
        print(f"new_distance:{distance_} ")
        print(f"f1_new:{f1}, f2__new:{f2}")
        print(f" f1-1:{quadric(a).function()}")
        print(f" f2-1:{quadric(b).function()}")
        #ax.scatter(quadric(a).x1,quadric(a).x2,color='b',alpha=0.1)
        x = [quadric(a).x1[0], quadric(b).x1[0]]
        y = [quadric(a).x2[0], quadric(b).x2[0]]
        z = [ count-1, count-1]
        ax.plot(x,y,z)

    else : 
        a = new_a
        new_a = new_b 
        new_b = a +(1-rho)*(b-a)
        f1 = f2
        f2 = quadric(new_b).function()
        distance = distance_array(a,b)
        print(f" a{count+2}_gs :{a}")
        print(f" b{count+2}_gs :{b}")
        print(f"new_a:{new_a}, new_b:{new_b}")
        print(f"distance _(a{count}-b{count}):{distance}")
        print(f" f1-1:{quadric(a).function()}")
        print(f" f2-1:{quadric(b).function()}")
        distance_ = distance_array(new_a,b)
        print(f"new_distance:{distance_} ")
        #ax.scatter(quadric(a).x1,quadric(a).x2,color='b',alpha=0.1)
        x = [quadric(a).x1[0], quadric(b).x1[0]]
        y = [quadric(a).x2[0], quadric(b).x2[0]]
        z = [ count-1, count-1]
        ax.plot(x,y,z)
    
    count +=1
    

plt.show()



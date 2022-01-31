from bracket import bracket_function
from golden_section import gold_section_function
import numpy as np
import function
import matplotlib.pyplot as plt

def g_(x_0):
    norm_g = function.quadric(x_0).norm_delfx()
    return norm_g

def d_(H,g):
    d = np.matmul(-H,g)
    d_x1 = d[0]
    d_x2 = d[1]
    norm = np.sqrt(d_x1**2+d_x2**2)
    norm_d = np.array ([d_x1/norm,d_x2/norm])
    return norm_d

def next_x(x_0, alpha,d):
    next_x_ = x_0 + alpha*d
    return next_x_

def normalization(g):
    g_x1 = g[0]
    g_x2 = g[1]
    norm = np.sqrt(g_x1**2+g_x2**2)
    return norm
"""algorithm start"""
x1_a0= -0.9
x2_a0= -0.5
x_0 = np.array ([[x1_a0],[x2_a0]])
f0 = function.quadric(x_0).function()
print(f"f0: {f0}")

H0 = np.identity(2)
def dfp_algorithm(H0,x_0):
    f0 = function.quadric(x_0).function()
    g0 = g_(x_0)
    d0 = d_(H0,g0)
    print(d0)
    alpha_range = bracket_function(x_0,d0)
    alpha0 = gold_section_function(alpha_range,x_0)
    print(f'alpha:{alpha0}')
    print(f"d0:{d0}")
    x1 = next_x(x_0,alpha0,d0)
    f1 = function.quadric(x1).function()

    delta_x0 =x1-x_0
    g1 = g_(x1)
    print(f"g1:{g1}")
    delta_g0 = g1-g0
    print(delta_g0)
    """ For compute H """
    a = np.matmul(delta_x0,delta_x0.T)
    b = np.matmul(delta_x0.T,delta_g0) #scalar
    c = np.matmul(H0,delta_g0)

    norm1 =a
    print(norm1)
    denorm1=b
    print(denorm1)
    norm2 = np.matmul(c,c.T)
    print(norm2)
    denorm2 = np.matmul(delta_g0.T,H0)
    denorm22 = np.matmul(denorm2,delta_g0) #scalar
    print(denorm22.item(0))

    H1 = H0 + norm1/denorm1.item(0)-norm2/denorm22
    print(f"H1 : {H1}")
    return H1,x1,f0,f1
"""graph for the function"""
fig = plt.figure()
ax = plt.axes(projection='3d')
def f(x_1, x_2):
    return  (x_2-x_1)**4 +12*x_1*x_2-x_1+x_2-3

x1 = np.linspace(-1, 1, 30)
x2 = np.linspace(-1, 1, 30)

X, Y = np.meshgrid(x1, x2)
Z = f(X, Y)
print(f"minimum Z value :{np.min(Z)}")
fig = plt.figure()
ax = plt.axes(projection='3d')

jet= plt.get_cmap('rainbow')
colors = iter(jet(np.linspace(0,1,30)))

ax.contour3D(X, Y, Z, 50, cmap='binary')
x1_a0= -0.9
x2_a0= -0.5
ax.scatter(x_0[0],x_0[1],f0, color='purple')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');

f0 = 1
f1 = 0
count = 0
norm_diff  = 1
while (norm_diff > 0.01 or f0 > f1 or count > 30):
    H1,x1,f0,f1 =dfp_algorithm(H0,x_0)
    print(f"x1 : {x1}")
    diff = (x1-x_0)
    norm_diff = normalization(diff)
    print(f'diff:{norm_diff}')
    ax.contour3D(X, Y, Z, 50, cmap='binary')

    ax.scatter(x1[0],x1[1],f1, color=next(colors))
    H0 = H1
    x_0 = x1
    count+=1
    print(f'count: {count}')
    print(f"f0:{f0}, f1:{f1}")

plt.show()


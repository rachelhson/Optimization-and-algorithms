import numpy as np
import matplotlib.pyplot as plt
x_1 = np.linspace(-1, 1, 30)
x_2 = np.linspace(-1, 1, 30)
def function(x1,x2):
        f = (x2-x1)**4 +12*x1*x2-x1+x2-3
        return f
"""Visualization"""
# Contour plot: With the global minimum showed as "X" on the plot
x, y = np.meshgrid(x_1, x_2)
z = function(x, y)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(x, y, z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')


# Create particles
numb_particles = 20
np.random.seed(100)
x = np.random.rand(2, numb_particles)
v = np.random.randn(2, numb_particles) * 0.5

# #k=0
p = x
fp = function(p[0],p[1])
g = p[:,fp.argmin()]
fg = fp.min()



def update(x,p,g,v):
        c1 = c2 = 0.3
        w = 0.8
        r,s = np.random.rand(2)
        v_new = w*v +c1*r*(p-x)+c2*s*(g.reshape(-1,1)-x)
        x_new = x+v_new
        fx_new = function(x_new[0],x_new[1])
        p[:,(fp>fx_new)]=x_new[:,(fp>fx_new)]
        fg = function(g[0],g[1])
        g_new = p[:,function(p[0],p[1]).argmin()]
        fg=fp.min()
        print(f"best: {g[0]},{g[1]},{fg}")

        return x_new,p,g_new,v_new

for i in range(100):
        x_new,p,g_new,v_new= update(x,p,g,v)
        x = x_new
        p = p
        g = g_new
        v = v_new
        fg = function(g[0],g[1])
        #ax.scatter(g[0],g[1],function(g[0],g[1]))

        ax.scatter(p[0],p[1],function(p[0],p[1]))

plt.show()

print(f" PSO best solution :{g[0],g[1],function(g[0],g[1])}")






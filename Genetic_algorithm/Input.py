import numpy as np
import matplotlib.pyplot as plt
x_1 = np.linspace(-1, 1, 30)
x_2 = np.linspace(-1, 1, 30)

def function(x1, x2):
    f = (x2 - x1) ** 4 + 12 * x1 * x2 - x1 + x2 - 3
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

#define resolution
generation_n = 50
population = 20
xover_rate = 0.75
mutate_rate = 0.01
bit_n = 16
#obj_fcn = function(x1,x2)
var_n = 2
range_ =[-1, 1]
popu = np.random.randint(0, 2, size=(population,bit_n*var_n))
print(popu)
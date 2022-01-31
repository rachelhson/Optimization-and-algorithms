import numpy as np
from function import *
import matplotlib.pyplot as plt


""" Main loop of GA"""
upper = []
average= []
lower =[]
x1_ =[]
x2_=[]
for i in range(generation_n):
    fun_value,numb1_,numb2_= evalpopu(popu,population,bit_n,range_)
    #fill obj function matrix
    x1 = numb1_[np.argmax(fun_value)]
    x2 = numb2_[np.argmax(fun_value)]
    upper.append(-np.max(fun_value))
    average.append(-np.average(fun_value))
    lower.append(-np.min(fun_value))
    popu = nextpopu(popu,fun_value,xover_rate,mutate_rate)
    x1_.append(x1)
    x2_.append(x2)





"""iteration vs. obj_value"""
fig = plt.figure()
iteration = np.arange(generation_n)
plt.plot(iteration,lower,color='blue', label = 'lower', marker='_', linestyle=':')
plt.plot(iteration,upper,color='red', label = 'upper',marker='o', linestyle='-')
plt.plot(iteration,average,color='orange', label='average',marker='+',linestyle=':')
plt.legend(loc='upper right')
plt.xlabel("Iteration")
plt.ylabel("Fitness value")
plt.show()


"""Visualization"""

fig = plt.figure()
ax = plt.axes(projection='3d')
# Contour plot: With the global minimum showed as "X" on the plot
x_1 = np.linspace(-1, 1, 50)
x_2 = np.linspace(-1, 1, 50)
x, y = np.meshgrid(x_1, x_2)
z = -function(x, y)
ax.contour3D(x, y, z, 50, cmap='binary')
ax.scatter(x1_,x2_,upper)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
print(len(x1_))
print(f" x1 : {x1_[generation_n-1]:.2f}, x2: {x2_[generation_n-1]:.2f}, f:{upper[generation_n-1]:.2f}")

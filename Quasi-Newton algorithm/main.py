from bracket import bracket_function
from golden_section import gold_section_function
import numpy as np
import function
import matplotlib.pyplot as plt

### normalized del
def g_(x):
    norm_g = function.quadric(x).norm_delfx()
    return norm_g

## normalized direction
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
x1_a0= 0.55
x2_a0= 0.7
x_0 = np.array ([[x1_a0],[x2_a0]])
f0 = function.quadric(x_0).function()
print(f"f0: {f0}")

H0 = np.identity(2)

def rank_one_algorithm(H0,x_0): # x_0 previous point H0 previous matrix
    f0 = function.quadric(x_0).function()
    ## compute direction
    g0 = g_(x_0)
    d0 = d_(H0,g0)
    ## select alpha
    alpha_range = bracket_function(x_0,d0)
    alpha0 = gold_section_function(alpha_range,x_0)
    print(f'alpha:{alpha_range}')
    print(f'alpha:{alpha0}')
    print(f"d0:{d0}")

    ## find next point
    x1 = next_x(x_0,alpha0,d0)
    f1 = function.quadric(x1).function()

    ## find new H matrix
    delta_x0 =x1-x_0
    g1 = g_(x1)
    print(f"g1:{g1}")
    delta_g0 = g1-g0
    a = (delta_x0-np.matmul(H0,delta_g0))
    b = (delta_x0-np.matmul(H0,delta_g0)).T
    norm = np.matmul(a,b)
    denorm = np.matmul(delta_g0.T,(delta_x0-np.matmul(H0,delta_g0)))
    delta_H0 = norm/denorm.item(0)
    print(f"delta_HO:{delta_H0}")
    H1 = H0 + delta_H0
    print(f"New_H:{H1}")

    return H1,x1,f0,f1,g1

"""graph for the function"""
def f(x_1, x_2):
    return  (x_2-x_1)**4 +12*x_1*x_2-x_1+x_2-3

x1 = np.linspace(-1.5, 1.5, 30)
x2 = np.linspace(-1.5, 1.5, 30)

fig = plt.figure()
#ax = plt.axes(projection='3d')
ax = plt.axes(projection='3d')
X, Y = np.meshgrid(x1, x2)
Z = f(X, Y)
print(f"minimum Z value :{np.min(Z)}")

jet= plt.get_cmap('rainbow')
colors = iter(jet(np.linspace(0,1,30)))
ax.contour3D(X, Y, Z, 50, cmap='binary')
## plot start point
x1_a0= 0.55
x2_a0= 0.7
ax.scatter(x1_a0,x2_a0,f0, color='purple')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');

f0 = 1
f1 = 0
count = 0
norm_diff  = 1
#while (f0 > f1):
while (norm_diff  >0.01):
#for count in range(22):
    H1,x1,f0,f1,g1 =rank_one_algorithm(H0,x_0)
    print(f"x1:{x1}, f1:{f1}")
    ## when H1 is positive definite
    diff = (x1-x_0)
    norm_diff = normalization(diff)
    print(f"x1-x0: {diff}")
    x = [x_0[0],x1[0]]
    y = [x_0[1],x1[1]]
    z = [f0,f1]
    #ax.plot(x,y,z)
    ax.scatter(x1[0],x1[1],f1, color=next(colors))
    #plt.annotate(x1[0],x1[1],'%s' % (str(count)), "red", color='red')
    H0 = H1
    x_0 = x1
    count+=1
    print(f'count: {count}')
    print(f"f0:{f0}, f1:{f1}")

plt.show()



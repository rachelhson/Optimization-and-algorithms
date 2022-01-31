import numpy as np
import matplotlib.pyplot as plt
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
ax.contour3D(X, Y, Z, 20, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');

""" initial point """
x1_a0= 0.55
x2_a0= 0.7
x_0 = np.array ([[x1_a0],[x2_a0]])
eps = 0.075

def next_x(x0,eps,norm_grad):
    x1 = x0 - eps*norm_grad 
    return x1 

def bracket(x_0):
    f0 = 1
    f1 = 0
    i = 1
    count = 1
    alpha_range =[]
    alpha_range_ =[]
    while (f0 > f1):
        print("=====bracket==========")
        x0 = quadric(x_0)
        del_x0x1,delx0x2,norm= x0.diff()
        norm_grad = x0.norm_delfx()
        eps = 0.075*i
        x1 = next_x(x_0,eps,norm_grad)
       
        print(f'norm_grad:{norm_grad}|eps:{eps}')
        print(f'f{count}:{f1}')
        f0 = x0.function()
        print(f'f{count-1}: {f0}')
        f1 = quadric(x1).function()
        i = 2*i
        count +=1
        print(eps)
        alpha_range.append(eps)
        print("==========================")
    if len(alpha_range) < 2 :
        x1 = x1
        f1 = quadric(x1).function()
        norm_grad = quadric(x1).norm_delfx()
        eps = 0.075 * 2
        x2 = next_x(x1,eps,norm_grad)
        f2 = quadric(x2).function()
        if f2 > f1 : 
            x2 = x1
            alpha_range.append(eps)
            
        print("===WARNING===")
    # if len(alpha_range) == 1:
    #     eps_ = 0.075*count*2
    #     alpha_range_.append(eps_)
    #     alpha_range.append(eps_)
    # else: 
    #     eps_ = 2**(len(alpha_range))*(0.075)
    #     alpha_range_.append(eps_)
    #     alpha_range.append(eps_) 
    return x1, alpha_range

#plt.show()
"""initial and last"""
x1_a0= 0.55
x2_a0= 0.7
x_0 = np.array ([[x1_a0],[x2_a0]])

""" golden section """
def gold_section(alpha_range):
    delta = 0.01
    rho = 0.382
    """compute N - the number of iteration"""
    a_0=alpha_range[0]
    #print(a_0)

    b_0=alpha_range[-1]

    del_ = b_0-a_0
    l = del_
    #print(l)

    n_ = math.log(delta/l)/math.log(0.618)
    #print(n_)
    n = math.ceil(n_)
    print (f"the number of iteration : {n}")
    fig = plt.figure()
    ax1 = plt.axes()
    a = a_0
    b = b_0
    print(f"a0 : {a}")
    print(f"b0 : {b}")
    new_a = a_0 + rho* (b_0-a_0)
    new_b = a_0 +(1-rho)*(b_0-a_0)
    x = quadric(x_0)
    norm_grad = x.norm_delfx()
    new_af = x_0-new_a*norm_grad 
    new_bf = x_0-new_b*norm_grad 
    print(f"a1 : {new_a}")
    print(f"b1 : {new_b}")
    f1 =quadric(new_af).function()
    f2 =quadric(new_bf).function()
    print(f1)
    print(f2)

    alpha = []
    for i in range(n):
        if f1 < f2:
            b = new_b
            new_b = new_a
            new_a = a + rho*(b-a)
            new_af = x_0-new_a*norm_grad
            f2 = f1
            f1 = quadric(new_af).function()
            print(f"range:{a,b}")
            print(f" f1-1:{f1}")
            print(f" f2-1:{f2}")
            alpha.append([a,b])
            x =[a,b]
            y =[i,i]
            ax1.plot(x,y)   
            
        else : 
            a = new_a
            new_a = new_b 
            new_b = a +(1-rho)*(b-a)
            new_bf =x_0-new_b*norm_grad
            f1=f2
            f2=quadric(new_bf).function()
            print(f"range:{a,b}")
            print(f" f1-2:{f1}")
            print(f" f2-2:{f2}")
            alpha.append([a,b])
            x =[a,b]
            y =[i,i]
            ax1.plot(x,y)
        print(f"========={i}==========")
    #plt.show()

    """ decide alpha """
    alpha = np.average(alpha[-1])
    print(f'alpha:{alpha}')
    return alpha

def distance(x_0,x_1):
    del_x = x_1[0]-x_0[0]
    del_y = x_1[1]-x_0[1]
    dist = np.sqrt(del_x**2+del_y**2)
    return dist 
dist = 0.8
iteration = 0

while (iteration < 10 or dist < 0.01):
    print(f"**********************{iteration}***********************")
    x1, alpha_range = bracket(x_0)
    print(f"alpha_range : {alpha_range}")
    
    if len(alpha_range) > 2: 
        alpha_range= alpha_range [-3:]     
        print(f"modified_alpha_range: {alpha_range}")
        alpha = gold_section(alpha_range)
        print(f'alpha:{alpha}')
        x0 = quadric(x_0)
        f0 = x0.function()
        del_x0x1,delx0x2,norm= x0.diff()
        norm_grad = x0.norm_delfx()
        print(norm)
        x_1 = next_x(x_0,alpha,norm_grad)
        x1= quadric(x_1)
        f1 = x1.function()
        print(x_1)
        ax.scatter(x_1[0][0],x_1[1][0],f1, color ='red')
        ax.scatter(x_0[0][0],x_0[1][0],f0, color='purple')
        x = [x_1[0][0],x_0[0][0]]
        y = [x_1[1][0],x_0[1][0]]
        z = [f1[0],f0[0]]
        print(x,y,z)
        ax.plot(x,y,z)
        dist = distance(x_0,x_1)
        x_0 = x_1
        print(f"dist : {dist}")
        iteration = iteration + 1
        print("***********************************************************")
    else: 
        alpha = gold_section(alpha_range)
        print(f'alpha:{alpha}')
        x0 = quadric(x_0)
        f0 = x0.function()
        del_x0x1,delx0x2,norm= x0.diff()
        norm_grad = x0.norm_delfx()
        print(norm)
        x_1 = next_x(x_0,alpha,norm_grad)
        x1= quadric(x_1)
        f1 = x1.function()
        print(x_1)
        ax.scatter(x_1[0][0],x_1[1][0],f1, color ='red')
        ax.scatter(x_0[0][0],x_0[1][0],f0, color='purple')
        x = [x_1[0][0],x_0[0][0]]
        y = [x_1[1][0],x_0[1][0]]
        z = [f1[0],f0[0]]
        ax.plot(x,y,z)
        dist = distance(x_0,x_1)
        x_0 = x_1
        print(f"dist : {dist}")
        iteration = iteration + 1
        print("***********************************************************")
        
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import function

""" finding minimum alpha """

""" initial point """
x1_a0= -0.9
x2_a0= -0.5
x_0 = np.array ([[x1_a0],[x2_a0]])
eps = 0.075

def next_x(x0,eps,norm_grad):
    x1 = x0 - eps*norm_grad
    return x1 

def bracket_function(x_0,norm_grad):
    f0 = 1
    f1 = 0
    i = 1
    count = 1
    alpha_range =[0]
    alpha_range_ =[]
    f =[]

    while (f0 > f1):
        # print(f"count: {count}")
        x0 = function.quadric(x_0)
        del_x0x1,delx0x2,norm= x0.diff()
        norm_grad = x0.norm_delfx()
        eps = 0.075*i
        x1 = next_x(x_0,eps,norm_grad)
        #print(f'norm_grad:{norm_grad}|eps:{eps}')
        # print(f'f{count}:{f1}')
        f0 = x0.function()
        # print(f'f{count-1}: {f0}')
        f1 = function.quadric(x1).function()
        f.append(f1)
        i = 2*i
        count +=1
        # print(eps)
        alpha_range.append(eps)
        # print("==========================")

    if len(alpha_range) == 1:
        eps_ = 0.075*count*2
        alpha_range_.append(eps_)
        alpha_range.append(eps_)
    else: 
        eps_ = 2**(len(alpha_range))*(0.075)
        alpha_range_.append(eps_)
        alpha_range.append(eps_)
    # print(f)
    return alpha_range

#plt.show()





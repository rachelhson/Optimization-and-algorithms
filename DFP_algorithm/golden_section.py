import numpy as np
import math
import matplotlib.pyplot as plt
import function


""" golden section """
def gold_section_function(alpha_range,x_0):
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
    # print (f"the number of iteration : {n}")
    #fig = plt.figure()
    #ax1 = plt.axes()
    a = a_0
    b = b_0
    # print(f"a0 : {a}")
    # print(f"b0 : {b}")
    new_a = a_0 + rho* (b_0-a_0)
    new_b = a_0 +(1-rho)*(b_0-a_0)
    x = function.quadric(x_0)
    norm_grad = x.norm_delfx()
    new_af = x_0-new_a*norm_grad
    new_bf = x_0-new_b*norm_grad
    # print(f"a1 : {new_a}")
    # print(f"b1 : {new_b}")
    f1 =function.quadric(new_af).function()
    f2 =function.quadric(new_bf).function()
    # print(f1)
    # print(f2)

    alpha = []
    for i in range(n):
        if f1 < f2:
            b = new_b
            new_b = new_a
            new_a = a + rho*(b-a)
            new_af = x_0-new_a*norm_grad
            f2 = f1
            f1 = function.quadric(new_af).function()
            # print(f"range:{a,b}")
            # print(f" f1-1:{f1}")
            # print(f" f2-1:{f2}")
            alpha.append([a,b])
            x =[a,b]
            y =[i,i]
            #ax1.plot(x,y)

        else :
            a = new_a
            new_a = new_b
            new_b = a +(1-rho)*(b-a)
            new_bf =x_0-new_b*norm_grad
            f1=f2
            f2=function.quadric(new_bf).function()
            # print(f"range:{a,b}")
            # print(f" f1-2:{f1}")
            # print(f" f2-2:{f2}")
            alpha.append([a,b])
            x =[a,b]
            y =[i,i]
            #ax1.plot(x,y)
        # print(f"========={i}==========")
    # plt.show()
    alpha = np.average(alpha[-1])
    return alpha



import numpy as np
from operator import xor

generation_n = 50
population = 20
xover_rate = 0.75
mutate_rate = 0.01
bit_n = 16
#obj_fcn = function(x1,x2)
var_n = 2
range_ =[-1, 1]
popu = np.random.randint(0, 2, size=(population,bit_n*var_n))

def function(x1, x2):
    f = -(x2 - x1) ** 4 - 12 * x1 * x2 + x1 - x2 + 3
    return f


def bit2numb(bit,range_): # bit should be list
     bit_str= ''.join(str(e) for e in bit) # change bit into string
     integer = int(bit_str,2)
     num = range_[0]+integer*(range_[1]-range_[0])/(2**len(bit)-1)

     return num

def evalpopu(popu,population,bit_n,range_):
    fitness_result =[]
    numb1_ = []
    numb2_ =[]
    for i in range(population):
        popu_ = popu[i]
        numb1 = bit2numb(popu_[:bit_n],range_)# from 0 to 16th - 16 bit
        numb2 = bit2numb(popu_[bit_n:],range_) #from 17th to 32th - 16 bit
        fitness = function(numb1,numb2)
        fitness_result.append(fitness)
        numb1_.append(numb1)
        numb2_.append(numb2)
    return fitness_result,numb1_,numb2_

def nextpopu(popu,fitness_result,xover_rate,mutate_rate):
    new_popu = popu
    popu_s = population
    string_length = bit_n
    # rescaling the fitness
    fitness_result = fitness_result - np.min(fitness_result)
    # find the best two and keep them
    tem_fitness = fitness_result
    # find the best
    best_index = np.argmax(tem_fitness)
    #print(f"best_index:{best_index}")
    best_obj = np.max(tem_fitness)
    tem_fitness[best_index]=min(tem_fitness)
    #next best
    next_index = np.argmax(tem_fitness)
    #print(f"next_index:{next_index}")
    #print(popu)
    new_popu[0] = popu[best_index]
    new_popu[1] = popu[next_index]

    total = sum(fitness_result)
    if total ==0:
        fitness_result_ = np.ones(population)/population
    cum_prob = np.cumsum(fitness_result)/total
    print(f"cumsum result:{np.cumsum(fitness_result)}")
    #selection and cross over
    for i in range(2,int(population/2)):
        temp = np.where(cum_prob-np.random.uniform(0, 1)>0)[0]

        parent1 = popu[temp[0]]
        #print(f"parent1:{parent1}")
        temp = np.where(cum_prob-np.random.uniform(0, 1)>0)[0]

        parent2 = popu[temp[0]]
        #print(f"parent2:{parent2}")
        # do cross over
        if np.random.uniform(0, 1)<xover_rate:
            #perform cross over
            xover_point = int(np.ceil(np.random.uniform(0,1)*(bit_n-1)))
            a= parent1[0:xover_point]
            b= parent2[xover_point:bit_n*var_n]
            #print(f"cross overed :{np.append(a,b)}")
            new_popu[2*i]= np.append(a,b)
            c=parent2[0:xover_point]
            d=parent1[xover_point:bit_n*var_n]
            new_popu[2*i+1]= np.append(c,d)
            #print(f"cross overed :{np.append(c, d)}")
            #print(f"new_popu: {new_popu}")
        # mutation
        mask =np.random.randint(0, 2, size=(population,bit_n*var_n))<0.01
        #print(f"mask:{mask}")
        new_popu_bool = new_popu<0.5
        #print(f"new_popu_bool:{new_popu_bool}")
        new_popu= np.multiply(xor(new_popu_bool,mask),1)
        # restore the elite
        new_popu[0] = popu[best_index]
        new_popu[1] = popu[next_index]
        #print(new_popu)
    return new_popu














import cg_class

n = 2 ## conjugate gradient algorithm iteration needs # of variable

""" first """
x0 = cg_class.cg(0, 0)
print(f" g0 = {x0.g()}")
d0 = x0.d()
print(f" d0 = {d0}")
print(f"alpha0 = {x0.alpha()}")
new_x1 = x0.next_x1()
print(f"x1 = {new_x1}")

x1 = cg_class.cg(new_x1.item(0), new_x1.item(1))
print(f" g1 = {x1.g()}")
g1 = x1.g()
if g1.item(0) == 0 :
    if g1.item(1) == 0 :
        print ("g1 is zero, so stop here")
    else:
        print("g1 is not zero, so keep going")

beta = x0.beta(g1)
print(f" beta0 = {beta}")
d1= x0.d1(g1)
print(f" d1 = {d1}")
alpha1 = x1.alpha1(d1,g1)
print(f"alpha1={alpha1}")
new_x = x1.new_x2(alpha1,d1)
print(f"x2 = {new_x}")
x2 = cg_class.cg(new_x.item(0), new_x.item(1))
g2 = x2.g()
print(f"g2: {g2}")
if g2.item(0) == 0 :
    if g2.item(1) == 0 :
        print ("g1 is zero, so stop here")
    else:
        print("g1 is not zero, so keep going")

# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file.
"""
import numpy as np

def T1(x,y):
    b2_4ac = np.sqrt((20*x+6*y)**2 - 436*(x**2+y**2-400))
    t1 = (20*x+6*y - b2_4ac) / 218
    return t1

def T2(x,y,t1):
    t2 = np.sqrt(400-(x-10-10*t1)**2)/3 + y/3 - t1
    return t2

# approximate version
V_list = []
for x in np.linspace(160,180,40):
    for y in np.linspace(45,60,30):
        t1 = T1(x,y)
        t2 = T2(x,y,t1)
        v2 = 0
        v3 = 5*(50-t1-t2)
        D_tot = 10*t1 + 10 + v3*(50-t1-t2)/2 - (v3-10)**2/10
        V_list.append(D_tot/50)
V_avg = np.average(V_list)

# precise version
V_list = []
for x in np.linspace(160,180,40):
    for y in np.linspace(45,60,30):
        for v in np.linspace(5,10,10):
            t1 = T1(x,y)
            t2 = T2(x,y,t1)
            if t1 + t2 < 50:
                v2 = 10 - 5*t2 if t2 < 2 else 0
                v3 = v2 + 5*(50-t1-t2)
                if v2 == 0:
                    D_tot = 10*t1 + 10 + v3*(50-t1-t2)/2
                    if v3 >= 10:
                        D_tot = D_tot - (v3-10)**2/10
                else:
                    D_tot = 10*t1 + (10+v2)*t2/2 + (v2+v3)*(50-t1-t2)/2
                    if v3 >= 10:
                        D_tot = D_tot - (v3-10)**2/10
            else:
                t2 = 50 - t1
                v2 = 10 - 5*t2 if t2 < 2 else 0
                if v2 == 0:
                    D_tot = 10*t1 + 10
                else:
                    D_tot = 10*t1 + (10+v2)*t2/2
            D_tot = D_tot - (10-v)**2/5/2
            V_list.append(D_tot/50)
V_avg = np.average(V_list)

#%% 
# np.random.seed(2023)
# for i in range(50):
#     C, P = [], []
#     C = np.random.uniform([-5, -5, 0, 5], [5, 5, 0, 10])
#     P = np.random.uniform([165, -55, 0, 3], [175, -50, 0, 3])
#     x = P[0] - C[0]
#     y = C[1] - P[1]
#     v = C[3]

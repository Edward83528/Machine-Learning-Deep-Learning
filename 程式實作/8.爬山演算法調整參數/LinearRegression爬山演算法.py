# -*- coding: utf-8 -*-
"""
@author: zhong-wei
"""
import numpy as np

# Training Data
x_data = [1.0, 2.0, 3.0]
y_data = [3.0, 6.0, 9.0]

w = 1.0  # a random guess: random value


# our model forward pass
def forward(x):
    return x * w


# Loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


# compute gradient
def gradient(x, y):  # d_loss/d_w
    return 2 * x * (x * w - y)


# Before training
print("Prediction (before training)",  4, forward(4))

# Training loop
w_list = []
mse_list = []
iterations=[]
local_loss=[]
is_stop=0


#三層迴圈 1:灑幾點 2:走幾步 3:灑幾次

for epoch in range(3):  #how many times restart path
    
    maxSteps=100    # max steps in a trajectory軌跡
    
    for step in range(maxSteps):    #step  for the specigic tractory
        print("step =",step,"=============")
        w_old=w
        local_loss=[]
        local_w=[]
        #for check in np.arange(-1,2): #[-1,0,1]
        for check in np.arange(-2,3): #[-2,-1,0,1,2]
            w=w_old+(0.1*check)
            l_sum=0
            #===============================
            for x_val,y_val in zip(x_data,y_data):
                l=loss(x_val,y_val)
                l_sum+=1
            local_loss.append(l_sum)
            local_w.append(w)
            
            mse_list.append(l_sum)
            print("step =",step,"w=",w,"loss=",l_sum)
            print("MSE =",l_sum)
            if check==2:
                print(local_loss)
        min_index=np.argmin(local_loss)
        
        w=local_w[min_index]
        w_list.append(w)
        print(min_index,w,'xxxxxxxxx')
        if min_index==2: #original middle
            print(' stop .... w no change!!!!')
            is_stop=1
            break;

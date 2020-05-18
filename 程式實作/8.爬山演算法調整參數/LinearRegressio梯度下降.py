# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 09:34:01 2019

@author: huan
"""
from optmlib.get_w import grad_methon
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

#每一個epoch會有一個w,再以這個w去找新的w
#看執行結果在第14次就loss為0,但我們還是去跑100回合
for epoch in range(100):
    for x_val, y_val in zip(x_data, y_data):
        # Compute derivative w.r.t to the learned weights
        # Update the weights
        # Compute the loss and print progress
        grad = gradient(x_val, y_val)
        
        #w = w - 0.01 * grad    #w更新
        w=grad_methon(w,grad);  #移到外面function
        
        print("\tgrad: ", x_val, y_val, round(grad, 2))
        l = loss(x_val, y_val)
    print("progress:", epoch, "w=", round(w, 2), "loss=", round(l, 2))
    iterations.append(epoch)
    w_list.append(w)
    mse_list.append(round(l, 2))
# After training
print("Predicted score (after training)",  "4 hours of studying: ", forward(4))
# Plot it all
plt.plot(iterations, mse_list)
plt.ylabel('Loss')
plt.xlabel('iterations')
plt.show()

plt.plot(iterations, w_list)
plt.ylabel('w')
plt.xlabel('iterations')
plt.show()
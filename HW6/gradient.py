#程式參考網站 https://blog.csdn.net/u010960155/article/details/113776715 ，有稍作修改並已理解程式邏輯與梯度下降概念

import numpy as np
import time
 
def cal_grad(A, b, x):
    left = np.dot(np.dot(A.T, A), x)
    right = np.dot(A.T, b)
    gradient = left - right
    return gradient
 
# iteration
def gradDecent(x, A, b, learning_rate, step):
    for i in range(step):
        gradient = cal_grad(A, b, x)
        delta = learning_rate * gradient
        x = x - delta
    print('近似解x = {a}'.format(a=x))

A = np.array([[1.0, -2.0, 1.0], [0.0, 2.0, -8.0], [-4.0, 5.0, 9.0]])
b = np.array([0.0, 8.0, -9.0])
# Giveb A and b，the solution x is [29, 16, 3]
 
x0 = np.array([1.0, 1.0, 1.0])
learning_rate = 0.01
step = 1000000
 
gradDecent(x0, A, b, learning_rate, step)
#測試結果
'''
近似解x = [28.98272933 15.99042465  2.99763054] 與已知正確答案相差無幾
'''

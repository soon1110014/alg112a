import math
from numpy import arange
#暴力法
def f(x):
    a = x**2 - 3*x +1
    return a
for x in arange(-100, 100, 0.001):
    if abs(f(x)) < 0.001:
        print("x = ", x, "f(x) = ", f(x))
#測試結果
'''
x =  0.3820000004793087 f(x) =  -7.600107173422188e-05
x =  2.6180000004899853 f(x) =  -7.599890439280443e-05
'''

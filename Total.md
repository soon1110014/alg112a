# 期末習題整理
```
課程名稱：演算法
學號：110910122
學生：陳振順
授課老師： 陳鍾誠 教授
用途： 期初至期末作業總結
```

## Hw1 : 費氏數列迴圈版
```
# 參考老師查表法，修改for迴圈部分。
from datetime import datetime
fib = [None]*10000
fib[0] = 0
fib[1] = 1

def fib_n(n):
    if n < 0: raise
    if not fib[n] is None: return fib[n]
    for i in range(2, n + 1):
        fib[i] = fib[i - 1] + fib[i - 2]
    return fib[n]

#採用迴圈與不採用迴圈相比能更節省記憶體，提高速度，原因是迴圈只需要處理最後兩個數字。
n = 100
startTime = datetime.now()
print(f'fib_n({n})={fib_n(n)}')
endTime = datetime.now()
seconds = endTime - startTime
print(f'time:{seconds}')
```
## Hw2 : power2n 四種實作方法
```
#原創
#方法一(運算子)
def p1(n):
    return 2**n
print(f'方法一(運算子): {p1(2)}')
print("----------")

#方法二(運算子)
def p2(n):
    result = 1
    for _ in range(n):
        result *= 2
    return result
print(f'方法二(迴圈): {p2(3)}')
print("----------")

#方法三(位元運算)
def p3(n):
    return 1 << n
print(f'方法三(位元運算): {p3(3)}')
print("----------")

#方法四(遞迴)
def p4(n):
    if n == 0:
        return 1
    else:
        return 2 * p4(n - 1)
print(f'方法四(遞迴): {p4(3)}')
print("----------")

#測試結果

方法一(運算子): 4
----------
方法二(迴圈): 8
----------
方法三(位元運算): 8
----------
方法四(遞迴): 8
----------
```
## Hw3 : 寫出可以列舉所有排列的程式
```
#參考老師permutation範例，自行撰寫。
def p(e, current_p=[]):
    # 如果所有元素都在目前排列中，則列印排列
    if not e:
        print(current_p)
        return

    # 對每個剩餘的元素進行遞迴
    for i in range(len(e)):
        # 選擇一個元素
        current_e = e[i]

        # 移除已選擇的元素
        remaining_e = e[:i] + e[i + 1:]

        # 遞迴呼叫
        p(remaining_e, current_p + [current_e])

# 測試
elements_to_permute = [1, 2, 3]
p(elements_to_permute)
#測試結果：
'''
[1, 2, 3]
[1, 3, 2]
[2, 1, 3]
[2, 3, 1]
[3, 1, 2]
[3, 2, 1]
'''
```
## Hw4 : 求解方程式(x^2 - 3x + 1 = 0)
##### 暴力法
```
#原創
import math
from numpy import arange

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
```
##### 迭代法（至少寫三個迭代式，其中至少有一個收斂）
```
#f1-f3參考ChatGPT 迴圈部分參考老師範例

# x^2 = 3x -1 (除以x) -> x = 3 - 1/x
# x^2 - 2x + 1 = x (x項移項) -> x = (x-1)^2
# x^2 + 1 = 3x (x移項方程式除以3) -> x = (x ^ 2 + 1) / 3 

f1 = lambda x: 3 - 1 / x 
f2 = lambda x: (x - 1) ** 2 
f3 = lambda x: (x ** 2 + 1) / 3 

x1 = x2 = x3 = 1.0

for i in range(20):
    x1, x2, x3 = f1(x1), f2(x2), f3(x3)
    print('x1:', x1, '   x2:', x2, '   x3:', x3)
#測試結果
'''
x1: 2.0    x2: 0.0    x3: 0.6666666666666666
x1: 2.5    x2: 1.0    x3: 0.48148148148148145
x1: 2.6    x2: 0.0    x3: 0.41060813900320076
x1: 2.6153846153846154    x2: 1.0    x3: 0.389533014605224
x1: 2.6176470588235294    x2: 0.0    x3: 0.3839119898224779
x1: 2.6179775280898876    x2: 1.0    x3: 0.3824628053098181
x1: 2.6180257510729614    x2: 0.0    x3: 0.3820925991484853
x1: 2.6180327868852458    x2: 1.0    x3: 0.3819982514413483
x1: 2.618033813400125    x2: 0.0    x3: 0.3819742213680825
x1: 2.6180339631667064    x2: 1.0    x3: 0.38196810192991765
x1: 2.618033985017358    x2: 0.0    x3: 0.381966543630648
x1: 2.618033988205325    x2: 1.0    x3: 0.3819661468177146
x1: 2.6180339886704433    x2: 0.0    x3: 0.3819660457715906
x1: 2.618033988738303    x2: 1.0    x3: 0.381966020040795
x1: 2.618033988748204    x2: 0.0    x3: 0.38196601348860165
x1: 2.618033988749648    x2: 1.0    x3: 0.38196601182012485
x1: 2.618033988749859    x2: 0.0    x3: 0.38196601139525727
x1: 2.6180339887498896    x2: 1.0    x3: 0.38196601128706725
x1: 2.618033988749894    x2: 0.0    x3: 0.38196601125951735
x1: 2.618033988749895    x2: 1.0    x3: 0.3819660112525019
可知f1、f3皆為收斂，f2為震盪
'''
```
## Hw5 : 寫一個爬山演算法程式可以找任何向量函數的山頂
```
#自行撰寫neighbor部分，其餘皆參考老師hillClimbing範例，使用ChatGPT輔助
import random, copy

def neighbor(f, p, h=0.01):
    """產生鄰近的解及其高度"""
    p1= p.copy()
    for i in range(len(p)):
        p1[i] = p1[i] +random.uniform(-h,h)
    f1 = f(p1)
    return p1, f1

def hillClimbing(f, p, h=0.01):
    failCount = 0                    
    while (failCount < 10000):       
        fnow = f(p)                  
        p1, f1 = neighbor(f, p, h)
        if f1 >= fnow:              
            fnow = f1                
            p = p1
            print('p=', p, 'f(p)=', fnow)
            failCount = 0            
        else:                        
            failCount = failCount + 1
    return (p,fnow)                 

def f(p):
    return -1*(p[0]**2+p[1]**2+p[2]**2)

hillClimbing(f, [2,1,3])
```
## Hw6 : 寫一個梯度下降法程式可以找任何向量函數的谷底
```
#程式未修改，程式來源：https://blog.csdn.net/u010960155/article/details/113776715
#已理解程式邏輯與梯度下降概念
#未修習內容參考：https://ind.ntou.edu.tw/~b0170/math/semester2/ch7/CH7.3.pdf ，能大致理解內容於程式作用

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
```
## Hw7 : 用 micrograd 的反傳遞算法算梯度
```
#程式複製顏駿葳同學及ChatGPT無修改，程式有看懂，反向傳遞法仍理解不足，僅透過老師提供資源約理解五成 

from engine import Value

# Define a more complex function
def my_function(a, b, c):
    return (a**2 + b**3) * c

# Initialize values
a = Value(2)
b = Value(3)
c = Value(4)

# Initialize parameters for the Adam optimizer
step = 0.01
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

# Initialize moment estimates
m_a = 0
m_b = 0
m_c = 0
v_a = 0
v_b = 0
v_c = 0
t = 0

for i in range(100):
    # Evaluate the more complex function
    f = my_function(a, b, c)
    
    # Initialize gradients to zero
    a.grad = 0
    b.grad = 0
    c.grad = 0
    
    # Backward pass
    f.backward()

    # Update parameters using Adam optimizer
    t += 1
    m_a = beta1 * m_a + (1 - beta1) * a.grad
    m_b = beta1 * m_b + (1 - beta1) * b.grad
    m_c = beta1 * m_c + (1 - beta1) * c.grad
    v_a = beta2 * v_a + (1 - beta2) * (a.grad ** 2)
    v_b = beta2 * v_b + (1 - beta2) * (b.grad ** 2)
    v_c = beta2 * v_c + (1 - beta2) * (c.grad ** 2)

    m_a_hat = m_a / (1 - beta1 ** t)
    m_b_hat = m_b / (1 - beta1 ** t)
    m_c_hat = m_c / (1 - beta1 ** t)
    v_a_hat = v_a / (1 - beta2 ** t)
    v_b_hat = v_b / (1 - beta2 ** t)
    v_c_hat = v_c / (1 - beta2 ** t)

    a.data -= step * m_a_hat / (v_a_hat ** 0.5 + epsilon)
    b.data -= step * m_b_hat / (v_b_hat ** 0.5 + epsilon)
    c.data -= step * m_c_hat / (v_c_hat ** 0.5 + epsilon)

print("Final values: a =", a.data, ", b =", b.data, ", c =", c.data)
```
## Hw8 : 選一位圖靈獎得主，詳細說明他的得獎原因
```
請參照<https://github.com/soon1110014/alg112a/blob/master/HW8/Turing%20Award.md>
```
## Hw9 : 請用搜尋法求解(老鼠走迷宮、《狼、羊、甘藍菜》過河的問題、八個皇后問題)其中的一個
##### 老鼠走迷宮
```
#程式參考老師java實作修改部分僅為python語法
#透過網路資源：https://ithelp.ithome.com.tw/articles/10281404，理解概念。
def matrix_print(m):
    for row in m:
        print(row)

def str_set(s, i, c):
    return s[:i] + c + s[i+1:]

def find_path(m, x, y):
    print("=========================")
    print(f"x={x} y={y}")
    matrix_print(m)

    if x >= 6 or y >= 8:
        return False
    if m[x][y] == '*':
        return False
    if m[x][y] == '+':
        return False

    if m[x][y] == ' ':
        m[x] = str_set(m[x], y, '.')

    if m[x][y] == '.' and (x == 5 or y == 7):
        return True

    if y < 7 and m[x][y+1] == ' ':  # 向右
        if find_path(m, x, y+1):
            return True
    if x < 5 and m[x+1][y] == ' ':  # 向下
        if find_path(m, x+1, y):
            return True
    if y > 0 and m[x][y-1] == ' ':  # 向左
        if find_path(m, x, y-1):
            return True
    if x > 0 and m[x-1][y] == ' ':  # 向上
        if find_path(m, x-1, y):
            return True

    m[x] = str_set(m[x], y, '+')
    return False

maze = ["********",
        "** * ***",
        "     ***",
        "* ******",
        "*     **",
        "***** **"]

find_path(maze, 2, 0)
print("=========================")
matrix_print(maze)
#測試結果：成功走出迷宮
'''
=========================
x=2 y=0
********
** * ***
     ***
* ******
*     **
***** **
=========================
x=2 y=1
********
** * ***
.    ***
* ******
*     **
***** **
=========================
x=2 y=2
********
** * ***
..   ***
* ******
*     **
***** **
=========================
x=2 y=3
********
** * ***
...  ***
* ******
*     **
***** **
=========================
x=2 y=4
********
** * ***
.... ***
* ******
*     **
***** **
=========================
x=1 y=4
********
** * ***
.....***
* ******
*     **
***** **
=========================
x=1 y=2
********
** *+***
...++***
* ******
*     **
***** **
=========================
x=3 y=1
********
**+*+***
..+++***
* ******
*     **
***** **
=========================
x=4 y=1
********
**+*+***
..+++***
*.******
*     **
***** **
=========================
x=4 y=2
********
**+*+***
..+++***
*.******
*.    **
***** **
=========================
x=4 y=3
********
**+*+***
..+++***
*.******
*..   **
***** **
=========================
x=4 y=4
********
**+*+***
..+++***
*.******
*...  **
***** **
=========================
x=4 y=5
********
**+*+***
..+++***
*.******
*.... **
***** **
=========================
x=5 y=5
********
**+*+***
..+++***
*.******
*.....**
***** **
=========================
********
**+*+***
..+++***
*.******
*.....**
*****.**
'''
```
## Hw10 : 寫一個程式可以求解 n 次多項式
##### 範例 : x^2 + 1 = 0 、x^8 + 3x^2 + 1 = 0

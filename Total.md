# 期末習題整理

###### 課程名稱：演算法
###### 學號：110910122
###### 學生：陳振順
###### 授課老師： 陳鍾誠 教授
###### 用途： 期初至期末作業總結


## Hw1 : 費氏數列迴圈版
##### 參考老師查表法，修改for迴圈部分。
```
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
##### 原創
```
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
##### 參考老師permutation範例，自行撰寫。
```
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
##### 原創
```
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

##### f1-f3參考ChatGPT做調整，迴圈部分參考老師範例無修改，已理解程式
```
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

##### 自行撰寫neighbor部分
##### 其餘皆參考老師hillClimbing範例(無修改，有看懂)，使用ChatGPT輔助理解
```
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

##### 程式未修改，程式來源：https://blog.csdn.net/u010960155/article/details/113776715
##### 已理解程式邏輯與梯度下降概念
##### 未修習內容參考：https://ind.ntou.edu.tw/~b0170/math/semester2/ch7/CH7.3.pdf ，理解內容於程式作用
```

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
##### 程式複製顏駿葳同學無修改以ChatGPT輔助，程式有看懂
##### [顏駿葳同學程式來源](https://github.com/Yan7668114/alg112ahw/blob/main/07/micrograd/micrograd/micrograd510.py)
##### 反向傳遞法仍理解不足，僅透過老師提供資源約理解五成 
```
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
#### 老鼠走迷宮
##### 程式參考老師java實作，修改部分僅為修改為python語法
##### [老師java實作來源](https://github.com/ccc112a/py2cs/blob/master/02-%E6%BC%94%E7%AE%97%E6%B3%95/02-%E6%96%B9%E6%B3%95/06-%E6%90%9C%E5%B0%8B%E6%B3%95/Q1-mouse/%E7%BF%92%E9%A1%8C%EF%BC%9A%E4%BB%A5%E6%B7%B1%E5%BA%A6%E5%84%AA%E5%85%88%E6%90%9C%E5%B0%8B%E8%A7%A3%E6%B1%BA%E8%80%81%E9%BC%A0%E8%B5%B0%E8%BF%B7%E5%AE%AE%E5%95%8F%E9%A1%8C.md)
##### 透過網路資源：https://ithelp.ithome.com.tw/articles/10281404，理解概念。
```
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
##### 自行撰寫，程式參考網上資源：https://blog.csdn.net/tanlangqie/article/details/86473480
```
import numpy as np

def solve_polynomial(coefficients):
    # 使用 numpy.roots 求解多項式的根
    roots = np.roots(coefficients)

    return roots

# 測試
coefficients = [1, 0, 0, 0, 0, 1] 
result = solve_polynomial(coefficients)
coefficients1 = [1, 0, 0, 0, 0, 0, 3, 0, 1] 
result1 = solve_polynomial(coefficients)

print("x^5 + 1 的根:", result)
print("x^8 + 3x^2 + 1 的根:", result1)
#測試結果
'''
x^5 + 1 的根: [-1.        +0.j         -0.30901699+0.95105652j -0.30901699-0.95105652j
  0.80901699+0.58778525j  0.80901699-0.58778525j]
  
x^8 + 3x^2 + 1 的根: [-1.07879081e+00+0.58413455j -1.07879081e+00-0.58413455j
  1.07879081e+00+0.58413455j  1.07879081e+00-0.58413455j
 -2.77555756e-17+1.14345358j -2.77555756e-17-1.14345358j
  2.77555756e-17+0.58109101j  2.77555756e-17-0.58109101j]
'''
```
## 挑戰：用遞迴寫最小編輯距離

##### 自行撰寫程式有參考老師[editDistance.py](https://github.com/ccc112a/py2cs/blob/master/02-%E6%BC%94%E7%AE%97%E6%B3%95/02-%E6%96%B9%E6%B3%95/08-%E5%8B%95%E6%85%8B%E8%A6%8F%E5%8A%83%E6%B3%95/editDistance/editDistance.py)
##### 透過網路資源https://rust-algo.club/levenshtein_distance/理解編輯距離

#### 遞迴方法
```
def recursive_edit_distance(str1, str2, m, n):
    if m == 0:
        return n
    if n == 0:
        return m
    if str1[m-1] == str2[n-1]:
        return recursive_edit_distance(str1, str2, m-1, n-1) 
    return 1 + min(
        recursive_edit_distance(str1, str2, m, n-1),  
        recursive_edit_distance(str1, str2, m-1, n),  
        recursive_edit_distance(str1, str2, m-1, n-1)  
    )
str1 = "kitten"
str2 = "sitting"
recursive_result = recursive_edit_distance(str1, str2, len(str1), len(str2))
print("遞迴法最小編輯距離:", recursive_result)
```
#### 動態規劃
```
def dp_edit_distance(str1, str2):
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1])
    return dp[m][n]
str1 = "kitten"
str2 = "sitting"
dp_result = dp_edit_distance(str1, str2)
print("動態規劃法最小編輯距離:", dp_result)
```
#### 測試結果
```
遞迴法最小編輯距離: 3
動態規劃法最小編輯距離: 3
```
## Hw11 : 請把從希爾伯特經圖靈到 NP-Complete 的故事寫下來
##   希爾伯特經圖靈到 NP-Complete 的故事
故事始於20世紀初的希爾伯特，他提出了一系列23個問題，旨在尋找數學的基礎和完備性。其中第二個問題涉及「可決定性問題」，即是否存在一種算法可以判斷所有數學命題的真偽性。

在此之後，哥德爾的不完備定理揭示了對於任何足夠強大的公理系統，總存在一個命題，該系統無法證明其真偽性。這顯示出數學體系的局限性，並為計算理論的發展奠定基礎。

Church的Lambda Calculus於1936年提出，這是一種用於描述計算過程的形式化方法。他證明了Lambda演算的等價性，並引入了「不可判定問題」的概念，表明存在一類問題，無法構建一個算法來決定其答案。

在相同的年份，圖靈提出了圖靈機，一種抽象的計算模型。他證明了對於這種計算模型，存在一類問題，即「停機問題」，對於這類問題，無法構建一個通用算法來判斷計算過程是否終止。

此後，喬姆斯基提出的語言的層次結構語言可以被分為不同的層次，每個層次具有特定的生成規則。這種層次結構的見解對於理解計算能力和語言的組織方式有著深刻的啟示。這種層次結構的見解有助於我們理解語言是如何組織和產生的。它同時為語言處理和語言學研究提供了框架，促使了對於語言現象和結構的更深入探討。這種語言的層次結構思想在計算語言學、人工智慧和語言處理等領域都有廣泛的應用。

最終，於1971年，Steven Cook提出了「非確定多項式時間完全性」（NP-Complete）的概念，並證明了「可滿足性問題」（SAT）是NP-Complete問題的一個代表。這一概念對於理解計算複雜性和算法效能提供了重要的視角，影響了計算機科學的許多方向。強調了某些問題的困難性。

隨後的年份中，許多計算問題被歸納為 NP-Complete，並且人們開始研究 NP-Complete 問題的特性和可能的解法。

整體而言，這個故事串聯了一系列的發現，從尋找數學基礎開始，經由不完備定理、Lambda Calculus、圖靈機、語言層次結構，最終到達了NP-Complete的概念。這一過程揭示了計算的極限，同時推動了計算機科學領域的許多重要進展。

參考資料：	
1. [希爾伯特第二問題](https://zh.wikipedia.org/zh-tw/%E5%B8%8C%E7%88%BE%E4%BC%AF%E7%89%B9%E7%AC%AC%E4%BA%8C%E5%95%8F%E9%A1%8C)
2. [哥德爾不完備定理](https://zh.wikipedia.org/wiki/%E5%93%A5%E5%BE%B7%E5%B0%94%E4%B8%8D%E5%AE%8C%E5%A4%87%E5%AE%9A%E7%90%86)
3. [圖靈機](https://zh.wikipedia.org/wiki/%E5%9B%BE%E7%81%B5%E6%9C%BA)
4. [Λ演算](https://zh.wikipedia.org/zh-tw/%CE%9B%E6%BC%94%E7%AE%97)
5. [NP-Complete](https://zh.wikipedia.org/zh-tw/NP%E5%AE%8C%E5%85%A8)

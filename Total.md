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
```
#原創
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
```

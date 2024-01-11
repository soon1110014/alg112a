#參考網址 https://zetria.tw/python/7d9a471162
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

#採用迴圈與不採用迴圈相比能更節省記憶體，提高速度
n = 100
startTime = datetime.now()
print(f'fib_n({n})={fib_n(n)}')
endTime = datetime.now()
seconds = endTime - startTime
print(f'time:{seconds}')
  

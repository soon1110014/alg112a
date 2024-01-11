# 原方程式： x^2 - 3x + 1 = 0

# x^2 = 3x -1 (除以x) -> x = 3 - 1/x
# x^2 - 2x + 1 = x (x項移項) -> x = (x-1)^2
# x^2 + 1 = 3x (x移項方程式除以3) -> x = (x ^ 2 + 1) / 3 

#f1-f3參考ChatGPT 迴圈部分參考老師範例

f1 = lambda x: 3 - 1 / x 
f2 = lambda x: (x - 1) ** 2 
f3 = lambda x: (x ** 2 + 1) / 3 

x1 = x2 = x3 = 1.0

for i in range(20):
    x1, x2, x3 = f1(x1), f2(x2), f3(x3)
    print('x1:', x1, '   x2:', x2, '   x3:', x3)

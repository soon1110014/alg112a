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

#梯度程式有參考老師範例，並參考網上資源<https://blog.csdn.net/u010960155/article/details/113776715>對後續程式進行撰寫與調整。已理解概念
rate = 0.01
# 函數 f 對變數 p[k] 的偏微分: df / dp[k]
def df(f, p, k):
    p1 = p.copy()
    p1[k] = p[k]+rate
    return (f(p1) - f(p)) / rate

# 梯度：函數 f 在點 p 上的梯度
def grad(f, p):
    gp = p.copy()
    for k in range(len(p)):
        gp[k] = df(f, p, k)
    return gp
    
def gradDescent(f, inpt, max=1000):
    p = np.array(initial_point)
    
    for i in range(max):
        gradient = grad(f, p)
        p = p - rate * grad
        if np.linalg.norm(gradient) < 1e-6:
            break
    return p, f(*p)

def f(x, y, z):
    return x**2 + y**2 + z**2

inpt = [8, 1, 5]
result = gradDescent(f, inpt)

print("最終結果： p =", result[0], "f(p) =", result[1])
    

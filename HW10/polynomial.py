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

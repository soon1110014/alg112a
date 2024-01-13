#程式參考顏駿葳同學及ChatGPT，反向傳遞法仍理解不足，透過老師提供資源約理解五成 

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

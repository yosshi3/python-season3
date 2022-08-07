# coding: utf-8
import numpy as np
import matplotlib.pylab as plt

def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)

def function_1(x):
    return x**2 + x 

def tangent_line(f, x):
    d = numerical_diff(f, x)
    print('傾き:',d)
    y = f(x) - d*x
    print('x:',x)
    print('f(x):',f(x))
    print('d*x:',d*x)
    return lambda t: d*t + y    # 傾きd、Y切片y

x = np.arange(0.0, 10.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")

tf = tangent_line(function_1, 5)
y2 = tf(x)

plt.plot(x, y)
plt.plot(x, y2)
plt.grid(True)
plt.show()

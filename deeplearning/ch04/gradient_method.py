# coding: utf-8
import numpy as np
import matplotlib.pylab as plt
from gradient_2d import numerical_gradient


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = np.append(init_x, function_2(init_x))
    print('x:',x)
    x_history = []
    for i in range(step_num):
        x_history.append( x.copy() )
        grad = numerical_gradient(f, x)
        x -= lr * grad
        x[2] = function_2(x)
    return x, np.array(x_history)


def function_2(x):
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0]) # 開始地点

lr = 0.1
step_num = 20
x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)

np.set_printoptions(precision=2)
print(x_history)

plt.plot( [-5, 5], [0,0], '--g') # グラフを書く。--破線で、greenを設定する。
plt.plot( [0,0], [-5, 5], '--b')
plt.plot(x_history[:,0], x_history[:,1], 'o')

plt.xlim(-5.5, 5.5)
plt.ylim(-5.5, 5.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()

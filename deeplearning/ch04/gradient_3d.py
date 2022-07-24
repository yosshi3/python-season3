# coding: utf-8
# cf.http://d.hatena.ne.jp/white_wheels/20100327/p3
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D


def _numerical_gradient_no_batch(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val  # 値を元に戻す
        
    return grad


def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)
        
        return grad


def function_2(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)


def tangent_line(f, x):
    d = numerical_gradient(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y


if __name__ == '__main__':
    x0 = np.arange(-2, 2.5, 0.5)
    x1 = np.arange(-2, 2.5, 0.5)
    X, Y = np.meshgrid(x0, x1)

    shp = X.shape
    X = X.flatten()
    Y = Y.flatten()

    grad = numerical_gradient(function_2, np.array([X, Y]).T).T

    fig = plt.figure()
    ax = Axes3D(fig)
    
    Z = function_2(np.array([X, Y]).T).T
    Z = Z.reshape(shp)
    X = X.reshape(shp)
    Y = Y.reshape(shp)

    ax.plot_wireframe(X,Y,Z, color='blue')
    ax.quiver(X,Y,Z,-grad[0].reshape(shp),-grad[1].reshape(shp),0
              , length=0.2, arrow_length_ratio=0.5, color='green')
    plt.xlim([-2.5, 2.5])
    plt.ylim([-2.5, 2.5])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()

    ax.view_init(elev=30, azim=30)

    plt.draw()
    plt.show()

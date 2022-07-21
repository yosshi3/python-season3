# coding: utf-8
from and_gate import AND
from or_gate import OR
from nand_gate import NAND
import numpy as np
import matplotlib.pyplot as plt

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

def plot2DContour(operator):
    '''0から１の2次元座標で、1なら赤、0なら青に色分けする'''
    xs = np.arange(-50,150,step=1) / 100
    ys = np.arange(-50,150,step=1) / 100
    xx,yy = np.meshgrid(xs,ys)
    color = np.zeros((len(xs),len(ys)), dtype=np.int8)

    for ix,x in enumerate(xs):
        for iy,y in enumerate(ys):
            color[ix][iy] =operator(x,y)

    # 等高線を描画する
    plt.contourf(xx, yy, color, cmap='rainbow')
    plt.colorbar(label="contour level")
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    plt.grid(color = "gray", linestyle="--")
    plt.show()

if __name__ == '__main__':
    plot2DContour(AND)
    plot2DContour(NAND)
    plot2DContour(OR)
    plot2DContour(XOR)
    
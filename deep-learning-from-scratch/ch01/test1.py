'''
問題１）1～１６の配列を利用して、２×２×４のnumpy配列を作成する。
※np.arange、x.reshapeを利用する

[[[ 1  2  3  4]
  [ 5  6  7  8]]

 [[ 9 10 11 12]
  [13 14 15 16]]]

問題２）問題１）の結果の２段目だけ取り出す。

[[ 9 10 11 12]
 [13 14 15 16]]

問題３）問題２）の結果を２倍する。

[[18 20 22 24]
 [26 28 30 32]]

問題４）問題３）の結果を使って、問題１）の結果の２段目を２倍する。

[[[ 1  2  3  4]
  [ 5  6  7  8]]

 [[18 20 22 24]
  [26 28 30 32]]]

問題５）問題４）の結果で０軸方向の最大値を取得する。
※np.max()を利用する

 [[18 20 22 24]
 [26 28 30 32]]

問題６）問題４）の結果で２軸方向の最大値を取得する。
※np.max()を利用する

[[[ 4]
  [ 8]]

 [[24]
  [32]]]

問題７）問題４）の結果から問題６）の結果を引く
[[[-3 -2 -1  0]
  [-3 -2 -1  0]]

 [[-6 -4 -2  0]
  [-6 -4 -2  0]]]

問題８）問題７）の結果にexp関数を適用する。
※np.exp()を利用する
[[[0.04978707 0.13533528 0.36787944 1.        ]
  [0.04978707 0.13533528 0.36787944 1.        ]]

 [[0.00247875 0.01831564 0.13533528 1.        ]
  [0.00247875 0.01831564 0.13533528 1.        ]]]

問題９）問題４）の結果に、以下の関数をsoftmax関数を適用する。
common\functions.py

[[[0.0320586  0.08714432 0.23688282 0.64391426]
  [0.0320586  0.08714432 0.23688282 0.64391426]]

 [[0.00214401 0.0158422  0.11705891 0.86495488]
  [0.00214401 0.0158422  0.11705891 0.86495488]]]

問題１０）問題９）の結果で、numpyの精度を少数２桁に変更する。
np.get_printoptions()
np.set_printoptions()

[[[0.03 0.09 0.24 0.64]
  [0.03 0.09 0.24 0.64]]

 [[0.   0.02 0.12 0.86]
  [0.   0.02 0.12 0.86]]]

問題１１）-5～5までの範囲(0.1刻み)で、
以下のstep_functionとsigmoid関数のグラフを表示する。
common\functions.py

'''

# coding: utf-8
import sys, os
sys.path.append(os.pardir)

import numpy as np
from common.functions import step_function,sigmoid,softmax
import matplotlib.pyplot as plt

np.set_printoptions(precision=8)
print(np.get_printoptions())

x = np.arange(1, 17)
x = x.reshape(2,2,4)

print(x[1] * 2)
print()
print()

x[1,:,:] = x[1] * 2

print(x)

print()
print("np.max(x, axis=0)")
print(np.max(x, axis=0))

print()
print("np.max(x, axis=-1, keepdims=True)")
print(np.max(x, axis=-1, keepdims=True))


print("x - np.max(x, axis=-1, keepdims=True)")
x1 = x - np.max(x, axis=-1, keepdims=True)
print(x1)

print()
print("np.exp()")
print(np.exp(x1))

y2 = softmax(x)

print()
print("softmax()")
print(y2)

np.set_printoptions(precision=2)
print(np.get_printoptions())

print(y2)

x = np.arange(-5, 6,step=0.1,dtype=float)
y1 = step_function(x)
y2 = sigmoid(x)

plt.plot(x, y1, label="step")
plt.plot(x, y2, linestyle = "--", label="softmax")
plt.xlabel("x") # x軸のラベル
plt.ylabel("y") # y軸のラベル
plt.title('step & softmax')
plt.legend()
plt.show()

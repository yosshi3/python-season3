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
http://arduinopid.web.fc2.com/N45.html
# $
{y_i = \frac{e^{x_i}}{\sum_{j=1}^N e^{x_j}} = \frac{e^{(x_i-x_{max})}}{\sum_{j=1}^N e^{(x_j-x_{max})}}}
$

[[[0.0320586  0.08714432 0.23688282 0.64391426]
  [0.0320586  0.08714432 0.23688282 0.64391426]]

 [[0.00214401 0.0158422  0.11705891 0.86495488]
  [0.00214401 0.0158422  0.11705891 0.86495488]]]

問題１０）問題９）の結果で、numpyの精度を少数２桁に変更する。
※以下を利用する
np.get_printoptions()
np.set_printoptions()

[[[0.03 0.09 0.24 0.64]
  [0.03 0.09 0.24 0.64]]

 [[0.   0.02 0.12 0.86]
  [0.   0.02 0.12 0.86]]]

問題１１）問題１０）の結果に、２軸方向で、総和を求める。
※np.sum()を利用する

[[1. 1.]
 [1. 1.]]

問題１２）-5～5までの範囲(0.1刻み)で、
以下のstep_functionとsigmoid関数のグラフを表示する。
common\functions.py

問題１３）y=x^3の３回微分のグラフを表示する。

'''

# coding: utf-8

import matplotlib.pyplot as plt
import torch

torch.set_printoptions(precision=5)

t = torch.arange(1.0, 17)
t = t.reshape(2,2,4)
x = t

print('x[1] * 2')
print(x[1] * 2)
print()

print('x[1] = x[1] * 2')
x[1] = x[1] * 2
print(x)

print()
print("torch.max(x, dim=0)-----0軸のMAX")
print(torch.max(x, dim=0))
print("-----")

print()
print("torch.max(x, dim=-1, keepdims=True)----- -1軸のMAX")
print(torch.max(x, dim=-1, keepdims=True))
print("-----")


print("x - torch.max(x, dim=-1, keepdims=True)[0]-----")
x1 = x - torch.max(x, dim=-1, keepdims=True)[0]
print(x1)
print("-----")

print()
print("torch.exp()")
print(torch.exp(x1))

y2 = torch.nn.Softmax(dim=-1)(x)

print()
print("torch.nn.Softmax(dim=-1)")
print(y2)

y3 = torch.sum(y2,dim=-1)

print()
print("np.sum(y2,axis=-1)")
print(y3)

# step & softmax
x = torch.arange(-5, 6,step=0.1,dtype=float,requires_grad=True)
y1 = torch.nn.ReLU()(x)
y1.backward(gradient=torch.ones(x.shape))
y1 = x.grad
y2 = torch.nn.Sigmoid()(x)

plt.plot(x.detach().numpy(), y1.detach().numpy(), label="ReLU")
plt.plot(x.detach().numpy(), y2.detach().numpy(), linestyle = "--", label="softmax")
plt.xlabel("x") # x軸のラベル
plt.ylabel("y") # y軸のラベル
plt.title('step & softmax')
plt.legend()
plt.show()

print()
print('1回微分')
torch.set_printoptions(precision=2)
x = torch.arange(-2, 2.1,step=0.1,dtype=float,requires_grad=True)
func = lambda x : x ** 3
y1 = func(x)
gradients = torch.ones(x.shape)
# 1回微分を求める
y1.backward(gradient=gradients) # backword()は、通常は１変数のみ。複数変数の時はgradientと内積をとる。
y2 = x.grad
plt.plot(x.detach().numpy(), y1.detach().numpy(), label="y=f(x)")
plt.plot(x.detach().numpy(), y2.numpy(), linestyle = "--", label="dy/dx")
plt.xlabel("x")
plt.ylabel("y")
plt.yticks(list(range(-14,14,2)))
plt.grid()
plt.title('dy/dx')
plt.legend()
plt.show()

# https://qiita.com/tmasada/items/9dee38e5bc1482217493
# torch.autograd.grad()を呼ぶときにcreate_graph=Trueとしているのがポイントです。
# こうすると、微分係数（上の場合は48）だけでなく、
# ffのxxに関する微分について計算グラフを作って、それも返してくれます。
# すると、その計算グラフを使うことで、2階の微分係数が計算できるようになります。
print()
print('2回微分')
x = torch.arange(-2, 2.1,step=0.1,dtype=float,requires_grad=True)
func = lambda x : x ** 3
y1 = func(x)
# 1回微分を求める
y2 = torch.autograd.grad(y1, x, create_graph=True, grad_outputs=torch.ones(x.shape))
# 2回微分を求める
y2[0].backward(torch.ones(x.shape))
y3 = x.grad
plt.plot(x.detach().numpy(), y1.detach().numpy(), label="y=f(x)")
plt.plot(x.detach().numpy(), y2[0].detach().numpy(), linestyle = "--", label="y=f'(x)")
plt.plot(x.detach().numpy(), y3.numpy(), linestyle = "--", label="y=f\"(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.yticks(list(range(-14,14,2)))
plt.grid()
plt.legend()
plt.show()

print()
print('3回微分')
x = torch.arange(-2, 2.1,step=0.1,dtype=float,requires_grad=True)
func = lambda x : x ** 3
y1 = func(x)
plt.plot(x.detach().numpy(), y1.detach().numpy(), label="y=f(x)")
for i in range(3):
    y1 = torch.autograd.grad(y1, x, create_graph=True, grad_outputs=torch.ones(x.shape))
    plt.plot(x.detach().numpy(), y1[0].detach().numpy(), 
             linestyle = "--", label="y=f" + "'" * (i+1) +  "(x)")

plt.xlabel("x")
plt.ylabel("y")
plt.yticks(list(range(-14,14,2)))
plt.grid()
plt.legend()
plt.show()







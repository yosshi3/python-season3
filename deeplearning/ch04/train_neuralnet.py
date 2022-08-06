# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print('x_train.shape:', x_train.shape)
print('t_train.shape:', t_train.shape)
print('x_test.shape:', x_test.shape)
print('t_test.shape:', t_test.shape)

network = TwoLayerNet(input_size=28*28, hidden_size=50, output_size=10)

iters_num = 3001  # 繰り返しの回数を適宜設定する
iters_num = 2  # 繰り返しの回数を適宜設定する
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
print('train_size: ', train_size)
print('batch_size: ', batch_size)
print('iter_per_epoch: ', iter_per_epoch)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size) # バッチサイズ分の乱数を発行する。
    x_batch = x_train[batch_mask]                         # バッチサイズ分取得する。
    t_batch = t_train[batch_mask]
    print('x_batch.shape:',x_batch.shape)
    print('t_batch.shape:',t_batch.shape)
    # 勾配の計算
    grad = network.numerical_gradient(x_batch, t_batch)
    #grad = network.gradient(x_batch, t_batch)
    print('grad[W1].shape:',grad['W1'].shape)
    print('grad[b1].shape:',grad['b1'].shape)
    print('grad[W2].shape:',grad['W2'].shape)
    print('grad[b2].shape:',grad['b2'].shape)
    # パラメータの更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print('i=',i,"train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# グラフの描画
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
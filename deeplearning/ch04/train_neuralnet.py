# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
from pprint import pprint

def figure(x_train):
    # MNISTデータの表示
    W = 22  # 横に並べる個数
    H = 5   # 縦に並べる個数
    fig = plt.figure(figsize=[12,3])
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1.0, hspace=0.05, wspace=0.05)
    for i in range(len(x_train)):
        ax = fig.add_subplot(H, W, i + 1, xticks=[], yticks=[])
        ax.imshow(x_train[i].reshape((28, 28)), cmap='gray')
    plt.show()

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
# x_train.shape: (60000, 784)
# t_train.shape: (60000, 10)
# x_test.shape: (10000, 784)
# t_test.shape: (10000, 10)

network = TwoLayerNet(input_size=28*28, hidden_size=50, output_size=10)

iters_num = 3001  # 繰り返しの回数を適宜設定する
#iters_num = 2  # 繰り返しの回数を適宜設定する
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
# train_size:  60000       訓練データの画像枚数
# batch_size:  100         バッチサイズ
# iter_per_epoch:  600.0   エポック

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size) # バッチサイズ分の乱数を発行する。
    x_batch = x_train[batch_mask]                         # バッチサイズ分取得する。
    t_batch = t_train[batch_mask]
    # x_batch.shape: (100, 784) バッチサイズ×入力層
    # t_batch.shape: (100, 10)  バッチサイズ×出力層
    # 勾配の計算
    #grad = network.numerical_gradient(x_batch, t_batch) # 数値微分
    grad = network.gradient(x_batch, t_batch) # 誤差逆伝播法
    # grad[W1].shape: (784, 50) 入力層×中間層
    # grad[b1].shape: (50,)     中間層
    # grad[W2].shape: (50, 10)  中間層×出力層
    # grad[b2].shape: (10,)     出力層
    # パラメータの更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc,_,_ = network.accuracy(x_train, t_train)
        test_acc,test_y,test_t = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print('i=',i,"train acc, test acc | " + str(train_acc) + ", " + str(test_acc))
        num=1000
        z=np.where((test_y[0:num]-test_t[0:num])!=0)
        print('不一致枚数:',z[0].shape[0])
        if z[0].shape[0]<110:
            figure(x_test[z])
        pprint(test_y[z])
        pprint(test_t[z])

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
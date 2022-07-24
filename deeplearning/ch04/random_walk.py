import matplotlib.pyplot as plt
from matplotlib import animation
import random

fig = plt.figure()
 
xlim = [0,100]
X, Y = [], []
 
def plot(data):
    plt.cla()                   # 前のグラフを削除
    Y.append(random.random())   # データを作成
    X.append(len(Y))
    
    if len(X) > 100:            # 描画範囲を更新
        xlim[0]+=1
        xlim[1]+=1
    
    plt.plot(X, Y)              # 次のグラフを作成
    plt.title("sample animation (real time)")
    plt.ylim(-1,2)
    plt.xlim(xlim[0],xlim[1])
    
 
# 10msごとにplot関数を呼び出してアニメーションを作成
ani = animation.FuncAnimation(fig, plot, interval=10)
ani.save('random_walk.gif', writer='pillow')


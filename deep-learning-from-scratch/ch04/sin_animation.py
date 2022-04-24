import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# 0 <=x < 2pi の範囲の点列を作成。
x = np.linspace(0, 2*np.pi, 101)[: -1]
# 各コマの画像を格納する配列
image_list = []

for i in range(100):
    # ずらしながらsinカーブを描写し、配列に格納
    y = np.sin(np.roll(x, -i))
    image = ax.plot(x, y)
    image_list.append(image)

# アニメーションを作成
ani = ArtistAnimation(fig, image_list, interval=1)
# gifに保存
ani.save('sin_animation.gif', writer='pillow')

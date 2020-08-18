# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# <a href="https://colab.research.google.com/github/yukinaga/lecture_pytorch/blob/master/python_basic/04_matplotlib.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# %% [markdown]
# # matplotlibの基礎
# グラフの描画や画像の表示、簡単なアニメーションの作成などを行うことができます。   
# %% [markdown]
# ## ●matplotlibのインポート
# グラフを描画するためには、matplotlibのpyplotというモジュールをインポートします。  
# pyplotはグラフの描画をサポートします。  
# データにはNumPyの配列を使いますので、NumPyもインポートします。  

# %%
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## ●linspace関数
# 
# matplotlibでグラフを描画する際に、NumPyのlinspace関数がよく使われます。  
# linspace関数は、ある区間を50に等間隔で区切ってNumPyの配列にします。  
# この配列を、グラフの横軸の値としてよく使います。  

# %%
import numpy as np

x = np.linspace(-5, 5)  # -5から5まで50に区切る

print(x)
print(len(x))  # xの要素数

# %% [markdown]
# この配列を使って、連続に変化する横軸の値を擬似的に表現します。
# %% [markdown]
# ## ●グラフの描画
# 
# 例として、pyplotを使って直線を描画します。  
# NumPyのlinspace関数でx座標のデータを配列として生成し、これに値をかけてy座標とします。  
# そして、pyplotのplotで、x座標、y座標のデータをプロットし、showでグラフを表示します。  

# %%
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5)  # -5から5まで
y = 2 * x  # xに2をかけてy座標とする

plt.plot(x, y)
plt.show()

# %% [markdown]
# ## ●グラフの装飾
# 軸のラベルやグラフのタイトル、凡例などを表示し、線のスタイルを変更してリッチなグラフにしましょう。

# %%
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5)
y_1 = 2 * x
y_2 = 3 * x

# 軸のラベル
plt.xlabel("x value")
plt.ylabel("y value")

# グラフのタイトル
plt.title("My Graph")

# プロット 凡例と線のスタイルを指定
plt.plot(x, y_1, label="y1")
plt.plot(x, y_2, label="y2", linestyle="dashed")
plt.legend() # 凡例を表示

plt.show()

# %% [markdown]
# ## ●散布図の表示
# scatter関数により散布図を表示することができます。  
# 以下のコードでは、x座標、y座標から散布図を描画しています。 

# %%
import numpy as np
import matplotlib.pyplot as plt

x = np.array([1.2, 2.4, 0.0, 1.4, 1.5])
y = np.array([2.4, 1.4, 1.0, 0.1, 1.7])

plt.scatter(x, y)  # 散布図のプロット
plt.show()

# %% [markdown]
# ## ●画像の表示
# pyplotのimshow関数は、配列を画像として表示することができます。  
# 以下のコードは、配列を画像として表示するサンプルです。

# %%
import numpy as np
import matplotlib.pyplot as plt

img = np.array([[0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10,11],
                [12,13,14,15]])

plt.imshow(img, "gray")  # グレースケールで表示
plt.colorbar()   # カラーバーの表示
plt.show()

# %% [markdown]
# この場合、0が黒、15が白を表し、その間の値はこれらの中間色を表します。  
# カラーバーを表示することもできます。


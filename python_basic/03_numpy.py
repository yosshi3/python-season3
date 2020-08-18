# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# <a href="https://colab.research.google.com/github/yukinaga/lecture_pytorch/blob/master/python_basic/03_numpy.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# %% [markdown]
# # NumPyの基礎
# NumPyはPythonの拡張モジュールで、シンプルな表記で効率的なデータの操作を可能にします。  
# 多次元配列を強力にサポートし、内部はC言語で実装されているため高速に動作します。   
# NumPyには様々な機能があるのですが、ここでは本講座で使用する範囲のみ解説します。
# %% [markdown]
# ## ●Numpyの導入
# %% [markdown]
# Pythonでは、importの記述によりモジュールを導入することができます。  
# NumPyはモジュールなので、NumPyを使用するためには、コードの先頭に例えば以下のように記述します。  

# %%
import numpy as np

# %% [markdown]
# asを使うことでモジュールに別の名前をつけることができます。  
# このように記述すると、これ以降npという名前でNumPyのモジュールを扱うことができます。  
# %% [markdown]
# ## ●Numpyの配列
# 
# 人工知能の計算にはベクトルや行列を多用しますが、これらを表現するのにNumPyの配列を用います。  
# ベクトルや行列についてはのちのセクションで改めて解説しますが、ここではとりあえずNumPyの配列とは数値が折り重なって並んだもの、と考えていただければ十分です。  
# 以降、単に配列と呼ぶ場合はNumPyの配列を指すことにします。  
# 
# NumPyの配列は、NumPyのarray関数を使うことでPythonのリストから簡単に作ることができます。 

# %%
import numpy as np

a = np.array([0, 1, 2, 3, 4, 5])  # PythonのリストからNumPyの配列を作る
print(a) 

# %% [markdown]
# このような配列が折り重なった、2次元の配列を作ることもできます。  
# 2次元配列は、要素がリストであるリスト（2重のリスト）から作ります。

# %%
import numpy as np

b = np.array([[0, 1, 2], [3, 4, 5]])  # 2重のリストからNumPyの2次元配列を作る
print(b)

# %% [markdown]
# 同様に、3次元の配列も作ることができます。  
# 3次元配列は2次元の配列がさらに折り重なったもので、3重のリストから作ります。

# %%
import numpy as np

c = np.array([[[0, 1, 2], [3, 4, 5]], [[5, 4, 3], [2, 1, 0]]])  # 3重のリストからNumPyの3次元配列を作る
print(c)

# %% [markdown]
# ## ●配列の演算
# 
# 以下の例では、配列と数値の間で演算を行なっています。  
# この場合、配列の各要素と数値の間で演算が行われます。

# %%
import numpy as np

a = np.array([[0, 1, 2], [3, 4, 5]])  # 2次元配列

print(a) 
print()
print(a + 3)  # 各要素に3を足す
print()
print(a * 3)  # 各要素に3をかける

# %% [markdown]
# また、以下は配列同士の演算の例です。  
# この場合は同じ位置の各要素同士で演算が行われます。 

# %%
b = np.array([[0, 1, 2], [3, 4, 5]])  # 2次元配列
c = np.array([[2, 0, 1], [5, 3, 4]])  # 2次元配列

print(b)
print()
print(c)
print()
print(b + c)
print()
print(b * c)

# %% [markdown]
# ブロードキャストという機能により、特定の条件を満たしていれば形状の異なる配列同士でも演算が可能です。

# %%
d = np.array([[1, 1],
              [1, 1]])  # 2次元配列
e = np.array([1, 2])  # 1次元配列

print(d + e)

# %% [markdown]
# ブロードキャストの厳密なルールは少々複雑で、全て記述すると長くなってしまうので、今回は必要最小限の解説としました。
# %% [markdown]
# ## ●形状の変換
# NumPyのshapeメソッドにより、配列の形状を得ることができます。  

# %%
import numpy as np

a = np.array([[0, 1, 2],
            [3, 4, 5]])

print(a.shape)

# %% [markdown]
# reshapeメソッドを使うと、配列の形状を変換することができます。  
# 以下の例では、要素数が8の1次元配列を 形状が(2, 4)の2次元配列に変換しています。  

# %%
b = np.array([0, 1, 2, 3, 4, 5, 6, 7])    # 配列の作成
c = b.reshape(2, 4)                       # (2, 4)の2次元配列に変換
print(c)

# %% [markdown]
# reshapeの引数を-1にすることで、どのような形状の配列でも1次元配列に変換することができます。

# %%
d = np.array([[[0, 1, 2],
                   [3, 4, 5]],
                  
                  [[5, 4, 3],
                   [2, 1, 0]]])  # 3重のリストからNumPyの3次元配列を作る


e = d.reshape(-1)
print(e)

# %% [markdown]
# ## ●要素へのアクセス
# %% [markdown]
# 配列の各要素へのアクセスは、リストの場合と同様にインデックスを利用します。  
# 一次元配列の場合、以下のように`[ ]`内にインデックスを指定することで、要素を取り出すことができます。

# %%
import numpy as np

a = np.array([0, 1, 2, 3, 4, 5])
print(a[2])

# %% [markdown]
# この場合は、先頭から0,1,2...とインデックスをつけた場合の、インデックスが2要素を取り出しています。  
# また、リストの場合と同様に、インデックスを指定して要素を入れ替えることができます。

# %%
a[2] = 9
print(a)

# %% [markdown]
# この場合は、インデックスが2の要素を9に置き換えています。  
# 
# 2次元配列の場合、要素を取り出す際にはインデックスを縦横で2つ指定します。  
# `,`（カンマ）区切りでインデックスを並べることも、インデックスを入れた`[ ]`を2つ並べることもできます。  

# %%
b = np.array([[0, 1, 2],
              [3, 4, 5]])

print(b[1, 2])  # b[1][2]と同じ

# %% [markdown]
# 縦のインデックスが1、横のインデックスが2の要素を取り出すことができました。  
# 要素を入れ替える際も、同様にインデックスを2つ指定します。

# %%
b[1, 2] = 9

print(b)

# %% [markdown]
# 2つのインデックスで指定した要素が入れ替わりました。  
# 3次元以上の配列の場合も同様に、インデックスを複数指定することで要素にアクセスすることができます。
# %% [markdown]
# ## ●関数と配列
# 
# 関数の引数や返り値としてNumPyの配列を使うことができます。  
# 以下の関数`my_func`は、引数として配列を受け取り、返り値として配列を返しています。

# %%
import numpy as np

def my_func(x):
    y = x * 2 + 1
    return y

a = np.array([[0, 1, 2],
              [3, 4, 5]])  # 2次元配列
b = my_func(a)  # 引数として配列を渡す

print(b)

# %% [markdown]
# ## ●NumPyの様々な機能
# 
# sumにより合計、averageにより平均、maxにより最大値、minにより最小値を得ることができます。

# %%
import numpy as np

a = np.array([[0, 1, 2],
              [3, 4, 5]])  # 2次元配列

print(np.sum(a))
print(np.average(a))
print(np.max(a))
print(np.min(a))

# %% [markdown]
# 引数にaxisを指定すると、特定の方向で演算を行うことができます。

# %%
import numpy as np

b = np.array([[0, 1, 2],
              [3, 4, 5]])  # 2次元配列

print(np.sum(b, axis=0))  # 縦方向で合計
print(np.sum(b, axis=1))  # 横方向で合計



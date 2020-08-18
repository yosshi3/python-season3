# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# <a href="https://colab.research.google.com/github/yukinaga/lecture_pytorch/blob/master/python_basic/02_python_basic_2.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# %% [markdown]
# # Pythonの基礎2
# Pythonの基礎として、関数やクラスについて解説します。
# %% [markdown]
# ## ●関数
# 関数を用いることで、複数行の処理をまとめることができます。  
# 関数はdefのあとに関数名を記述し、()の中に引数を記述します。  
# returnのあとの値が返り値になります。  

# %%
def add(a, b):          # 関数の定義
    c = a + b
    return c

print(add(3, 4))        # 関数の実行

# %% [markdown]
# 引数にはデフォルト値を設定できます。  
# デフォルト値を設定すると、関数を呼び出す際にその引数を省略できます。  
# 以下の例では、第2引数にデフォルト値が設定されています。  

# %%
def add(a, b=4):        # 第2引数にデフォルト値を設定
    c = a + b
    return c

print(add(3))           # 第2引数は指定しない

# %% [markdown]
# また、`*`（アスタリスク）を付けたタプルを用いて、複数の引数を一度に渡すことができます。

# %%
def add(a, b ,c):
    d = a + b + c
    print(d)

e = (1, 2, 3)
add(*e)           # 複数の引数を一度に渡す

# %% [markdown]
# ## ●変数のスコープ
# 関数内で定義された変数がローカル変数、関数外で定義された変数がグローバル変数です。  
# ローカル変数は同じ関数内からのみ参照できますが、グローバル変数はどこからでも参照できます。  

# %%
a = 123         # グローバル変数

def showNum():
    b = 456     # ローカル変数
    print(a, b)
    
showNum()

# %% [markdown]
# Pythonでは、関数内でグローバル変数に値を代入しようとすると、新しいローカル変数とみなされます。  
# 以下の例では、関数内でグローバル変数aに値を代入しても、グローバル変数aの値は変わっていません。

# %%
a = 123

def setLocal():
    a = 456         # aはローカル変数とみなされる
    print("Local:", a)
    
setLocal()
print("Global:", a)

# %% [markdown]
# グローバル変数の値を変更するためには、`global`を用いて、変数がローカルではないことを明記する必要があります。  

# %%
a = 123

def setGlobal():
    global a            # nonlocalでも可
    a = 456
    print("Global:", a)
    
setGlobal()
print("Global:", a)

# %% [markdown]
# ## ●クラス
# Pythonでは、オブジェクト指向プログラミングが可能です。  
# オブジェクト指向は、オブジェクト同士の相互作用として、システムの振る舞いをとらえる考え方です。  
# 
# オブジェクト指向には、クラスとインスタンスという概念があります。  
# クラスは設計図のようなもので、インスタンスは実体です。  
# クラスから複数のインスタンスを生成することができます。  
# クラスとインスタンスを総称して、オブジェクトといいます。  
# 
# Pythonでクラスを定義するためには、`class`の表記を用います。  
# クラスを用いると、複数のメソッドをまとめることができます。  
# メソッドは関数に似ており、defで記述を開始します。  
# 
# 以下の例では、`Calc`クラス内に`__init__`メソッド、`add`メソッド、`multiply`メソッドが実装されています。

# %%
class Calc:
    def __init__(self, a):
        self.a = a
   
    def add(self, b):
        print(self.a + b)
        
    def multiply(self, b):
        print(self.a * b)

# %% [markdown]
# Pythonのメソッドは引数として`self`を受け取るという特徴があります。  
# このselfを用いて、インスタンス変数にアクセスすることができます。  
# インスタンス変数は、クラスからインスタンスを生成し、そちらの方でアクセスする変数です。  
# 
# 　`__init__`は特殊なメソッドで、コンストラクタと呼ばれています。  
#  このメソッドで、インスタンスの初期設定を行います。  
#  上記のクラスでは、`self.a = a`で引数として受け取った値をインスタンス変数`a`に代入します。
# 
# `add`メソッドと`multiply`メソッドでは、引数として受け取った値をインスタンス変数`a`と演算しています。  
# このように、一度メソッドで値が代入されたインスタンス変数は、同じインスタンスのどのメソッドからでも`self`を用いてアクセスすることができます。 
# 
# 上記のクラスCalcから、以下のようにインスタンスを生成しメソッドを呼び出すことができます。  
# この場合、`Calc(3)`でインスタンスを生成し、変数`calc`に代入しています。  

# %%
calc = Calc(3)
calc.add(4)
calc.multiply(4)

# %% [markdown]
# 初期化時に3という値をインスタンスに渡し、addメソッドとmultiplyメソッドを呼び出します。  
# 実行すると、4+3と4x3、それぞれの計算結果を得ることができます。  
# 
# また、クラスには継承という概念があります。  
# クラスを継承することで、既存のクラスを引き継いで新たなクラスを定義することができます。  
# 以下の例では、`Calc`クラスを継承して`CalcPlus`クラスを定義しています。  

# %%
class CalcPlus(Calc):     # Calcを継承
    def subtract(self, b):
        print(self.a - b)
        
    def divide(self, b):
        print(self.a / b)

# %% [markdown]
# `subtract`メソッドと、`divide`メソッドが新たに追加されています。  
# それでは、CalcPlusメソッドからインスタンスを生成し、メソッドを呼び出してみましょう。  

# %%
calc_plus = CalcPlus(3)
calc_plus.add(4)
calc_plus.multiply(4)
calc_plus.subtract(4)
calc_plus.divide(4)

# %% [markdown]
# 継承元の`Calc`クラスで定義されたメソッドも、これを継承した`CalcPlus`クラスで定義されたメソッドも、同じように呼び出すことができます。  
# このようなクラスの継承を利用すれば、複数のクラスの共通部分を継承元のクラスにまとめることができます。  
# %% [markdown]
# ## ●\_\_call\_\_メソッド
# \_\_init\_\_の他に、\_\_call\_\_という特殊なメソッドがあります。  
# このメソッドをクラス内に実装すると、インスタンス名からメソッドを呼び出すことができます。
# 

# %%
class Hello:
    def __init__(self, name):
        self.name = name

    def __call__(self):
        print("Hello " + self.name + "!")

h = Hello("AI")
h()  # インスタンス名hを使って__call__メソッドを呼ぶ

Hello("AI")()  # 上に同じ



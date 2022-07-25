import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data


class Function:
    def __call__(self, input):   # callするとforward()する
        x = input.data
        y = self.forward(x)
        output = Variable(y)     # Variable()にくるむ
        return output

    def forward(self, in_data):
        raise NotImplementedError()


class Square(Function):    # Functionを継承する
    def forward(self, x):
        return x ** 2


x = Variable(np.array(10))
f = Square()
y = f(x)
print(type(y))
print(y.data)
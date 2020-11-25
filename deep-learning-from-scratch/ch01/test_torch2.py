# coding: utf-8
import sys, os
sys.path.append(os.pardir)

import numpy as np
#from common.functions import step_function,sigmoid,softmax
import matplotlib.pyplot as plt
import torch

torch.set_printoptions(precision=2)

x = torch.arange(-2, 2.1,step=0.1,dtype=float,requires_grad=True)
func = lambda x : x ** 3
y1 = func(x)

gradients = torch.ones(x.shape)
y1.backward(gradients)
y2 = x.grad

plt.plot(x.detach().numpy(), y1.detach().numpy(), label="y=f(x)")
plt.plot(x.detach().numpy(), y2.numpy(), linestyle = "--", label="dy/dx")
plt.xlabel("x")
plt.ylabel("y")
plt.yticks(np.arange(-10,13,step=2))
plt.grid()
plt.title('dy/dx')
plt.legend()
plt.show()

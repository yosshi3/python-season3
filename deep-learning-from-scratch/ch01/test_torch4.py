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

plt.plot(x.detach().numpy(), y1.detach().numpy(), label="y=f(x)")

for i in range(2):
    y1 = torch.autograd.grad(y1, x, create_graph=True, grad_outputs=torch.ones(x.shape))
    plt.plot(x.detach().numpy(), y1[0].detach().numpy(), 
             linestyle = "--", label="y=f" + "'" * (i+1) +  "(x)")

plt.xlabel("x")
plt.ylabel("y")
plt.yticks(np.arange(-14,14,step=2))
plt.grid()
plt.legend()
plt.show()

# coding: utf-8
import sys, os
sys.path.append(os.pardir)

import numpy as np
from common.functions import step_function,sigmoid,softmax

print(np.get_printoptions())
np.set_printoptions(precision=2)

x = np.arange(-5, 6,step=0.1,dtype=float)

y1 = step_function(x)
y2 = sigmoid(x)

z1 = np.stack([y1,y2])

z2 = np.concatenate([y1,y2])
z3 = z2.reshape(2,-1)

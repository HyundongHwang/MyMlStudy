import numpy as np
import matplotlib.pyplot as plt



def sigmoid(x):
    res = 1 / (1 + np.exp(-x))
    return res



def relu(x):
    res = np.maximum(0, x)
    return res



def identify_function(x):
    res = x
    return res
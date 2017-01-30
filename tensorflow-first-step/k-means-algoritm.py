import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import seaborn as sns



num_points = 2000
vectors_set = []

for i in range(num_points) :
    if np.random.random() > 0.5 :
        vectors_set.append([np.random.normal(0, 0.9), np.random.normal(0, 0.9)])
    else :
        vectors_set.append([np.random.normal(3, 0.5), np.random.normal(1, 0.5)])


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

df = pd.DataFrame({
    "x": [v[0] for v in vectors_set], 
    "y": [v[1] for v in vectors_set]})

sns.lmplot("x", "y", data=df, fit_reg=False, size=6)
plt.show()
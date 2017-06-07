import numpy as np

for x in range(100):
    if x in [0, 1, 2, 4, 8, 16]:
        print(x)

for x in [0, 1, 2, 4, 8, 16]:
    print("x = %d" % (x))

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import tensorflow as tf
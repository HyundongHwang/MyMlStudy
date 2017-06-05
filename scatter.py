import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

fig = plt.figure()
ax = fig.gca(projection='3d')

x = np.array([0, 0, 1, 1])
y = np.array([0, 1, 0, 1])
z = np.array([1, 0, 0, 1])

ax.scatter(x, y, z)

plt.xlim(-10, 10)
plt.ylim(-10, 10)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.title("test")
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

fig = plt.figure()
ax = fig.gca(projection='3d')

x = np.arange(-10, 10, 1)
y = np.arange(-10, 10, 1)
x, y = np.meshgrid(x, y)

z = 2 * x**2 + 3 * y**3

ax.plot_surface(x, y, z)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('x')
plt.title("test")
plt.show()
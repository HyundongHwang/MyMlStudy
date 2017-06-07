import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import tensorflow as tf



################################################################################
# original function
def func(x0, x1) :
    y = 10 * (x0 ** 2) + 20 * (x1 ** 2)
    return y



################################################################################
# original data
# -10 to +10 in increments of 0.5 40 * 40 = 1600
x0 = np.arange(-10, 10, 0.5)
x1 = np.arange(-10, 10, 0.5)
y = func(x0, x1)

xMat = np.zeros((1600, 2))
yMat = np.zeros((1600, 1))
idx = 0

for v0 in x0:
    for v1 in x1:
        xMat[idx][0] = v0
        xMat[idx][1] = v1
        yMat[idx][0] = func(v0, v1)
        idx = idx + 1



################################################################################
# original data plot
fig = plt.figure()
ax = fig.gca(projection='3d')
x0, x1 = np.meshgrid(x0, x1)
ax.plot_surface(x0, x1, y)
ax.set_xlabel('x0')
ax.set_ylabel('x1')
ax.set_zlabel('y')
plt.title("original ( y = 10*x0**2 + 20*x1**2 )")
plt.savefig("original")



################################################################################
# Nueral Network model with the original data
#   layer count : 4
#   hidden layer size : 20
#   activation function : relu
#   optimizer : Adam
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hidden_layer_size = 20;

W1 = tf.Variable(tf.random_uniform([2, hidden_layer_size], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([hidden_layer_size, hidden_layer_size], -1.0, 1.0))
W3 = tf.Variable(tf.random_uniform([hidden_layer_size, hidden_layer_size], -1.0, 1.0))
W4 = tf.Variable(tf.random_uniform([hidden_layer_size, 1], -1.0, 1.0))

b1 = tf.Variable(tf.zeros([hidden_layer_size]), name="bias1")
b2 = tf.Variable(tf.zeros([hidden_layer_size]), name="bias2")
b3 = tf.Variable(tf.zeros([hidden_layer_size]), name="bias3")
b4 = tf.Variable(tf.zeros([1]), name="bias4")

L2 = tf.nn.relu(tf.matmul(X, W1) + b1)
L3 = tf.nn.relu(tf.matmul(L2, W2) + b2)
L4 = tf.nn.relu(tf.matmul(L3, W3) + b3)
y_ = tf.matmul(L4, W4) + b4
cost = tf.reduce_mean(tf.square(y_ - Y))



################################################################################
# training NN model ...
#   train count : 10000
#   minimum cost : 30!!!
train = tf.train.AdamOptimizer(0.1).minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(10000):
    sess.run(train, feed_dict={X:xMat, Y:yMat})
    if step in [0, 1, 2, 3, 10, 20, 30, 100, 200, 300, 999]:
        print("step : ", step)
        thisCost = sess.run(cost, feed_dict={X:xMat, Y:yMat})
        print("cost : ", thisCost)

        ################################################################################
        # trained NN plot
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        x_0 = xMat[:,0]
        x_1 = xMat[:,1]
        thisY = sess.run(y_, feed_dict={X:xMat})
        ax.scatter(x_0, x_1, thisY)
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        ax.set_xlabel('x0')
        ax.set_ylabel('x1')
        ax.set_zlabel('y')
        plt.title("ML result step(%d) cost(%d) ( y = 10*x0**2 + 20*x1**2 )" % (step, thisCost))
        plt.savefig("ML result step(%d)" % (step))



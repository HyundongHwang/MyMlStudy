import numpy as np

def func(x0, x1) :
    y = 10 * (x0 ** 2) + 20 * (x1 ** 2)
    return y

x0 = np.arange(-10, 10, 0.1)
x1 = np.arange(-10, 10, 0.1)
y = func(x0, x1)

xMat = np.zeros((40000, 2))
yMat = np.zeros((40000, 1))
idx = 0

for v0 in x0:
    for v1 in x1:
        xMat[idx][0] = v0
        xMat[idx][1] = v1
        yMat[idx][0] = func(v0, v1)
        idx = idx + 1



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

fig = plt.figure()
ax = fig.gca(projection='3d')
x0, x1 = np.meshgrid(x0, x1)
ax.plot_surface(x0, x1, y)
ax.set_xlabel('x[0]')
ax.set_ylabel('x[1]')
ax.set_zlabel('y')
plt.title("정답")
plt.show()




import tensorflow as tf

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
train = tf.train.AdamOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(30000):
    sess.run(train, feed_dict={X:xMat, Y:yMat})
    if step % 100 == 0 :
        print("step : ", step)
        print("cost : ", sess.run(cost, feed_dict={X:xMat, Y:yMat}))
        #print("W1 : ", sess.run(W1))
        #print("W2 : ", sess.run(W2))
        #print("W3 : ", sess.run(W3))



yRun = sess.run(y_, feed_dict={X:xMat})

fig = plt.figure()
ax = fig.gca(projection='3d')
x_0 = xMat[:,0]
x_1 = xMat[:,1]
ax.scatter(x_0, x_1, sess.run(y_, feed_dict={X:xMat}))
plt.xlim(-10, 10)
plt.ylim(-10, 10)
ax.set_xlabel('x[0]')
ax.set_ylabel('x[1]')
ax.set_zlabel('y')
plt.title("ML결과")
plt.show()

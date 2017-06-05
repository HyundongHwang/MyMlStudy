import numpy as np
import sys, os

scriptDir = os.path.dirname(os.path.realpath(__file__))
dataFilePath = os.path.join(scriptDir, "xor-deep-network.txt")
xy = np.loadtxt(dataFilePath, unpack=True)
x_data = np.transpose(xy[0:-1])
y_data = np.reshape(xy[-1], (4,1))



import tensorflow as tf

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hidden_layer_size = 10;

W1 = tf.Variable(tf.random_uniform([2, hidden_layer_size], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([hidden_layer_size, 1], -1.0, 1.0))

b1 = tf.Variable(tf.zeros([hidden_layer_size]), name="bias1")
b2 = tf.Variable(tf.zeros([1]), name="bias2")

L2 = tf.sigmoid(tf.matmul(X, W1) + b1)
hypothesis = tf.sigmoid(tf.matmul(L2, W2) + b2)

cost = -tf.reduce_mean( Y * tf.log(hypothesis) + (1-Y) * tf.log(1.0 - hypothesis) )

alpha = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(alpha)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(3000):
    sess.run(train, feed_dict={X:x_data, Y:y_data})
    if step % 100 == 0 :
        print("step : ", step)
        print("cost : ", sess.run(cost, feed_dict={X:x_data, Y:y_data})) # ???
        # print("W1 : ", sess.run(W1))
        # print("W2 : ", sess.run(W2))
        print("")


print("")
print("")
print("")

print("cost : ", sess.run(cost, feed_dict={X:x_data, Y:y_data})) # ???
print("W1 : ", sess.run(W1))
print("W2 : ", sess.run(W2))

correct_prediction = tf.equal(tf.floor(hypothesis + 0.5), Y)
print("correct_prediction : ", sess.run(correct_prediction, feed_dict={X:x_data, Y:y_data}))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print("accuracy : ", sess.run(accuracy, feed_dict={X:x_data, Y:y_data}))
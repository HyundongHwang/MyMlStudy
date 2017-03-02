import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)



x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))


xW = tf.matmul(x, W)
hyperthesis = tf.nn.softmax(xW + b)
y = tf.placeholder(tf.float32, [None, 10])

cost = -tf.reduce_sum(y * tf.log(hyperthesis))
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train, feed_dict={x: batch_xs, y:batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(hyperthesis, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print ( sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels}) )
import tensorflow as tf
import numpy as np

xy = np.loadtxt("xor-deep-network.txt", unpack=True)
x_data = xy[0:-1]
y_data = xy[-1]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([1, len(x_data)], -1.0, 1.0))

h = tf.matmul(W, X)
hypothesis = tf.div(1., 1.+tf.exp(-h))

cost = -tf.reduce_mean( Y * tf.log(hypothesis) + (1-Y) * tf.log(1 - hypothesis) )

alpha = tf.Variable(0.01)
optimizer = tf.train.GradientDescentOptimizer(alpha)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train, feed_dict={X:x_data, Y:y_data})
    if step % 20 == 0 :
        print("step : ", step)
        print("cost : ", sess.run(cost, feed_dict={X:x_data, Y:y_data})) # ???
        # print("W : ", sess.run(W))
        print("")


print("")
print("")
print("")

print("cost : ", sess.run(cost, feed_dict={X:x_data, Y:y_data})) # ???
print("W : ", sess.run(W))

correct_prediction = tf.equal(tf.floor(hypothesis + 0.5), Y)
print("correct_prediction : ", sess.run(correct_prediction, feed_dict={X:x_data, Y:y_data}))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print("accuracy : ", sess.run(accuracy, feed_dict={X:x_data, Y:y_data}))
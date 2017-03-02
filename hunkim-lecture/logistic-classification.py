import tensorflow as tf
import numpy as np

xy = np.genfromtxt("train-logistic-classification.txt", dtype="float32", unpack=True)

x_data = xy[0:-1]
y_data = xy[-1]

print("x_data : ", x_data)
print("y_data : ", y_data)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([1, len(x_data)], -1.0, 1.0))

h = tf.matmul(W, X)
hypothesis = tf.div(1.0, 1.0 + tf.exp(-h))

cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))

a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train, feed_dict={X:x_data, Y:y_data})
    if step % 20 == 0 :
        print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W))




print ("---------------------------------------------------------------------------")
print ("Ask to ML")
# 상수, 스터디시간, 출석횟수 => 합/불 여부
print ( sess.run(hypothesis, feed_dict={X:[[1], [2], [2]]}) > 0.5 ) # 1=상수, 2=2시간스터디, 2=출석2회
print ( sess.run(hypothesis, feed_dict={X:[[1], [5], [5]]}) > 0.5 ) # 1=상수, 5=5시간스터디, 5=출석5회
print ( sess.run(hypothesis, feed_dict={X:[[1, 1], [4, 3], [3, 5]]}) > 0.5 )
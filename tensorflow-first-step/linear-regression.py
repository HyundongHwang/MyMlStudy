import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf



num_points = 1000
vectors_set = []

for i in range(num_points):
    x1 = np.random.normal(loc=0, scale=0.55)
    y1 = x1 * 0.1 + 0.3 + np.random.normal(loc=0, scale=0.03)
    vectors_set.append([x1, y1])

x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

plt.plot(x_data, y_data, "ro")
plt.legend()
plt.show()



W = tf.Variable(tf.random_uniform(shape=[1], minval=-1.0, maxval=1.0))
b = tf.Variable(tf.zeros(shape=[1]))
y = W * x_data + b



loss = tf.reduce_mean(tf.square(y - y_data))

print("W : ", W)
print("b : ", b)
print("y : ", y)
print("y - y_data : ", y - y_data)
print("tf.square(y - y_data) : ", tf.square(y - y_data))
print("loss : ", loss)

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(8):
    sess.run(train)
    plt.plot(x_data, y_data, "ro")
    plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
    plt.legend()
    plt.xlabel("x")
    plt.xlabel("y")
    plt.title("step : %d" % step)
    plt.show()

print(sess.run(W), sess.run(b))


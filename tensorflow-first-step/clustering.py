import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

points = []

for i in range(2000):
    newPoint = [np.random.randint(-10, 10), np.random.randint(-10, 10)]
    points.append(newPoint)

vectors = tf.constant(points)
expanded_vectors = tf.expand_dims(vectors, 0)

print("points : ", points)
print("vectors : ", vectors)
print("expanded_vectors : ", expanded_vectors)
print("expanded_vectors.get_shape() : ", expanded_vectors.get_shape())

a = tf.placeholder(dtype=tf.float32)
b = tf.placeholder(dtype=tf.float32)
y = tf.mul(a, b)
sess = tf.Session()
sess_run_result = sess.run(fetches=y, feed_dict={a:3, b:5})
print("a : ", a)
print("b : ", b)
print("y : ", y)
print("sess_run_result : ", sess_run_result)
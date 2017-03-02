import tensorflow as tf

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

add = tf.add(a, b)
mul = tf.mul(a, b)

sess = tf.Session()

addResult = sess.run(add, {a: 2, b: 3})
print("addResult : ", addResult)

mulResult = sess.run(mul, feed_dict={a: 2, b: 3})
print("mulResult : ", mulResult)
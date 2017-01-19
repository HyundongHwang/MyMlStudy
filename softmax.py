import tensorflow as tf
import numpy as np

xy = np.genfromtxt("train-softmax.txt", unpack=True, dtype="float32")

# 샘플입력데이타
# 0컬럼부터 3개의 컬럼 = [n행 3열]
x_data = np.transpose(xy[0:3])

# 샘플결과데이타
# 3컬럼부터 끝개의 컬럼 : [n행 3열]
y_data = np.transpose(xy[3:])

# 추정 입력 데이타
# [3행 n열]
X = tf.placeholder("float", [None, 3])

# 추정 결과 데이타
# [3행 n열]
Y = tf.placeholder("float", [None, 3])

# 추정 파라미터 데이타
# [3행 3열]
W = tf.Variable(tf.zeros([3, 3]))

# 예측함수 h
# softmax 적용
# XW = [3행 n열][3행 3열]
#       = [3행][3열]
#       = [3행 3열]
hypothesis = tf.nn.softmax(tf.matmul(X, W))

# 추정 스텝 알파값
learning_rate = 0.001

# 코스트함수
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), reduction_indices=1))

# 옵티마이저는 GradientDescent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)



init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(2001):
        sess.run(optimizer, feed_dict={X:x_data, Y:y_data})
        print( "step : ", step );
        print( "cost : ", cost );
        print( "sess.run(cost, feed_dict={X:x_data, Y:y_data}) : ", sess.run(cost, feed_dict={X:x_data, Y:y_data}) );
        print( "W : ", W );
        print( "sess.run(W) : ", sess.run(W) );
        print( "" );
        print( "" );
        print( "" );
        # if step % 200 == 0:
            # print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W))

    a = sess.run(hypothesis, feed_dict={X:[[1, 11, 7]]})
    print("a : ", a, sess.run(tf.arg_max(a, 1)))

    b = sess.run(hypothesis, feed_dict={X:[[1, 3, 4]]})
    print("b : ", b, sess.run(tf.arg_max(b, 1)))

    c = sess.run(hypothesis, feed_dict={X:[[1, 1, 0]]})
    print("c : ", c, sess.run(tf.arg_max(c, 1)))
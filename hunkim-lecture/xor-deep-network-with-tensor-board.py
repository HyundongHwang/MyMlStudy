import tensorflow as tf
import numpy as np

xy = np.loadtxt("xor-deep-network.txt", unpack=True)
x_data = np.transpose(xy[0:-1])
y_data = np.reshape(xy[-1], (4,1))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([2, 10], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([10, 1], -1.0, 1.0))

b1 = tf.Variable(tf.zeros([10]), name="bias1")
b2 = tf.Variable(tf.zeros([1]), name="bias2")

with tf.name_scope("layer2") as scope:
    L2 = tf.sigmoid(tf.matmul(X, W1) + b1)

with tf.name_scope("layer3") as scope:
    hypothesis = tf.sigmoid(tf.matmul(L2, W2) + b2)

with tf.name_scope("cost") as scope:
    cost = -tf.reduce_mean( Y * tf.log(hypothesis) + (1-Y) * tf.log(1.0 - hypothesis) )
    cost_summ = tf.scalar_summary("cost", cost)

with tf.name_scope("train") as scope:
    alpha = tf.Variable(0.1)
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train = optimizer.minimize(cost)

with tf.name_scope("accuracy") as scope:
    correct_prediction = tf.equal(tf.floor(hypothesis + 0.5), Y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    accuracy_summ = tf.scalar_summary("accuracy", accuracy)

w1_hist = tf.histogram_summary("weights1", W1)
w2_hist = tf.histogram_summary("weights2", W2)

b1_hist = tf.histogram_summary("biases1", b1)
b2_hist = tf.histogram_summary("biases2", b2)

y_hist = tf.histogram_summary("y", Y)

init = tf.global_variables_initializer()



with tf.Session() as sess:
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("./logs/xor_logs", sess.graph_def)
    sess.run(init)

    for step in range(2001):
        sess.run(train, feed_dict={X:x_data, Y:y_data})

        if step % 20 == 0 :
            summary = sess.run(merged, feed_dict={X:x_data, Y:y_data})
            writer.add_summary(summary, step)

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
    print("correct_prediction : ", sess.run(correct_prediction, feed_dict={X:x_data, Y:y_data}))
    print("accuracy : ", sess.run(accuracy, feed_dict={X:x_data, Y:y_data}))



# 
# 아래와 같이 텐서보드 서버실행
# PS> tensorboard.exe --logdir="C:\project\170109_MlStudy\hunkim-lecture\logs\xor_logs"
# 
# http://localhost:6006 에서 확인가능
# 
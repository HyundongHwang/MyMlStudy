# (고전적인) MNIST 데이터를 활용한 필기 숫자의 분류(classification)를 위해 데이터를 어떻게 다운로드 받아야 하는지
# https://tensorflowkorea.gitbooks.io/tensorflow-kr/content/g3doc/tutorials/mnist/download/
# 각 압축파일을 받아서 MNIST_data/ 에 복사해둔다
# 이렇게 하면 mnist 객체가 완성됨.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf



X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

W1 = tf.Variable(tf.random_normal([784, 256]))
W2 = tf.Variable(tf.random_normal([256, 256]))
W3 = tf.Variable(tf.random_normal([256, 10]))

B1 = tf.Variable(tf.random_normal([256]))
B2 = tf.Variable(tf.random_normal([256]))
B3 = tf.Variable(tf.random_normal([10]))

L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), B1))
L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), B2))
hypothesis = tf.add(tf.matmul(L2, W3), B3)

# 크로스 엔트로피 코스트함수
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypothesis, Y))

# 텐서플로우는 당신이 선택한 최적화 알고리즘을 적용하여 변수를 수정하고 손실을 줄일 수 있습니다.
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(hypothesis,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



tf.histogram_summary("W1", W1)
tf.histogram_summary("W2", W2)
tf.histogram_summary("W3", W3)
tf.histogram_summary("B1", B1)
tf.histogram_summary("B2", B2)
tf.histogram_summary("B3", B3)
tf.histogram_summary("Y", Y)

tf.scalar_summary("cost", cost)
tf.scalar_summary("accuracy", accuracy)



init = tf.global_variables_initializer()



with tf.Session() as sess:
    sess.run(init)
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("./logs/mnist-neural-network", sess.graph_def)

    # 학습을 시킵시다 -- 여기선 학습을 1000번 시킬 겁니다!
    for i in range(1000):
        # 학습 데이터셋에서 무작위로 선택된 100개의 데이터로 구성된 "배치(batch)"를 가져옵니다.
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys})

        if i % 100 == 0:
            summary = sess.run(merged, feed_dict={X:batch_xs, Y:batch_ys})
            writer.add_summary(summary, i)
            print("i : ", i)
            print("W1 : ", sess.run(W1))
            print("W2 : ", sess.run(W2))
            print("W3 : ", sess.run(W3))
            print("B1 : ", sess.run(B1))
            print("B2 : ", sess.run(B2))
            print("B3 : ", sess.run(B3))
            print("cost : ", sess.run(cost, feed_dict={X: batch_xs, Y: batch_ys}))
            print("")
            print("")
            print("")



    print("total accuracy : ", sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))



    randxs, randys = mnist.train.next_batch(100)
    print("random correct_prediction : ", sess.run(correct_prediction, feed_dict={X: randxs, Y: randys}))
    print("random accuracy : ", sess.run(accuracy, feed_dict={X: randxs, Y: randys}))



# 
# 아래와 같이 텐서보드 서버실행
# PS> tensorboard.exe --logdir="C:\project\170109_MlStudy\hunkim-lecture\logs\mnist-neural-network"
# 
# http://localhost:6006 에서 확인가능
# 
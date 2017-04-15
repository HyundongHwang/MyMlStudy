# (고전적인) MNIST 데이터를 활용한 필기 숫자의 분류(classification)를 위해 데이터를 어떻게 다운로드 받아야 하는지
# https://tensorflowkorea.gitbooks.io/tensorflow-kr/content/g3doc/tutorials/mnist/download/
# 각 압축파일을 받아서 MNIST_data/ 에 복사해둔다
# 이렇게 하면 mnist 객체가 완성됨.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf



# x = [n행 784열]
# 샘플 입력값 이미지들
x = tf.placeholder(tf.float32, [None, 784])

# y = [n행 10열]
# 샘플 결과값 레이블들
y = tf.placeholder(tf.float32, [None, 10])

# W = [784행 10열]
W = tf.Variable(tf.zeros([784, 10]))

# b = [10행 1열]
b = tf.Variable(tf.zeros([10]))



# 모델 생성
# hypothesis = xw + b
#   = [n행 784열][784행 10열] + [10행 1열]
#   = [n행][10열] + [10행 1열]
hypothesis = tf.nn.softmax(tf.matmul(x, W) + b)

# 크로스 엔트로피 코스트함수
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), reduction_indices=[1]))

# 텐서플로우는 당신이 선택한 최적화 알고리즘을 적용하여 변수를 수정하고 손실을 줄일 수 있습니다.
train = tf.train.GradientDescentOptimizer(0.5).minimize(cost)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(hypothesis,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



init = tf.global_variables_initializer()



with tf.Session() as sess:
    sess.run(init)

    # 학습을 시킵시다 -- 여기선 학습을 1000번 시킬 겁니다!
    for i in range(1000):
        # 학습 데이터셋에서 무작위로 선택된 100개의 데이터로 구성된 "배치(batch)"를 가져옵니다.
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train, feed_dict={x: batch_xs, y: batch_ys})

        if i % 10 == 0:
            print("i : ", i)
            print("W : ", sess.run(W))
            print("b : ", sess.run(b))
            print("cost : ", sess.run(cost, feed_dict={x: batch_xs, y: batch_ys}))
            print("")
            print("")
            print("")



    print("total accuracy : ", sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))



    randxs, randys = mnist.train.next_batch(100)
    print("random correct_prediction : ", sess.run(correct_prediction, feed_dict={x: randxs, y: randys}))
    print("random accuracy : ", sess.run(accuracy, feed_dict={x: randxs, y: randys}))
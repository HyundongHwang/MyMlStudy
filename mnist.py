# (고전적인) MNIST 데이터를 활용한 필기 숫자의 분류(classification)를 위해 데이터를 어떻게 다운로드 받아야 하는지
# https://tensorflowkorea.gitbooks.io/tensorflow-kr/content/g3doc/tutorials/mnist/download/
# 각 압축파일을 받아서 MNIST_data/ 에 복사해둔다
# 이렇게 하면 mnist 객체가 완성됨.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print("mnist.test.images : ", mnist.test.images)
print("mnist.test.labels : ", mnist.test.labels)
print("mnist.test.images[0] : ", mnist.test.images[0])
print("mnist.test.labels[0] : ", mnist.test.labels[0])
print("len mnist.test.images : ", len(mnist.test.images))
print("len mnist.test.labels : ", len(mnist.test.labels))

exit()

import tensorflow as tf

# x = [n행 784열]
# 샘플 입력값 이미지들
x = tf.placeholder(tf.float32, [None, 784])

# W = [784행 10열]
W = tf.Variable(tf.zeros([784, 10]))

# b = [10행 1열]
b = tf.Variable(tf.zeros([10]))

# y = xW + b
#   = [n행 784열][784행 10열] + [10행 1열]
#   = [n행][10열] + [10행 1열]
# hypothesis
y = tf.nn.softmax(tf.matmul(x, W) + b)

# y_ = [n행 10열]
# 샘플 결과값 레이블들
y_ = tf.placeholder(tf.float32, [None, 10])

# 크로스 엔트로피 코스트함수
# yy 는 우리가 예측한 확률 분포이며, y′y′는 실제 분포(우리가 입력하는 원-핫 벡터) 입니다.
cost_cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# 텐서플로우는 당신이 선택한 최적화 알고리즘을 적용하여 변수를 수정하고 손실을 줄일 수 있습니다.
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cost_cross_entropy)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

# 학습을 시킵시다 -- 여기선 학습을 1000번 시킬 겁니다!
for i in range(1000):
    # 학습 데이터셋에서 무작위로 선택된 100개의 데이터로 구성된 "배치(batch)"를 가져옵니다.
    # 무작위 데이터의 작은 배치를 사용하는 방법을 확률적 학습(stochastic training)이라고 부릅니다 -- 여기서는 확률적 경사 하강법입니다. 이상적으로는 학습의 매 단계마다 전체 데이터를 사용하고 싶지만(그렇게 하는게 우리가 지금 어떻게 하는게 좋을지에 대해 더 잘 알려줄 것이므로), 그렇게 하면 작업이 무거워집니다. 따라서 그 대신에 매번 서로 다른 부분집합을 사용하는 것입니다. 이렇게 하면 작업 내용은 가벼워지지만 전체 데이터를 쓸 때의 이점은 거의 다 얻을 수 있기 때문입니다.+
    batch_xs, batch_ys = mnist.train.next_batch(100)
    tran_step_res = sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    # print("sess.run(W) : ", sess.run(W))
    # print("sess.run(b) : ", sess.run(b))
    # print("sess.run(cost_cross_entropy) : ", sess.run(cost_cross_entropy, feed_dict={x: batch_xs, y_: batch_ys}))
    # print("")
    # print("")
    # print("")


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

correct_prediction_res = sess.run(correct_prediction, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
print("correct_prediction_res : ", correct_prediction_res)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

idx_0_res_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images[0], y_: mnist.test.labels[0]})
print("idx_0_res_accuracy : ", idx_0_res_accuracy)

# print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

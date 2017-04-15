# (고전적인) MNIST 데이터를 활용한 필기 숫자의 분류(classification)를 위해 데이터를 어떻게 다운로드 받아야 하는지
# https://tensorflowkorea.gitbooks.io/tensorflow-kr/content/g3doc/tutorials/mnist/download/
# 각 압축파일을 받아서 MNIST_data/ 에 복사해둔다
# 이렇게 하면 mnist 객체가 완성됨.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf


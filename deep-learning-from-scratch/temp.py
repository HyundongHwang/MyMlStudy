def hhd_log(expr):
    print("%s : %s" % (expr, eval(expr)))



if False:



    ################################################################################
    # p31_basic
    print("hello")
    print(type(10))
    print(type("hello"))

    hungry = True

    if hungry :
        print("hungry")
    else :
        print("not hungry")

    for x in [1, 2, 3]:
        print(x)



    ################################################################################
    # p33_function
    def hello():
        print("hello")

    hello()

    def hello(str):
        print("hello " + str)

    hello("world")



    ################################################################################
    # p34_class
    class Man:
        def __init__(self, name):
            self.name = name
            print("__init__")

        def hello(self):
            print("hello " + self.name)

    m = Man("will")
    m.hello()



    ################################################################################
    # p36_numpy
    import numpy as np
    x = np.array([1, 2, 3])
    print(x)



    ################################################################################
    # ex_python_path
    import sys
    print(sys.executable)



    ################################################################################
    # ex_pickle_serialize
    import pickle
 
    class Rectangle:
        def __init__(self, width, height):
            self.width = width
            self.height = height
            self.area = width * height
 
    rect = Rectangle(10, 20)
 
    # 사각형 rect 객체를 직렬화 (Serialization)
    with open('rect.data', 'wb') as f:
        pickle.dump(rect, f)
 
 
    # 역직렬화 (Deserialization)
    with open('rect.data', 'rb') as f:
        r = pickle.load(f)
 
    print("%d x %d" % (r.width, r.height))




    ################################################################################
    # p37_numpy_calc
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    z = x + y
    print("z : ", z)

    A = np.array([[1,2],[3,4], [5,6]])
    hhd_log("A")
    hhd_log("A.shape")
    hhd_log("A[0]")
    hhd_log("A[0][1]")
    hhd_log("A[A > 2]")



    ################################################################################
    # p42_pyplot_simple
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.arange(0, 6, 0.1)
    y = np.sin(x)

    plt.plot(x, y)
    plt.show()



    ################################################################################
    # p43_pyplot_complex
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.arange(0, 6, 0.1)
    y1 = np.sin(x)
    y2 = np.cos(x)

    plt.plot(x, y1, label="sin")
    plt.plot(x, y2, linestyle="--", label="cos")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("sin and cos")
    plt.legend()
    plt.show()



    ################################################################################
    # p44_img_load
    import matplotlib.pyplot as plt
    from matplotlib.image import imread

    img = imread("lena.png")
    plt.imshow(img)
    plt.show()



    ################################################################################
    # p51_perceptron_simple_and
    def AND(x1, x2):
        w1, w2, theta = 0.5, 0.5, 0.7
        tmp = x1 * w1 + x2 * w2
        if tmp <= theta:
            return 0
        elif tmp > theta:
            return 1

    hhd_log("AND(0, 0)")
    hhd_log("AND(0, 1)")
    hhd_log("AND(1, 0)")
    hhd_log("AND(1, 1)")



    ################################################################################
    # p53_w_b_and
    import numpy as np

    def AND(x1, x2):
        x = np.array([x1, x2])
        w = np.array([0.5, 0.5])
        b = -0.7
        tmp = np.sum(w * x) + b
        if tmp <= 0:
            return 0
        elif tmp > 0:
            return 1

    hhd_log("AND(0, 0)")
    hhd_log("AND(0, 1)")
    hhd_log("AND(1, 0)")
    hhd_log("AND(1, 1)")



    ################################################################################
    # p49_nand_or
    import numpy as np

    def NAND(x1, x2):
        x = np.array([x1, x2])
        w = np.array([-0.5, -0.5])
        b = 0.7
        tmp = np.sum(w * x) + b
        if tmp <= 0:
            return 0
        elif tmp > 0:
            return 1

    hhd_log("NAND(0, 0)")
    hhd_log("NAND(0, 1)")
    hhd_log("NAND(1, 0)")
    hhd_log("NAND(1, 1)")

    def OR(x1, x2):
        x = np.array([x1, x2])
        w = np.array([0.5, 0.5])
        b = -0.7
        tmp = np.sum(w * x) + b
        if tmp <= 0:
            return 0
        elif tmp > 0:
            return 1

    hhd_log("OR(0, 0)")
    hhd_log("OR(0, 1)")
    hhd_log("OR(1, 0)")
    hhd_log("OR(1, 1)")



    ################################################################################
    # p59_xor
    import numpy as np

    def XOR(x1, x2):
        s1 = NAND(x1, x2)
        s2 = OR(x1, x2)
        y = AND(s1, s2)
        return y

    hhd_log("XOR(0, 0)")
    hhd_log("XOR(0, 1)")
    hhd_log("XOR(1, 0)")
    hhd_log("XOR(1, 1)")



    ################################################################################
    # p69_step_function
    import numpy as np

    def step_function(x):
        if x > 0:
            return 1
        else:
            return 0

    def step_function(x):
        y = x > 0
        return y.astype(np.int)

    x = np.array([-1.0, 1.0, 2.0])
    y = x > 0
    hhd_log("y")




    ################################################################################
    # p70_step_function_plt
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.arange(-5, 5, 0.1)
    y = step_function(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()



    ################################################################################
    # p72_sigmoid
    import numpy as np
    import matplotlib.pyplot as plt

    def sigmoid(x):
        res = 1 / (1 + np.exp(-x))
        return res

    x = np.arange(-5, 5, 0.1)
    y = sigmoid(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()



    ################################################################################
    # p76_relu
    import numpy as np
    import matplotlib.pyplot as plt

    def relu(x):
        res = np.maximum(0, x)
        return res

    x = np.arange(-5, 5, 0.1)
    y = relu(x)
    plt.plot(x, y)
    plt.show()



    ################################################################################
    # p77_mat_dot
    import numpy as np
    import matplotlib.pyplot as plt

    A = np.array([1, 2, 3, 4])
    hhd_log("A")
    hhd_log("np.ndim(A)")
    hhd_log("A.shape")
    hhd_log("A.shape[0]")

    B = np.array([[1,2],[3,4],[5,6]])
    hhd_log("B")
    hhd_log("np.ndim(B)")
    hhd_log("B.shape")

    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    hhd_log("np.dot(A, B)")

    A = np.array([[1, 2, 3], [4, 5, 6]])
    B = np.array([[1, 2], [3, 4], [5, 6]])
    hhd_log("np.dot(A, B)")



    ################################################################################
    # p82_nn_dot
    import numpy as np
    import matplotlib.pyplot as plt

    X = np.array([1, 2])
    hhd_log("X.shape")

    W = np.array([[1, 3, 5], [2, 4, 6]])
    hhd_log("W")
    hhd_log("W.shape")

    Y = np.dot(X, W)
    hhd_log("Y")



    ################################################################################
    # p83_nn_3depth
    import numpy as np
    import matplotlib.pyplot as plt
    import hhd_utils


    X = np.array([1, 0.5])
    W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    B1 = np.array([0.1, 0.2, 0.3])

    hhd_log("X.shape")
    hhd_log("W1.shape")
    hhd_log("B1.shape")

    A1 = np.dot(X, W1) + B1
    Z1 = hhd_utils.sigmoid(A1)

    hhd_log("A1")
    hhd_log("Z1")

    def identify_function(x):
        return x

    W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    B2 = np.array([0.1, 0.2])

    hhd_log("W2.shape")
    hhd_log("B2.shape")

    A2 = np.dot(Z1, W2) + B2
    Z2 = hhd_utils.sigmoid(A2)

    hhd_log("A2")
    hhd_log("Z2")

    W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
    B3 = np.array([0.1, 0.2])

    hhd_log("W3.shape")
    hhd_log("B3.shape")

    A3 = np.dot(Z2, W3) + B3
    Y = identify_function(A3)

    hhd_log("A3")
    hhd_log("Y")



    ################################################################################
    # p89_nn_impl_summary
    import numpy as np
    import matplotlib.pyplot as plt
    import hhd_utils

    def init_network():
        network = {}
        network["W1"] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
        network["b1"] = np.array([0.1, 0.2, 0.3])
        network["W2"] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
        network["b2"] = np.array([0.1, 0.2])
        network["W3"] = np.array([[0.1, 0.3], [0.2, 0.4]])
        network["b3"] = np.array([0.1, 0.2])
        return network

    def forward(network, x):
        W1, W2, W3 = network["W1"], network["W2"], network["W3"]
        b1, b2, b3 = network["b1"], network["b2"], network["b3"]

        a1 = np.dot(x, W1) + b1
        z1 = hhd_utils.sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = hhd_utils.sigmoid(a2)
        a3 = np.dot(z2, W3) + b3
        y = hhd_utils.identify_function(a3)

        return y

    network = init_network()
    x = np.array([1, 0.5])
    y = forward(network, x)
    hhd_log("y")



    ################################################################################
    # p94_softmax_feature
    import numpy as np
    import matplotlib.pyplot as plt
    import hhd_utils

    def softmax(a):
        c = np.max(a)
        exp_a = np.exp(a - c)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        return y

    a = np.array([0.3, 2.9, 4])
    y = softmax(a)
    hhd_log("y")



################################################################################
# p00_
import numpy as np
import matplotlib.pyplot as plt
import hhd_utils

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class exercise(object):
    #写一个可以控制几次方的多项式模拟程序，模拟函数为sin(x)
    def __init__(self, n):
        self.learning_rate = 0.000001
        self.n = n

        self.xs = np.linspace(-3.0, 3.0, 100)
        self.ys = np.sin(self.xs)

        self.X = tf.placeholder(tf.float32)
        self.Y = tf.placeholder(tf.float32)

        self.W = tf.Variable(tf.random_normal([n]) * 0.001)
        self.b = tf.Variable(tf.random_normal([1]) * 0.001)
        self.y_pred = 0
        self.y_pred = tf.add(tf.multiply(self.W[0], self.X), self.b)
        print(self.y_pred)
        for i in range(n - 1):
            self.y_pred =  tf.add(self.y_pred,tf.multiply(self.W[i + 1], tf.pow(self.X, i + 1)))

        loss = tf.square(self.Y - self.y_pred)
        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for k in range(100):
                total_loss = 0
                for j in range(100):
                    o, l = sess.run([optimizer, loss],feed_dict={self.X:self.xs[j], self.Y:self.ys[j]})
                    total_loss += l
                if k % 5 == 0:
                    print('Epoch {}:loss{}'.format(k, total_loss))
            print(sess.run(self.W))


a = exercise(10)

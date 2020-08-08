import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

learning_rate = tf.constant(0.001)

xs = tf.linspace(-3.0, 3.0, 100)
ys = tf.sin(xs) + tf.random_normal([100])
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
W = tf.Variable(tf.random_normal([1]))
b = tf.Variable(tf.random_normal([1]))
y_pred = tf.add(tf.multiply(W, X), b)

loss = tf.square(Y - y_pred)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.device('/gpu:0'):
    sess = tf.Session(config = tf.ConfigProto(log_device_placement = True))
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(r'E:\code\七月第一部分\tensorboard', sess.graph)

    xs, ys = sess.run([xs, ys])

    for i in range(50):
        total_loss = 0
        for j in range(100):

            _, l = sess.run([optimizer, loss], feed_dict= {X:xs[j], Y:ys[j]})
            total_loss += l
        if i % 5 == 0:
            print('Epoch {}:loss {}'.format(i, total_loss/100))

writer.close()
W, b = sess.run([W, b])
print(W, b)





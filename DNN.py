#写一个具有三层结构的普通全连接神经网络,笔记本带不起来，没有验证正确性
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets(r'E:\code\七月第一部分\数据结构\mnist_dataset', one_hot=True, )

batch_size = 2
hidden_1 = 32
hidden_2 = 32
hidden_3 = 32
n_classes = 10
learning_rate =0.001
n_epochs = 30

X = tf.placeholder(tf.float32,[batch_size, 784])
Y = tf.placeholder(tf.float32,[batch_size, 10])

weights = {
        'W0': tf.Variable(tf.random_normal([784, hidden_1])),
        'W1': tf.Variable(tf.random_normal([hidden_1, hidden_2])),
        'W2': tf.Variable(tf.random_normal([hidden_2, hidden_3])),
        'W3': tf.Variable(tf.random_normal([hidden_3, n_classes])),
}
bias = {
        'b0': tf.Variable(tf.random_normal([hidden_1])),
        'b1': tf.Variable(tf.random_normal([hidden_2])),
        'b2': tf.Variable(tf.random_normal([hidden_3])),
        'b3': tf.Variable(tf.random_normal([n_classes]))
}

def DNN_3(X, weights, bias):
    z_1_0 = tf.add(tf.matmul(X, weights['W0']), bias['b0'])
    z_1_1 = tf.nn.relu(z_1_0)
    z_2_0 = tf.add(tf.matmul(z_1_1, weights['W1']), bias['b1'])
    z_2_1 = tf.nn.relu(z_2_0)
    z_3_0 = tf.add(tf.matmul(z_2_1, weights['W2']), bias['b2'])
    z_3_1 = tf.nn.relu(z_3_0)
    z_3_0 = tf.add(tf.matmul(z_3_1, weights['W3']), bias['b3'])
    z_3_1 = tf.nn.relu(z_3_0)
    return z_3_1
pred = DNN_3(X, weights, bias)

entropy = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y)
loss = tf.reduce_mean(entropy)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:

    for i in range(n_epochs):
        total_loss = 0
        n_batch = mnist.train.images.shape[0] // batch_size
        for j in range(n_batch):
            batch_X, batch_Y = mnist.train.next_batch(batch_size)
            o, l = sess.run([optimizer, loss], feed_dict= {X:batch_X, Y:batch_Y})
            total_loss += l / n_batch
        if i % 5 == 0:
            print('Epoch {}: loss {}'.format(i, total_loss))

    n_batch_test = mnist.test.images.shape[0] // batch_size
    test_pred = tf.nn.softmax(pred)
    correct = tf.equal(tf.argmax(test_pred, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    for k in range(n_batch_test):
        batch_X_test, batch_Y_test = mnist.test.next_batch(batch_size)
        r,l = sess.run([test_pred, accuracy], feed_dict={X:batch_X_test, Y:batch_Y_test})
        print(r, l)

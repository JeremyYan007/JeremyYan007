import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets(r'E:\code\七月第一部分\数据结构\mnist_dataset', one_hot=True, )
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)

batch_size = 32
learning_rate = 0.001

X = tf.placeholder(tf.float32, [batch_size, 784])
Y = tf.placeholder(tf.float32, [batch_size, 10])

W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.random_normal([1, 10]))

Y_pred = tf.add(tf.matmul(X, W), b)
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Y_pred, labels=Y)

loss = tf.reduce_mean(entropy)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    n_batch = mnist.train.images.shape[0] // batch_size

    for i in range(50):
        total_loss = 0
        for j in range(50):
            X_batch, Y_batch = mnist.train.next_batch(batch_size)
            o, l = sess.run([optimizer, loss], feed_dict={X: X_batch, Y: Y_batch})
            total_loss += l
        print('aaaaaaaaaaaaaa' + str(i))
        if i % 5 == 0:
            print('Epoch {}:{}'.format(i, total_loss))

    possible_pred = tf.nn.softmax(Y_pred)
    final_pred = tf.argmax(possible_pred, 1)
    accuracy = tf.reduce_sum(tf.cast(tf.equal(final_pred, tf.argmax(Y, 1)), tf.float32))

    n_batch_test = mnist.test.images.shape[0] // batch_size

    total_loss_test = 0
    for u in range(n_batch_test):
        X_batch, Y_batch = mnist.test.next_batch(batch_size)
        l, l_p = sess.run([accuracy, final_pred], feed_dict={X: X_batch, Y: Y_batch})
        total_loss_test += l
        print(total_loss_test, l_p)
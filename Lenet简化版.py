# 搭一个lenet简化版本模型（单层cnn单层pooling接FC层），mnist数据集

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import os
#设置训练参数
batch_size = 100
learning_rate = 0.1
learning_rate_decay = 0.9
regularizeation_rate = 0.001
moving_average_decay = 0.99
epochs = 50

model_save_path = r'./'
model_name = 'lenet_model.ckpt'


# 输入层参数设置
image_size = 28
image_channel = 1
classes_number = 10

# 卷积参数设置
layer_1_kernel_size = 5
layer_1_kernel_number = 32


# 定义一个变量定义函数，保证权重都加到正则项中
def get_varables_weights(shape, regularizer):
    weights = tf.get_variable('weights',
                              shape,
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('loss', regularizer(weights))
    return weights


# 定义卷积神经网络
def model(inputdata, regularizer):
    # 定义一个卷积层
    with tf.variable_scope('layer_1_conv'):
        layer_1_conv_kernel = get_varables_weights(shape=[layer_1_kernel_size,
                                                          layer_1_kernel_size,
                                                          image_channel,
                                                          layer_1_kernel_number], regularizer=regularizer)
        layer_1_conv_bias = tf.get_variable('bias',
                                            shape=layer_1_kernel_number,
                                            initializer=tf.truncated_normal_initializer())
        layer_1_conv_output = tf.add(tf.nn.conv2d(inputdata,
                                                  layer_1_conv_kernel,
                                                  strides=[1, 1, 1, 1],
                                                  padding='SAME'), layer_1_conv_bias)
        layer_1_relu_output = tf.nn.relu(layer_1_conv_output)
    # 定义一个池化层
    with tf.variable_scope('layer_1_pooling'):
        layer_1_pooling = tf.nn.max_pool(layer_1_relu_output,
                                         ksize=[1, 2, 2, 1],
                                         strides=[1, 2, 2, 1],
                                         padding='SAME')
    # reshape
    layer_pooling_shape = layer_1_pooling.get_shape().list()  # 四维张量，第一维是batch
    layer_pooling_reshape_1 = layer_pooling_shape[1] * layer_pooling_shape[2] * layer_pooling_shape[3]
    FC_input = tf.reshape(layer_1_pooling, [layer_pooling_shape[0], layer_pooling_reshape_1])

    # 定义一个fc层
    with tf.variable_scope('layer_2_FC'):
        layer_2_FC_weights = get_varables_weights(shape=[layer_pooling_reshape_1, classes_number],
                                                  regularizer=regularizer)
        layer_2_FC_bias = tf.get_variable('bias', shape=[classes_number], initializer=tf.constant_initializer(0.0))
        layer_2_FC_output = tf.add(tf.matmul(FC_input, layer_2_FC_weights), layer_2_FC_bias)
        layer_2_relu_output = tf.nn.relu(layer_2_FC_output)
    return layer_2_relu_output

def train(mnist):
    X = tf.placeholder(tf.float32, [None, image_size, image_size, image_channel])
    Y = tf.placeholder(tf.float32, [None, classes_number])
    regularizer = tf.contrib.layers.l2_regularizer(regularizeation_rate)
    y_pred = model(mnist, regularizer)

    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay,global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(Y, y_pred)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = tf.add(cross_entropy_mean, tf.add_n(tf.get_collection('loss')))\

    learning_rate_1 = tf.train.exponential_decay(learning_rate = learning_rate,
                                               global_step=global_step,
                                               decay_steps=mnist.train.num_examples//batch_size,
                                               decay_rate= learning_rate_decay)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate_1).minimize(loss, global_step= global_step)
    optimizer_op = tf.group(train, variable_averages_op)

    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        for j in range(epochs):
            total_loss = 0
            batch_number = mnist.train.num_examples // batch_size
            for i in range(batch_number):
                xs, ys = mnist.train.next_batch(batch_size)
                reshaped_xs = tf.reshape(xs, [batch_size, image_size, image_size,image_channel])
                o, l = sess.run([optimizer_op, global_step], feed_dict={X:reshaped_xs, Y:ys})
                total_loss += l / batch_number
            if j % 5 == 0:
                print(total_loss)
                saver.save(sess, os.path.join(model_save_path, model_name))


def main():
    mnist = input_data.read_data_sets('E:\code')
    train(mnist)

if __name__ == '__main__':
    tf.app.run()
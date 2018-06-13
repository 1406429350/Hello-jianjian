import tensorflow as tf
from sklearn import metrics
import numpy as np
import tensorflow.contrib.slim as slim

train_files = ["train250_all.tfrecords"]
test_files = ["test250_all.tfrecords"]

# 解析 tfrecords文件的数据
def parse_exmp(serial_exmp):

    feats = tf.parse_single_example(serial_exmp, features={'class': tf.FixedLenFeature([], tf.string),
                                                           'label': tf.FixedLenFeature([], tf.int64)})
    Class = tf.decode_raw(feats['class'], tf.float32)
    Class = tf.reshape(Class, [250, 1])
    #label = tf.cast(feats['label'], tf.int32)
    label = feats['label']
    return Class, label
def CNNnet(inputs, n_class):
    conv1 = tf.layers.conv1d(inputs=inputs, filters=4, kernel_size=31, strides=1, padding='same', activation=tf.nn.relu)

    avg_pool_1 = tf.layers.average_pooling1d(inputs=conv1, pool_size=5, strides=5, padding='same')

    conv2 = tf.layers.conv1d(inputs=avg_pool_1, filters=8, kernel_size=6, strides=1, padding='same', activation=tf.nn.relu)

    avg_pool_2 = tf.layers.average_pooling1d(inputs=conv2, pool_size=5, strides=5, padding='same')

    flat = tf.reshape(avg_pool_2, (-1, int(250 / 5 / 5 * 8)))

    logits = tf.layers.dense(inputs=flat, units=n_class, activation=None)

    logits = tf.nn.softmax(logits)

    return logits


buffer_size = 10000   # 一波打乱数据的大小
batch_size = 100    # 每一批次数据的大小
learning_rate = 0.01


filenames = tf.placeholder(tf.string, shape=[None])    # 用于转换目录
epochs = tf.placeholder(tf.int64)         # 训练次数

dataset = tf.data.TFRecordDataset(filenames)   # 读取数据

dataset = dataset.map(parse_exmp)      #解析数据
dataset = dataset.repeat(epochs)
dataset = dataset.batch(batch_size)

iterator = dataset.make_initializable_iterator()     #定义数据的迭代器

features, label = iterator.get_next()         # 获取每一批次数据

label_batch = tf.one_hot(label, depth=4)      # 转化标签

logit = CNNnet(features, 4)         # 网络层

cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=label_batch))  # 分类加交叉熵

train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)  # 优化

correct_prediction_train = tf.equal(tf.argmax(logit, 1), tf.argmax(label_batch, 1))     #求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction_train, tf.float32))

with tf.Session() as sess:
    sess.run((tf.global_variables_initializer(),
              tf.local_variables_initializer()))
    sess.run(iterator.initializer, feed_dict={filenames: train_files, epochs:30})
    i = 0
    while 1:
        try:
            i = i+1
            if i%100 == 0:
                train_accuracy = accuracy.eval()
                print(" 训练完了{0}批训练集数据,本批数据的识别率 train_pre :{1} ".format(i, train_accuracy))
            #train_step.run(feed_dict={keep_prob: 0.8})
            train_step.run()
        except tf.errors.OutOfRangeError:
            break
    sess.run(iterator.initializer, feed_dict={filenames: test_files, epochs: 1})
    i = 0
    while 1:
        try:
            i = i+1
            if i%100 == 0:
                test_accuracy = accuracy.eval()
                print(" 第{0}批测试集数据的识别率 test_pre :{1} ".format(i, test_accuracy))
            #train_step.run(feed_dict={keep_prob: 0.8})
        except tf.errors.OutOfRangeError:
            break


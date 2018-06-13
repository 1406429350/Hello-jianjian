import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import scipy.io as sio

def _bytes_feature(value):  #转换为字符串进行存储
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))

def _float_feature(value):  #转换为字符串进行存储
    return tf.train.Feature(float_list = tf.train.FloatList(value=[value]))


def _int64_feature(value):  #转换为整数列表进行存储
    return tf.train.Feature(int64_list= tf.train.Int64List(value=[value]))


root_path = 'E:\\Super\\search\\Demo\\data\\'
tfrecords_filename = root_path + 'tfrecords/testdata_all.tfrecords'
#------------------------------------------写入-----------------------------------------------------------------------------
#将每一列数据分开存入（同图片）
writer = tf.python_io.TFRecordWriter(tfrecords_filename)
try:
    Class1 = sio.loadmat(root_path + '/mat/' + 'final_data.mat')  #train_3000
    print(Class1)
    Class2 = Class1['data']   # 根据print（class1）中打印的元素名来填索引
    # Class=Class2.tostring()
    m, l = Class2.shape
    print(Class2.shape)
    a = int(0.7*l)
    for k in range(a, l):
        data = np.float32(Class2[0:1200, k]) # [x[i] for x in Class2] #取第i列存入line中（一列数据）
        label = np.int64(Class2[1200, k])
        if k%100==0:
            print(label)
            print(data.shape)
        line = data.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'class': _bytes_feature(line),
            'label': _int64_feature(label)}))

        writer.write(example.SerializeToString())
except IndexError:
    print("finish!")

writer.close()

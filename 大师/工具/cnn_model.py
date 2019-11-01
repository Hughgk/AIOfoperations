import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tools.image_processing import *
from sklearn.model_selection import StratifiedShuffleSplit
import os

SYMBOL = {0: '0',
          1: '1',
          2: '2',
          3: '3',
          4: '4',
          5: '5',
          6: '6',
          7: '7',
          8: '8',
          9: '9',
          10:'+',
          11:'-',
          12:'*',
          13:'/',
          14:'(',
          15:')'}

#数据预处理代码
class train_test(object):
    def __init__(self):
        self.images = None
        self.labels = None
        self.offset = 0

    def next_batch(self, batch_size):
        #将数据分批次进行测试
        #构造函数返回训练数据或测试数据的下一批次
        if self.offset + batch_size <= self.images.shape[0]:
            batch_images = self.images[self.offset:self.offset + batch_size]
            batch_labels = self.labels[self.offset:self.offset + batch_size]
            self.offset = (self.offset + batch_size) % self.images.shape[0]
        else:
            new_offset = self.offset + batch_size - self.images.shape[0]
            batch_images = self.images[self.offset:-1]
            batch_labels = self.labels[self.offset:-1]
            batch_images = np.r_[batch_images, self.images[0:new_offset]]
            batch_labels = np.r_[batch_labels, self.labels[0:new_offset]]
            self.offset = new_offset
        return batch_images, batch_labels

class digit_data(object):
    def __init__(self):
        self.train = train_test()
        self.test = train_test()

    def input_data(self):
        #读取MINIST数据集并将训练数据和测试数据进行整合
        mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
        images = np.r_[mnist.train.images, mnist.test.images]
        #labels内的数据是标记每个图片对应的数值
        #用来标注每个图片上的数字是几。把图片和标签放在一起，称为“样本”。通过样本就可以实现一个有监督信号的深度学习。
        labels = np.r_[mnist.train.labels, mnist.test.labels]
        #扩大标签数据的维度，增加6个符号维度
        #创建一个全0数组，labels.shape[0]行 6列
        zeros = np.zeros((labels.shape[0], 6))
        #矩阵相加 ， 合并
        labels = np.c_[labels, zeros]

        print("Loading the operators' datasets....")
        # 读取符号数据集并与MINIST数据集进行合并
        op_images, op_labels = get_images_labels()
        #将数字图片与运算符图片矩阵合并， label合并
        images, labels = np.r_[images, op_images], np.r_[labels, op_labels]
        print("Generating the train_data and test_data....")
        #使用sklearn中的数据划分函数生成训练数据和测试数据
        #分成16组，每组train/test ： 0.85/0.15 ，random_state随机数据打乱
        sss = StratifiedShuffleSplit(n_splits=16, test_size=0.15, random_state=23)
        for train_index, test_index in sss.split(images, labels):
            self.train.images, self.test.images = images[train_index], images[test_index]
            self.train.labels, self.test.labels = labels[train_index], labels[test_index]


class model(object):
    def __init__(self, batch_size=100, hidden_size=1024, n_output=16):
        self.HIDDEN_SIZE = hidden_size
        self.BATCH_SIZE = batch_size
        self.N_OUTPUT = n_output
        self.N_BATCH = 0

    def weight_variable(self, shape):
        #定义权重函数，并使用随机初始化权重
        # 正态分布，标准差为0.1，默认最大为1，最小为-1，均值为0
        initial = tf.truncated_normal(shape, stddev=0.10)  
        return tf.Variable(initial, name="w")

    def bias_variable(self, shape):
        #定义偏置函数，并初始化为0.1
        # 创建一个结构为shape矩阵也可以说是数组shape声明其行列，初始化所有值为0.1
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name="b")

    def conv2d(self, x, W):
        # 定义卷积函数
        # 卷积遍历各方向步数为1，SAME：边缘外自动补0，遍历相乘
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        #定义池化函数
        # 池化卷积结果（conv2d）池化层采用kernel大小为2*2，步数也为2，周围补0，取最大值。数据量缩小了4倍
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    #构造训练函数
    def train_model(self, EPOCH=21, learning_rate=1e-4, regular_coef=5e-4, model_dir='./model/', model_name='model'):
        #读取训练数据和测试数据
        mnist_operator = digit_data()
        mnist_operator.input_data()

        #计算批次数目
        #训练样本数/64
        self.N_BATCH = mnist_operator.train.images.shape[0] // self.BATCH_SIZE

        #定义输入 ： 行为定，列为784
        # 声明一个占位符，None表示输入图片的数量不定，28*28图片分辨率
        x = tf.placeholder(tf.float32, [None, 784], name='image_input')
        #列为16 ：符号+数字
        # 类别是0-9总共10个类别，对应输出分类结果
        y = tf.placeholder(tf.float32, [None, self.N_OUTPUT])
        #一维
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        #-1表示任意数量的样本数, 大小为28x28深度为一的张量
        # x_image又把xs reshape成了28*28*1的形状，因为是灰色图片，所以通道是1.作为训练时的input，-1代表图片数量不定
        x_image = tf.reshape(x, [-1, 28, 28, 1])

        #定义第一个卷积池化层
        with tf.variable_scope("conv1"):
            # 第一二参数值得卷积核尺寸大小，即patch，
            # 第三个参数是图像通道数，第四个参数是卷积核的数目，代表会出现多少个卷积特征图像;
            W_conv1 = self.weight_variable([5, 5, 1, 32])
            # 对于每一个卷积核都有一个对应的偏置量。
            b_conv1 = self.bias_variable([32])
            # 图片乘以卷积核，并加上偏执量，卷积结果28x28x32
            h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
            # 池化结果14x14x32 卷积结果乘以池化卷积核
            h_pool1 = self.max_pool_2x2(h_conv1)
        # 定义第二个卷积池化层
        with tf.variable_scope("conv2"):
            # 32通道卷积，卷积出64个特征
            W_conv2 = self.weight_variable([5, 5, 32, 64])
            # 64个偏执数据
            b_conv2 = self.bias_variable([64])
            # 注意h_pool1是上一层的池化结果，#卷积结果14x14x64
            h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
            # 池化结果7x7x64
            h_pool2 = self.max_pool_2x2(h_conv2)
            # 原图像尺寸28*28，第一轮图像缩小为14*14，共有32张，第二轮后图像缩小为7*7，共有64张
        # 定义第一个全连接层
        with tf.variable_scope("fc1"):
            # 二维张量，第一个参数7*7*64的patch，也可以认为是只有一行7*7*64个数据的卷积，第二个参数代表卷积个数共1024个
            W_fc1 = self.weight_variable([7 * 7 * 64, self.HIDDEN_SIZE])
            # 1024个偏执数据
            b_fc1 = self.bias_variable([self.HIDDEN_SIZE])
            # 将第二层卷积池化结果reshape成只有一行7*7*64个数据# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
            # 卷积操作，结果是1*1*1024，单行乘以单列等于1*1矩阵，
            # matmul实现最基本的矩阵相乘，不同于tf.nn.conv2d的遍历相乘，自动认为是前行向量后列向量
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
            # 使用占位符，由dropout自动确定scale，也可以自定义，比如0.5，
            # 根据tensorflow文档可知，程序中真实使用的值为1/0.5=2，也就是某些输入乘以2，同时某些输入乘以0

            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) #对卷积结果执行dropout操作
        # 定义第二个全连接层
        with tf.variable_scope("fc2"):
            # 二维张量，1*1024矩阵卷积，共16个卷积，对应我们开始的y长度为16
            W_fc2 = self.weight_variable([self.HIDDEN_SIZE, self.N_OUTPUT])
            b_fc2 = self.bias_variable([self.N_OUTPUT])

            h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        #定义正则项
        regularizers = (tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1) + tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(b_fc1))
        # 最后的分类，结果为1*1*10 softmax和sigmoid都是基于logistic分类算法，一个是多分类一个是二分类
        prediction = tf.nn.softmax(h_fc2, name="prediction")
        predict_op = tf.argmax(prediction, 1, name="predict_op")

        #定义损失函数
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction))
        loss_re = loss + 5e-4 * regularizers

        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_re)

        correct_prediction = tf.equal(predict_op, tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        saver = tf.train.Saver()
        tf.add_to_collection("predict_op", predict_op)

        print("Start training....")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in tqdm(range(EPOCH * self.N_BATCH)):
                epoch = i // self.N_BATCH
                batch_xs, batch_ys = mnist_operator.train.next_batch(self.BATCH_SIZE)
                sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7})
                if epoch % 10 == 0 and (i+1) % self.N_BATCH == 0:
                    acc = []
                    for i in range(mnist_operator.test.labels.shape[0]//self.BATCH_SIZE):
                        batch_xs_test, batch_ys_test = mnist_operator.test.next_batch(self.BATCH_SIZE)
                        test_acc = sess.run(accuracy, feed_dict={x: batch_xs_test, y: batch_ys_test, keep_prob: 1.0})
                        acc.append(test_acc)
                    print()
                    print("Iter" + str(epoch) + ",Testing Accuracy = " + str(sum(acc) / len(acc)))
                    if not os.path.exists(model_dir):
                        os.mkdir(model_dir)
                    saver.save(sess, model_dir + '/'+  model_name, global_step=epoch)

    #加载模型
    def load_model(self, meta, path):
        self.sess = tf.Session()
        saver = tf.train.import_meta_graph(meta)
        saver.restore(self.sess, tf.train.latest_checkpoint(path))

    #构造预测函数
    def predict(self, X):
        predict = tf.get_collection('predict_op')[0]
        graph = tf.get_default_graph()
        input_X = graph.get_operation_by_name("image_input").outputs[0]
        keep_prob = graph.get_operation_by_name("keep_prob").outputs[0]
        return self.sess.run(predict, feed_dict={input_X: X, keep_prob: 1.0})[0:]
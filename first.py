import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNIST_data 代表当前程序文件所在的目录中，用于存放MNIST数据的文件夹，如果没有则新建，然后下载．
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

''''#获取第二张图片
image = mnist.train.images[1,:]
#将图像数据还原成28*28的分辨率
image = image.reshape(28,28)
#打印对应的标签
print(mnist.train.labels[1]) '''

# x是特征值
x = tf.placeholder(tf.float32, [None, 784])
# w表示每一个特征值（像素点）会影响结果的权重
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b
# 是图片实际对应的值
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# mnist.train 训练数据
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 取得y得最大概率对应的数组索引来和y_的数组索引对比，如果索引相同，则表示预测正确
correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

print('fighting hcc')
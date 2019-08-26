import tensorflow as tf
import numpy as np

# 使用numpy，生成100个随机点
x_data = np.random.rand(100)
y_data = x_data*0.1 + 0.2

# 构造一个线性模型
# 先初始化变量，括号中的值为初始值
b = tf.Variable(1.1)
k = tf.Variable(0.5)
y = k*x_data + b

# 定义代价函数
loss = tf.reduce_mean(tf.square(y_data-y))
# 定义一个梯度下降法优化器,学习率为0.2
optimizer = tf.train.GradientDescentOptimizer(0.2)
# 最小化代价函数
train = optimizer.minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)
        if step%20 == 0:
            print(step, sess.run([k,b]))
'''
运行结果：
    0 [0.29582477, 0.6691145]
    20 [0.08723096, 0.20607442]
    40 [0.092676796, 0.20348318]
    60 [0.09580018, 0.2019976]
    80 [0.097591415, 0.20114562]
    100 [0.09861867, 0.20065701]
    120 [0.09920781, 0.2003768]
    140 [0.09954567, 0.2002161]
    160 [0.09973945, 0.20012394]
    180 [0.09985056, 0.20007108]
    200 [0.0999143, 0.20004077]
'''
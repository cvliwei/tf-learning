import tensorflow as tf


###---Independent part01--------------------------------------------------------------------
# x = tf.Variable([1,2])
# a = tf.constant([3,3])
# # 增加一个减法op
# sub = tf.subtract(x,a)
# # 增加一个加法op
# add = tf.add(a,sub)

# # 全局变量初始化
# init = tf.global_variables_initializer()

# with tf.Session() as sess:
#     # 存在变量时，需要先进行初始化
#     sess.run(init)
#     print(x.value)
#     print(sess.run(sub))
#     print(sess.run(add))
'''
运行结果：
    <bound method Variable.value of <tf.Variable 'Variable:0' shape=(2,) dtype=int32_ref>>
    [-2 -1]
    [1 2]
'''

###---Independent part02--------------------------------------------------------------------
# # 创建一个变量，初始化为0
# state = tf.Variable(0, name='counter')
# # 创建一个op
# new_value = tf.add(state,1)
# # 赋值op
# update = tf.assign(state,new_value)
# # 初始化全局变量
# init = tf.global_variables_initializer()

# # 变量初始化为0，循环更新并打印
# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(state))
#     for _ in range(5):
#         sess.run(update)
#         print(sess.run(state))
'''
运行结果：
    0
    1
    2
    3
    4
    5
'''

###---Independent part03--------------------------------------------------------------------
# fetch可以在一个会话中同时执行多个op
# input1 = tf.constant(3.0)
# input2 = tf.constant(2.0)
# input3 = tf.constant(5.0)

# add = tf.add(input2,input3)
# mul = tf.multiply(input1,add)

# with tf.Session() as sess:
#     result = sess.run([mul, add])
#     print((result))
'''
运行结果：
    [21.0, 7.0]
'''

###---Independent part04--------------------------------------------------------------------
# feed可以每次给图传入不同的数据
# 创建占位符
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)

with tf.Session() as sess:
    # feed数据以字典形式传入
    print(sess.run(output, feed_dict={input1:[2.0],input2:[6.0]}))
'''
运行结果：
    [12.]
'''
#通过Tensorflow 来对矩阵进行定义和运算

'''
矩阵的定义
'''
import tensorflow as tf
data1 = tf.constant([[6,6]])  #矩阵 一行两列
data2 = tf.constant([[2],
                     [2]])    #矩阵 两行一列
data3 = tf.constant([[1,2],
                     [3,4],
                     [5,6]])  #矩阵 三行两列

with tf.Session() as sess :
    print(sess.run(data1))
    print(sess.run(data3[0]))   #打印第一行
    print(sess.run(data3[:,0])) #打印第一列
    print(sess.run(data3[0,0])) #打印第一行第一列
##################################################
'''
矩阵的运算
'''
data4 = tf.constant([[6,6]])    #一行两列的 矩阵
data5 = tf.constant([[2,2]])
data6 = tf.constant([[2],       #两行一列的居然
                     [1]])

matAdd1 = tf.add(data4,data5)    #一行两列的矩阵加法
matMul2 = tf.matmul(data5,data6) #一行两列 乘以 两行一列
with tf.Session() as sess :
    print('***********')
    print(sess.run(matAdd1))
    print(sess.run(matMul2))
    print("一次打印多个结果：")
    print(sess.run([matAdd1,matMul2]))
##############################################
'''
定义特殊矩阵

'''
mat0 = tf.zeros([2,3])      #定义一个 两行三列矩阵（元素全为0）
mat1 = tf.ones([3,2])       #定义一个 三行两列矩阵（元素全为1）
mat2 = tf.fill([2,3],15)    #定义一个 两行三列矩阵（元素全为15）
mat3 = tf.zeros_like(mat0)  #定义一个 跟mat0维度一样的矩阵（元素全为0）
mat4 = tf.linspace(0.0,2.0,11)        #将0到2分成10份，用矩阵表示
mat5 = tf.random_uniform([2,3],-1,2)  #定义一个 两行三列的随机矩阵，数值选择从-1到2
with tf.Session() as sess:
    print("*******************")
    print(sess.run(mat0))
    print(sess.run(mat1))
    print(sess.run(mat2))
    print(sess.run(mat3))
    print(sess.run(mat4))
    print(sess.run(mat5))
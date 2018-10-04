import tensorflow as tf
data1 = tf.constant(2,dtype=tf.int32)       #常数  shape纬度  类型
data2 = tf.Variable(10,name='var')          #变量  shape纬度  整型
print(data1)
print(data2)
init = tf.global_variables_initializer()    #将变量初始化(所有的变量都必须要初始化)
sess = tf.Session()
with sess:  #用完之后关闭Session
    sess.run(init)
    print(sess.run(data2))

'''
tensorflow 的本质 = 张量(tensor) +(op)  计算图(grahps)

tensor ：本质就是数据 (包括 常量constant 和 变量variable) 纬度可以是n维

op     : operation操作 例如常见的 四则运算 或者 两个tensor之间的加法等

grahps : tensor 进行 op 之后 就形成 grahps(数据的操作过程)

tensorflow的运行 ： 所有的grahps 都要通过Session(会话)中进行

在用tensorflow的时候 可以进行以下分析 ：① 分析有哪些tensor ② 分析有哪些grahps ③ 使用Session

'''
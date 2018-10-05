#通过matplotib 来对 numpy定义的数组进行绘图


import numpy as np
import matplotlib.pyplot as plt

x1 = np.array([1,2,3,4,5,6,7,8])
y1 = np.array([3,5,7,6,2,6,10,15])
plt.plot(x1,y1,'g',lw=10)             ####折线图 x1,y1为坐标,'g'为绿色,lw=10为折线线粗

x2 = np.array([1,2,3,4,5,6,7,8])
y2 = np.array([13,25,17,36,21,16,10,15])
plt.bar(x2,y2,0.5,alpha=1,color='b')  ####柱状图 x2,y2为坐标,0.5为折线线粗，alpha为透明度
plt.show()
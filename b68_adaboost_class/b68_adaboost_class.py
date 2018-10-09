'''
adaboost 训练
1.初始化数据权值分布
    苹果  苹果 苹果 香蕉
权值:0.1  0.1  0.1 0.1

2.遍历阈值
    会算出一系列的误差概率  找到最小的minP
    minP对应的权值 t
3.G1(x)
    计算权重系数
4.更新 权值分布 update
    苹果  苹果 苹果 香蕉
权值:0.2  0.2  0.2 0.7

5.训练终止条件 1.for循环结束 2.P满足了情况，就结束
'''

'''
基于Haar+Adaboost 人脸识别
1.加载 .xml文件
2.加载图片
3.haar训练(官方已经完成) 将图片灰度
4.检测 人脸、眼睛
5.用方框圈住  脸和眼睛

'''
import cv2
import numpy as np
# 加载xml文件
face_xml = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_xml = cv2.CascadeClassifier('haarcascade_eye.xml')

# 加载图片 灰度图片
img = cv2.imread('face.jpg')
cv2.imshow('src',img)

gray =  cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces = face_xml.detectMultiScale(gray,1.3,5) #灰度图片，(BP缩放)缩放系数，目标大小(不得小于5个像素)
print('face = ',len(faces))                   #打印人脸数
# 画方框
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #画矩形（图片，起点，终点，颜色，宽度）
    roi_face = gray[y:y+h,x:x+w]                   #在灰度图片中，获取脸部位置
    roi_color = img[y:y+h,x:x+w]                   #在彩色图片中，获取脸部位置

    eyes = eye_xml.detectMultiScale(roi_face)      #检测眼睛
    print('eyes = ',len(eyes))

    for (e_x,e_y,e_w,e_h) in eyes:
        cv2.rectangle(roi_color,(e_x,e_y),(e_x+e_w,e_y+e_h),(0,255,0),2)
cv2.imshow('dst',img)
cv2.waitKey(0)

'''
可以看出，识别率并不高，把鼻子也当成眼睛了
'''
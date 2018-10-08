import cv2
img = cv2.imread('image1.jpg')
imgInof = img.shape              #先得到照片的大小，从而得到视频的大小
size = (imgInof[1],imgInof[0])
print(size)
videoWrite = cv2.VideoWriter('2.mp4',-1,5,size)  #写入对象，参数：名字，编码器，帧率，视频大小
for i in range(1,11):
    fileName = 'image'+str(i)+'.jpg'
    img = cv2.imread(fileName)
    videoWrite.write(img)       #写入方法
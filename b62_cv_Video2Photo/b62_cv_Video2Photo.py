import cv2
cap = cv2.VideoCapture("1.mp4") #获取一个视频
isOpened = cap.isOpened #判断是否打开
print(isOpened)
fps = cap.get(cv2.CAP_PROP_FPS) #帧率
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #获得帧的宽
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(fps,width,height) #打印帧数 和他们的帧数的信息

i = 0 #储存所获得的帧数
while(isOpened): #当视频能打开时
    if i == 10:
        break
    else:
        i = i+1
    (flag,frame) = cap.read()  #读取每一张 flag图片是否读取成功  frame图片的内容
    fileName = 'image'+str(i)+'.jpg'   #每一张都给它取个名字
    print(fileName)
    if flag == True :
        cv2.imwrite(fileName,frame,[cv2.IMWRITE_JPEG_QUALITY,100])  #写入    照片质量为100
print('end!')
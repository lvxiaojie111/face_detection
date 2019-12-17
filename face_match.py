import cv2
import easygui as g

#人脸匹配阈值
threshold=0.39

#读取人脸匹配样本1
mask0_img = cv2.imread('./111.png',1)
img0 = cv2.cvtColor(mask0_img, cv2.COLOR_RGB2BGR)
#读取人脸匹配样本2
mask1_img = cv2.imread('./112.png',1)
img1 = cv2.cvtColor(mask1_img, cv2.COLOR_RGB2BGR)


# 灰度直方图算法
# 计算单通道的直方图的相似值
def calculate(image1, image2):

    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
    # 计算直方图的重合度
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + \
                (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    degree = degree / len(hist1)
    return degree

 # RGB每个通道的直方图相似度
# 将图像resize后，分离为RGB三个通道，再计算每个通道的相似值
def classify_hist_with_split(image1, image2, size=(256, 256)):

    image1 = cv2.resize(image1, size)
    image2 = cv2.resize(image2, size)
    sub_image1 = cv2.split(image1)
    sub_image2 = cv2.split(image2)
    sub_data = 0
    for im1, im2 in zip(sub_image1, sub_image2):
        sub_data += calculate(im1, im2)
    sub_data = sub_data / 3
    return sub_data
#进行人脸识别
def face_rec():
    #调用摄像头读取数据
    vid = cv2.VideoCapture(0)
    return_value, frame = vid.read()
    dis0 = classify_hist_with_split(img0, frame)
    dis1 = classify_hist_with_split(img1, frame)
    while True:
        return_value, frame = vid.read()
        if return_value:
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#将图片转化成灰度
        else:
            raise ValueError("No image!")
        #读取的视频帧送入人脸特征分类器进行分类
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
        face_cascade.load('./haarcascade_frontalface_alt2.xml')#haar人脸特征分类器'''
        #获取检测到的人脸框
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            frame1 = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            # frame1=frame
            frame1=frame1[y:y + h,x:x +w]
            #与两个预存样本进行人脸相似度匹配
            dis0 = classify_hist_with_split(img0, frame1)
            dis1 = classify_hist_with_split(img1, frame1)
            #保存人脸图像
            cv2.imwrite("face.jpg", frame1)
           #距离大于阈值，表明人脸匹配成功，进行开门操作
            if dis0 > threshold or dis1 > threshold:
                cv2.putText(frame, "The door is open", (120, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                choice = g.buttonbox(msg="                   人脸匹配成功，锁已经成功打开，请关门", image="./face.jpg",title="人脸匹配结果", choices=("取消",'关门'))
                if choice == "取消":
                    pass
                if choice == "关门":
                    pass
                    # cv2.putText(frame, "The door is closed", (120, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                choice = g.buttonbox(msg="                     人脸匹配失败，您无权开锁，谢谢", image="./face.jpg", title="人脸匹配结果",
                                     choices=("取消", '退出'))
                if choice == "取消":
                    pass
                if choice == "退出":
                    pass
        #实时显示摄像内容
        cv2.putText(frame, "Real time monitoring screen", (220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow('Real Monitor ', frame)
        #按“q”键 退出程序
        if cv2.waitKey(30) & 0xFF == ord('q'): break
if __name__ == '__main__':
     face_rec()

# 202331223060 张曼

import cv2  # 导入OpenCV库

def resize_image(img, max_width=1200):
    """按比例缩小图片，确保宽度不超过max_width（默认1200像素）"""
    height, width = img.shape[:2]  # 获取图片原始宽高
    if width > max_width:  # 若宽度超过最大值，按比例缩小
        scale = max_width / width  # 计算缩放比例
        new_height = int(height * scale)  # 计算缩小后的高度
        img = cv2.resize(img, (max_width, new_height))  # 执行缩小
    return img

# 加载人脸检测模型（Haar级联分类器）
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 读取图片（替换为你的图片路径，若图片在项目根目录，直接写文件名）
img = cv2.imread('face.jpg')  # 读取彩色图片
if img is None:
    print("无法读取图片，请检查路径是否正确！")
    exit()  # 若图片读取失败，退出程序

# 将图片转为灰度图（人脸检测对灰度图更高效）
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 检测人脸
# 参数说明：
# gray：灰度图
# scaleFactor：缩放因子（1.1表示每次缩小10%）
# minNeighbors：最小邻居数（控制检测精度，5为推荐值）
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# 在检测到的人脸周围画矩形框
for (x, y, w, h) in faces:
    # 参数：图片、左上角坐标、右下角坐标、颜色（BGR格式，这里是蓝色）、线宽
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# 6. 显示结果图片
cv2.namedWindow('Face Detection', cv2.WINDOW_NORMAL)  # 创建可缩放窗口
cv2.imshow('Face Detection', img)  # 在可缩放窗口中显示图片
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(0)  # 等待用户按任意键关闭窗口
cv2.destroyAllWindows()  # 关闭所有窗口
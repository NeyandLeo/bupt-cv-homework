import cv2
import numpy as np

# 加载图片
image = cv2.imread('coin.jpg')

# 将图片转换为灰度图，因为边缘检测通常在灰度图上进行
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 应用高斯模糊，减少图像噪声
gray_blurred = cv2.GaussianBlur(gray, (3, 3), 0)

# 进行Canny边缘检测
edges = cv2.Canny(gray_blurred, threshold1=100, threshold2=200)

# 使用霍夫圆变换捕捉圆形
circles = cv2.HoughCircles(gray_blurred,
                           cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                           param1=50, param2=100, minRadius=100, maxRadius=200)

# 如果检测到圆形，绘制圆形
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # 绘制圆心
        cv2.circle(image, (i[0], i[1]), 1, (0, 100, 100), 3)
        # 绘制圆轮廓
        cv2.circle(image, (i[0], i[1]), i[2], (255, 0, 255), 3)

# 保存边缘检测的结果
cv2.imwrite('results/cv2_edges_all.jpg', edges)

# 保存圆形检测的结果
cv2.imwrite('results/cv2_Hough.jpg', image)

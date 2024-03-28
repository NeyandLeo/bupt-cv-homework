# 钱币定位系统
## 简介
本项目旨在解决钱币定位问题。通过使用canny边缘检测算法和HoughCircle算法，项目定位钱币的能力能够达到接近opencv库中API的效果。
项目的目录结构如下所示，主要包括源代码、文档和测试代码等部分。
## 项目结构
以下是本项目的目录结构说明，帮助您快速了解各部分文件的存放位置及作用：
```plaintext
.
├── ohter_results/        # 存放各种参数对应的实验结果图，图片命名格式为：sigma值-低阈值-高阈值
├── results/              # 存放实验的最优结果以及调用opencv库所得的结果图
├── canny.py              # canny边缘检测算法的python源码
├── Hough_circle.py       # HoughCircle算法的python源码
├── opencv.py             # 调用opencv库中处理图片的python源码
├── coin.jpg              # 钱币图片
└── README.md             # 项目说明文件
import numpy as np
from skimage import io
import os
import matplotlib.pyplot as plt
from canny import convolve2d

def optimized_hough_circle_transform(edge_image, gradient_direction, radius_range, angle_range):
    """
        执行优化的霍夫圆变换来检测图像中的圆。

        输入:
        - edge_image: 边缘检测后的二值图像。
        - gradient_direction: 每个像素点的梯度方向。
        - radius_range: 圆半径的范围，为一个序列。
        - angle_range: 在梯度方向上考虑的角度范围，单位为度。

        输出:
        - accumulator: 三维累加器数组，记录圆心位置和半径大小的投票数。
    """
    accumulator = np.zeros((edge_image.shape[0], edge_image.shape[1], len(radius_range)))

    edge_points = np.argwhere(edge_image > 0)  # 获取边缘点的坐标
    angle_offset = np.deg2rad(angle_range)  # 将角度范围转换为弧度

    # 遍历每一个边缘点
    for x, y in edge_points:
        edge_direction = gradient_direction[x, y]
        # 在梯度方向的正负5度范围内枚举
        for delta in np.linspace(-angle_offset, angle_offset, num=angle_range * 2 + 1):
            direction = edge_direction + delta
            for radius_index, radius in enumerate(radius_range):
                # 根据梯度方向和半径计算圆心坐标
                a = int(x - radius * np.sin(direction))
                b = int(y + radius * np.cos(direction))
                # 检查计算出的圆心是否在图像范围内
                if 0 <= a < edge_image.shape[0] and 0 <= b < edge_image.shape[1]:
                    accumulator[a, b, radius_index] += 1

    return accumulator


# 计算图像的梯度方向
def compute_gradient_direction(image):
    """
        计算图像的梯度方向。

        参数:
        - image: 原始图像。

        返回:
        - gradient_direction: 图像的梯度方向，大小与输入图像相同。
    """
    # Sobel核在x方向
    sobel_kernel_x = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]])

    # Sobel核在y方向
    sobel_kernel_y = np.array([[-1, -2, -1],
                               [0, 0, 0],
                               [1, 2, 1]])

    dx = convolve2d(image,sobel_kernel_x)
    dy = convolve2d(image,sobel_kernel_y)
    gradient_direction = np.arctan2(dy, dx)
    return gradient_direction

def detect_circles_by_threshold(accumulator, radius_range, threshold):
    """
        根据阈值从累加器中检测高于阈值的圆。

        参数:
        - accumulator: 霍夫变换的累加器。
        - radius_range: 圆半径的范围，为一个序列。
        - threshold: 投票数的阈值，只有大于此阈值的圆会被检测出来。

        返回:
        - detected_circles: 检测到的圆的列表，每个元素为一个元组(y, x, r)，其中y和x为圆心坐标，r为半径索引。
    """
    detected_circles = []
    while True:
        # 找到累加器中的最大值及其索引
        i, j, k = np.unravel_index(accumulator.argmax(), accumulator.shape)
        max_votes = accumulator[i, j, k]

        # 如果最大值小于阈值，则停止搜索
        if max_votes <= threshold:
            break

        # 添加找到的圆
        detected_circles.append((i, j, k))

        # 非极大值抑制：创建一个区域将其清零
        mask_radius = radius_range[k]  # 也可以根据需要调整这个范围
        i_min = max(0, i - mask_radius)
        i_max = min(accumulator.shape[0], i + mask_radius + 1)
        j_min = max(0, j - mask_radius)
        j_max = min(accumulator.shape[1], j + mask_radius + 1)

        # 清零该区域
        accumulator[i_min:i_max, j_min:j_max, :] = 0

    return detected_circles


def draw_circles(image, circles, radius_range, save_path=None):
    """
        在图像上绘制检测到的圆并显示。

        参数:
        - image: 原始图像或边缘图像。
        - circles: 检测到的圆的列表，格式同detect_circles函数的输出。
        - radius_range: 圆半径的范围，为一个序列。
        - save_path: 可选，保存绘制圆的图像的路径。

        返回:
        - 无直接输出，函数执行将显示和/或保存图像。
    """
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    for y, x, r_idx in circles:
        radius = radius_range[r_idx]
        circle = plt.Circle((x, y), radius, color='red', fill=False)
        center = plt.scatter(x, y, color='red')
        ax.add_artist(circle)
        ax.add_artist(center)
        print("Detected circle center: (", y, ",", x, ")")
        print("Detected circle radius: ", radius)
    ax.set_axis_off()  # 可以选择关闭坐标轴显示
    plt.tight_layout()  # 减少图片周围不必要的边缘空间

    if save_path is not None:
        path = os.path.join(save_path,"Hough.jpg")
        plt.savefig(path, bbox_inches='tight', pad_inches=0.0)
        print(f"Image saved to {path}")
    plt.show()  # 显示图片

def Huff_process_and_display(image_path, radius_range, threshold,save_path):
    """
        处理图像，执行霍夫圆变换，检测圆，并显示及保存结果。

        参数:
        - image_path: 要处理的图像的路径。
        - radius_range: 圆半径的范围，为一个序列。
        - threshold: 检测圆时使用的投票阈值。
        - save_path: 保存结果图像的路径。

        返回:
        - 无直接输出，此函数将通过调用其他函数来处理图像，检测圆，绘制并保存结果图像。
    """
    edge_image = io.imread(image_path, as_gray=True)
    coin_image = io.imread('coin.jpg',as_gray=True)
    gradient_direction = compute_gradient_direction(edge_image)

    # 执行霍夫圆变换
    accumulator = optimized_hough_circle_transform(edge_image, gradient_direction, radius_range, 5)


    # 检测圆
    circles = detect_circles_by_threshold(accumulator,radius_range,threshold)

    # 可视化
    draw_circles(coin_image, circles, radius_range,save_path)

if __name__ == "__main__":
    save_path="results"
    # 调用Huff_process_and_display函数，这里需要替换成实际的参数
    Huff_process_and_display('results/edges.jpg', range(150, 180), 15,save_path)

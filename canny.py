import matplotlib.pyplot as plt
from skimage import color, io
import numpy as np
import os
def gaussian_derivative_kernels(sigma, size):
    """生成高斯一阶导数卷积核。
        参数:
        - sigma: 高斯核的标准差
        - size: 卷积核的尺寸
        返回:
        - gaussian_kernel_dx, gaussian_kernel_dy：高斯一阶导数卷积核在x和y方向上的两个数组
    """
    size = int(size) if size % 2 == 1 else int(size) + 1  # 确保核的大小是奇数
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    gaussian_kernel = np.exp(-0.5 * (xx ** 2 + yy ** 2) / sigma ** 2) #先生成一个高斯核
    gaussian_kernel /= gaussian_kernel.sum()

    # 直接利用高斯函数求导的结果生成高斯一阶导kernel
    gaussian_kernel_dx = -xx / sigma ** 2 * gaussian_kernel
    gaussian_kernel_dy = -yy / sigma ** 2 * gaussian_kernel

    # 由于卷积核必须整体和为0，需要对核进行标准化
    gaussian_kernel_dx -= gaussian_kernel_dx.mean()
    gaussian_kernel_dy -= gaussian_kernel_dy.mean()
    #返回x，y方向的高斯一阶导卷积核
    return gaussian_kernel_dx, gaussian_kernel_dy


def convolve2d(image, kernel):
    """
    对图像进行二维卷积操作，保持输出图像的尺寸与输入图像相同。
    参数:
    - image: 输入的图像，二维数组
    - kernel: 卷积核，二维数组
    返回:
    - output：卷积后的图像，大小与输入图像相同
    """
    kernel_height, kernel_width = kernel.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # 对图像进行边缘填充
    image_padded = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='edge')

    output_height, output_width = image.shape
    output = np.zeros((output_height, output_width))

    # 更新卷积操作，考虑填充
    for y in range(output_height):
        for x in range(output_width):
            output[y, x] = np.sum(image_padded[y:y + kernel_height, x:x + kernel_width] * kernel)

    return output


def apply_gaussian_derivative(image, sigma):
    """应用高斯一阶导卷积核于图像，进行边缘增强。
    参数:
    - image: 输入图像
    - sigma: 高斯核的标准差
    返回:
    - G, Theta：图像的梯度幅度和梯度方向
    """
    # 生成高斯核的导数
    kernel_dx, kernel_dy = gaussian_derivative_kernels(sigma,6 * sigma + 1)

    # 在两个方向上分别卷积
    Ix = convolve2d(image, kernel_dx)
    Iy = convolve2d(image, kernel_dy)

    G = np.sqrt(Ix ** 2 + Iy ** 2)
    Theta = np.arctan2(Iy, Ix)

    return (G, Theta)


def non_max_suppression(gradient_magnitude, gradient_direction):
    """
    对图像应用非极大值抑制，以细化边缘。
    参数:
    - gradient_magnitude: 梯度幅度
    - gradient_direction: 梯度方向
    返回:
    - Z：经过非极大值抑制处理后的图像
    """
    M, N = gradient_magnitude.shape
    Z = np.zeros((M, N), dtype=np.float32)
    angle = gradient_direction * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255

                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = gradient_magnitude[i, j + 1]
                    r = gradient_magnitude[i, j - 1]
                elif (22.5 <= angle[i, j] < 67.5):
                    q = gradient_magnitude[i + 1, j - 1]
                    r = gradient_magnitude[i - 1, j + 1]
                elif (67.5 <= angle[i, j] < 112.5):
                    q = gradient_magnitude[i + 1, j]
                    r = gradient_magnitude[i - 1, j]
                elif (112.5 <= angle[i, j] < 157.5):
                    q = gradient_magnitude[i - 1, j - 1]
                    r = gradient_magnitude[i + 1, j + 1]

                if (gradient_magnitude[i, j] >= q) and (gradient_magnitude[i, j] >= r):
                    Z[i, j] = gradient_magnitude[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass

    return Z


def threshold(img, low_threshold, high_threshold):
    """
    对图像应用双阈值法进行边缘检测。
    参数:
    - img: 输入图像
    - low_threshold: 低阈值
    - high_threshold: 高阈值
    返回:
    - res, weak, strong：处理后的图像，弱边缘和强边缘的阈值
    """
    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= high_threshold)
    zeros_i, zeros_j = np.where(img < low_threshold)

    weak_i, weak_j = np.where((img <= high_threshold) & (img >= low_threshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return (res, weak, strong)


def hysteresis(img, weak, strong=255):
    """
    通过滞后过程连接弱边缘和强边缘。
    参数:
    - img: 输入图像
    - weak: 弱边缘标记值
    - strong: 强边缘标记值
    返回:
    - img：经过强弱边缘连接处理后的图像
    """
    M, N = img.shape
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if (img[i, j] == weak):
                if ((img[i + 1, j - 1] == strong) or (img[i + 1, j] == strong) or (img[i + 1, j + 1] == strong)
                        or (img[i, j - 1] == strong) or (img[i, j + 1] == strong)
                        or (img[i - 1, j - 1] == strong) or (img[i - 1, j] == strong) or (img[i - 1, j + 1] == strong)):
                    img[i, j] = strong
                else:
                    img[i, j] = 0
    return img


def canny_edge_detection(image, sigma, low_threshold_ratio=0.05, high_threshold_ratio=0.10):
    """
    实现Canny边缘检测算法。
    参数:
    - image: 输入图像
    - sigma: 高斯核的标准差
    - low_threshold_ratio: 低阈值比率
    - high_threshold_ratio: 高阈值比率
    返回:
    - img_final：经过Canny边缘检测处理后的图像
    """
    gradient_magnitude, gradient_direction = apply_gaussian_derivative(image,sigma=sigma)#直接用高斯一阶导卷积核卷积，和下面两行代码等价
    non_max_img = non_max_suppression(gradient_magnitude, gradient_direction)#非最大化抑制
    threshold_img, weak, strong = threshold(non_max_img, low_threshold_ratio * np.max(non_max_img),
                                            high_threshold_ratio * np.max(non_max_img))#双门限step1
    img_final = hysteresis(threshold_img, weak, strong)#双门限step2，根据上一步的结果连接strong与weak的交界
    return img_final #返回处理后的图像


def load_image_as_gray(image_path):
    """
    加载图像并将其转换为灰度图。
    参数:
    - image_path: 图像的路径
    返回:
    - image:转换为灰度的图像
    """
    image = io.imread(image_path)
    return color.rgb2gray(image)

def process_image_with_canny(image_path, sigma=6, low_threshold_ratio=0.1, high_threshold_ratio=0.35):
    """
    使用Canny边缘检测处理图像。
    参数:
    - image_path: 图像的路径
    - sigma: 高斯核的标准差
    - low_threshold_ratio: 低阈值比率
    - high_threshold_ratio: 高阈值比率
    返回:
    - edges_detected:经过Canny边缘检测处理后的图像
    """
    gray_image = load_image_as_gray(image_path)
    edges_detected = canny_edge_detection(gray_image, sigma=sigma, low_threshold_ratio=low_threshold_ratio, high_threshold_ratio=high_threshold_ratio)
    return edges_detected

def canny_process_and_display(path,save_path=None):
    """
        处理并显示图像的Canny边缘检测结果，可选地保存结果图像。
        参数:
        - path: 输入图像的路径
        - save_path: 结果图像的保存路径（可选）
        返回：
        - 无返回结果，将可视化边缘检测后的图像
    """
    image_path = path
    edges = process_image_with_canny(image_path, sigma=6, low_threshold_ratio=0.1, high_threshold_ratio=0.35)
    edges_all = process_image_with_canny(image_path,sigma=1,low_threshold_ratio=0.1,high_threshold_ratio=0.23)
    if save_path is not None:
        path1 = os.path.join(save_path,'edges.jpg')
        path2 = os.path.join(save_path,'edges_all.jpg')
        plt.imsave(path1, edges, cmap='gray')
        plt.imsave(path2, edges_all, cmap='gray')
    # 显示原图和处理后的图像
    plt.figure(figsize=(12, 6))
    plt.subplot(131)
    plt.imshow(io.imread(image_path))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(edges_all, cmap='gray')
    plt.title('ALL Edges Detected')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(edges, cmap='gray')
    plt.title('Edges Detected')
    plt.axis('off')

    plt.show()


def get_other_results(path, save_path):
    """
    遍历sigma值和所有高低阈值比例组合，应用Canny边缘检测，并保存所有结果图像。
    参数:
    - path: 输入图像的路径
    - save_path: 结果图像的保存路径
    返回：
    - 无直接返回，将保存各个参数组合下的结果图于save_path下
    """
    sigma_values = np.arange(1, 7)  # Sigma从1到6
    threshold_ratios = np.arange(0.05, 0.55, 0.1)  # 阈值比例从0.05遍历到0.45，以0.1为增量

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for sigma in sigma_values:
        for low_threshold_ratio in threshold_ratios:
            for high_threshold_ratio in threshold_ratios:
                if low_threshold_ratio < high_threshold_ratio:
                    # 处理图像，使用Canny边缘检测
                    edges_detected = process_image_with_canny(path, sigma=sigma,
                                                              low_threshold_ratio=low_threshold_ratio,
                                                              high_threshold_ratio=high_threshold_ratio)
                    # 构建文件名并保存结果图像
                    filename = f"{sigma}-{low_threshold_ratio:.2f}-{high_threshold_ratio:.2f}.jpg"
                    filepath = os.path.join(save_path, filename)
                    plt.imsave(filepath, edges_detected, cmap='gray')
                    print(f"Saved: {filename}")


if __name__ ==  "__main__":
    image_path = 'coin.jpg'
    save_path = 'results'
    canny_process_and_display(image_path,save_path)
    #other_result_path = 'other_results' #做参数调整的实验才需要下面这两行代码，正常运行请注释掉
    #get_other_results('coin.jpg',other_result_path)


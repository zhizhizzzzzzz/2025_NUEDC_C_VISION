import cv2
import numpy as np

def resize_to_height(image, target_height):
    """
    按比例调整图像高度
    :param image: 输入图像
    :param target_height: 目标高度
    :return: 调整后的图像
    """
    h, w = image.shape[:2]
    ratio = target_height / float(h)
    dim = (int(w * ratio), target_height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

def rotate_image(image, angle):
    """
    旋转图像（保持内容完整）
    :param image: 输入图像
    :param angle: 旋转角度（度）
    :return: 旋转后的图像
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 计算新边界尺寸
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # 调整旋转矩阵
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    return cv2.warpAffine(image, M, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
import cv2
import yaml
import numpy as np

# 二值化 阈值50，找轮廓，找到粗校准角点4个
import cv2
import numpy as np
from util_leaveup import *
def find_corner_points(image):
    # 图像处理
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 获取层级关系
    hierarchy = hierarchy[0]
    img_area = image.shape[0] * image.shape[1]
    img_center = np.array([image.shape[1] // 2, image.shape[0] // 2])
    
    # 找出所有叶子节点索引
    leaf_indices = [i for i, h in enumerate(hierarchy) if h[2] == -1]
    
    # 筛选符合条件的叶子轮廓
    def valid_contour(c):
        area = cv2.contourArea(c)
        if area < 0.005 * img_area or area > 0.666 * img_area: 
            return False
        M = cv2.moments(c)
        if M["m00"] == 0: 
            return False
        cx, cy = M["m10"]/M["m00"], M["m01"]/M["m00"]
        return np.linalg.norm([cx-img_center[0], cy-img_center[1]]) < image.shape[0]/2
    
    filtered_leaf_indices = [i for i in leaf_indices if valid_contour(contours[i])]
    # 子集按面积從小到大 排序
    filtered_leaf_indices.sort(key=lambda i: cv2.contourArea(contours[i]))
    # 获取父轮廓索引
    parent_indices = {hierarchy[i][3] for i in filtered_leaf_indices if hierarchy[i][3] != -1}
    
    # 多边形近似函数
    def approx_points(c):
        epsilon = 0.005 * cv2.arcLength(c, True)
        return [[p[0], p[1]] for p in cv2.approxPolyDP(c, epsilon, True).squeeze()]
    
    # 生成轮廓点集
    parent_points = [approx_points(contours[i]) for i in parent_indices]
    child_points = [approx_points(contours[i]) for i in filtered_leaf_indices]

    if parent_points and child_points:
        return [parent_points, child_points]
    return None

def classify_shape(points, frame):
    shape_type = None
    circle_center = None
    if len(points) == 4:
        shape_type = "Square"

    elif len(points) == 3:
        shape_type = "Triangle"
    
    elif len(points) > 5:
        # 先判断是否为圆形
        # 计算质点
        M = cv2.moments(np.array(points).reshape(-1, 1, 2).astype(np.int32))
        # 绘制质点
        circle_center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
        cv2.circle(frame, circle_center, 1, (0, 0, 255), -1)
        
        # 如果所有点到质点距离的方差小于一定值，则认为是圆形
        if np.var([np.linalg.norm(np.array(p) - np.array(circle_center)) for p in points]) < 1000:
            shape_type = "Circle"
        else:            
            shape_type, develop_pts = classify_leaveup_shape(points, frame)
            # 中心是develop_pts的质点
            if develop_pts is not None:
                M = cv2.moments(np.array(develop_pts).reshape(-1, 1, 2).astype(np.int32))
                if M['m00'] != 0:
                    circle_center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
                    if develop_pts is not None:
                        return shape_type, circle_center, develop_pts
            
    # 不论传入多少点,在这些点的质心标记形状
    if shape_type is not None:
        M = cv2.moments(np.array(points).reshape(-1, 1, 2).astype(np.int32))
        circle_center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
    
    return shape_type, circle_center, points

def shi_tomasi_refinement(gray, rough_point, window_size=15):
    """
    在现有角点位置周围小窗口内，使用Shi-Tomasi方法精确定位
    :param gray: 灰度图像
    :param rough_point: 粗略角点位置 [x, y]
    :param window_size: 搜索窗口大小
    :return: 精确定位后的单个角点位置
    """
    x, y = map(int, rough_point)
    half = window_size // 2
    
    # 确保坐标在图像范围内
    x = max(half, min(x, gray.shape[1] - half - 1))
    y = max(half, min(y, gray.shape[0] - half - 1))
    
    # 创建小ROI
    roi = gray[y-half:y+half+1, x-half:x+half+1].copy()
    
    # 确保ROI足够大
    if roi.shape[0] < 3 or roi.shape[1] < 3:
        return rough_point
    
    # 在ROI上执行Shi-Tomasi检测
    corners = cv2.goodFeaturesToTrack(
        roi,
        maxCorners=1,  # 只检测一个最佳角点
        qualityLevel=0.1,  # 较高灵敏度
        minDistance=window_size,  # 确保只返回一个点
        blockSize=3,
        useHarrisDetector=False
    )
    
    if corners is not None and len(corners) > 0:
        cx, cy = corners[0].ravel()
        return [x - half + cx, y - half + cy]
    
    # 如果未检测到，返回原始点
    return rough_point

def subpixel_refinement(frame, corner_points_list):
    """使用Shi-Tomasi方法精确移动现有角点"""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    refined_list = []
    
    # corner_points_list 结构: [parent_points, child_points]
    for contour_group in corner_points_list:
        refined_group = []
        for contour in contour_group:
            refined_contour = []
            for point in contour:
                # 精炼每个点
                refined_point = shi_tomasi_refinement(gray_frame, point)
                refined_contour.append(refined_point)
            refined_group.append(refined_contour)
        refined_list.append(refined_group)
    
    # 画出角点
    for contour_group in refined_list:
        for contour in contour_group:
            for point in contour:
                int_pt = (int(round(point[0])), int(round(point[1])))
                cv2.circle(frame, int_pt, 1, (0, 255, 0), -1)  # 绿色点
    
    return refined_list

def sort_points_clockwise(points):
    # 确保输入是 NumPy 数组
    points = np.array(points, dtype="float32")
    
    # 按x坐标排序，分成左右两列
    x_sorted_indices = np.argsort(points[:, 0])
    x_sorted = points[x_sorted_indices]
    left_col = x_sorted[:2, :]  # 左侧的两个点
    right_col = x_sorted[2:, :]  # 右侧的两个点

    # 左侧点按y坐标排序（从小到大：上→下）
    left_col = left_col[np.argsort(left_col[:, 1])]
    tl, bl = left_col  # 左上 (top-left), 左下 (bottom-left)

    # 右侧点按y坐标排序（从小到大：上→下）
    right_col = right_col[np.argsort(right_col[:, 1])]
    tr, br = right_col  # 右上 (top-right), 右下 (bottom-right)

    # 计算左右边的长度
    left_length = np.linalg.norm(tl - bl)
    right_length = np.linalg.norm(tr - br)

        # 确保左边是长边（竖摆矩形）
    if left_length < 0.7 * right_length:
        tl, tr, bl, br = bl, br, tr, tl  # 交换左右边，使左边为长边

    # 返回顺时针顺序：左下(起点)→右下→右上→左上
    return np.array([bl, br, tr, tl], dtype="float32")

def load_camera_params(config_path):
    """从YAML文件加载相机参数"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    camera_matrix = np.array(config['camera_matrix']['data']).reshape(3, 3)
    dist_coeffs = np.array(config['distortion_coefficients']['data'])

    return camera_matrix, dist_coeffs

def sort_points_by_orientation(points):
    """
    根据矩形方向（竖摆）对四个点进行排序
    顺序：左上 -> 右上 -> 右下 -> 左下（顺时针）
    """
    # 将点转换为NumPy数组
    points = np.array(points, dtype="float32")
    
    # 1. 计算中心点
    centroid = np.mean(points, axis=0)
    
    # 2. 将点分为左右两组
    left_points = []
    right_points = []
    
    for point in points:
        if point[0] < centroid[0]:
            left_points.append(point)
        else:
            right_points.append(point)
    
    # 3. 在每组中按y坐标排序（从上到下）
    left_points = sorted(left_points, key=lambda p: p[1])
    right_points = sorted(right_points, key=lambda p: p[1])
    
    # 4. 确定长边（竖边）
    if len(left_points) == 2 and len(right_points) == 2:
        # 计算左右边的长度
        left_length = np.linalg.norm(left_points[0] - left_points[1])
        right_length = np.linalg.norm(right_points[0] - right_points[1])
        
        # 计算上下边的长度
        top_length = np.linalg.norm(left_points[0] - right_points[0])
        bottom_length = np.linalg.norm(left_points[1] - right_points[1])
        
        # 5. 验证竖边是长边（应该明显长于横边）
    if min(left_length, right_length) > 0.7 * max(top_length, bottom_length):
    # 左上 -> 右上 -> 右下 -> 左下
            return np.array([left_points[0], right_points[0], right_points[1], left_points[1]], 
                           dtype="float32")
    
    # 如果无法确定方向，使用角度排序作为后备方案
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    sorted_indices = np.argsort(angles)
    return points[sorted_indices]


def calculate_actual_distance(image_points):
    camera_matrix, dist_coeffs = load_camera_params('config.yaml')
    
    # A4纸尺寸 (毫米)
    a4_height = 257  # 长边（竖摆时的高度）
    a4_width = 170   # 短边
    
    # 世界坐标点（Y轴向上）
    half_h = a4_height / 2
    half_w = a4_width / 2
    world_points = np.array([
        [-half_w, -half_h, 0],   # 左上
        [half_w, -half_h, 0],    # 右上
        [half_w, half_h, 0],     # 右下
        [-half_w, half_h, 0]     # 左下
    ], dtype=np.float32)
    
    # 使用方向感知排序
    sorted_points = sort_points_by_orientation(image_points)
    
    # 使用SOLVEPNP_IPPE方法
    success, rvec, tvec = cv2.solvePnP(
        world_points,
        sorted_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE
    )
    
    if not success:
        return 0, None
    
    # 计算重投影误差
    reprojected_points, _ = cv2.projectPoints(
        world_points, rvec, tvec, camera_matrix, dist_coeffs)
    reprojected_points = reprojected_points.reshape(-1, 2)
    
    # 计算实际距离（Z轴深度）
    distance = np.linalg.norm(tvec)
    
    return distance, rvec, tvec

def calculate_distance(points, frame):
    # 确保有4个点
    if len(points) != 4:
        return 0, None
        
    # 按正确顺序排序
    sorted_points = sort_points_clockwise(points)
    
    # 计算距离
    distance, rvec, position = calculate_actual_distance(sorted_points)
    
    # 显示距离信息
    cv2.putText(frame, f'Distance: {distance:.3f} mm', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return distance, rvec, position

def calculate_shape_size(frame, shape_type, shape_points, circle_center, rvec, tvec, reference_size):
    """
    :param reference_size: 参考物体的实际物理尺寸（单位：毫米）
    """
    
    camera_matrix, dist_coeffs = load_camera_params('config.yaml')
    # 步骤1：计算参考平面的比例因子
    # 使用参考尺寸创建参考点
    ref_obj_points = np.array([
        [0, 0, 0],
        [reference_size, 0, 0],
        [reference_size, reference_size, 0],
        [0, reference_size, 0]
    ], dtype=np.float32)
    # 投影参考物体得到理论图像点
    ref_img_points, _ = cv2.projectPoints(ref_obj_points, rvec, tvec, camera_matrix, dist_coeffs)
    ref_img_points = ref_img_points.reshape(-1, 2)
    # 步骤2：计算实际检测点与理论点的比例因子
    # 计算参考物体在图像中的像素尺寸
    ref_pixel_width = np.linalg.norm(ref_img_points[0] - ref_img_points[1])
    scale_factor = reference_size / ref_pixel_width
    # 步骤3：测量未知物体（与参考物体共面）
    if shape_type == 'Square' and len(shape_points) == 4:
        # 计算检测到的四边形边长（像素单位）
        pixel_side = np.linalg.norm(np.array(shape_points[0]) - np.array(shape_points[1]))
        return pixel_side * scale_factor  # 返回实际边长
    
    elif shape_type == 'Circle' and len(shape_points) > 0:
        print("ok")
        # 圆心到边缘点的距离（像素）
        pixel_radius = np.linalg.norm(np.array(shape_points[0]) - np.array(circle_center))
        # 返回直径
        return 2 * pixel_radius * scale_factor
    
    elif shape_type == 'Triangle' and len(shape_points) == 3:
        # 计算边的长度（像素）
        pixel_side = np.linalg.norm(np.array(shape_points[0]) - np.array(shape_points[1]))
        return pixel_side * scale_factor  # 返回实际边长
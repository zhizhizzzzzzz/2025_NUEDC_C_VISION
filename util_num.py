import cv2
import numpy as np
import math
import logger
import mathutils
import imgutils

def main_num(frame):
    is_debug = True
    is_debug_contour = False and is_debug
    is_debug_corner = False and is_debug
    
    tbox_size = (297-40, 210-40)  # Size of the target box (mm)
    tbox_ratio = (tbox_size[0] / tbox_size[1])  # Aspect ratio of the target box

    result = frame.copy()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    _, binary_frame = cv2.threshold(
        gray_frame, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU
    )
    contours, hierarchy = cv2.findContours(
        binary_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    
    target_id = -1
    target_polygon = None
    refined_target = None
    inner_ids = []
    refined_inner_polygons = []
    
    for i, contour in enumerate(contours):
        
        hierarchy_info = hierarchy[0][i]
        
        if cv2.contourArea(contour) < 500: continue
        polygon = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        
        if is_debug_contour: cv2.polylines(result, [polygon], True, (0, 0, 128), 1)
        if (len(polygon) != 4) or (hierarchy_info[3] != -1) or (not cv2.isContourConvex(polygon)): continue
        
        if is_debug_contour: cv2.polylines(result, [polygon], True, (128, 0, 0), 2)
        opposide_angles = [mathutils.arrangle((polygon[1+i]-polygon[0+i]).flatten(), (polygon[2+i]-polygon[(3+i)%4]).flatten()) for i in range(2)]
        if not all(angle < 10 for angle in opposide_angles): continue
        
        if is_debug_contour: cv2.polylines(result, [polygon], True, (0, 128, 0), 2)
        corner_angles = [mathutils.arrangle((polygon[i]-polygon[(i+1)%4]).flatten(), (polygon[i]-polygon[(i+3)%4]).flatten()) for i in range(4)]
        if not all(abs(90-angle) < 10 for angle in corner_angles): continue
        
        if is_debug_contour: cv2.polylines(result, [polygon], True, (128, 0, 128), 2)
        edges = [polygon[i]-polygon[(i+1)%4] for i in range(4)]
        lengths = [np.linalg.norm(edge) for edge in edges]
        if not (tbox_ratio * 0.9 < max(lengths[2:]) / min(lengths[2:]) < tbox_ratio * 1.1): continue
        
        roi = cv2.boundingRect(polygon)
        roi = hsv_frame[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
        mask = np.zeros_like(roi[:, :, 0])
        cv2.drawContours(mask, [polygon], -1, 255, -1)
        mean_saturation = cv2.mean(roi[:, :, 1], mask=mask)[0]
        if is_debug_contour: cv2.imshow("SaturationChannel", roi[1:, :, 1])
        if mean_saturation > 30: continue
        
        if target_id == -1 or cv2.contourArea(contours[i]) > cv2.contourArea(contours[target_id]):
            target_id = i
            target_polygon = polygon

        imgutils.draw_out_box(result, polygon, i, opposide_angles, corner_angles)

    if target_id == -1:
        logger.warn("No valid target found.")
    else:
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) < 500: continue
            if hierarchy[0][i][3] == target_id:
                inner_ids.append(i)

        refined_target = cv2.cornerSubPix(
            gray_frame, np.float32(target_polygon), (5, 5), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        if is_debug_corner:
            for p in refined_target:
                imgutils.draw_AA_circle(result, p[0], (0, 0, 255), 1, -1)
            
    for inner_id in inner_ids:
        inner_polygon = cv2.approxPolyDP(contours[inner_id], 0.005 * cv2.arcLength(contours[inner_id], True), True)
        is_convex = cv2.isContourConvex(inner_polygon)
        if not is_convex: 
            inner_polygon = cv2.approxPolyDP(contours[inner_id], 0.03*(1/len(inner_polygon)) * cv2.arcLength(contours[inner_id], True), True)
        refined_inner_polygon = cv2.cornerSubPix(
            gray_frame, np.float32(inner_polygon), (5, 5), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        refined_inner_polygons.append(refined_inner_polygon)

        # cv2.polylines(result, [inner_polygon], True, (0, 255, 255), 1)
        for p in inner_polygon:
            cv2.circle(result, tuple(p[0]), 12, (0, 255, 255), 2)
        for p in refined_inner_polygon:
            imgutils.draw_AA_circle(result, p[0], (0, 0, 255), 1, -1)
        

        imgutils.draw_inner_box_numbers(result, contours, hierarchy, inner_id)
    
    if refined_target is not None and len(refined_inner_polygons) > 0:
        
        # 1. 提取ROI区域
            x, y, w, h = cv2.boundingRect(inner_polygon)
            inner_roi = frame[y:y+h, x:x+w].copy()
            
        # 4. 创建处理区域（灰度+掩码）
            gray_roi = cv2.cvtColor(inner_roi, cv2.COLOR_BGR2GRAY)
            
            # 5. 二值化处理
            _, digit_region = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            
            # 6. 调整大小并归一化
            digit_region = cv2.resize(digit_region, (128, 128))
            
            # 7. 模板匹配
            digit_val = None
            max_val = 0
            best_i = -1
            
            # 加载模板（确保模板路径正确）
            digit_imgs = []
            for i in range(10):
                template_path = f'digits/{i}.png'
                
                template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                if template is None:
                    logger.error(f"无法加载模板: {template_path}")
                    continue
                # 调整模板尺寸并二值化
                _, template = cv2.threshold(template, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

                template = cv2.resize(template, (128, 128))

                digit_imgs.append((i, template))
            for i, template in digit_imgs:
                # 确保模板和区域尺寸相同
                if digit_region.shape != template.shape:
                    logger.warn(f"尺寸不匹配: ROI {digit_region.shape} vs 模板 {template.shape}")
                    continue
                    
                # 使用多种方法提高匹配鲁棒性
                methods = [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED]
                current_max_val = 0
                for method in methods:
                    res = cv2.matchTemplate(digit_region, template, method)
                    _, local_max_val, _, _ = cv2.minMaxLoc(res)
                    if local_max_val > current_max_val:
                        current_max_val = local_max_val
                
                if current_max_val > max_val:
                    max_val = current_max_val
                    digit_val = i
                    best_template = template
            
            # 8. 显示结果
            if digit_val is not None and max_val > 0.3:  # 降低阈值
                print(f"识别到的数字: {digit_val}, 置信度: {max_val:.2f}")
                # 创建彩色覆盖层
                overlay = inner_roi.copy()
                # 找到数字轮廓
                _, thresh = cv2.threshold(best_template, 30, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # 绘制数字轮廓
                    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
                
                # 添加数字文本
                cv2.putText(overlay, str(digit_val), (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 半透明覆盖
                cv2.addWeighted(overlay, 0.7, inner_roi, 0.3, 0, inner_roi)
                result[y:y+h, x:x+w] = inner_roi
                
                logger.info(f'Inner ID: {inner_id}, Digit: {digit_val}, Confidence: {max_val:.2f}')

    result = frame.copy()
    # cv2.imshow('Result', result)
    cv2.waitKey(1)
    
    
    return result,refined_target,refined_inner_polygons
    

if __name__ == "__main__":
    cap = cv2.VideoCapture("output.mp4")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = main_num(frame)
        cv2.imshow('Result', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
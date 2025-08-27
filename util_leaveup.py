import cv2
import numpy as np

def classify_leaveup_shape(points, frame):
    points = [tuple(map(int, pt)) if isinstance(pt, (list, tuple, np.ndarray)) else pt for pt in points]
    # 画所有点并标出当前顺序
    for i, pt in enumerate(points):
        cv2.circle(frame, pt, 1, (0, 255, 0), -1)
        cv2.putText(frame, str(i), pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    n = len(points)
    if n < 3:
        return "Insufficient Points", []  # Need at least 3 points
    
    corner_points = []
    break_points = []
    vectors = []
    
    # Classify points from index 1 to n-1
    for i in range(n):
        curr_pt = points[i]
        next_pt = points[(i + 1) % n]
        vector = (next_pt[0] - curr_pt[0], next_pt[1] - curr_pt[1])
        vectors.append(vector)
        
        if i > 0:  # Skip first point for initial cross product
            prev_pt = points[i - 1]
            v1 = np.array([curr_pt[0] - prev_pt[0], curr_pt[1] - prev_pt[1]])
            v2 = np.array(vector)
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            
            if cross < 0:  # Clockwise turn -> corner
                corner_points.append(curr_pt)
            else:  # Counter-clockwise or collinear -> break
                break_points.append(curr_pt)
    
    # Classify the first point using the last vector
    if n >= 3:
        prev_pt = points[-1]
        curr_pt = points[0]
        next_pt = points[1]
        v1 = np.array([curr_pt[0] - prev_pt[0], curr_pt[1] - prev_pt[1]])
        v2 = np.array([next_pt[0] - curr_pt[0], next_pt[1] - curr_pt[1]])
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if cross < 0:
            corner_points.append(curr_pt)
        else:
            break_points.append(curr_pt)
    
    # Mark break points (red) and corner points (blue)
    for pt in break_points:
        cv2.circle(frame, pt, 1, (0, 0, 255), -1)  # Red
    for pt in corner_points:
        cv2.circle(frame, pt, 1, (255, 0, 0), -1)  # Blue
    
    # Create a set for efficient lookup
    break_set = set(break_points)
    break_indices = [i for i, pt in enumerate(points) if pt in break_set]
    
    segments = []
    n_breaks = len(break_indices)
    
    if n_breaks == 0:
        segments.append(points)  # Entire contour as one segment
    else:
        # Segment the contour using break points
        for i in range(n_breaks):
            start_idx = break_indices[i]
            end_idx = break_indices[(i + 1) % n_breaks]  # Wrap around for last segment
            
            if start_idx <= end_idx:
                segment = points[start_idx:end_idx + 1]  # Inclusive slice
            else:
                segment = points[start_idx:] + points[:end_idx + 1]  # Wrap around contour
            segments.append(segment)
    
    # 颜色字典
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    # 绘制所有分段,每个分段用不同的颜色
    for i, segment in enumerate(segments):
        color = colors[i % len(colors)]  # Cycle through colors
        # cv2.polylines(frame, [np.array(segment, dtype=np.int32)], False, color, 2)
    
    failed_segments = []
    min_area = 10000000
    min_area_rect = None
    min_area_box = None
    
    # 分类每个分段
    for i, segment in enumerate(segments):
        # 如果分段里有4个及以上的点，则拟合最小外接正方形
        if len(segment) >= 4:
            rect = cv2.minAreaRect(np.array(segment, dtype=np.int32))
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # 顺便计算一下面积
            area = cv2.contourArea(box)
            # 顺便看看自己是不是目前最小的
            if area < min_area:
                min_area = area
                min_area_rect = rect
                min_area_box = box
                
        # 如果分段里有3个或两个点，保存分段到数组
        else:
            min_area_box = None
            failed_segments.append(segment)
            # cv2.polylines(frame, [np.array(segment, dtype=np.int32)], False, (0, 255, 255), 2)
    
    # 对失败分段进行配对处理（第一个和最后一个匹配，第二个和倒数第二个匹配）
    n_failed = len(failed_segments)
    print(f"Failed segments: {n_failed}")
    
    # 修复：将处理逻辑移到循环内部
    for i in range(n_failed // 2):
        seg1 = failed_segments[i]
        seg2 = failed_segments[n_failed - 1 - i]
        
        # 情况1：两个分段都含有3个点
        if len(seg1) == 3 and len(seg2) == 3:
            # 获取两个分段中的第二个点（索引1）
            pt1 = seg1[1]  # 第一个分段的中间点
            pt2 = seg2[1]  # 第二个分段的中间点
            
            
            # 计算对角线中点作为正方形中心
            cx = int((pt1[0] + pt2[0]) / 2)
            cy = int((pt1[1] + pt2[1]) / 2)
            center = (cx, cy)
            
            # 计算对角线长度（用于确定正方形边长）
            dx = pt2[0] - pt1[0]
            dy = pt2[1] - pt1[1]
            diagonal_length = np.sqrt(dx**2 + dy**2)
            side_length = diagonal_length / np.sqrt(2)  # 正方形的边长
            
            # 计算对角线角度（用于确定正方形方向）
            angle = np.degrees(np.arctan2(dy, dx))
            
            # 创建正方形描述（中心点、边长、角度）
            square_rect = (center, (side_length, side_length), angle - 45)
            
            # 计算正方形的四个角点
            box = cv2.boxPoints(square_rect)
            box = np.int0(box)
            
            # 计算正方形面积
            area = side_length * side_length
            
            # 更新最小面积正方形
            if area < min_area:
                min_area = area
                min_area_rect = square_rect
                min_area_box = box
                
        # 情况2：两个分段都含有2个点
        elif len(seg1) == 2 and len(seg2) == 2:
            # 计算两个分段之间的距离（中点距离）
            mid1 = ((seg1[0][0] + seg1[1][0]) // 2, (seg1[0][1] + seg1[1][1]) // 2)
            mid2 = ((seg2[0][0] + seg2[1][0]) // 2, (seg2[0][1] + seg2[1][1]) // 2)
            distance = np.sqrt((mid1[0]-mid2[0])**2 + (mid1[1]-mid2[1])**2)
            
            # 合并四个点
            combined_points = seg1 + seg2
            combined_array = np.array(combined_points, dtype=np.float32)
            
            # 拟合最小外接矩形
            rect = cv2.minAreaRect(combined_array)
            center, size, angle = rect
            
            # 强制调整为正方形
            square_size = max(size[0], size[1])
            square_rect = (center, (square_size, square_size), angle)
            box = cv2.boxPoints(square_rect)
            box = np.int0(box)
            area = square_size * square_size
            
            # 绘制拟合的正方形（粉色）和距离线
            # cv2.line(frame, mid1, mid2, (200, 200, 0), 2)
            cv2.putText(frame, f"2+2 Dist: {distance:.1f}", tuple(mid1), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            
            # 更新最小面积正方形
            if area < min_area:
                min_area = area
                min_area_rect = square_rect
                min_area_box = box
    
        # 处理奇数个分段的情况（3v3或2v2）
        if n_failed % 2 == 1 and n_failed > 0:
            # 计算每个分段的中点
            segment_midpoints = []
            for segment in failed_segments:
                if len(segment) > 0:
                    # 计算分段的中点
                    midpoint = np.mean(segment, axis=0).astype(int)
                    segment_midpoints.append(midpoint)
                else:
                    segment_midpoints.append(None)
            
            # 移除无效的中点
            valid_midpoints = [mp for mp in segment_midpoints if mp is not None]
            n_midpoints = len(valid_midpoints)
            
            # 确保有足够的中点
            if n_midpoints >= 3:
                # 创建组合点集：所有中点+最后一个分段的前一个点
                combined_points = []
                for i in range(n_midpoints - 1):  # 除了最后一个分段
                    combined_points.append(valid_midpoints[i])
                
                # 添加最后一个分段的前一个点
                last_segment = failed_segments[-1]
                if len(last_segment) > 0:
                    # 取最后一个分段的第一个点作为"前一个点"
                    prev_point = last_segment[0]
                    combined_points.append(prev_point)
                
                # 对除了最后一个点之外的点进行两两配对
                square_candidates = []
                n_combined = len(combined_points)
                
                # 配对点并形成封闭图形
                for i in range(0, n_combined - 1, 2):
                    if i + 1 >= n_combined:
                        break
                        
                    mid1 = combined_points[i]
                    mid2 = combined_points[i + 1]
                    
                    # 计算两点距离
                    distance = np.linalg.norm(np.array(mid1) - np.array(mid2))
                    
                    # 在图像上绘制连线
                    cv2.line(frame, tuple(mid1), tuple(mid2), (200, 200, 0), 2)
                    cv2.putText(frame, f"2+2 Dist: {distance:.1f}", tuple(mid1), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                    
                    # 获取配对分段对应的原始点
                    seg_idx1 = i // 2
                    seg_idx2 = (i + 1) // 2
                    
                    # 确保索引在范围内
                    if seg_idx1 < len(failed_segments) and seg_idx2 < len(failed_segments):
                        seg1 = failed_segments[seg_idx1]
                        seg2 = failed_segments[seg_idx2]
                        
                        # 组合两个分段的点
                        merged_points = seg1 + seg2
                        
                        # 添加连接点形成封闭图形
                        closed_points = merged_points + [mid1, mid2]
                        
                        # 拟合最小外接矩形
                        if len(closed_points) >= 3:
                            rect = cv2.minAreaRect(np.array(closed_points, dtype=np.int32))
                            (_, _), (w, h), _ = rect
                            
                            # 检查是否为正方形（宽高比在0.9-1.1之间）
                            ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0
                            if ratio >= 0.9:
                                area = w * h
                                box = cv2.boxPoints(rect)
                                box = np.int0(box)
                                
                                # 绘制候选正方形
                                
                                # 更新最小面积正方形
                                if area < min_area:
                                    min_area = area
                                    min_area_rect = rect
                                    min_area_box = box
                
                # 处理最后一个分段
                last_seg = failed_segments[-1]
                if len(last_seg) >= 3:  # 至少3个点才能形成形状
                    rect_last = cv2.minAreaRect(np.array(last_seg, dtype=np.int32))
                    (_, _), (w, h), _ = rect_last
                    ratio_last = min(w, h) / max(w, h) if max(w, h) > 0 else 0
                    if ratio_last >= 0.9:
                        area_last = w * h
                        box_last = cv2.boxPoints(rect_last)
                        box_last = np.int0(box_last)
                        
                        # 绘制候选正方形
                        
                        # 更新最小面积正方形
                        if area_last < min_area:
                            min_area = area_last
                            min_area_rect = rect_last
                            min_area_box = box_last

        return "Square", min_area_box
    return "Square", min_area_box
import cv2
import logger
import numpy as np
def draw_out_box(frame, polygon, contour_id, opposide_angles, corner_angles):
    # cv2.polylines(frame, [polygon], True, (255, 0, 255), 3)
    for j in range(4):
        cv2.circle(frame, tuple(polygon[j][0]), 12, (0, 255, 0), 2)
        cv2.putText(frame, f"id: {j} A: {corner_angles[j]:.2f}", tuple(polygon[j][0]+(15,0)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
    cv2.putText(frame, f"ID: {contour_id}", 
                (polygon[0][0][:2]+(0,15)), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (0, 200, 0), 1)
    cv2.putText(frame, f"OSA: {opposide_angles[0]:.2f}, {opposide_angles[1]:.2f}", #Opposite Sides Angles
                (polygon[0][0][:2]+(0,-15)), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (0, 200, 0), 1)
    
    
def draw_inner_box_numbers(frame, contours, hierarchy, inner_id):
    for i, contour in enumerate(contours):
                    if cv2.contourArea(contour) < 100: continue
                    if hierarchy[0][i][3] == inner_id:
                        cv2.drawContours(frame, contours, i, (255, 255, 0), -1)
                        if hierarchy[0][i][2] != -1:
                            for j, contour_in in enumerate(contours):
                                if cv2.contourArea(contour_in) < 100: continue
                                if hierarchy[0][j][3] == i:
                                    cv2.drawContours(frame, contours, j, (255, 0, 255), -1)
                        
def draw_AA_circle(img, center, color, radius, thickness=1):
    shift = 3  # 绘制目标中心(抗锯齿圆形)
    factor = 1 << shift
    cv2.circle(img, (int(center[0] * factor + 0.5), int(center[1] * factor + 0.5)),
               radius * factor, color, thickness, cv2.LINE_AA, shift,)

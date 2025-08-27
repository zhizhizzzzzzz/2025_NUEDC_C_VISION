import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from util_base import *
from util_leaveup import *
from util_num import *

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera Measurement")
        self.root.geometry("1600x900")
        # 强行全屏 
        self.root.attributes('-fullscreen', True)
        # self.cap = cv2.VideoCapture("output.mp4")
        self.cap = cv2.VideoCapture(0)
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, 1500)
        
        self.main_frame = ttk.Frame(root)
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=4) 
        self.main_frame.grid_columnconfigure(1, weight=1) 
        
        self.video_frame = ttk.Frame(self.main_frame)
        self.video_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.grid(row=0, column=1, sticky="nsew")
        self.black_screen = False
        
        self.control_frame.grid_rowconfigure(0, weight=1)  
        self.control_frame.grid_rowconfigure(1, weight=1)  
        self.black_screen_frame = ttk.Frame(self.control_frame)
        self.black_screen_frame.grid(row=2, column=0, sticky="se", pady=(10, 5))
        
        self.toggle_btn = ttk.Button(
            self.black_screen_frame,
            text="休眠",  
            command=self.toggle_black_screen,
            width=8
        )
        self.toggle_btn.pack(side=tk.RIGHT, padx=5, pady=5)
        self.toggle_btn.pack(side=tk.RIGHT, padx=5, pady=5)
        self.digit_frame = ttk.LabelFrame(self.control_frame, text="数字输入", padding=5)
        self.digit_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 5))
        
        self.input_var = tk.StringVar()
        self.input_display = ttk.Entry(
            self.digit_frame, 
            textvariable=self.input_var, 
            font=("Arial", 36, "bold"), 
            justify="center",
            state="readonly",
            width=2
        )
        self.input_display.pack(pady=(10, 15), ipady=10)
        
        button_frame = ttk.Frame(self.digit_frame)
        button_frame.pack(fill=tk.BOTH, expand=True)
        
        buttons = [
            '1', '2', '3',
            '4', '5', '6',
            '7', '8', '9',
            '0', '⌫', '✓'
        ]
        
        for i in range(4): 
            row_frame = ttk.Frame(button_frame)
            row_frame.pack(fill=tk.BOTH, expand=True, pady=2)
            for j in range(3):  
                btn_text = buttons[i*3 + j]
                btn = ttk.Button(
                    row_frame, 
                    text=btn_text, 
                    width=3,
                    command=lambda t=btn_text: self.button_action(t)
                )
                btn.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
                
                if btn_text == '✓':
                    btn.configure(style="Accent.TButton")
        
        self.info_frame = ttk.LabelFrame(self.control_frame, text="测量结果", padding=5)
        self.info_frame.grid(row=1, column=0, sticky="nsew")
        
        info_style = ttk.Style()
        info_style.configure("Info.TLabel", font=("Arial", 14))
        info_style.configure("Result.TLabel", font=("Arial", 20, "bold"), foreground="blue")
        
        shape_frame = ttk.Frame(self.info_frame)
        shape_frame.pack(fill=tk.X, pady=5)
        ttk.Label(shape_frame, text="形状:", style="Info.TLabel").pack(side=tk.LEFT)
        self.shape_var = tk.StringVar(value="")
        ttk.Label(shape_frame, textvariable=self.shape_var, style="Info.TLabel").pack(side=tk.LEFT, padx=10)
        
        size_frame = ttk.Frame(self.info_frame)
        size_frame.pack(fill=tk.X, pady=5)
        ttk.Label(size_frame, text="尺寸:", style="Info.TLabel").pack(side=tk.LEFT)
        self.size_var = tk.StringVar(value="")
        ttk.Label(size_frame, textvariable=self.size_var, style="Info.TLabel").pack(side=tk.LEFT, padx=10)
        
        diameter_frame = ttk.Frame(self.info_frame)
        diameter_frame.pack(fill=tk.X, pady=5)
        ttk.Label(diameter_frame, text="直径:", style="Info.TLabel").pack(side=tk.LEFT)
        self.diameter_var = tk.StringVar(value="")
        self.diameter_label = ttk.Label(
            diameter_frame, 
            textvariable=self.diameter_var, 
            style="Result.TLabel"
        )
        
        self.diameter_label.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(self.info_frame).pack(fill=tk.BOTH, expand=True)
        
        self.style = ttk.Style()
        self.style.configure("Accent.TButton", font=("Arial", 12, "bold"), foreground="green")
        self.style.configure("TButton", font=("Arial", 16, "bold"), padding=10)
        
        self.shape_type = None
        self.circle_center = None
        self.rvec = None
        self.position = None
        self.excellent_points = None
        
        self.target_digit = None
        self.digit_contour = None
        self.centroid = None
        self.radius_pixels = 0
        self.diameter_mm = 0
        self.measuring_radius = False
        self.scale_factor = None  
        
        self.update_video()
        
        self.root.bind('<e>', self.exit_fullscreen)
        self.root.bind('<f>', self.fullscreen_window)
        self.root.bind('<Key>', self.key_press)
        
    def toggle_black_screen(self):
        """切换黑屏状态"""
        self.black_screen = not self.black_screen
        
        if self.black_screen:
            self.toggle_btn.configure(text="显示")
        else:
            self.toggle_btn.configure(text="休眠")
    
            
    def exit_fullscreen(self, event):
        """退出全屏窗口"""
        self.root.attributes('-fullscreen', False)
        
    def fullscreen_window(self, event):
        """全屏窗口"""
        self.root.attributes('-fullscreen', True)
        
    def key_press(self, event):
        """处理键盘事件"""
        if event.char == 'q':  
            self.root.attributes('-fullscreen', False)
            self.cap.release()
            self.root.destroy()
        
    def button_action(self, text):
        """处理按钮点击"""
        if text in '0123456789':
            self.set_digit(text)
        elif text == '⌫':
            self.delete_digit()
        elif text == '✓':
            self.confirm_input()
    
    def set_digit(self, digit):
        """设置当前输入的数字"""
        self.input_var.set(digit)
    
    def delete_digit(self):
        """删除当前输入的数字"""
        self.input_var.set("")
    
    def confirm_input(self):
        """确认输入的数字"""
        digit = self.input_var.get()
        if digit:
            print(f"Confirmed digit: {digit}")
            self.target_digit = int(digit)
            self.measuring_radius = True
            self.diameter_var.set("12.4636")
            self.input_var.set("")
    
    def update_video(self):
        """更新摄像头画面"""
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.root.after(30, self.update_video)
            return

        if self.black_screen:
            processed_frame = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        else:
            processed_frame = self.process_frame(frame.copy())

        img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        label_width = self.video_label.winfo_width()
        label_height = self.video_label.winfo_height()

        if label_width > 1 and label_height > 1:
            img_ratio = img.width / img.height
            label_ratio = label_width / label_height

            if img_ratio > label_ratio:
                new_width = label_width
                new_height = int(new_width / img_ratio)
            else:
                new_height = label_height
                new_width = int(new_height * img_ratio)

            img = img.resize((new_width, new_height), Image.LANCZOS)

        img_tk = ImageTk.PhotoImage(image=img)

        self.video_label.configure(image=img_tk)
        self.video_label.image = img_tk

        self.root.after(30, self.update_video)
    
    def find_digit_contour(self, frame, shape_contour):
        """在形状轮廓区域内查找数字轮廓"""
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [shape_contour], -1, 255, -1)
        
        # 获取ROI
        x, y, w, h = cv2.boundingRect(shape_contour)
        roi = frame[y:y+h, x:x+w]
        mask_roi = mask[y:y+h, x:x+w]
        
        # 转换灰度,自适应阈值
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 21, 5
        )
        
        thresh = cv2.bitwise_and(thresh, thresh, mask=mask_roi)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        digit_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x_cnt, y_cnt, w_cnt, h_cnt = cv2.boundingRect(cnt)
            
            if area > 150 and w_cnt > 10 and h_cnt > 20 and 0.3 < w_cnt/h_cnt < 1.2:
                adjusted_cnt = cnt + np.array([x, y])
                digit_contours.append(adjusted_cnt)
        
        digit_contours = sorted(digit_contours, key=cv2.contourArea, reverse=True)[:3]
        
        return digit_contours
    
    def calculate_centroid(self, contour):
        """计算轮廓的质心"""
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
        return None
    
    def calculate_radius(self, centroid, shape_contour):
        """计算从质心到形状轮廓的最小距离（半径）"""
        distance = cv2.pointPolygonTest(shape_contour, centroid, True)
        return abs(distance)
    
    def pixels_to_mm(self, pixels, distance):
        """将像素距离转换为毫米距离"""
        # 假设摄像头焦距为f（像素），物体距离为d（毫米）
        # 则实际尺寸 = (像素尺寸 * 物体距离) / 焦距
        # 焦距估计值（需要根据摄像头校准）
        focal_length = 1200  
        
        mm_size = (pixels * distance) / focal_length
        return mm_size
    
    def measure_radius(self, frame, shape_contour_list, distance, rvec, tvec):
        """遍历所有同级轮廓，寻找数字并测量"""
        found = False
        for shape_contour in shape_contour_list:
            digit_contours = self.find_digit_contour(frame, shape_contour)
            if not digit_contours:
                continue
            for digit_contour in digit_contours:
                centroid = self.calculate_centroid(digit_contour)
                if centroid is None:
                    continue
                if self.is_target_digit(digit_contour, frame):
                    radius_px = self.calculate_radius(centroid, shape_contour)
                    circle_point = (int(centroid[0] + radius_px), int(centroid[1]))
                    actual_size = calculate_shape_size(
                        frame, "Circle", circle_point, centroid, rvec, tvec, 170
                    )
                    cv2.drawContours(frame, [digit_contour], -1, (255, 0, 255), 2)
                    cv2.circle(frame, centroid, 5, (0, 0, 255), -1)
                    cv2.circle(frame, centroid, int(radius_px), (0, 255, 255), 2)
                    cv2.putText(frame, f"Dia: {actual_size:.2f} mm",
                                (centroid[0] + 20, centroid[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    found = True
                    return frame, radius_px, actual_size
        if not found:
            return frame, None, None
        
    def is_target_digit(self, digit_contour, frame):
        """判断是否为目标数字"""
        perimeter = cv2.arcLength(digit_contour, True)
        approx = cv2.approxPolyDP(digit_contour, 0.04 * perimeter, True)
        if len(approx) == 4:
            return True
        return False
    
    ##########################################################################
    def process_frame(self, frame):
        """处理视频帧"""
        shape_type = None
        circle_center = None
        rvec = None
        position = None    
        excellent_points = None
        min_area = 100000000
        corner_points = find_corner_points(frame)
        develop_pts = []
        shape_contour = None
        distance = None  
        
        if corner_points is not None:
            excellent_points = subpixel_refinement(frame, corner_points)
            
            if len(excellent_points) > 1 and len(excellent_points[1]) > 0:
                for excellent_point in excellent_points[1]:
                    shape_type, circle_center, current_pts = classify_shape(excellent_point, frame)
                    
                    if shape_type == 'Square' and current_pts is not None:
                        
                        np_current_pts = np.array(current_pts).reshape(-1, 1, 2).astype(np.float32)
                        
                        if len(np_current_pts) > 0:
                            area = cv2.contourArea(np_current_pts)
                            if area < min_area:
                                min_area = area
                                develop_pts = current_pts
                                shape_contour = np.array(develop_pts, dtype=np.int32)
                    
            if len(excellent_points[0]) > 0 and len(excellent_points[0][0]) == 4:
                img_points = np.array(sort_points_clockwise(excellent_points[0][0]), dtype=np.float32)
                distance, rvec, position = calculate_distance(excellent_points[0][0], frame)
                cv2.putText(frame, f"Distance: {distance:.2f} mm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # 标记点
                for i, point in enumerate(excellent_points[0][0]):
                    int_point = (int(round(point[0])), int(round(point[1])))

            if shape_type is not None and rvec is not None:
                if len(develop_pts) > 0:
                    actual_size = calculate_shape_size(
                        frame, 
                        shape_type, 
                        develop_pts, 
                        circle_center,
                        rvec,
                        position,
                        170
                    )
                    if actual_size is not None:
                        cv2.drawContours(frame, [np.int32(develop_pts)], -1, (0, 255, 0), 2)
                        cv2.putText(frame, f"{shape_type} {actual_size:.2f} mm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        self.shape_var.set(shape_type)
                        self.size_var.set(f"{actual_size:.2f} mm")

                else:
                    actual_size = calculate_shape_size(
                            frame, 
                            shape_type, 
                            excellent_points[1][0], 
                            circle_center, 
                            rvec, 
                            position, 
                            170
                        )
                    
                    if actual_size is not None:
                        cv2.drawContours(frame, [np.int32(excellent_points[1][0])], -1, (0, 255, 0), 2)
                        cv2.putText(frame, f"{shape_type} {actual_size:.2f} mm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        self.shape_var.set(shape_type)
                        self.size_var.set(f"{actual_size:.2f} mm")
            
            if self.measuring_radius and shape_contour is not None and distance is not None:
                frame, radius_px, diameter_mm = self.measure_radius(frame, shape_contour, distance, rvec, position)
                if diameter_mm is not None:
                    self.diameter_mm = diameter_mm
                    self.diameter_var.set(f"{diameter_mm:.2f} mm")
                    self.measuring_radius = False
                else:
                    self.diameter_var.set("未检测到")
                    self.measuring_radius = False
            
            return frame
        else:
            frame, corner_num_points, shape_points = main_num(frame)
            shape_type = None
            circle_center = None
            rvec = None
            position = None
            min_area = 100000000
            develop_pts = [] 
            shape_contour = None
            distance = None
            
            converted_shape_points = []
            if shape_points is not None:
                for contour in shape_points:
                    new_contour = []
                    for pt in contour:
                        if isinstance(pt, np.ndarray) and pt.size >= 2:
                            new_contour.append([pt.ravel()[0], pt.ravel()[1]])
                        elif (isinstance(pt, (list, tuple))) and len(pt) >= 2:
                            new_contour.append([pt[0], pt[1]])
                        else:
                            continue
                    converted_shape_points.append(new_contour)
                shape_points = converted_shape_points
            
            if shape_points is not None:
                for pts in shape_points:
                    st, cc, cp = classify_shape(pts, frame)
                    shape_type = st
                    circle_center = cc
                    current_pts = cp
                            
                    if shape_type == 'Square' and current_pts is not None:
                        np_current_pts = np.array(current_pts).reshape(-1, 1, 2).astype(np.float32)
                        if len(np_current_pts) > 0:
                            area = cv2.contourArea(np_current_pts)
                            if area < min_area:
                                min_area = area
                                develop_pts = current_pts
                                shape_contour = np.array(develop_pts, dtype=np.int32)
                                        
            if corner_num_points is not None and len(corner_num_points) == 4:
                points = np.array(corner_num_points).reshape(4, 2)
                points = np.array(sort_points_clockwise(points), dtype=np.float32)
                distance, rvec, position = calculate_distance(points, frame)
                cv2.putText(frame, f"Distance: {distance:.2f} mm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if shape_type is not None and rvec is not None:
                if len(develop_pts) > 0:
                    actual_size = calculate_shape_size(
                        frame, 
                        shape_type, 
                        develop_pts, 
                        circle_center,
                        rvec,
                        position,
                        170
                    )
                    if actual_size is not None:
                        cv2.drawContours(frame, [np.int32(develop_pts)], -1, (0, 255, 0), 2)
                        cv2.putText(frame, f"{shape_type} {actual_size:.2f} mm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        self.shape_var.set(shape_type)
                        self.size_var.set(f"{actual_size:.2f} mm")
                elif shape_points and len(shape_points) > 0:
                    actual_size = calculate_shape_size(
                        frame, 
                        shape_type, 
                        shape_points[0], 
                        circle_center,
                        rvec,
                        position,
                        170
                    )
                    if actual_size is not None:
                        cv2.drawContours(frame, [np.int32(shape_points[0])], -1, (0, 255, 0), 2)
                        cv2.putText(frame, f"{shape_type} {actual_size:.2f} mm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        self.shape_var.set(shape_type)
                        self.size_var.set(f"{actual_size:.2f} mm")
            
            if self.measuring_radius and excellent_points is not None and len(excellent_points) > 1 and len(excellent_points[1]) > 0 and distance is not None:
                shape_contour_list = [np.array(pts, dtype=np.int32) for pts in excellent_points[1]]
                frame, radius_px, diameter_mm = self.measure_radius(frame, shape_contour_list, distance, rvec, position)
                if diameter_mm is not None:
                    self.diameter_mm = diameter_mm
                    self.diameter_var.set(f"{diameter_mm:.2f} mm")
                    self.measuring_radius = False
                else:
                    self.diameter_var.set("未检测到")
                    self.measuring_radius = False
            
            if self.circle_center is not None:
                cv2.circle(frame, (int(self.circle_center[0]), int(self.circle_center[1])), int(radius_px), (0, 255, 0), 2)
            return frame
    #####################################################################
    def on_closing(self):
        """关闭窗口时的清理工作"""
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()

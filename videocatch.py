import cv2

# 设置视频参数
width = 1920
height = 1080
fps = 60.0
output_filename = 'output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4编码器

# 初始化摄像头
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FPS, fps)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 初始化视频写入器
out = None
recording = False

print("按 'k' 开始录制，按 'q' 停止录制并退出")

while True:
    # 读取摄像头帧
    ret, frame = cap.read()
    if not ret:
        print("无法获取帧，退出...")
        break
    
    # 显示当前帧
    cv2.imshow('Video Recorder', frame)
    
    # 检查按键
    key = cv2.waitKey(1) & 0xFF
    
    # 按下 'k' 开始录制
    if key == ord('k'):
        if not recording:
            out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
            recording = True
            print("开始录制...")
    
    # 按下 'q' 停止录制并退出
    if key == ord('q'):
        if recording:
            print("停止录制并保存视频...")
        break
    
    # 如果正在录制，写入帧
    if recording:
        out.write(frame)

# 释放资源
if out is not None:
    out.release()
cap.release()
cv2.destroyAllWindows()

print(f"视频已保存为 {output_filename}")
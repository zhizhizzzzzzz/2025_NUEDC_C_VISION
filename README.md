# 2025_NUEDC_C_VISION

本项目是一个基于计算机视觉的测量与识别系统，支持摄像头实时视频处理、形状识别、数字识别以及距离测量等功能。

## 功能概述

1. **实时视频录制**：通过摄像头捕获视频并保存为 MP4 文件。
2. **形状与数字识别**：识别图像中的形状（如正方形、圆形）和数字。
3. **距离与尺寸测量**：基于摄像头校准参数，计算物体的实际距离和尺寸。
4. **图像采集**：支持拍照保存，用于校准或训练数据集。
5. **串口通信**：解析串口数据，支持与外部设备交互。

---

## 安装依赖

1. 安装 Python 依赖：
    ```bash
    pip install opencv-python-headless numpy pillow pyserial
    ```
2. 确保系统中安装了以下库：
    - OpenCV
    - Tkinter（用于图形界面）
    - Pillow（用于图像处理）

## 使用方法
1. 启动主程序
运行主程序 main.py，启动图形界面：
    ``` bash
    python3 [main.py](http://_vscodecontentref_/13)
    ```

2. 视频录制
运行 videocatch.py，按下 k 键开始录制，按下 q 键停止录制并保存视频：
    ``` bash
    python3 [videocatch.py](http://_vscodecontentref_/14)
    ```

3. 图像采集
运行 catch.py，按下 k 键拍照保存，按下 q 键退出程序：
    ``` bash
    python3 [catch.py](http://_vscodecontentref_/15)
    ```

4. 数字模板生成
运行 train.py，生成数字模板文件（存储在 digits/ 文件夹中）：
    ``` bash
    python3 [train.py](http://_vscodecontentref_/16)
    ```

5. 串口通信
运行 cd_serial.py，解析串口数据：
    ``` bash
    python3 [cd_serial.py](http://_vscodecontentref_/17)
    ```

## 自启动配置
在 Ubuntu 24.04 中，使用 Startup Applications 配置程序自启动：

1. 打开 Startup Applications。
2. 点击 Add，添加以下条目：
- Name: 2025_NUEDC_C_VISION
- Command:
    ``` bash
    python3 /path/to/main.py
    ```
- Comment: 启动视觉测量系统
3. 保存配置。

## 注意事项
1. 相机配置：确保相机已正确连接，且设备编号与代码中的 cv2.VideoCapture 参数一致。
2. 模板文件：运行 train.py 生成数字模板文件后，确保模板文件存储在 digits 文件夹中。
3. 权限问题：某些功能（如串口通信）可能需要管理员权限运行。
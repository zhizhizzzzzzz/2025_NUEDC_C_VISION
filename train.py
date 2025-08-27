import os
from PIL import Image, ImageDraw, ImageFont

# 创建digits目录
output_dir = "digits"
os.makedirs(output_dir, exist_ok=True)

# 配置参数 - 改为正方形尺寸
size = 28  # 正方形尺寸
background_color = 0      # 黑色背景
text_color = 255          # 白色数字

try:
    # 尝试加载系统字体（适用于不同平台）
    font_paths = [
        # Windows
        "C:/Windows/Fonts/arialbd.ttf",  # Arial Bold
        "C:/Windows/Fonts/arial.ttf",    # Arial Regular
        
        # macOS
        "/Library/Fonts/Arial Bold.ttf",
        "/Library/Fonts/Arial.ttf",
        
        # Linux
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        
        # 通用备选
        "arialbd.ttf",
        "Arial.ttf"
    ]
    
    font = None
    for path in font_paths:
        try:
            # 修正1：字体大小调整为画布的70%
            font = ImageFont.truetype(path, int(size * 1))
            break
        except OSError:
            continue
    
    if font is None:
        print("警告：未找到系统字体，使用默认小号字体")
        font = ImageFont.load_default()
        font_size = min(size, 20)  # 默认字体最大支持20px
        font = ImageFont.load_default()

    # 生成0-9的数字图像
    for digit in range(10):
        img = Image.new('L', (size, size), background_color)
        draw = ImageDraw.Draw(img)
        
        text = str(digit)
        
        # 修正2：使用更准确的锚点定位
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # 计算居中位置（修正基线偏移）
        x = (size - text_width) / 2
        y = (size - text_height) / 2 - bbox[1]  # 调整基线偏移
        
        draw.text((x, y), text, fill=text_color, font=font)
        img.save(os.path.join(output_dir, f"{digit}.png"))
        print(f"生成: {output_dir}/{digit}.png")

    print("所有数字模板已生成完毕！")

except Exception as e:
    print(f"错误: {e}")
    print("请确保已安装Pillow库: pip install Pillow")
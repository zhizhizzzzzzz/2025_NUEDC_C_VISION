import serial
import struct
import time

PORT = '/dev/ttyUSB0'      
BAUDRATE = 9600   
TIMEOUT = 1        

def parse_float_data(data):
    """解析12字节数据为3个浮点数"""
    if len(data) != 12:
        raise ValueError("必须为12字节")
    
    # 使用struct模块解包字节数据（小端模式：'<'，3个浮点数：'fff'）
    return struct.unpack('<fff', data)

def main():
    ser = serial.Serial(PORT, BAUDRATE, timeout=TIMEOUT)
    print(f"连接到串口: {ser.name}")

    try:
        while True:
            raw_data = ser.read(12)
            
            if len(raw_data) == 12:
                try:
                    #
                    float1, float2, float3 = parse_float_data(raw_data)
                    print(f"解析结果: {float1:.2f}, {float2:.2f}, {float3:.2f}")
                except ValueError as e:
                    print(f"解析错误: {e}")
            else:
                print(f"数据不足12字节，收到: {len(raw_data)}字节")
            
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\n程序终止")
    finally:
        ser.close()
        print("串口已关闭")

if __name__ == "__main__":
    main()
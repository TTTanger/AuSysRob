#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器人误差补偿模块
专门处理机械臂本身的精度补偿，不涉及手眼标定
"""

import json
import os
import numpy as np

# ==================== 基础补偿参数 ====================
X_POSITION_OFFSET = -14
Y_POSITION_OFFSET = 26
Z_POSITION_OFFSET = 0.0

# 关节角度补偿
JOINT1_OFFSET = 0.0
JOINT2_OFFSET = 0.0
JOINT3_OFFSET = 0.0
JOINT4_OFFSET = 0.0
JOINT5_OFFSET = 0.0

# ==================== 系统配置参数 ====================
# 夹持器参数
GRIP_HEIGHT = 25.0  # 夹持高度 (mm)
GRIPPER_OPEN_ANGLE = 0  # 夹持器开启角度
GRIPPER_CLOSE_ANGLE = 90  # 夹持器闭合角度
GRIPPER_GRAB_ANGLE = 30  # 夹持器抓取角度

# 移动参数
MOVE_TIME = 200  # 移动时间 (ms)

# 摄像头参数
CAMERA_INDEX = 1  # 摄像头索引

# 高度参数
SAFE_HEIGHT = 100.0  # 安全高度 (mm)
WORKTABLE_HEIGHT = 0.0  # 工作台高度 (mm)

# 调试参数
DEBUG_MODE = False  # 调试模式
SHOW_CAMERA_FEED = True  # 显示摄像头画面
SAVE_DETECTION_IMAGES = False  # 保存检测图像
IMAGE_SAVE_PATH = "detection_images"  # 图像保存路径

# 串口参数
SERIAL_PORT = 'COM5'  # 串口号
BAUD_RATE = 115200  # 波特率
TIMEOUT = 5  # 超时时间

# ==================== 误差数据库 ====================
ERROR_DATABASE = {}

def add_error_data(target_position, actual_position):
    """
    添加误差数据到数据库
    
    Args:
        target_position: 目标位置 (x, y, z)
        actual_position: 实际到达位置 (x, y, z)
    """
    x_error = actual_position[0] - target_position[0]
    y_error = actual_position[1] - target_position[1]
    z_error = actual_position[2] - target_position[2]
    
    ERROR_DATABASE[target_position] = (x_error, y_error, z_error)
    print(f"✅ 误差数据已添加: {target_position} -> 误差({x_error:.2f}, {y_error:.2f}, {z_error:.2f})")

def get_error_data(target_position):
    """
    获取指定位置的误差数据
    
    Args:
        target_position: 目标位置 (x, y, z)
        
    Returns:
        tuple: 误差数据 (x_error, y_error, z_error) 或 None
    """
    return ERROR_DATABASE.get(target_position)

def find_nearest_error_data(target_position, max_distance=50.0):
    """
    查找最近的误差数据
    
    Args:
        target_position: 目标位置 (x, y, z)
        max_distance: 最大搜索距离 (mm)
        
    Returns:
        tuple: 最近的误差数据或 None
    """
    if not ERROR_DATABASE:
        return None
    
    min_distance = float('inf')
    nearest_error = None
    
    for position, error in ERROR_DATABASE.items():
        distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(target_position, position)))
        if distance < min_distance and distance <= max_distance:
            min_distance = distance
            nearest_error = error
    
    return nearest_error

def data_driven_position_compensation(x, y, z):
    """
    基于数据的位置补偿
    
    Args:
        x, y, z: 目标位置坐标
        
    Returns:
        tuple: 补偿后的位置坐标
    """
    target_position = (x, y, z)
    
    # 1. 尝试获取精确匹配的误差数据
    error_data = get_error_data(target_position)
    
    if error_data:
        # 使用精确匹配的误差数据
        x_error, y_error, z_error = error_data
        compensated_x = x - x_error
        compensated_y = y - y_error
        compensated_z = z - z_error
        
        print(f"🎯 精确补偿: 目标({x:.1f}, {y:.1f}, {z:.1f}) -> 补偿({compensated_x:.1f}, {compensated_y:.1f}, {compensated_z:.1f})")
        return (compensated_x, compensated_y, compensated_z)
    
    # 2. 尝试获取最近的误差数据
    nearest_error = find_nearest_error_data(target_position)
    
    if nearest_error:
        # 使用最近的误差数据
        x_error, y_error, z_error = nearest_error
        compensated_x = x - x_error
        compensated_y = y - y_error
        compensated_z = z - z_error
        
        print(f"📏 近似补偿: 目标({x:.1f}, {y:.1f}, {z:.1f}) -> 补偿({compensated_x:.1f}, {compensated_y:.1f}, {compensated_z:.1f})")
        return (compensated_x, compensated_y, compensated_z)
    
    # 3. 使用固定补偿
    compensated_x = x + X_POSITION_OFFSET
    compensated_y = y + Y_POSITION_OFFSET
    compensated_z = z + Z_POSITION_OFFSET
    
    print(f"🔧 固定补偿: 目标({x:.1f}, {y:.1f}, {z:.1f}) -> 补偿({compensated_x:.1f}, {compensated_y:.1f}, {compensated_z:.1f})")
    return (compensated_x, compensated_y, compensated_z)

def data_driven_joint_compensation(j1, j2, j3, j4, j5):
    """
    基于数据的关节补偿
    
    Args:
        j1, j2, j3, j4, j5: 目标关节角度
        
    Returns:
        tuple: 补偿后的关节角度
    """
    compensated_j1 = j1 + JOINT1_OFFSET
    compensated_j2 = j2 + JOINT2_OFFSET
    compensated_j3 = j3 + JOINT3_OFFSET
    compensated_j4 = j4 + JOINT4_OFFSET
    compensated_j5 = j5 + JOINT5_OFFSET
    
    return (compensated_j1, compensated_j2, compensated_j3, compensated_j4, compensated_j5)

# ==================== 数据库管理 ====================
def save_error_database(filename="error_database.json"):
    """保存误差数据库到文件"""
    try:
        # 将元组转换为字符串键和列表值
        serializable_data = {}
        for position, error in ERROR_DATABASE.items():
            position_key = f"{position[0]},{position[1]},{position[2]}"
            serializable_data[position_key] = list(error)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 误差数据库已保存到 {filename}")
        return True
    except Exception as e:
        print(f"❌ 保存数据库失败: {e}")
        return False

def load_error_database(filename="error_database.json"):
    """从文件加载误差数据库"""
    try:
        if not os.path.exists(filename):
            print(f"⚠️ 数据库文件 {filename} 不存在")
            return False
        
        with open(filename, 'r', encoding='utf-8') as f:
            serializable_data = json.load(f)
        
        # 将字符串键和列表值转换回元组
        ERROR_DATABASE.clear()
        for position_key, error_list in serializable_data.items():
            position_values = [float(x) for x in position_key.split(',')]
            position = tuple(position_values)
            error = tuple(error_list)
            ERROR_DATABASE[position] = error
        
        print(f"✅ 误差数据库已从 {filename} 加载")
        return True
    except Exception as e:
        print(f"❌ 加载数据库失败: {e}")
        return False

def print_error_database():
    """打印误差数据库内容"""
    print("📊 误差数据库内容:")
    print("=" * 50)
    
    if not ERROR_DATABASE:
        print("数据库为空")
        return
    
    for i, (position, error) in enumerate(ERROR_DATABASE.items(), 1):
        print(f"{i}. 位置{position} -> 误差{error}")

def get_database_info():
    """获取数据库信息"""
    return {
        'record_count': len(ERROR_DATABASE),
        'positions': list(ERROR_DATABASE.keys()),
        'errors': list(ERROR_DATABASE.values())
    }

# ==================== 传统补偿函数（保持兼容性） ====================
def apply_position_compensation(x, y, z):
    """应用位置补偿（传统方法）"""
    return (x + X_POSITION_OFFSET, y + Y_POSITION_OFFSET, z + Z_POSITION_OFFSET)

def apply_joint_compensation(j1, j2, j3, j4, j5):
    """应用关节补偿（传统方法）"""
    return (j1 + JOINT1_OFFSET, j2 + JOINT2_OFFSET, j3 + JOINT3_OFFSET, 
            j4 + JOINT4_OFFSET, j5 + JOINT5_OFFSET)

def print_compensation_status():
    """打印补偿状态"""
    print("🔧 机器人误差补偿模块状态:")
    print(f"  - 位置补偿: X={X_POSITION_OFFSET:.1f}, Y={Y_POSITION_OFFSET:.1f}, Z={Z_POSITION_OFFSET:.1f}")
    print(f"  - 关节补偿: J1={JOINT1_OFFSET:.1f}, J2={JOINT2_OFFSET:.1f}, J3={JOINT3_OFFSET:.1f}, J4={JOINT4_OFFSET:.1f}, J5={JOINT5_OFFSET:.1f}")
    print(f"  - 误差数据库记录数: {len(ERROR_DATABASE)}")

if __name__ == "__main__":
    print("🤖 机器人误差补偿模块")
    print("=" * 40)
    print("1. 查看当前数据库")
    print("2. 手动添加误差数据")
    print("3. 保存数据库")
    print("4. 加载数据库")
    print("5. 退出")
    
    while True:
        choice = input("\n请选择操作 (1-5): ")
        
        if choice == '1':
            print_error_database()
        elif choice == '2':
            try:
                print("\n请输入目标位置:")
                target_x = float(input("目标X坐标: "))
                target_y = float(input("目标Y坐标: "))
                target_z = float(input("目标Z坐标: "))
                target_position = (target_x, target_y, target_z)
                
                print("\n请输入实际到达位置:")
                actual_x = float(input("实际X坐标: "))
                actual_y = float(input("实际Y坐标: "))
                actual_z = float(input("实际Z坐标: "))
                actual_position = (actual_x, actual_y, actual_z)
                
                add_error_data(target_position, actual_position)
            except ValueError:
                print("❌ 输入格式错误，请输入数字")
        elif choice == '3':
            save_error_database()
        elif choice == '4':
            load_error_database()
        elif choice == '5':
            print("👋 退出")
            break
        else:
            print("❌ 无效选项")
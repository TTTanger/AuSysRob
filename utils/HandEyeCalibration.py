import cv2
import numpy as np
import time
import json
import os
from typing import Tuple, Optional
from .RobotCompensation import CAMERA_INDEX

class HandEyeCalibration:
    """
    手眼标定类
    用于校准摄像头像素坐标和机械臂基座物理坐标之间的转换关系
    """
    
    def __init__(self, camera_index: int = CAMERA_INDEX):
        """
        初始化手眼标定类
        
        Args:
            camera_index: 摄像头索引，默认为0
        """
        self.camera_index = camera_index
        self.cap = None
        self.compensation_vector = None
        self.scale_factor = None
        self.calibrated = False
        
    def open_camera(self) -> bool:
        """
        打开摄像头
        
        Returns:
            bool: 是否成功打开摄像头
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                print(f"❌ 无法打开摄像头 {self.camera_index}")
                return False
            print(f"✅ 摄像头 {self.camera_index} 打开成功")
            return True
        except Exception as e:
            print(f"❌ 打开摄像头时发生错误: {e}")
            return False
    
    def close_camera(self):
        """关闭摄像头"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            print("📷 摄像头已关闭")
    
    def detect_red_lego(self, frame) -> Optional[Tuple[int, int, float]]:
        """
        检测红色乐高积木
        
        Args:
            frame: 输入图像帧
            
        Returns:
            Optional[Tuple[int, int, float]]: 检测到的红色积木中心像素坐标和角度，
            格式为 (x, y, angle)，如果未检测到则返回None
        """
        # 转换到HSV色彩空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 定义红色范围（HSV）
        # 红色在HSV中跨越0度和180度，所以需要两个范围
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        # 创建红色掩码
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2
        
        # 形态学操作去除噪声
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 找到最大的轮廓（假设是乐高积木）
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # 面积阈值，过滤掉太小的区域
            if area > 1000:  # 可以根据实际情况调整
                # 计算轮廓的中心点
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # 新增：计算积木角度
                    rect = cv2.minAreaRect(largest_contour)
                    angle = rect[2]  # 获取角度
                    
                    # 角度标准化：确保角度在-90到90度之间
                    if angle < -45:
                        angle += 90
                    elif angle > 45:
                        angle -= 90
                    
                    return (cx, cy, angle)
        
        return None
    
    def capture_and_detect(self) -> Optional[Tuple[int, int, float]]:
        """
        拍摄图片并检测红色乐高积木
        
        Returns:
            Optional[Tuple[int, int, float]]: 检测到的像素坐标和角度，
            格式为 (x, y, angle)，如果未检测到则返回None
        """
        if not self.cap or not self.cap.isOpened():
            print("Camera not opened")
            return None
        
        print("📷 正在拍摄图片...")
        ret, frame = self.cap.read()
        if not ret:
            print("Cannot read camera frame")
            return None
        
        # 检测红色乐高积木
        result = self.detect_red_lego(frame)
        
        if result:
            x, y, angle = result
            print(f"Detected red Lego, position: ({x}, {y}), angle: {angle:.1f}°")
            
            # 在图像上标记检测到的位置和角度
            cv2.circle(frame, (x, y), 10, (0, 255, 0), 2)
            cv2.putText(frame, f"({x}, {y})", 
                    (x + 15, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 显示角度信息
            cv2.putText(frame, f"Angle: {angle:.1f}°", 
                    (x + 15, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 绘制角度指示线
            line_length = 40
            end_x = int(x + line_length * np.cos(np.radians(angle)))
            end_y = int(y + line_length * np.sin(np.radians(angle)))
            cv2.line(frame, (x, y), (end_x, end_y), (255, 0, 0), 2)
            
            # 显示图像
            cv2.imshow("检测结果", frame)
            cv2.waitKey(2000)  # 显示2秒
            cv2.destroyAllWindows()
            
            return result
        else:
            print("❌ 未检测到红色乐高积木")
            return None
    
    def calibrate(self, physical_coord: Tuple[float, float]) -> bool:
        """
        执行手眼标定
        
        Args:
            physical_coord: 物理坐标 (x, y)，相对于机械臂基座中心点
            
        Returns:
            bool: 标定是否成功
        """
        print("🔧 开始手眼标定...")
        print("请将红色乐高积木放置在摄像头视野内，然后按任意键继续...")
        
        # 打开摄像头
        if not self.open_camera():
            return False
        
        try:
            # 实时显示摄像头画面
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Cannot read camera frame")
                    break
                
                # 检测红色积木并实时显示
                result = self.detect_red_lego(frame)
                if result:
                    x, y, angle = result
                    cv2.circle(frame, (x, y), 10, (0, 255, 0), 2)
                    cv2.putText(frame, f"Detected red Lego: ({x}, {y}), Angle: {angle:.1f}°", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.putText(frame, "Press 'c' to confirm, press 'q' to exit", 
                           (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow("Hand-Eye Calibration", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    break
                elif key == ord('q'):
                    print("❌ User cancelled calibration")
                    return False
            
            # 拍摄并检测
            result = self.capture_and_detect()
            if not result:
                return False

            x, y, angle = result
            pixel_coord = (x, y)  # 只使用位置信息进行标定

            # 计算补偿向量
            self.compensation_vector = (
                physical_coord[0] - pixel_coord[0],
                physical_coord[1] - pixel_coord[1]
            )   
            
            # 计算缩放因子（可选，用于更精确的转换）
            # 这里假设像素和物理坐标的比例关系
            self.scale_factor = (390 / 680)  # 可以根据实际情况调整
            
            self.calibrated = True
            print(f"✅ Calibration completed!")
            print(f"   Pixel coordinates: {pixel_coord}")
            print(f"   Physical coordinates: {physical_coord}")
            print(f"   Compensation vector: {self.compensation_vector}")
            
            return True
            
        finally:
            self.close_camera()
    
    def pixel_to_physical(self, pixel_coord: Tuple[int, int]) -> Tuple[float, float]:
        """
        将像素坐标转换为物理坐标
        
        Args:
            pixel_coord: 像素坐标 (x, y)
            
        Returns:
            Tuple[float, float]: 物理坐标 (x, y)
        """
        if not self.calibrated:
            raise ValueError("❌ Calibration not completed, please call calibrate() first")
        
        physical_x = (pixel_coord[0] + self.compensation_vector[0]) * self.scale_factor
        physical_y = (pixel_coord[1] + self.compensation_vector[1]) * self.scale_factor
        
        return (-physical_x, physical_y)
    
    def get_calibration_info(self) -> dict:
        """
        获取标定信息
        
        Returns:
            dict: 包含标定信息的字典
        """
        return {
            'calibrated': self.calibrated,
            'compensation_vector': self.compensation_vector,
            'scale_factor': self.scale_factor
        }
    
    def save_calibration(self, filename="hand_eye_calibration.json"):
        """
        保存标定数据到文件
        
        Args:
            filename: 保存文件名
            
        Returns:
            bool: 保存是否成功
        """
        if not self.calibrated:
            print("❌ 尚未完成标定，无法保存")
            return False
        
        try:
            calibration_data = {
                'calibrated': self.calibrated,
                'compensation_vector': list(self.compensation_vector),
                'scale_factor': self.scale_factor,
                'camera_index': self.camera_index,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(calibration_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 标定数据已保存到 {filename}")
            return True
            
        except Exception as e:
            print(f"❌ 保存标定数据失败: {e}")
            return False
    
    def load_calibration(self, filename="hand_eye_calibration.json"):
        """
        从文件加载标定数据
        
        Args:
            filename: 加载文件名
            
        Returns:
            bool: 加载是否成功
        """
        try:
            if not os.path.exists(filename):
                print(f"⚠️ 标定文件 {filename} 不存在")
                return False
            
            with open(filename, 'r', encoding='utf-8') as f:
                calibration_data = json.load(f)
            
            self.calibrated = calibration_data.get('calibrated', False)
            if self.calibrated:
                compensation_vector = calibration_data.get('compensation_vector', [0, 0])
                self.compensation_vector = tuple(compensation_vector)
                self.scale_factor = calibration_data.get('scale_factor', 1.0)
                self.camera_index = calibration_data.get('camera_index', self.camera_index)
                
                timestamp = calibration_data.get('timestamp', '未知')
                print(f"✅ 标定数据已从 {filename} 加载")
                print(f"   标定时间: {timestamp}")
                print(f"   补偿向量: {self.compensation_vector}")
                print(f"   缩放因子: {self.scale_factor}")
                return True
            else:
                print("❌ 加载的标定数据无效")
                return False
                
        except Exception as e:
            print(f"❌ 加载标定数据失败: {e}")
            return False
    
    def print_calibration_status(self):
        """打印标定状态"""
        print("📊 手眼标定状态:")
        print(f"  - 标定状态: {'已完成' if self.calibrated else '未完成'}")
        if self.calibrated:
            print(f"  - 补偿向量: {self.compensation_vector}")
            print(f"  - 缩放因子: {self.scale_factor}")
        print(f"  - 摄像头索引: {self.camera_index}") 
    
    def test_angle_detection(self):
        """
        测试角度检测功能
        
        使用方法：
        1. 将红色乐高积木放在摄像头前
        2. 运行此函数
        3. 观察检测结果和可视化效果
        """
        print("🧪 开始角度检测测试...")
        print("=" * 50)
        print("测试步骤：")
        print("1. 请将红色乐高积木放在摄像头视野内")
        print("2. 按 'c' 确认开始检测")
        print("3. 按 'q' 退出测试")
        print("4. 可以旋转积木观察角度变化")
        print("=" * 50)
        
        # 打开摄像头
        if not self.open_camera():
            print("❌ 无法打开摄像头")
            return False
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ 无法读取摄像头画面")
                    break
                
                # 检测红色乐高积木
                result = self.detect_red_lego(frame)
                
                # 创建显示用的图像副本
                display_frame = frame.copy()
                
                if result:
                    x, y, angle = result
                    
                    # 在图像上标记检测结果
                    # 1. 标记中心点
                    cv2.circle(display_frame, (x, y), 10, (0, 255, 0), 2)
                    
                    # 2. 显示位置信息
                    cv2.putText(display_frame, f"Position: ({x}, {y})", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # 3. 显示角度信息
                    cv2.putText(display_frame, f"Angle: {angle:.1f}°", 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # 4. 绘制角度指示线
                    line_length = 50
                    end_x = int(x + line_length * np.cos(np.radians(angle)))
                    end_y = int(y + line_length * np.sin(np.radians(angle)))
                    cv2.line(display_frame, (x, y), (end_x, end_y), (255, 0, 0), 3)
                    
                    # 5. 在控制台输出检测结果
                    print(f"\r检测结果: 位置({x}, {y}), 角度: {angle:.1f}°", end="")
                    
                else:
                    # 未检测到积木
                    cv2.putText(display_frame, "No red Lego detected", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    print("\r未检测到红色乐高积木", end="")
                
                # 显示操作提示
                cv2.putText(display_frame, "Press 'c' to capture, 'q' to quit", 
                        (10, display_frame.shape[0] - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 显示图像
                cv2.imshow("角度检测测试", display_frame)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    if result:
                        x, y, angle = result
                        print(f"\n✅ 捕获检测结果:")
                        print(f"   位置: ({x}, {y})")
                        print(f"   角度: {angle:.1f}°")
                        
                        # 保存检测结果图像
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = f"angle_test_{timestamp}.jpg"
                        cv2.imwrite(filename, display_frame)
                        print(f"   图像已保存为: {filename}")
                    else:
                        print("\n❌ 未检测到积木，无法捕获")
                        
                elif key == ord('q'):
                    print("\n👋 测试结束")
                    break
        
        finally:
            self.close_camera()
            cv2.destroyAllWindows()
        
        return True
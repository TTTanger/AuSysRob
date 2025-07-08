#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Error Compensation Test Module
Specialized for testing and collecting mechanical arm error data
"""

import sys
import os
import cv2
import numpy as np

# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LegoGraspingSystem import LegoGraspingSystem
from utils import RobotCompensation

class ErrorCompensationTester:
    """Error Compensation Tester"""
    
    def __init__(self):
        self.system = None
        self.cap = None
    
    def initialize_system(self):
        """Initialize system (without hand-eye calibration)"""
        print("🔧 Initialize system...")
        
        try:
            self.system = LegoGraspingSystem()
            
            # Initialize basic components, without hand-eye calibration
            if not self.system.initialize_system():
                print("❌ System initialization failed")
                return False
            
            # Check if already calibrated
            if not hasattr(self.system, 'calibration') or not self.system.calibration:
                print("❌ System not calibrated, please run hand-eye calibration first")
                return False
            
            print("✅ System initialization completed")
            return True
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            return False
    
    def manual_error_test(self):
        """Manual error test"""
        print("\nManual error test")
        print("=" * 50)
        print("Test steps:")
        print("1. Input target position")
        print("2. Move the robot arm to the target position")
        print("3. Measure the actual position reached by the robot arm")
        print("4. Record error data")
        print("=" * 50)
        
        try:
            # Input target position
            print("\nPlease input target position:")
            target_x = float(input("Target X coordinate (mm): "))
            target_y = float(input("Target Y coordinate (mm): "))
            target_z = float(input("Target Z coordinate (mm): "))
            target_position = (target_x, target_y, target_z)
            
            print(f"\n🎯 Target position: {target_position}")
            
            # Confirm move
            confirm = input("Move the robot arm to the target position? (y/n): ")
            if confirm.lower() != 'y':
                print("❌ Cancel test")
                return False
            
            # Move the robot arm to the target position
            print("The robot arm is moving to the target position...")
            
            # Calculate inverse kinematics
            from utils.InverseKinematic import inverse_kinematics
            joint_angles = inverse_kinematics(target_position)
            
            if joint_angles is None:
                print("❌ Inverse kinematics calculation failed, position unreachable")
                return False
            
            # Apply joint compensation
            compensated_angles = RobotCompensation.data_driven_joint_compensation(*joint_angles)
            
            # Move the robot arm step by step
            if self.system and self.system.robot:
                try:
                    # Step 1: Move to safe height and open gripper
                    print("🔄 Step 1: Move to safe height and open gripper...")
                    
                    # Calculate the position of safe height (keep x,y unchanged, set z to safe height)
                    safe_position = (target_x, target_y, RobotCompensation.SAFE_HEIGHT)
                    safe_joint_angles = inverse_kinematics(safe_position)
                    
                    if safe_joint_angles is None:
                        print("❌ Safe height position unreachable")
                        return False
                    
                    # Move to safe height and open gripper
                    safe_compensated_angles = RobotCompensation.data_driven_joint_compensation(*safe_joint_angles)
                    self.system.robot.move_to_angles(*safe_compensated_angles, RobotCompensation.GRIPPER_OPEN_ANGLE, RobotCompensation.MOVE_TIME)
                    print(f"✅ Moved to safe height {RobotCompensation.SAFE_HEIGHT}mm, gripper opened to {RobotCompensation.GRIPPER_OPEN_ANGLE}°")
                    
                    # Wait for the action to complete
                    import time
                    time.sleep(1)
                    
                    # Step 2: Move to target position
                    print("🔄 Step 2: Move to target position...")
                    self.system.robot.move_to_angles(*compensated_angles)
                    print("✅ Robot arm moved to target position")
                    
                except Exception as e:
                    print(f"❌ Robot arm movement failed: {e}")
                    return False
            else:
                print("⚠️ Robot arm not connected, simulate step-by-step movement")
                print(f"Step 1: Safe height {RobotCompensation.SAFE_HEIGHT}mm, gripper angle {RobotCompensation.GRIPPER_OPEN_ANGLE}°")
                print(f"Step 2: Target position, joint angles: {compensated_angles}")
            
            # Wait for user to measure actual position
            input("\nPlease measure the actual position reached by the robot arm, then press Enter to continue...")
            
            # Input actual position
            print("\nPlease input the actual position reached by the robot arm:")
            actual_x = float(input("Actual X coordinate (mm): "))
            actual_y = float(input("Actual Y coordinate (mm): "))
            actual_z = float(input("Actual Z coordinate (mm): "))
            actual_position = (actual_x, actual_y, actual_z)
            
            # Add error data
            RobotCompensation.add_error_data(target_position, actual_position)
            
            # Calculate error
            x_error = actual_x - target_x
            y_error = actual_y - target_y
            z_error = actual_z - target_z
            
            print(f"\n📊 Error analysis:")
            print(f"Target position: {target_position}")
            print(f"Actual position: {actual_position}")
            print(f"Error: ({x_error:.2f}, {y_error:.2f}, {z_error:.2f})")
            print(f"Total error: {np.sqrt(x_error**2 + y_error**2 + z_error**2):.2f} mm")
            
            return True
            
        except ValueError:
            print("❌ Input format error, please enter numbers")
            return False
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False
    
    def camera_based_error_test(self):
        """Camera-based error test"""
        print("\n📷 Camera-based error test")
        print("=" * 50)
        print("Test steps:")
        print("1. Place a red Lego brick in the camera's field of view")
        print("2. Press 't' to perform detection and error test")
        print("3. Press 'v' to view the error database")
        print("4. Press 'q' to exit the test")
        print("=" * 50)
        
        if not self.system or not self.system.cap:
            print("❌ System not initialized or camera not opened")
            return False
        
        # Check if hand-eye calibration has been completed
        if not self.system.is_calibrated:
            print("❌ System not calibrated, please run hand-eye calibration first")
            print("Suggestion: Run the main system for hand-eye calibration before using this test")
            return False
        
        print("✅ Start camera-based error test...")
        
        while True:
            ret, frame = self.system.cap.read()
            if not ret:
                print("❌ Unable to read camera frame")
                break
            
            # Create a copy of the image for display
            display_frame = frame.copy()
            
            # Detect and display the Lego brick
            lego_result = self.system.detect_red_lego(frame)
            if lego_result:
                lego_x, lego_y, lego_angle = lego_result
                # Mark the Lego brick position on the image
                cv2.circle(display_frame, (lego_x, lego_y), 10, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Lego: ({lego_x}, {lego_y})", 
                           (lego_x + 15, lego_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display the camera frame
            cv2.imshow('Camera Error Test - Press t to test, v to view, q to quit', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("Exit camera-based error test")
                break
            elif key == ord('t'):
                self.perform_camera_error_test(frame)
            elif key == ord('v'):
                RobotCompensation.print_error_database()
        
        cv2.destroyAllWindows()
        return True
    def detect_red_tape(self, frame):
        """
        Detect the red tape on the robot arm
        
        Args:
            frame: Input image frame
            
        Returns:
            Optional[Tuple[int, int, float, list]]: The center pixel coordinates, angle, and contour information of the detected red tape,
            formatted as (x, y, angle, contour_info), if not detected, return None
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define the range of red tape
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        # Create red mask
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2
        
        # Morphological operations to remove noise
        kernel = np.ones((3, 3), np.uint8)  # Use a smaller kernel because the tape is smaller
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Sort by area, find the appropriate size of the contour (tape is smaller than the Lego brick)
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            for contour in sorted_contours:
                area = cv2.contourArea(contour)
                
                # Tape area range: approximately 1 square centimeter, approximately 100-800 pixels in the image
                # Lego brick area is usually greater than 1000 pixels
                if 100 <= area <= 800:  # Tape area range
                    # Calculate the center point of the contour
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Calculate the angle of the tape
                        rect = cv2.minAreaRect(contour)
                        angle = rect[2]  # Get the angle
                        
                        # Angle normalization: ensure the angle is between -90 and 90 degrees
                        if angle < -45:
                            angle += 90
                        elif angle > 45:
                            angle -= 90
                        
                        # Return the detection result and contour information
                        contour_info = {
                            'contour': contour,
                            'rect': rect,
                            'area': area,
                            'center': (cx, cy)
                        }
                        
                        return (cx, cy, angle, contour_info)
        
        return None
        
    def automated_error_test(self):
        """Automated error test process"""
        print("\n🤖 Automated error test process")
        print("=" * 50)
        print("Test steps:")
        print("1. Detect the position of the red Lego brick")
        print("2. Move the robot arm to the Lego brick position")
        print("3. Detect the position of the red tape on the robot arm")
        print("4. Calculate the error and store it in the database")
        print("=" * 50)
        
        if not self.system or not self.system.cap:
            print("❌ System not initialized or camera not opened")
            return False
        
        # Check if hand-eye calibration has been completed
        if not self.system.is_calibrated:
            print("❌ System not calibrated, please run hand-eye calibration first")
            return False
        
        print("✅ Start automated error test...")
        
        while True:
            ret, frame = self.system.cap.read()
            if not ret:
                print("❌ Unable to read camera frame")
                break
            
            # Create a copy of the image for display
            display_frame = frame.copy()
            
            # Detect and display the Lego brick
            lego_result = self.system.detect_red_lego(frame)
            if lego_result:
                lego_x, lego_y, lego_angle = lego_result
                # Mark the Lego brick position on the image
                cv2.circle(display_frame, (lego_x, lego_y), 10, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Lego: ({lego_x}, {lego_y})", 
                           (lego_x + 15, lego_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Detect and display the tape
            tape_result = self.detect_red_tape(frame)
            if tape_result:
                tape_x, tape_y, tape_angle, contour_info = tape_result
                # Mark the tape position on the image
                cv2.circle(display_frame, (tape_x, tape_y), 8, (255, 0, 0), 2)
                cv2.putText(display_frame, f"Tape: ({tape_x}, {tape_y})", 
                           (tape_x + 15, tape_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Display the camera frame
            cv2.imshow('Automated Error Test - Press t to test, q to quit', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                self.system.robot.move_to_home()
                print("Exit automated error test")
                break
            elif key == ord('t'):
                self.perform_automated_error_test(frame)
        
        cv2.destroyAllWindows()
        return True

    def perform_automated_error_test(self, frame):
        """Perform automated error test"""
        print("\n🔍 Perform automated error test...")
        
        # Check if the system is initialized
        if not self.system:
            print("❌ System not initialized")
            return False
        
        # Step 1: Detect the red Lego brick
        print("🔍 Step 1: Detect the red Lego brick...")
        lego_result = self.system.detect_red_lego(frame)
        
        if not lego_result:
            print("❌ No red Lego brick detected")
            return False
        
        lego_x, lego_y, lego_angle = lego_result
        lego_pixel_coord = (lego_x, lego_y)
        
        print(f"Detected Lego brick position: {lego_pixel_coord}")
        print(f"Detected Lego brick angle: {lego_angle:.1f}°")
        
        # Calculate the physical coordinates of the Lego brick
        if not self.system.calibration:
            print("❌ System not calibrated")
            return False
        
        try:
            lego_physical_coord = self.system.calibration.pixel_to_physical(lego_pixel_coord)
            if not lego_physical_coord:
                print("❌ Unable to calculate the physical coordinates of the Lego brick")
                return False
        except ValueError as e:
            print(f"❌ Calibration error: {e}")
            return False
        
        # 应用基础补偿得到目标位置
        target_coord = self.system.calculate_grasp_position(lego_pixel_coord)
        if not target_coord:
            print("❌ 无法计算目标坐标")
            return False
        
        print(f"积木物理坐标: {lego_physical_coord}")
        print(f"目标移动坐标: {target_coord}")
        
        # 确认移动
        confirm = input("是否让机械臂移动到积木位置？(y/n): ")
        if confirm.lower() == 'n':
            print("❌ 取消测试")
            return False
        
        # 第二步：机械臂移动到目标位置
        print("🤖 第二步：机械臂移动到目标位置...")
        
        try:
            # 计算逆运动学
            from utils.InverseKinematic import inverse_kinematics
            joint_angles = inverse_kinematics(target_coord)
            
            if joint_angles is None:
                print("❌ 逆运动学计算失败，位置不可达")
                return False
            
            # 应用关节补偿
            compensated_angles = RobotCompensation.data_driven_joint_compensation(*joint_angles)
            
            # 移动到目标位置
            self.system.robot.move_to_angles(*compensated_angles, RobotCompensation.GRIPPER_OPEN_ANGLE, RobotCompensation.MOVE_TIME)
            print("✅ 机械臂移动完成")
        except Exception as e:
            print(f"❌ 机械臂移动失败: {e}")
            return False
        
        # 等待机械臂稳定
        print("⏳ 等待机械臂稳定...")
        import time
        time.sleep(2)
        
        # 第三步：检测红色胶带
        print("🔍 第三步：检测机械臂上的红色胶带...")
        
        tape_result = None
        max_attempts = 5  # 最大尝试次数
        
        for attempt in range(max_attempts):
            print(f"\n🔄 第 {attempt + 1} 次尝试检测胶带...")
            
            # 重新拍摄图像
            if not self.system.cap:
                print("❌ 摄像头未打开")
                return False
                
            ret, new_frame = self.system.cap.read()
            if not ret:
                print("❌ 无法读取新的摄像头画面")
                return False
            
            # 检测红色胶带
            tape_result = self.detect_red_tape(new_frame)
            
            if not tape_result:
                print("❌ 未检测到红色胶带")
                if attempt < max_attempts - 1:
                    retry = input("是否重新检测？(y/n): ")
                    if retry.lower() == 'n':
                        print("❌ 用户取消检测")
                        self.system.robot.move_to_home()
                        return False
                    continue
                else:
                    print("❌ 达到最大尝试次数，检测失败")
                    print("请确保机械臂上贴有红色胶带标记")
                    self.system.robot.move_to_home()
                    return False
            
            tape_x, tape_y, tape_angle, contour_info = tape_result
            tape_pixel_coord = (tape_x, tape_y)
            
            print(f"检测到胶带位置: {tape_pixel_coord}")
            print(f"检测到胶带角度: {tape_angle:.1f}°")
            
            # 在主窗口显示检测结果
            display_frame = new_frame.copy()
            
            # 绘制胶带轮廓
            cv2.drawContours(display_frame, [contour_info['contour']], -1, (255, 0, 0), 2)
            
            # 绘制中心点
            cv2.circle(display_frame, (tape_x, tape_y), 5, (255, 0, 0), -1)
            
            # 绘制最小外接矩形
            box = cv2.boxPoints(contour_info['rect'])
            box = np.array(box, dtype=np.int32)
            cv2.drawContours(display_frame, [box], 0, (0, 255, 0), 2)
            
            # 添加文本信息
            cv2.putText(display_frame, f"Tape: ({tape_x}, {tape_y})", 
                       (tape_x + 10, tape_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(display_frame, f"Angle: {tape_angle:.1f}°", 
                       (tape_x + 10, tape_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(display_frame, f"Area: {contour_info['area']:.0f}", 
                       (tape_x + 10, tape_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # 显示检测结果
            cv2.imshow('Automated Error Test - Press t to test, q to quit', display_frame)
            cv2.waitKey(1000)  # 显示1秒
            
            # 人工确认检测结果
            confirm = input("检测到的红色胶带是否正确？(y/n): ")
            if confirm.lower() == 'y':
                print("✅ 用户确认检测结果正确")
                break
            elif attempt < max_attempts - 1:
                print("🔄 重新检测...")
                tape_result = None  # 重置检测结果，确保下次循环重新检测
                continue
            else:
                print("❌ 达到最大尝试次数，检测失败")
                return False
        
        # 计算胶带的物理坐标
        try:
            tape_physical_coord_2d = self.system.calibration.pixel_to_physical(tape_pixel_coord)
            if not tape_physical_coord_2d:
                print("❌ 无法计算胶带物理坐标")
                return False
            # 添加Z坐标（使用目标位置的Z坐标）
            tape_physical_coord = (tape_physical_coord_2d[0], tape_physical_coord_2d[1], target_coord[2])
        except ValueError as e:
            print(f"❌ 标定错误: {e}")
            return False
        
        # 第四步：计算误差并存储
        print("📊 第四步：计算误差并存储...")
        
        # 计算误差（胶带位置 - 目标位置）
        x_error = tape_physical_coord[0] - target_coord[0]
        y_error = tape_physical_coord[1] - target_coord[1]
        z_error = tape_physical_coord[2] - target_coord[2] if len(tape_physical_coord) > 2 else 0
        
        print(f"目标位置: {target_coord}")
        print(f"实际位置（胶带）: {tape_physical_coord}")
        print(f"误差: ({x_error:.2f}, {y_error:.2f}, {z_error:.2f})")
        print(f"总误差: {np.sqrt(x_error**2 + y_error**2 + z_error**2):.2f} mm")
        
        # 添加误差数据到数据库
        RobotCompensation.add_error_data(target_coord, tape_physical_coord)
        self.system.robot.move_to_home()
        
        print("✅ 误差数据已添加到数据库")
        
        # 显示当前数据库信息
        info = RobotCompensation.get_database_info()
        print(f"数据库现在包含 {info['record_count']} 条记录")
        
        return True

    def perform_camera_error_test(self, frame):
        """执行摄像头误差测试"""
        print("\n🔍 执行摄像头误差测试...")
        
        # 检查系统是否已初始化
        if not self.system:
            print("❌ 系统未初始化")
            return False
        
        # 检测红色乐高积木
        result = self.system.detect_red_lego(frame)
        
        if not result:
            print("❌ 未检测到红色乐高积木")
            return False
        
        x, y, angle = result
        pixel_coord = (x, y)
        
        print(f"检测到积木位置: {pixel_coord}")
        print(f"检测到积木角度: {angle:.1f}°")
        
        # 计算物理坐标
        if not self.system.calibration:
            print("❌ 系统未标定")
            return False
        
        try:
            physical_coord = self.system.calibration.pixel_to_physical(pixel_coord)
            if not physical_coord:
                print("❌ 无法计算物理坐标")
                return False
        except ValueError as e:
            print(f"❌ 标定错误: {e}")
            print("请先运行主系统完成手眼标定")
            return False
        
        # 应用基础补偿
        compensated_coord = self.system.calculate_grasp_position(pixel_coord)
        if not compensated_coord:
            print("❌ 无法计算补偿后坐标")
            return False
        
        print(f"计算得到目标物理坐标: {physical_coord}")
        print(f"应用补偿后坐标: {compensated_coord}")
        
        # 确认移动
        confirm = input("是否让机械臂移动到检测位置？(y/n): ")
        if confirm.lower() != 'y':
            print("❌ 取消测试")
            return False
        
        # 让机械臂移动到目标位置
        print("�� 机械臂正在移动到目标位置...")
        
        try:
            # 使用系统的抓取序列（但不实际抓取）
            success = self.system.execute_grasp_sequence(compensated_coord, angle)
            if success:
                print("✅ 机械臂移动完成")
            else:
                print("❌ 机械臂移动失败")
                return False
        except Exception as e:
            print(f"❌ 机械臂移动失败: {e}")
            return False
        
        # 等待用户测量实际位置
        input("\n请测量机械臂实际到达位置，然后按回车继续...")
        
        # 输入实际位置
        try:
            print("\n请输入机械臂实际到达位置:")
            actual_x = float(input("实际X坐标 (mm): "))
            actual_y = float(input("实际Y坐标 (mm): "))
            actual_z = float(input("实际Z坐标 (mm): "))
            actual_position = (actual_x, actual_y, actual_z)
            
            # 添加误差数据
            RobotCompensation.add_error_data(compensated_coord, actual_position)
            
            # 计算误差
            x_error = actual_x - compensated_coord[0]
            y_error = actual_y - compensated_coord[1]
            z_error = actual_z - compensated_coord[2]
            
            print(f"\n📊 误差分析:")
            print(f"目标位置: {compensated_coord}")
            print(f"实际位置: {actual_position}")
            print(f"误差: ({x_error:.2f}, {y_error:.2f}, {z_error:.2f})")
            
            return True
            
        except ValueError:
            print("❌ 输入格式错误，请输入数字")
            return False
    
    def test_compensation_effect(self):
        """测试补偿效果"""
        print("\n🧪 测试补偿效果")
        print("=" * 50)
        
        if not RobotCompensation.ERROR_DATABASE:
            print("❌ 没有误差数据，无法测试补偿效果")
            return False
        
        print("✅ 发现误差数据，开始测试补偿效果")
        
        # 显示数据库信息
        info = RobotCompensation.get_database_info()
        print(f"数据库包含 {info['record_count']} 条记录")
        
        # 测试几个位置
        test_positions = [
            (0, 0, 25),
            (50, 50, 25),
            (-50, -50, 25)
        ]
        
        for position in test_positions:
            print(f"\n测试位置: {position}")
            
            # 传统补偿
            traditional = RobotCompensation.apply_position_compensation(*position)
            print(f"传统补偿: {traditional}")
            
            # 数据化补偿
            data_driven = RobotCompensation.data_driven_position_compensation(*position)
            print(f"数据化补偿: {data_driven}")
            
            # 比较差异
            diff_x = data_driven[0] - traditional[0]
            diff_y = data_driven[1] - traditional[1]
            diff_z = data_driven[2] - traditional[2]
            print(f"补偿差异: ({diff_x:.2f}, {diff_y:.2f}, {diff_z:.2f})")
    
    def run_interactive_test(self):
        """运行交互式测试"""
        print("🧪 误差补偿交互式测试")
        print("=" * 60)
        
        # 初始化系统
        if not self.initialize_system():
            return False
        
        # 加载现有数据库
        try:
            RobotCompensation.load_error_database()
        except:
            print("⚠️ 未找到现有数据库文件，将创建新的数据库")
        
        while True:
            print("\n请选择测试选项:")
            print("1. 手动误差录入")
            print("2. 摄像头误差测试 (需要先进行手眼标定)")
            print("3. 自动化误差测试 (需要先进行手眼标定)")
            print("4. 测试补偿效果")
            print("5. 查看误差数据库")
            print("6. 保存数据库")
            print("7. 加载数据库")
            print("8. 退出")
            
            choice = input("请输入选项 (1-8): ")
            
            if choice == '1':
                self.manual_error_test()
            elif choice == '2':
                self.camera_based_error_test()
            elif choice == '3':
                self.automated_error_test()
            elif choice == '4':
                self.test_compensation_effect()
            elif choice == '5':
                print("\n📊 当前误差数据库:")
                RobotCompensation.print_error_database()
            elif choice == '6':
                RobotCompensation.save_error_database()
            elif choice == '7':
                RobotCompensation.load_error_database()
            elif choice == '8':
                print("👋 退出测试")
                break
            else:
                print("❌ 无效选项，请重新选择")

def main():
    """主函数"""
    tester = ErrorCompensationTester()
    tester.run_interactive_test()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lego brick recognition and grasping system
Integrates hand-eye calibration, inverse kinematics, forward control, and mechanical error compensation
"""

import cv2
import numpy as np
import time
import os
from typing import Tuple, Optional

# Import custom modules
from utils.HandEyeCalibration import HandEyeCalibration
from utils.ForwardController import BraccioRobot
from utils.InverseKinematic import inverse_kinematics
from utils.RobotCompensation import (
    apply_joint_compensation, apply_position_compensation,
    GRIP_HEIGHT, GRIPPER_OPEN_ANGLE, GRIPPER_CLOSE_ANGLE, GRIPPER_GRAB_ANGLE,
    MOVE_TIME, CAMERA_INDEX, SAFE_HEIGHT, WORKTABLE_HEIGHT,
    DEBUG_MODE, SHOW_CAMERA_FEED, SAVE_DETECTION_IMAGES, IMAGE_SAVE_PATH,
    SERIAL_PORT, BAUD_RATE, TIMEOUT, print_compensation_status
)

class LegoGraspingSystem:
    """
    Main class for Lego brick recognition and grasping system
    """
    
    def __init__(self, camera_index: int = CAMERA_INDEX, 
                 serial_port: str = SERIAL_PORT,
                 baud_rate: int = BAUD_RATE,
                 timeout: int = TIMEOUT):
        """
        Initialize the system
        
        Args:
            camera_index: Camera index
            serial_port: Serial port
            baud_rate: Baud rate
            timeout: Timeout
        """
        self.camera_index = camera_index
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.timeout = timeout
        
        # Initialize components
        self.calibration = HandEyeCalibration(camera_index)
        self.robot = BraccioRobot(serial_port, baud_rate, timeout)
        self.cap = None
        
        # System status
        self.is_calibrated = False
        self.is_robot_connected = False
        self.is_camera_open = False
        
        # Try to load existing calibration data
        if self.calibration.load_calibration():
            self.is_calibrated = True
            print("✅ Existing hand-eye calibration data loaded")
            # Calibration information is already printed in the load_calibration method
        else:
            print("⚠️ No existing calibration data found, need to re-calibrate")
        
        # Create image save directory
        if SAVE_DETECTION_IMAGES:
            os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)
        
        print("🤖 Lego brick recognition and grasping system initialized")
        print_compensation_status()
    
    def initialize_system(self) -> bool:
        """
        Initialize system components
        
        Returns:
            bool: Whether initialization is successful
        """
        print("🔧 Initializing system components...")
        
        # Check if the robot arm is connected
        if self.robot.s and self.robot.s.is_open:
            self.is_robot_connected = True
            print("✅ Robot arm connected successfully")
        else:
            print("❌ Robot arm connection failed")
            return False
        
        # Open the camera
        if self.open_camera():
            self.is_camera_open = True
            print("✅ Camera opened successfully")
        else:
            print("❌ Camera opening failed")
            return False
        
        return True
    
    def open_camera(self) -> bool:
        """
        Open the camera
        
        Returns:
            bool: Whether the camera is successfully opened
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                print(f"❌ Unable to open camera {self.camera_index}")
                return False
            return True
        except Exception as e:
            print(f"❌ Error opening camera: {e}")
            return False
    
    def close_camera(self):
        """Close the camera"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.is_camera_open = False
            print("📷 Camera closed")
    
    def detect_red_lego(self, frame) -> Optional[Tuple[int, int, float]]:
        """
        Detect the red Lego brick
        
        Args:
            frame: Input image frame
            
        Returns:
            Optional[Tuple[int, int, float]]: The center pixel coordinates and angle of the detected red Lego brick,
            formatted as (x, y, angle), if not detected, return None
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define the red range (HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        # Create red mask
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2
        
        # Morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area = 0
        if contours:
            # Find the largest contour (assuming it is a Lego brick)
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # Area threshold, filter out too small areas
            if area > 500:  # Can be adjusted according to actual situation
                # Calculate the center point of the contour
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # New: Calculate the angle of the Lego brick
                    rect = cv2.minAreaRect(largest_contour)
                    angle = rect[2]  # Get the angle
                    
                    # Angle normalization: ensure the angle is between -90 and 90 degrees
                    if angle < -45:
                        angle += 90
                    elif angle > 45:
                        angle -= 90
                    
                    return (cx, cy, angle)
        
        return None
    
    def perform_hand_eye_calibration(self) -> bool:
        """
        Perform hand-eye calibration
        
        Returns:
            bool: Whether the calibration is successful
        """
        print("\n" + "="*60)
        print("🔧 Start hand-eye calibration process")
        print("="*60)
        
        print("📋 Calibration steps:")
        print("1. Please place the red Lego brick in the camera's field of view")
        print("2. The system will take a picture and recognize the Lego brick position")
        print("3. Please measure the physical coordinates of the Lego brick relative to the center point of the robot arm base")
        print("4. Input the physical coordinates to complete the calibration")
        print("-"*60)
        
        # Get the user input physical coordinates
        try:
            print("Please input the physical coordinates of the red Lego brick (relative to the center point of the robot arm base):")
            x = float(input("X coordinate (millimeters): "))
            y = float(input("Y coordinate (millimeters): "))
            physical_coord = (x, y)
            
            print(f"📏 Input physical coordinates: ({x}, {y}) millimeters")
            
        except ValueError:
            print("❌ Input format error, please enter valid numbers")
            return False
        
        # Perform calibration
        if self.calibration.calibrate(physical_coord):
            self.is_calibrated = True
            print("✅ Hand-eye calibration completed!")
            
            # Automatically save calibration data
            if self.calibration.save_calibration():
                print("✅ Calibration data saved automatically")
            
            return True
        else:
            print("❌ Hand-eye calibration failed")
            return False
    
    def calculate_grasp_position(self, pixel_coord: Tuple[int, int]) -> Optional[Tuple[float, float, float]]:
        """
        Calculate grasp position
        
        Args:
            pixel_coord: Pixel coordinates
            
        Returns:
            Optional[Tuple[float, float, float]]: Grasp position (x, y, z)
        """
        if not self.is_calibrated:
            print("❌ Hand-eye calibration not completed")
            return None
        
        # Convert pixel coordinates to physical coordinates
        physical_coord_origin = self.calibration.pixel_to_physical(pixel_coord)
        x, y = physical_coord_origin
        z = WORKTABLE_HEIGHT + GRIP_HEIGHT  # Define z coordinate

        from utils.RobotCompensation import data_driven_position_compensation
        compensated = data_driven_position_compensation(x, y, z)
        
        return (compensated[0], compensated[1], compensated[2])
    
    def check_reachability(self, position: Tuple[float, float, float]) -> bool:
        """
        Check if position is reachable
        
        Args:
            position: Target position (x, y, z)
            
        Returns:
            bool: Whether reachable
        """
        
        # Check if position is within reasonable range
        x, y, z = position
        r = np.sqrt(x**2 + y**2)
        r_min = 120
        r_max = 450
        z_min = 0
        z_max = 300

        if not (r_min <= r <= r_max) or not (z_min <= z <= z_max):
            print(f"⚠️ Position {position} exceeds the robot arm's working circle range")
            print(f"Suggested range: r: [{r_min}, {r_max}], Z: [{z_min}, {z_max}]")
            return False
        
        else :
            return True
    
    def execute_grasp_sequence(self, position: Tuple[float, float, float], angle: float = 0) -> bool:
        """
        Execute grasp sequence
        
        Args:
            position: Target position (x, y, z)
            angle: Brick angle, used to adjust gripper orientation (degrees)
            
        Returns:
            bool: Whether grasp is successful
        """
        print(f"🤖 Starting grasp sequence, target position: {position}, brick angle: {angle:.1f}°")
        
        try:
            # 1. Calculate inverse kinematics
            joint_angles = inverse_kinematics(position)
            if joint_angles is None:
                print("❌ Inverse kinematics calculation failed, position not reachable")
                return False
            
            # 2. Apply data-driven joint compensation
            from utils.RobotCompensation import data_driven_joint_compensation
            compensated_angles = data_driven_joint_compensation(*joint_angles)
            
            print(f"🔧 Original joint angles: {joint_angles}")
            print(f"🔧 Compensated joint angles: {compensated_angles}")
            
            # 3. Calculate gripper angle adjustment
            # Use the 5th joint (twist joint) to adjust gripper orientation
            base, shoulder, elbow, wrist, twist = compensated_angles
            
            # Adjust twist joint based on brick angle
            if abs(angle) > 5:  # Only adjust if angle deviation is greater than 5 degrees
                adjusted_twist = twist + angle
                # Ensure angle is within reasonable range
                if adjusted_twist > 180:
                    adjusted_twist -= 180
                elif adjusted_twist < 0:
                    adjusted_twist += 180
                
                compensated_angles = (base, shoulder, elbow, wrist, adjusted_twist)
                print(f"🔄 Adjusting gripper angle: original twist angle {twist}° -> adjusted {adjusted_twist}°")
            
            # 4. Move to safe height
            safe_position = (position[0], position[1], position[2] + SAFE_HEIGHT)
            safe_angles = inverse_kinematics(safe_position)
            if safe_angles:
                safe_compensated = data_driven_joint_compensation(*safe_angles)
                self.robot.move_to_angles(safe_compensated[0], safe_compensated[1], safe_compensated[2], safe_compensated[3], safe_compensated[4], GRIPPER_OPEN_ANGLE, MOVE_TIME)
                time.sleep(2)
            
            # 5. Move to grasp position (using adjusted angles)
            self.robot.move_to_angles(*compensated_angles, GRIPPER_OPEN_ANGLE, MOVE_TIME)
            time.sleep(2)
            
            # 6. Close gripper
            self.robot.move_to_angles(*compensated_angles, GRIPPER_CLOSE_ANGLE, MOVE_TIME)
            time.sleep(1)
            
            # 7. Move to safe height
            if safe_angles:
                self.robot.move_to_angles(safe_compensated[0], safe_compensated[1], safe_compensated[2], safe_compensated[3], safe_compensated[4], GRIPPER_CLOSE_ANGLE, MOVE_TIME)
                time.sleep(2)

            # 8. Return to initial position
            self.robot.move_to_angles(0, 90, 180, 90, 0, GRIPPER_CLOSE_ANGLE, MOVE_TIME)
            time.sleep(2)

            # 9. Release gripper
            self.robot.move_to_angles(0, 90, 180, 90, 0, GRIPPER_OPEN_ANGLE, MOVE_TIME)
            time.sleep(2)
            
            print("✅ Grasp sequence execution completed")
            return True
            
        except Exception as e:
            print(f"❌ Grasp sequence execution failed: {e}")
            return False
    
    def run_detection_loop(self):
        """
        Run detection and grasping loop
        """
        print("\n" + "="*60)
        print("🔄 Starting detection and grasping loop")
        print("="*60)
        print("Press 'q' to quit loop, press 'r' to recalibrate")
        
        if not self.is_calibrated:
            print("❌ Hand-eye calibration not completed, please calibrate first")
            return
        
        while True:
            # Read camera frame
            if not self.cap:
                print("❌ Camera not opened")
                break
                
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Unable to read camera frame")
                break
            
            # Detect red Lego brick
            result = self.detect_red_lego(frame)

            # Display detection results
            if SHOW_CAMERA_FEED:
                display_frame = frame.copy()
                
                if result:
                    x, y, angle = result
                    pixel_coord = (x, y)  # Maintain backward compatibility
                    
                    # Mark detected brick
                    cv2.circle(display_frame, pixel_coord, 15, (0, 255, 0), 3)
                    cv2.putText(display_frame, f"Lego: ({x}, {y}), Angle: {angle:.1f}°", 
                            (pixel_coord[0] + 20, pixel_coord[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Draw angle indicator line
                    line_length = 50
                    end_x = int(x + line_length * np.cos(np.radians(angle)))
                    end_y = int(y + line_length * np.sin(np.radians(angle)))
                    cv2.line(display_frame, (x, y), (end_x, end_y), (255, 0, 0), 3)
                    
                    # Calculate physical coordinates
                    grasp_position = self.calculate_grasp_position(pixel_coord)
                    if grasp_position:
                        cv2.putText(display_frame, f"Pos: ({grasp_position[0]:.1f}, {grasp_position[1]:.1f})", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Add control hints
                cv2.putText(display_frame, "Press 'g' to grasp, 'q' to quit, 'r' to recalibrate", 
                           (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow("Lego Brick Detection", display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("👋 User exited detection loop")
                break
            elif key == ord('r'):
                print("🔄 Recalibrating")
                if self.perform_hand_eye_calibration():
                    continue
                else:
                    break
            elif key == ord('g') and result:
                print("🤖 Starting grasp...")
                
                # Calculate grasp position
                x, y, angle = result
                pixel_coord = (x, y)
                grasp_position = self.calculate_grasp_position(pixel_coord)
                if not grasp_position:
                    print("❌ Unable to calculate grasp position")
                    continue
                
                # Check reachability
                if not self.check_reachability(grasp_position):
                    print("⚠️ Target position not reachable, skipping this grasp")
                    continue
                
                # Execute grasp
                if self.execute_grasp_sequence(grasp_position, angle):
                    print("✅ Grasp successful!")
                    # Wait for a while before next detection
                    time.sleep(3)
                else:
                    print("❌ Grasp failed")
        
        cv2.destroyAllWindows()
    
    def run(self):
        """
        Run main program
        """
        print("🚀 Starting Lego brick recognition and grasping system")
        
        # Initialize system
        if not self.initialize_system():
            print("❌ System initialization failed")
            return
        
        try:
            # Check if hand-eye calibration data already exists
            if self.is_calibrated:
                print("✅ Existing hand-eye calibration data detected, using directly")
                print("💡 To recalibrate, press 'r' key in detection loop")
            else:
                print("⚠️ No hand-eye calibration data detected, need to recalibrate")
                # Perform hand-eye calibration
                if not self.perform_hand_eye_calibration():
                    print("❌ Hand-eye calibration failed, program exiting")
                    return
            
            # Run detection and grasping loop
            self.run_detection_loop()
            
        except KeyboardInterrupt:
            print("\n⚠️ User interrupted program")
        except Exception as e:
            print(f"❌ Program runtime error: {e}")
        finally:
            # Clean up resources
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("🧹 Cleaning up resources...")
        
        # Close camera
        self.close_camera()
        
        # Close robot arm connection
        if self.robot:
            self.robot.close_serial()
        
        # Close all OpenCV windows
        cv2.destroyAllWindows()
        
        print("✅ Resource cleanup completed")



    def simple_error_test(self):
        """
        Simple error test function
        
        Usage:
        1. Place target object at known position
        2. Run this function
        3. Manually input actual reached position
        4. Record error data
        """
        print("🧪 Starting simple error test...")
        print("=" * 50)
        print("Test steps:")
        print("1. Place target object at known position")
        print("2. Press 't' to perform error test")
        print("3. Press 'v' to view error database")
        print("4. Press 'q' to exit test")
        print("=" * 50)
        
        # Initialize system (without recalibration)
        if not self.initialize_system():
            print("❌ System initialization failed")
            return False
        
        # Check if already calibrated
        if not hasattr(self, 'calibration') or not self.calibration:
            print("❌ System not calibrated, please run normal hand-eye calibration first")
            return False
        
        print("✅ System initialization completed, starting error test...")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("Exiting error test")
                break
            elif key == ord('t'):
                # Perform error test
                print("\n🎯 Please input target position:")
                try:
                    target_x = float(input("Target X coordinate (mm): "))
                    target_y = float(input("Target Y coordinate (mm): "))
                    target_z = float(input("Target Z coordinate (mm): "))
                    target_position = (target_x, target_y, target_z)
                    self.test_single_position(target_position)
                except ValueError:
                    print("❌ Input format error, please enter numbers")
            elif key == ord('v'):
                # View error database
                from utils.RobotCompensation import print_error_database
                print_error_database()
            
            # Display camera frame
            if self.cap:
                ret, frame = self.cap.read()
                if ret:
                    cv2.imshow('Error Test - Press t to test, v to view, q to quit', frame)
        
        cv2.destroyAllWindows()
        return True

    def test_single_position(self, target_position):
        """
        Test accuracy of single position
        
        Args:
            target_position: Target position
            
        Returns:
            bool: Whether test is successful
        """
        try:
            print(f"🎯 Testing position accuracy: {target_position}")
            
            # 1. Calculate inverse kinematics
            joint_angles = inverse_kinematics(target_position)
            if joint_angles is None:
                print("❌ Inverse kinematics calculation failed")
                return False
            
            # 2. Apply joint compensation
            compensated_angles = apply_joint_compensation(*joint_angles)
            
            print(f"🔧 Target position: {target_position}")
            print(f"  Joint angles: {compensated_angles}")
            
            # 3. Move to target position
            self.robot.move_to_angles(*compensated_angles, GRIPPER_OPEN_ANGLE, MOVE_TIME)
            time.sleep(3)
            
            # 4. Get user input for actual position
            print("\n📏 Please measure the actual position reached by robot arm end:")
            try:
                actual_x = float(input("Actual X coordinate (mm): "))
                actual_y = float(input("Actual Y coordinate (mm): "))
                actual_z = float(input("Actual Z coordinate (mm): "))
                actual_position = (actual_x, actual_y, actual_z)
                
                # 5. Add error data to database
                from utils.RobotCompensation import add_error_data
                add_error_data(target_position, actual_position)
                
                print(f"✅ Error data recorded")
                
                # 6. Return to initial position
                self.robot.move_to_angles(0, 90, 180, 90, 0, GRIPPER_OPEN_ANGLE, MOVE_TIME)
                time.sleep(2)
                
                return True
                
            except ValueError:
                print("❌ Input format error")
                # Return to initial position even if input is wrong
                self.robot.move_to_angles(0, 90, 180, 90, 0, GRIPPER_OPEN_ANGLE, MOVE_TIME)
                time.sleep(2)
                return False
            
        except Exception as e:
            print(f"❌ Position accuracy test failed: {e}")
            # Return to initial position even when exception occurs
            try:
                self.robot.move_to_angles(0, 90, 180, 90, 0, GRIPPER_OPEN_ANGLE, MOVE_TIME)
                time.sleep(2)
            except:
                pass
            return False
    
def main():
    """Main function"""
    print("="*60)
    print("🤖 Lego Brick Recognition and Grasping System")
    print("="*60)
    
    # Create system instance
    system = LegoGraspingSystem()
    
    # Run system
    system.run()

if __name__ == "__main__":
    main() 
# AuSysRob: Lego Brick Recognition and Grasping System

## Overview

**AuSysRob1** is an integrated system for recognizing and grasping Lego bricks using a robotic arm. It combines computer vision, hand-eye calibration, inverse kinematics, forward control, and mechanical error compensation to achieve robust and precise pick-and-place operations.

## Features

- **Lego Brick Detection:** Real-time recognition of red Lego bricks using OpenCV.
- **Hand-Eye Calibration:** Interactive calibration process to map camera coordinates to robot coordinates.
- **Inverse Kinematics:** Calculates joint angles for target positions.
- **Forward Control:** Sends commands to the Braccio robotic arm via serial communication.
- **Error Compensation:** Data-driven and traditional compensation for both position and joint errors.
- **Error Database:** Collects and manages error data to improve compensation accuracy.
- **Interactive GUI:** Visual feedback and keyboard controls for grasping and calibration.

## Project Structure

```
AuSysRob/
│
├── LegoGraspingSystem.py         # Main system logic and entry point
├── CorrectionTool.py             # Error correction and compensation tools
├── error_database.json           # Collected error data for compensation
├── hand_eye_calibration.json     # Saved hand-eye calibration data
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
│
└── utils/
    ├── __init__.py
    ├── ForwardController.py      # Robot forward kinematics and control
    ├── HandEyeCalibration.py     # Hand-eye calibration logic
    ├── InverseKinematic.py       # Inverse kinematics calculations
    └── RobotCompensation.py      # Error compensation and database management
```

## Requirements

- Python 3.7+
- Hardware: Braccio robotic arm (or compatible), camera
- OS: Windows 10+ (other OS may require serial port adjustments)

### Python Dependencies

All required packages are listed in `requirements.txt`. Main dependencies include:
- `opencv-python`
- `numpy`
- `sympy`
- `pyserial`

Install with:
```bash
pip install -r requirements.txt
```

## Getting Started

### 1. Hardware Setup

- Connect the Braccio robotic arm to your PC via USB.
- Connect and position the camera so it can view the workspace.

### 2. Calibration

Before running the main system, perform hand-eye calibration to align the camera and robot coordinate systems.

### 3. Running the System

```bash
python LegoGraspingSystem.py
```

- The system will attempt to load existing calibration data.
- If not found, it will guide you through the calibration process.
- The main loop will display the camera feed and detected Lego bricks.
- Keyboard controls:
  - `g`: Grasp the detected brick
  - `r`: Recalibrate hand-eye
  - `q`: Quit

### 4. Error Compensation

- The system supports both traditional and data-driven error compensation.
- Use the built-in error test functions to collect and update error data for improved accuracy.

## Main Functionalities

- **Hand-Eye Calibration:**  
  Interactive process to input physical coordinates for a detected brick and save the mapping.

- **Detection and Grasping Loop:**  
  Continuously detects red Lego bricks, calculates grasp positions, checks reachability, and executes the grasp sequence.

- **Error Testing:**  
  Tools for testing and recording the robot's actual vs. target positions to build an error compensation database.

## Customization

- Modify color detection parameters in `LegoGraspingSystem.py` for different brick colors.
- Adjust robot parameters in `utils/ForwardController.py` and `utils/InverseKinematic.py` for different hardware.

## Troubleshooting

- **Serial Port Issues:**  
  Ensure the correct port is set in `SERIAL_PORT` (default in `RobotCompensation.py`).
- **Camera Not Detected:**  
  Check `CAMERA_INDEX` and ensure the camera is properly connected.
- **Grasping Inaccuracy:**  
  Use the error test and compensation tools to improve accuracy.

## License

This project is for academic and research purposes. For commercial use, please contact the author.

---
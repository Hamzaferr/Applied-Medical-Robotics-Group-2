# Applied Medical Robotics – Group 2
Benchtop 2-link planar robot arm + Dash GUI + Arduino PID control, developed as a proxy for automated dermatological laser probe scanning.

## Repository contents (high level)
This repo contains:
- **Python GUI + kinematics + trajectory tools** (PC side)
- **Arduino firmware** for joint control and trajectory playback
- **CAD export** of the robot body

## Files and what they do

### Main robot pipeline (PC / Python)
- `arduino_gui.py`  
  Main Dash GUI for selecting targets and generating paths, calculating IK, and sending trajectories to the Arduino.

- `coords.py`  
  Coordinate mapping utilities (Paper frame ↔ Robot frame).

- `FK.py`  
  Forward kinematics functions for the 2-link planar arm.

- `IK.py`  
  Closed-form (analytical) inverse kinematics for elbow-up / elbow-down solutions.

- `jacobian_ik.py`  
  Jacobian-based (differential) iterative IK solver.

- `shapes.py`  
  Path/shape primitives (e.g., line, square) in paper coordinates.

- `trajectory_planner.py`  
  Waypoint generation, timing assignment, and any corner-slowdown / smoothing logic used for execution.

- `serial_reader.py`  
  Serial communication helper (send/receive data to/from Arduino).

- `workspace_plotter.py`  
  Plotting utilities for workspace/targets/paths (used for debugging/visualisation).

- `evaluation.py`  
  Analysis scripts for generating metrics/plots used in the report (e.g., PID RMSE, point-to-point error summaries).

### PID test utilities (Python)
These scripts were created specifically for the standalone PID step-response experiments reported in the write-up:
- `pid_module.py`  
  Shared PID analysis utilities / processing functions.
- `pid_gui_tabbed.py`  
  PID testing interface/scripts used to run and record step tests.

### Arduino firmware
- `robot_arduino.ino`  
  Main Arduino firmware: encoder reading, 100 Hz PID control, and timed trajectory playback (waypoint interpolation + feedforward).

### CAD
- `CAD Model.stl`  
  Exported STL model of the assembled robot.  
  **Note:** the original STEP file could not be uploaded due to GitHub file size limits. If needed, a downloadable STEP can be provided separately.

## How to run (minimal)
1. Upload `robot_arduino.ino` to an Arduino Uno (wired to motors/encoders via the motor driver as per the report wiring diagram).
2. Run the GUI on PC:
   - Ensure required Python packages are installed (Dash + common scientific stack).
   - Launch `arduino_gui.py` and use the interface to compute IK and send trajectories.

## Notes
- The project is a **benchtop proof-of-concept** and does not include laser hardware.
- The paper workspace is 12 × 12 cm with a paper-to-robot coordinate mapping defined in the report.

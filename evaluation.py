"""
Author:Hamza
Robot Arm Evaluation Module

Provides comprehensive testing and evaluation:
1. Joint Space Repeatability - test each motor's accuracy and repeatability
2. Cartesian Space Accuracy - test IK positioning accuracy
3. Trajectory Deviation - measure deviation from ideal paths

Author: Yagmur
Date: December 2025
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import json

from coords import L1, L2, paper_to_robot, robot_to_paper
from FK import get_end_effector_position
from IK import inverse_kinematics


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class JointTestResult:
    """Result of a single joint movement test."""
    motor: int
    target_angle: float
    actual_angle: float
    error: float
    settle_time_ms: float
    
    def to_dict(self):
        return {
            'motor': self.motor,
            'target': self.target_angle,
            'actual': self.actual_angle,
            'error': self.error,
            'settle_time': self.settle_time_ms
        }


@dataclass
class CartesianTestResult:
    """Result of a single Cartesian positioning test."""
    target_paper_x: float
    target_paper_y: float
    actual_paper_x: float
    actual_paper_y: float
    error_x: float
    error_y: float
    error_euclidean: float
    target_theta1: float
    target_theta2: float
    actual_theta1: float
    actual_theta2: float
    theta1_error: float
    theta2_error: float
    
    def to_dict(self):
        return {
            'target_x': self.target_paper_x,
            'target_y': self.target_paper_y,
            'actual_x': self.actual_paper_x,
            'actual_y': self.actual_paper_y,
            'error_x': self.error_x,
            'error_y': self.error_y,
            'error_euclidean': self.error_euclidean,
            'target_theta1': self.target_theta1,
            'target_theta2': self.target_theta2,
            'actual_theta1': self.actual_theta1,
            'actual_theta2': self.actual_theta2,
            'theta1_error': self.theta1_error,
            'theta2_error': self.theta2_error
        }


@dataclass
class TrajectoryTestResult:
    """Result of a trajectory deviation test."""
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    num_samples: int
    deviations: List[float]  # Perpendicular distance from ideal line
    max_deviation: float
    mean_deviation: float
    rms_deviation: float
    
    def to_dict(self):
        return {
            'start': self.start_point,
            'end': self.end_point,
            'num_samples': self.num_samples,
            'max_deviation': self.max_deviation,
            'mean_deviation': self.mean_deviation,
            'rms_deviation': self.rms_deviation
        }


@dataclass
class EvaluationResults:
    """Complete evaluation results."""
    timestamp: str = ""
    joint_tests: List[JointTestResult] = field(default_factory=list)
    cartesian_tests: List[CartesianTestResult] = field(default_factory=list)
    trajectory_tests: List[TrajectoryTestResult] = field(default_factory=list)
    
    # Summary statistics
    joint_mean_error_m1: float = 0.0
    joint_mean_error_m2: float = 0.0
    joint_max_error_m1: float = 0.0
    joint_max_error_m2: float = 0.0
    joint_std_m1: float = 0.0  # Repeatability
    joint_std_m2: float = 0.0
    
    cartesian_mean_error: float = 0.0
    cartesian_max_error: float = 0.0
    cartesian_std: float = 0.0
    
    def compute_statistics(self):
        """Compute summary statistics from test results."""
        # Joint statistics
        m1_errors = [t.error for t in self.joint_tests if t.motor == 1]
        m2_errors = [t.error for t in self.joint_tests if t.motor == 2]
        
        if m1_errors:
            self.joint_mean_error_m1 = np.mean(np.abs(m1_errors))
            self.joint_max_error_m1 = np.max(np.abs(m1_errors))
            self.joint_std_m1 = np.std(m1_errors)
        
        if m2_errors:
            self.joint_mean_error_m2 = np.mean(np.abs(m2_errors))
            self.joint_max_error_m2 = np.max(np.abs(m2_errors))
            self.joint_std_m2 = np.std(m2_errors)
        
        # Cartesian statistics
        if self.cartesian_tests:
            euclidean_errors = [t.error_euclidean for t in self.cartesian_tests]
            self.cartesian_mean_error = np.mean(euclidean_errors)
            self.cartesian_max_error = np.max(euclidean_errors)
            self.cartesian_std = np.std(euclidean_errors)
    
    def to_dict(self):
        return {
            'timestamp': self.timestamp,
            'joint_tests': [t.to_dict() for t in self.joint_tests],
            'cartesian_tests': [t.to_dict() for t in self.cartesian_tests],
            'trajectory_tests': [t.to_dict() for t in self.trajectory_tests],
            'summary': {
                'joint_m1': {
                    'mean_error': self.joint_mean_error_m1,
                    'max_error': self.joint_max_error_m1,
                    'repeatability_std': self.joint_std_m1
                },
                'joint_m2': {
                    'mean_error': self.joint_mean_error_m2,
                    'max_error': self.joint_max_error_m2,
                    'repeatability_std': self.joint_std_m2
                },
                'cartesian': {
                    'mean_error': self.cartesian_mean_error,
                    'max_error': self.cartesian_max_error,
                    'std': self.cartesian_std
                }
            }
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# =============================================================================
# TEST CONFIGURATIONS
# =============================================================================

# Default test angles for joint repeatability
DEFAULT_JOINT_TEST_ANGLES = [30, 60, 90, -30, -60]

# Default test points for Cartesian accuracy (paper coordinates)
DEFAULT_CARTESIAN_TEST_POINTS = [
    (4, 4),   # Bottom-left region
    (8, 4),   # Bottom-right region
    (4, 8),   # Top-left region
    (8, 8),   # Top-right region
    (6, 6),   # Center
    (3, 6),   # Left edge
    (9, 6),   # Right edge
    (6, 3),   # Bottom region
    (6, 9),   # Top region
]

# Number of repetitions for each test
DEFAULT_REPETITIONS = 3

# Settle time (ms) to wait after sending command
DEFAULT_SETTLE_TIME = 1500


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def perpendicular_distance(point: Tuple[float, float], 
                          line_start: Tuple[float, float],
                          line_end: Tuple[float, float]) -> float:
    """
    Calculate perpendicular distance from a point to a line segment.
    
    Uses formula: |((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)| / sqrt((y2-y1)^2 + (x2-x1)^2)
    """
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
    denominator = np.sqrt((y2-y1)**2 + (x2-x1)**2)
    
    if denominator < 1e-10:
        # Line has zero length, return distance to start point
        return np.sqrt((x0-x1)**2 + (y0-y1)**2)
    
    return numerator / denominator


def compute_actual_paper_position(theta1: float, theta2: float) -> Tuple[float, float]:
    """Compute paper coordinates from joint angles using FK."""
    robot_x, robot_y = get_end_effector_position(L1, L2, theta1, theta2)
    paper_x, paper_y = robot_to_paper(robot_x, robot_y)
    return paper_x, paper_y


def compute_target_angles(paper_x: float, paper_y: float, 
                         elbow: str = 'down') -> Optional[Tuple[float, float]]:
    """Compute target joint angles from paper coordinates using IK."""
    robot_x, robot_y = paper_to_robot(paper_x, paper_y)
    result = inverse_kinematics(L1, L2, robot_x, robot_y, elbow=elbow)
    
    if result is None:
        return None
    
    return float(result[0]), float(result[1])


# =============================================================================
# EVALUATION CLASS
# =============================================================================

class RobotEvaluator:
    """
    Manages robot evaluation tests.
    
    Works with the serial reader to send commands and read positions.
    """
    
    def __init__(self):
        self.results = EvaluationResults()
        self.is_running = False
        self.current_test = ""
        self.progress = 0
        self.status_callback = None
    
    def set_status_callback(self, callback):
        """Set callback for status updates: callback(message, progress_percent)"""
        self.status_callback = callback
    
    def _update_status(self, message: str, progress: int):
        """Update status and progress."""
        self.current_test = message
        self.progress = progress
        if self.status_callback:
            self.status_callback(message, progress)
    
    def run_joint_test_single(self, reader, motor: int, target_angle: float,
                              settle_time_ms: int = DEFAULT_SETTLE_TIME) -> Optional[JointTestResult]:
        """
        Run a single joint positioning test.
        
        Args:
            reader: SerialReader instance
            motor: 1 or 2
            target_angle: Target angle in degrees
            settle_time_ms: Time to wait for settling
            
        Returns:
            JointTestResult or None if failed
        """
        if not reader.is_connected():
            return None
        
        # Send command based on which motor
        if motor == 1:
            reader.send_angles(target_angle, 0)
        else:
            reader.send_angles(0, target_angle)
        
        # Wait for settling
        time.sleep(settle_time_ms / 1000.0)
        
        # Read actual position
        state = reader.get_state()
        
        if motor == 1:
            actual_angle = state.theta1_current
        else:
            actual_angle = state.theta2_current
        
        error = actual_angle - target_angle
        
        return JointTestResult(
            motor=motor,
            target_angle=target_angle,
            actual_angle=actual_angle,
            error=error,
            settle_time_ms=settle_time_ms
        )
    
    def run_joint_repeatability_test(self, reader, 
                                     test_angles: List[float] = None,
                                     repetitions: int = DEFAULT_REPETITIONS,
                                     settle_time_ms: int = DEFAULT_SETTLE_TIME) -> List[JointTestResult]:
        """
        Run full joint repeatability test for both motors.
        
        Tests each angle multiple times, returning to home between tests.
        """
        if test_angles is None:
            test_angles = DEFAULT_JOINT_TEST_ANGLES
        
        results = []
        total_tests = 2 * len(test_angles) * repetitions * 2  # 2 motors, angles, reps, go+return
        current = 0
        
        self.is_running = True
        
        for motor in [1, 2]:
            for angle in test_angles:
                for rep in range(repetitions):
                    if not self.is_running:
                        return results
                    
                    # Test going TO the angle
                    self._update_status(f"M{motor}: {angle}° (rep {rep+1}/{repetitions})", 
                                       int(100 * current / total_tests))
                    result = self.run_joint_test_single(reader, motor, angle, settle_time_ms)
                    if result:
                        results.append(result)
                    current += 1
                    
                    # Test returning to HOME
                    self._update_status(f"M{motor}: return to 0° (rep {rep+1}/{repetitions})",
                                       int(100 * current / total_tests))
                    result = self.run_joint_test_single(reader, motor, 0, settle_time_ms)
                    if result:
                        results.append(result)
                    current += 1
        
        self.is_running = False
        self._update_status("Joint test complete", 100)
        return results
    
    def run_cartesian_test_single(self, reader, paper_x: float, paper_y: float,
                                  elbow: str = 'down',
                                  settle_time_ms: int = DEFAULT_SETTLE_TIME) -> Optional[CartesianTestResult]:
        """
        Run a single Cartesian positioning test.
        """
        if not reader.is_connected():
            return None
        
        # Compute target angles
        target_angles = compute_target_angles(paper_x, paper_y, elbow)
        if target_angles is None:
            return None
        
        target_theta1, target_theta2 = target_angles
        
        # Send command
        reader.send_angles(target_theta1, target_theta2)
        
        # Wait for settling
        time.sleep(settle_time_ms / 1000.0)
        
        # Read actual position
        state = reader.get_state()
        actual_theta1 = state.theta1_current
        actual_theta2 = state.theta2_current
        
        # Compute actual paper position
        actual_paper_x, actual_paper_y = compute_actual_paper_position(actual_theta1, actual_theta2)
        
        # Compute errors
        error_x = actual_paper_x - paper_x
        error_y = actual_paper_y - paper_y
        error_euclidean = np.sqrt(error_x**2 + error_y**2)
        
        return CartesianTestResult(
            target_paper_x=paper_x,
            target_paper_y=paper_y,
            actual_paper_x=actual_paper_x,
            actual_paper_y=actual_paper_y,
            error_x=error_x,
            error_y=error_y,
            error_euclidean=error_euclidean,
            target_theta1=target_theta1,
            target_theta2=target_theta2,
            actual_theta1=actual_theta1,
            actual_theta2=actual_theta2,
            theta1_error=actual_theta1 - target_theta1,
            theta2_error=actual_theta2 - target_theta2
        )
    
    def run_cartesian_accuracy_test(self, reader,
                                    test_points: List[Tuple[float, float]] = None,
                                    elbow: str = 'down',
                                    repetitions: int = DEFAULT_REPETITIONS,
                                    settle_time_ms: int = DEFAULT_SETTLE_TIME) -> List[CartesianTestResult]:
        """
        Run full Cartesian accuracy test.
        """
        if test_points is None:
            test_points = DEFAULT_CARTESIAN_TEST_POINTS
        
        results = []
        total_tests = len(test_points) * repetitions
        current = 0
        
        self.is_running = True
        
        for px, py in test_points:
            for rep in range(repetitions):
                if not self.is_running:
                    return results
                
                self._update_status(f"Point ({px}, {py}) rep {rep+1}/{repetitions}",
                                   int(100 * current / total_tests))
                
                result = self.run_cartesian_test_single(reader, px, py, elbow, settle_time_ms)
                if result:
                    results.append(result)
                current += 1
                
                # Return to home between tests
                reader.send_home()
                time.sleep(settle_time_ms / 1000.0)
        
        self.is_running = False
        self._update_status("Cartesian test complete", 100)
        return results
    
    def run_line_trajectory_test(self, reader,
                                 start_point: Tuple[float, float],
                                 end_point: Tuple[float, float],
                                 sample_interval_ms: int = 100,
                                 settle_time_before_ms: int = 2000,
                                 velocity: float = 2.0,
                                 elbow: str = 'down') -> Optional[TrajectoryTestResult]:
        """
        Run a line trajectory test using BATCH MODE (smooth drawing).
        
        This uses the same trajectory system as shape drawing,
        so it accurately tests the robot's path-following capability.
        """
        if not reader.is_connected():
            return None
        
        self.is_running = True
        
        # First, move to start position using direct command
        self._update_status(f"Moving to start ({start_point[0]}, {start_point[1]})", 10)
        start_angles = compute_target_angles(start_point[0], start_point[1], elbow)
        if start_angles is None:
            self.is_running = False
            return None
        
        reader.send_angles(start_angles[0], start_angles[1])
        time.sleep(settle_time_before_ms / 1000.0)
        
        # Generate line trajectory using batch mode
        self._update_status("Generating trajectory...", 20)
        
        # Calculate line parameters
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        line_length = np.sqrt(dx**2 + dy**2)
        
        # Generate waypoints along the line (every 0.5cm or so)
        num_points = max(10, int(line_length / 0.3))  # ~0.3cm spacing for smooth line
        num_points = min(num_points, 50)  # Cap at 50 points
        
        trajectory = []
        total_time_ms = int((line_length / velocity) * 1000)  # Time based on velocity
        
        for i in range(num_points):
            t_normalized = i / (num_points - 1) if num_points > 1 else 0
            
            # Paper coordinates
            px = start_point[0] + dx * t_normalized
            py = start_point[1] + dy * t_normalized
            
            # Convert to angles
            angles = compute_target_angles(px, py, elbow)
            if angles is None:
                continue
            
            # Timestamp
            t_ms = int(t_normalized * total_time_ms)
            
            trajectory.append((t_ms, angles[0], angles[1]))
        
        if len(trajectory) < 2:
            self.is_running = False
            return None
        
        # Send trajectory using batch protocol
        self._update_status(f"Sending trajectory ({len(trajectory)} points)...", 30)
        
        success = reader.send_trajectory(trajectory)
        if not success:
            self._update_status("Failed to send trajectory", 0)
            self.is_running = False
            return None
        
        # Sample positions during trajectory playback
        self._update_status("Drawing line, sampling positions...", 40)
        
        samples = []
        sample_times = []
        start_time = time.time()
        
        # Sample for trajectory duration + some buffer
        total_sample_time = (total_time_ms / 1000.0) + 1.0
        
        while (time.time() - start_time) < total_sample_time:
            if not self.is_running:
                return None
            
            state = reader.get_state()
            actual_x, actual_y = compute_actual_paper_position(
                state.theta1_current, state.theta2_current
            )
            samples.append((actual_x, actual_y))
            sample_times.append(time.time() - start_time)
            
            # Update progress
            elapsed_ratio = (time.time() - start_time) / total_sample_time
            progress = min(90, 40 + int(50 * elapsed_ratio))
            self._update_status(f"Sampling... ({len(samples)} points)", progress)
            
            time.sleep(sample_interval_ms / 1000.0)
        
        # Filter samples to only include those during actual motion
        # (skip first few while accelerating to line, skip last few after stopping)
        if len(samples) > 10:
            # Skip first 10% and last 10% of samples
            skip = max(1, len(samples) // 10)
            motion_samples = samples[skip:-skip] if skip > 0 else samples
        else:
            motion_samples = samples
        
        # Compute perpendicular deviations from ideal line
        deviations = [perpendicular_distance(s, start_point, end_point) for s in motion_samples]
        
        if not deviations:
            self.is_running = False
            return None
        
        result = TrajectoryTestResult(
            start_point=start_point,
            end_point=end_point,
            num_samples=len(motion_samples),
            deviations=deviations,
            max_deviation=max(deviations),
            mean_deviation=float(np.mean(deviations)),
            rms_deviation=float(np.sqrt(np.mean(np.array(deviations)**2)))
        )
        
        self.is_running = False
        self._update_status("Line test complete", 100)
        return result
    
    def run_circle_trajectory_test(self, reader,
                                   center: Tuple[float, float],
                                   radius: float = 2.0,
                                   sample_interval_ms: int = 100,
                                   settle_time_before_ms: int = 2000,
                                   velocity: float = 2.0,
                                   num_points: int = 72,
                                   elbow: str = 'down') -> Optional[TrajectoryTestResult]:
        """
        Run a circle trajectory test using BATCH MODE.
        
        Draws a circle and measures deviation from ideal circle.
        This tests smooth curved motion - the hardest case for the robot.
        """
        if not reader.is_connected():
            return None
        
        self.is_running = True
        
        # Generate circle waypoints
        self._update_status("Generating circle trajectory...", 10)
        
        # Circle circumference and timing
        circumference = 2 * np.pi * radius
        total_time_ms = int((circumference / velocity) * 1000)
        
        trajectory = []
        circle_points = []  # Store ideal points for deviation calculation
        
        # Start at rightmost point (0 degrees)
        for i in range(num_points + 1):  # +1 to close the circle
            angle_rad = 2 * np.pi * i / num_points
            
            # Paper coordinates
            px = center[0] + radius * np.cos(angle_rad)
            py = center[1] + radius * np.sin(angle_rad)
            
            circle_points.append((px, py))
            
            # Convert to joint angles
            angles = compute_target_angles(px, py, elbow)
            if angles is None:
                continue
            
            # Timestamp (linear for circles - no corner slowdown!)
            t_ms = int((i / num_points) * total_time_ms)
            
            trajectory.append((t_ms, angles[0], angles[1]))
        
        if len(trajectory) < 10:
            self.is_running = False
            return None
        
        # Move to start position first
        self._update_status(f"Moving to circle start...", 20)
        if trajectory:
            reader.send_angles(trajectory[0][1], trajectory[0][2])
            time.sleep(settle_time_before_ms / 1000.0)
        
        # Send trajectory
        self._update_status(f"Sending circle trajectory ({len(trajectory)} points)...", 30)
        
        success = reader.send_trajectory(trajectory)
        if not success:
            self._update_status("Failed to send trajectory", 0)
            self.is_running = False
            return None
        
        # Sample positions during trajectory
        self._update_status("Drawing circle, sampling positions...", 40)
        
        samples = []
        start_time = time.time()
        total_sample_time = (total_time_ms / 1000.0) + 1.5
        
        while (time.time() - start_time) < total_sample_time:
            if not self.is_running:
                return None
            
            state = reader.get_state()
            actual_x, actual_y = compute_actual_paper_position(
                state.theta1_current, state.theta2_current
            )
            samples.append((actual_x, actual_y))
            
            elapsed_ratio = (time.time() - start_time) / total_sample_time
            progress = min(90, 40 + int(50 * elapsed_ratio))
            self._update_status(f"Sampling circle... ({len(samples)} points)", progress)
            
            time.sleep(sample_interval_ms / 1000.0)
        
        # Calculate deviations from ideal circle
        # For each sample, find distance from center and compare to radius
        deviations = []
        for sx, sy in samples:
            dist_from_center = np.sqrt((sx - center[0])**2 + (sy - center[1])**2)
            deviation = abs(dist_from_center - radius)
            deviations.append(deviation)
        
        if not deviations:
            self.is_running = False
            return None
        
        # Create result (using start/end as circle center + radius info)
        result = TrajectoryTestResult(
            start_point=center,
            end_point=(radius, num_points),  # Store radius and num_points here
            num_samples=len(samples),
            deviations=deviations,
            max_deviation=max(deviations),
            mean_deviation=float(np.mean(deviations)),
            rms_deviation=float(np.sqrt(np.mean(np.array(deviations)**2)))
        )
        
        self.is_running = False
        self._update_status("Circle test complete", 100)
        return result
    
    def run_full_evaluation(self, reader, 
                           run_joint: bool = True,
                           run_cartesian: bool = True,
                           run_trajectory: bool = True,
                           run_circle: bool = False,
                           joint_angles: List[float] = None,
                           cartesian_points: List[Tuple[float, float]] = None,
                           trajectory_line: Tuple[Tuple[float, float], Tuple[float, float]] = None,
                           circle_params: Tuple[Tuple[float, float], float] = None,
                           velocity: float = 2.0,
                           repetitions: int = DEFAULT_REPETITIONS,
                           settle_time_ms: int = DEFAULT_SETTLE_TIME) -> EvaluationResults:
        """
        Run complete evaluation suite.
        """
        from datetime import datetime
        
        self.results = EvaluationResults()
        self.results.timestamp = datetime.now().isoformat()
        
        total_tests = sum([run_joint, run_cartesian, run_trajectory, run_circle])
        test_num = 0
        
        if run_joint:
            self._update_status("Running joint tests...", int(100 * test_num / total_tests))
            self.results.joint_tests = self.run_joint_repeatability_test(
                reader, joint_angles, repetitions, settle_time_ms
            )
            test_num += 1
        
        if run_cartesian:
            self._update_status("Running Cartesian tests...", int(100 * test_num / total_tests))
            self.results.cartesian_tests = self.run_cartesian_accuracy_test(
                reader, cartesian_points, 'down', repetitions, settle_time_ms
            )
            test_num += 1
        
        if run_trajectory and trajectory_line:
            self._update_status("Running line trajectory test...", int(100 * test_num / total_tests))
            result = self.run_line_trajectory_test(
                reader, trajectory_line[0], trajectory_line[1],
                velocity=velocity
            )
            if result:
                self.results.trajectory_tests.append(result)
            test_num += 1
        
        if run_circle and circle_params:
            self._update_status("Running circle trajectory test...", int(100 * test_num / total_tests))
            center, radius = circle_params
            result = self.run_circle_trajectory_test(
                reader, center, radius,
                velocity=velocity
            )
            if result:
                self.results.trajectory_tests.append(result)
            test_num += 1
        
        # Compute summary statistics
        self.results.compute_statistics()
        
        self._update_status("Evaluation complete!", 100)
        return self.results
    
    def stop(self):
        """Stop any running test."""
        self.is_running = False


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_evaluator = None

def get_evaluator() -> RobotEvaluator:
    """Get the singleton evaluator instance."""
    global _evaluator
    if _evaluator is None:
        _evaluator = RobotEvaluator()
    return _evaluator

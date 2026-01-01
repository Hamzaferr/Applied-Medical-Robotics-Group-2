"""
Shape Trajectory Generator for 2-Link Robot Arm

Generates time-stamped trajectories for Arduino batch upload.
Arduino handles interpolation at 100Hz.

Author: Yagmur
Date: December 2025
"""

import numpy as np
from typing import List, Tuple, Dict, Any

# Import robot modules
from coords import (L1, L2, paper_to_robot, is_reachable_robot)
from trajectory_planner import (
    generate_line_path, generate_polygon_path,
    generate_square, generate_rectangle, generate_circle, 
    generate_triangle, generate_star
)
from IK import analytic_ik_trajectory
from jacobian_ik import jacobian_ik_trajectory


# =============================================================================
# CONFIGURATION
# =============================================================================

# Default velocity (cm/s in paper space)
DEFAULT_VELOCITY = 3.0

# Corner slowdown settings
CORNER_DETECTION_THRESHOLD = 30.0  # degrees - angle change to detect corner
CORNER_SLOWDOWN_FACTOR = 0.3       # slow to 30% speed at sharp corners

# Trajectory settings
MIN_SEGMENT_TIME = 20              # ms - minimum time between waypoints
MAX_TRAJECTORY_POINTS = 100        # Arduino buffer limit

# Quality presets - controls number of waypoints
QUALITY_PRESETS = {
    'fast': {
        'points_per_side': 8,
        'num_circle_points': 48,
        'velocity': 4.0,
    },
    'normal': {
        'points_per_side': 12,
        'num_circle_points': 72,
        'velocity': 3.0,
    },
    'smooth': {
        'points_per_side': 25,
        'num_circle_points': 96,
        'velocity': 2.5,
    },
    'ultra': {
        'points_per_side': 35,
        'num_circle_points': 100,
        'velocity': 2.0,
    },
}


# =============================================================================
# CORNER DETECTION (for optional corner slowdown)
# =============================================================================

def detect_corners(angles: List[Tuple[float, float]], 
                   threshold: float = CORNER_DETECTION_THRESHOLD) -> List[int]:
    """
    Detect corner indices where direction changes sharply in joint space.
    Returns list of indices that are corners.
    """
    if len(angles) < 3:
        return []
    
    corners = []
    
    for i in range(1, len(angles) - 1):
        if angles[i][0] is None or angles[i-1][0] is None or angles[i+1][0] is None:
            continue
        
        # Direction vectors (in joint space)
        d1_before = angles[i][0] - angles[i-1][0]
        d2_before = angles[i][1] - angles[i-1][1]
        d1_after = angles[i+1][0] - angles[i][0]
        d2_after = angles[i+1][1] - angles[i][1]
        
        # Check for direction reversal
        if d1_before * d1_after < 0 and max(abs(d1_before), abs(d1_after)) > 1.0:
            corners.append(i)
            continue
        if d2_before * d2_after < 0 and max(abs(d2_before), abs(d2_after)) > 1.0:
            corners.append(i)
            continue
        
        # Check for large angle change
        angle_change = abs(d1_after - d1_before) + abs(d2_after - d2_before)
        if angle_change > threshold:
            corners.append(i)
    
    return corners


def compute_corner_speed_factors(n_points: int, corners: List[int], 
                                 slowdown: float = CORNER_SLOWDOWN_FACTOR,
                                 zone_radius: int = 2) -> List[float]:
    """
    Compute speed factor for each segment (1.0 = full speed, <1.0 = slow).
    Slows down near corners to allow sharp turns without overshooting.
    """
    factors = [1.0] * n_points
    
    for corner_idx in corners:
        for offset in range(-zone_radius, zone_radius + 1):
            idx = corner_idx + offset
            if 0 <= idx < n_points:
                distance = abs(offset)
                if distance == 0:
                    factors[idx] = slowdown
                else:
                    t = distance / (zone_radius + 1)
                    factors[idx] = min(factors[idx], slowdown + (1.0 - slowdown) * t)
    
    return factors


# =============================================================================
# TIMESTAMP COMPUTATION
# =============================================================================

def compute_timestamps(waypoints_paper: List[Tuple[float, float]],
                       angles: List[Tuple[float, float]],
                       velocity: float = DEFAULT_VELOCITY,
                       corner_slowdown: bool = True) -> List[int]:
    """
    Compute timestamps (in ms) for each waypoint.
    
    HOW VELOCITY WORKS:
    - velocity is in cm/s (centimeters per second in paper space)
    - For each segment, we calculate: time = distance / velocity
    - Example: 1cm segment at 2 cm/s = 500ms
    
    Args:
        waypoints_paper: Cartesian waypoints in paper coordinates
        angles: Joint angles for each waypoint (used for corner detection)
        velocity: Speed in cm/s (e.g., 2.0 = 2 centimeters per second)
        corner_slowdown: If True, slow down at detected corners
    
    Returns:
        List of timestamps in milliseconds
    """
    if len(waypoints_paper) < 2:
        return [0]
    
    # Detect corners (optional)
    corners = []
    if corner_slowdown:
        corners = detect_corners(angles)
    
    # Compute speed factors (1.0 everywhere if no corners)
    speed_factors = compute_corner_speed_factors(len(angles), corners)
    
    # Compute timestamps
    timestamps = [0]  # First point at t=0
    
    for i in range(1, len(waypoints_paper)):
        # Distance in paper space (cm)
        px1, py1 = waypoints_paper[i-1]
        px2, py2 = waypoints_paper[i]
        dist_cm = np.sqrt((px2 - px1)**2 + (py2 - py1)**2)
        
        # Time = distance / (velocity * speed_factor)
        effective_velocity = velocity * speed_factors[i]
        effective_velocity = max(effective_velocity, 0.5)  # Minimum 0.5 cm/s
        
        segment_time_ms = int((dist_cm / effective_velocity) * 1000)
        segment_time_ms = max(segment_time_ms, MIN_SEGMENT_TIME)
        
        timestamps.append(timestamps[-1] + segment_time_ms)
    
    return timestamps


# =============================================================================
# TRAJECTORY GENERATOR CLASS
# =============================================================================

class TrajectoryGenerator:
    """
    Generates time-stamped trajectories for Arduino.
    
    Usage:
        gen = TrajectoryGenerator()
        result = gen.generate_square(6, 6, 4, velocity=3.0)
        
        if result['valid']:
            trajectory = result['trajectory']  # List of (time_ms, theta1, theta2)
    """
    
    def __init__(self, l1=L1, l2=L2):
        self.l1 = l1
        self.l2 = l2
    
    def generate_trajectory(self, 
                           waypoints_paper: List[Tuple[float, float]],
                           method: str = 'analytic',
                           elbow: str = 'down',
                           velocity: float = DEFAULT_VELOCITY,
                           corner_slowdown: bool = True,
                           initial_theta1: float = 0.0,
                           initial_theta2: float = 0.0) -> Dict[str, Any]:
        """
        Generate a complete trajectory from paper-space waypoints.
        
        Args:
            waypoints_paper: List of (x, y) in paper coordinates
            method: 'analytic' or 'jacobian' IK
            elbow: 'up' or 'down' configuration
            velocity: Speed in cm/s
            corner_slowdown: Slow down at corners
            initial_theta1/2: Starting angles for Jacobian IK
        
        Returns:
            Dictionary with trajectory data
        """
        result = {
            'valid': False,
            'trajectory': [],
            'angles': [],
            'waypoints_paper': waypoints_paper,
            'waypoints_robot': [],
            'total_time_ms': 0,
            'path_length': 0.0,
            'corners': [],
            'errors': []
        }
        
        if len(waypoints_paper) < 2:
            result['errors'].append("Need at least 2 waypoints")
            return result
        
        # Check trajectory length
        if len(waypoints_paper) > MAX_TRAJECTORY_POINTS:
            result['errors'].append(f"Too many points ({len(waypoints_paper)}), max is {MAX_TRAJECTORY_POINTS}")
            step = len(waypoints_paper) // MAX_TRAJECTORY_POINTS + 1
            waypoints_paper = waypoints_paper[::step]
        
        # Convert to robot coordinates
        waypoints_robot = []
        for i, (px, py) in enumerate(waypoints_paper):
            rx, ry = paper_to_robot(px, py)
            
            if not is_reachable_robot(rx, ry):
                result['errors'].append(f"Point {i} at Paper({px:.1f}, {py:.1f}) not reachable")
                return result
            
            waypoints_robot.append((rx, ry))
        
        result['waypoints_robot'] = waypoints_robot
        
        # Compute path length
        total_length = 0.0
        for i in range(1, len(waypoints_paper)):
            px1, py1 = waypoints_paper[i-1]
            px2, py2 = waypoints_paper[i]
            total_length += np.sqrt((px2 - px1)**2 + (py2 - py1)**2)
        result['path_length'] = total_length
        
        # Compute IK
        if method == 'analytic':
            angles, all_valid, failed = analytic_ik_trajectory(
                self.l1, self.l2, waypoints_robot,
                elbow=elbow, prefer_closest=True,
                initial_theta1=initial_theta1, initial_theta2=initial_theta2
            )
            if not all_valid:
                result['errors'].append(f"IK failed for {len(failed)} points")
                return result
        else:
            angles, all_converged = jacobian_ik_trajectory(
                self.l1, self.l2, waypoints_robot,
                initial_theta1=initial_theta1, initial_theta2=initial_theta2,
                tolerance=0.05, gain=0.5
            )
            if not all_converged:
                result['errors'].append("Some points did not converge (may still work)")
        
        result['angles'] = angles
        
        # Detect corners
        corners = detect_corners(angles) if corner_slowdown else []
        result['corners'] = corners
        
        # Compute timestamps
        timestamps = compute_timestamps(waypoints_paper, angles, velocity, corner_slowdown)
        result['total_time_ms'] = timestamps[-1] if timestamps else 0
        
        # Build trajectory: (time_ms, theta1, theta2)
        trajectory = []
        for i, (t1, t2) in enumerate(angles):
            if t1 is not None and t2 is not None:
                trajectory.append((timestamps[i], t1, t2))
        
        result['trajectory'] = trajectory
        result['valid'] = len(trajectory) >= 2
        
        return result
    
    # =========================================================================
    # SHAPE GENERATORS
    # =========================================================================
    
    def generate_square(self, center_x: float, center_y: float, size: float,
                        method: str = 'analytic', elbow: str = 'down',
                        velocity: float = DEFAULT_VELOCITY,
                        quality: str = 'normal',
                        corner_slowdown: bool = True,
                        **kwargs) -> Dict[str, Any]:
        """Generate square trajectory."""
        preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS['normal'])
        velocity = velocity or preset['velocity']
        
        waypoints = generate_square(center_x, center_y, size, preset['points_per_side'])
        
        return self.generate_trajectory(waypoints, method=method, elbow=elbow,
                                        velocity=velocity, corner_slowdown=corner_slowdown,
                                        **kwargs)
    
    def generate_circle(self, center_x: float, center_y: float, radius: float,
                        method: str = 'analytic', elbow: str = 'down',
                        velocity: float = DEFAULT_VELOCITY,
                        quality: str = 'normal',
                        **kwargs) -> Dict[str, Any]:
        """Generate circle trajectory."""
        preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS['normal'])
        velocity = velocity or preset['velocity']
        
        waypoints = generate_circle(center_x, center_y, radius, preset['num_circle_points'])
        
        return self.generate_trajectory(waypoints, method=method, elbow=elbow,
                                        velocity=velocity, corner_slowdown=False,
                                        **kwargs)
    
    def generate_triangle(self, center_x: float, center_y: float, size: float,
                          method: str = 'analytic', elbow: str = 'down',
                          velocity: float = DEFAULT_VELOCITY,
                          quality: str = 'normal',
                          corner_slowdown: bool = True,
                          **kwargs) -> Dict[str, Any]:
        """Generate equilateral triangle trajectory."""
        preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS['normal'])
        velocity = velocity or preset['velocity']
        
        waypoints = generate_triangle(center_x, center_y, size, preset['points_per_side'])
        
        return self.generate_trajectory(waypoints, method=method, elbow=elbow,
                                        velocity=velocity, corner_slowdown=corner_slowdown,
                                        **kwargs)
    
    def generate_rectangle(self, center_x: float, center_y: float, 
                           width: float, height: float,
                           method: str = 'analytic', elbow: str = 'down',
                           velocity: float = DEFAULT_VELOCITY,
                           quality: str = 'normal',
                           corner_slowdown: bool = True,
                           **kwargs) -> Dict[str, Any]:
        """Generate rectangle trajectory."""
        preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS['normal'])
        velocity = velocity or preset['velocity']
        
        waypoints = generate_rectangle(center_x, center_y, width, height, preset['points_per_side'])
        
        return self.generate_trajectory(waypoints, method=method, elbow=elbow,
                                        velocity=velocity, corner_slowdown=corner_slowdown,
                                        **kwargs)
    
    def generate_star(self, center_x: float, center_y: float, outer_radius: float,
                      method: str = 'analytic', elbow: str = 'down',
                      velocity: float = DEFAULT_VELOCITY,
                      num_points: int = 5, inner_ratio: float = 0.4,
                      quality: str = 'normal',
                      corner_slowdown: bool = True,
                      **kwargs) -> Dict[str, Any]:
        """Generate star trajectory."""
        preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS['normal'])
        velocity = velocity or preset['velocity']
        
        points_per_segment = max(3, preset['points_per_side'] // 2)
        waypoints = generate_star(center_x, center_y, outer_radius,
                                  num_points, inner_ratio, points_per_segment)
        
        return self.generate_trajectory(waypoints, method=method, elbow=elbow,
                                        velocity=velocity, corner_slowdown=corner_slowdown,
                                        **kwargs)
    
    def generate_line(self, x1: float, y1: float, x2: float, y2: float,
                      method: str = 'analytic', elbow: str = 'down',
                      velocity: float = DEFAULT_VELOCITY,
                      num_points: int = 20,
                      **kwargs) -> Dict[str, Any]:
        """Generate straight line trajectory."""
        waypoints = generate_line_path((x1, y1), (x2, y2), num_points)
        
        return self.generate_trajectory(waypoints, method=method, elbow=elbow,
                                        velocity=velocity, corner_slowdown=False,
                                        **kwargs)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Trajectory Generator Test")
    print("=" * 60)
    
    gen = TrajectoryGenerator()
    
    # Test square
    print("\n--- Square (4cm, velocity=3 cm/s) ---")
    result = gen.generate_square(6, 6, 4, velocity=3.0, quality='normal')
    print(f"Valid: {result['valid']}")
    print(f"Points: {len(result['trajectory'])}")
    print(f"Path length: {result['path_length']:.1f} cm")
    print(f"Total time: {result['total_time_ms']}ms ({result['total_time_ms']/1000:.1f}s)")
    print(f"Effective speed: {result['path_length'] / (result['total_time_ms']/1000):.1f} cm/s")
    
    # Test line
    print("\n--- Line (6cm, velocity=2 cm/s) ---")
    result = gen.generate_line(3, 6, 9, 6, velocity=2.0)
    print(f"Valid: {result['valid']}")
    print(f"Points: {len(result['trajectory'])}")
    print(f"Path length: {result['path_length']:.1f} cm")
    print(f"Total time: {result['total_time_ms']}ms ({result['total_time_ms']/1000:.1f}s)")
    print(f"Effective speed: {result['path_length'] / (result['total_time_ms']/1000):.1f} cm/s")
    
    print("\n" + "=" * 60)
    print("Done!")
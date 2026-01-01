"""
Trajectory Planner for 2-Link Planar Robot Arm

SIMPLIFIED VERSION - Only path generation, no velocity profiling.
Arduino handles interpolation and timing.

Author: Yagmur
Date: December 2025
"""

import numpy as np


# =============================================================================
# BASIC PATH GENERATION
# =============================================================================

def generate_line_path(start, end, num_points=50):
    """
    Generate linearly interpolated waypoints between two points.
    
    Parameters:
        start: (x, y) starting position
        end: (x, y) ending position
        num_points: Number of waypoints (including start and end)
    
    Returns:
        List of (x, y) tuples
    """
    if num_points < 2:
        num_points = 2
    
    x_start, y_start = start
    x_end, y_end = end
    
    waypoints = []
    for i in range(num_points):
        t = i / (num_points - 1)
        x = x_start + t * (x_end - x_start)
        y = y_start + t * (y_end - y_start)
        waypoints.append((x, y))
    
    return waypoints


def generate_arc_path(center, radius, start_angle, end_angle, num_points=50):
    """
    Generate waypoints along a circular arc.
    
    Parameters:
        center: (x, y) center of arc
        radius: Arc radius
        start_angle: Starting angle in degrees (0° = +X direction)
        end_angle: Ending angle in degrees
        num_points: Number of waypoints
    
    Returns:
        List of (x, y) tuples
    """
    if num_points < 2:
        num_points = 2
    
    cx, cy = center
    
    waypoints = []
    for i in range(num_points):
        t = i / (num_points - 1)
        angle = start_angle + t * (end_angle - start_angle)
        angle_rad = np.radians(angle)
        
        x = cx + radius * np.cos(angle_rad)
        y = cy + radius * np.sin(angle_rad)
        waypoints.append((x, y))
    
    return waypoints


def generate_polygon_path(vertices, points_per_side=20):
    """
    Generate waypoints along a polygon with sharp corners.
    
    Parameters:
        vertices: List of (x, y) tuples defining polygon corners
        points_per_side: Number of points per side
    
    Returns:
        List of (x, y) tuples forming closed polygon path
    """
    if len(vertices) < 2:
        return vertices
    
    waypoints = []
    n = len(vertices)
    
    for i in range(n):
        start = vertices[i]
        end = vertices[(i + 1) % n]
        
        # Generate line segment
        line_points = generate_line_path(start, end, points_per_side + 1)
        if waypoints:
            line_points = line_points[1:]  # Skip first to avoid duplicate
        waypoints.extend(line_points[:-1])  # Exclude last point
    
    # Close the polygon
    waypoints.append(vertices[0])
    
    return waypoints


def path_length(waypoints):
    """Calculate total length of a path."""
    if len(waypoints) < 2:
        return 0.0
    
    total = 0.0
    for i in range(len(waypoints) - 1):
        x1, y1 = waypoints[i]
        x2, y2 = waypoints[i + 1]
        total += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    return total


# =============================================================================
# SHAPE GENERATORS
# =============================================================================

def generate_square(center_x, center_y, size, points_per_side=15):
    """
    Generate a square path centered at the given position.
    
    Parameters:
        center_x, center_y: Center of square (in Paper coordinates)
        size: Side length of the square
        points_per_side: Number of points per side
    
    Returns:
        List of (x, y) waypoints (closed path)
    """
    half = size / 2
    vertices = [
        (center_x - half, center_y - half),  # Bottom-left
        (center_x + half, center_y - half),  # Bottom-right
        (center_x + half, center_y + half),  # Top-right
        (center_x - half, center_y + half),  # Top-left
    ]
    return generate_polygon_path(vertices, points_per_side)


def generate_rectangle(center_x, center_y, width, height, points_per_side=15):
    """
    Generate a rectangle path centered at the given position.
    
    Parameters:
        center_x, center_y: Center of rectangle
        width: Width (X dimension)
        height: Height (Y dimension)
        points_per_side: Number of points per side
    
    Returns:
        List of (x, y) waypoints (closed path)
    """
    hw = width / 2
    hh = height / 2
    vertices = [
        (center_x - hw, center_y - hh),  # Bottom-left
        (center_x + hw, center_y - hh),  # Bottom-right
        (center_x + hw, center_y + hh),  # Top-right
        (center_x - hw, center_y + hh),  # Top-left
    ]
    return generate_polygon_path(vertices, points_per_side)


def generate_circle(center_x, center_y, radius, num_points=60):
    """
    Generate a circle path centered at the given position.
    
    Parameters:
        center_x, center_y: Center of circle
        radius: Radius of the circle
        num_points: Total number of points around the circle
    
    Returns:
        List of (x, y) waypoints (closed path)
    """
    waypoints = generate_arc_path(
        (center_x, center_y), 
        radius, 
        0, 360, 
        num_points + 1  # +1 to close the circle
    )
    return waypoints


def generate_triangle(center_x, center_y, size, points_per_side=15):
    """
    Generate an equilateral triangle path centered at the given position.
    
    Parameters:
        center_x, center_y: Center of triangle
        size: Side length
        points_per_side: Number of points per side
    
    Returns:
        List of (x, y) waypoints (closed path)
    """
    h = size * np.sqrt(3) / 2  # Height
    r = size / np.sqrt(3)       # Circumradius
    
    # Vertices (pointing up)
    vertices = [
        (center_x, center_y + r),                           # Top
        (center_x - size/2, center_y - h/3),               # Bottom-left
        (center_x + size/2, center_y - h/3),               # Bottom-right
    ]
    return generate_polygon_path(vertices, points_per_side)


def generate_star(center_x, center_y, outer_radius, num_points_star=5, 
                  inner_ratio=0.4, points_per_segment=10):
    """
    Generate a star path centered at the given position.
    
    Parameters:
        center_x, center_y: Center of star
        outer_radius: Radius to outer points
        num_points_star: Number of star points (5 = classic star)
        inner_ratio: Ratio of inner to outer radius (default 0.4)
        points_per_segment: Points per line segment
    
    Returns:
        List of (x, y) waypoints (closed path)
    """
    inner_radius = outer_radius * inner_ratio
    
    vertices = []
    total_points = num_points_star * 2
    
    for i in range(total_points):
        angle = (i * 360 / total_points) - 90  # Start from top
        angle_rad = np.radians(angle)
        
        if i % 2 == 0:
            r = outer_radius
        else:
            r = inner_radius
        
        x = center_x + r * np.cos(angle_rad)
        y = center_y + r * np.sin(angle_rad)
        vertices.append((x, y))
    
    return generate_polygon_path(vertices, points_per_segment)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Trajectory Planner Test")
    print("=" * 60)
    
    # Test line
    print("\nLine (0,0) to (10,10), 5 points:")
    line = generate_line_path((0, 0), (10, 10), 5)
    for p in line:
        print(f"  ({p[0]:.1f}, {p[1]:.1f})")
    print(f"  Length: {path_length(line):.2f}")
    
    # Test square
    print("\nSquare at (6,6), size=4:")
    square = generate_square(6, 6, 4, points_per_side=5)
    print(f"  {len(square)} points, length: {path_length(square):.1f}")
    
    # Test circle
    print("\nCircle at (6,6), radius=2:")
    circle = generate_circle(6, 6, 2, num_points=12)
    print(f"  {len(circle)} points, length: {path_length(circle):.1f}")
    
    print("\n" + "=" * 60)
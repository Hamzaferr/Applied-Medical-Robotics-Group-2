"""
Author:Hamza
Coordinate System Module for 2-Link Planar Robot Arm

=== COORDINATE CONVENTIONS ===

Robot Coordinate System:
    - Origin (0, 0) at robot base
    - +Y axis: Direction arm points when θ1=0°, θ2=0° (fully extended)
    - +X axis: 90° clockwise from +Y (to the right when facing +Y)
    - At θ1=0°, θ2=0°: End effector at (0, 24)

Paper Coordinate System:
    - 12x12 cm imaginary paper
    - Paper (0, 0): Bottom-left corner
    - Paper (12, 12): Top-right corner  
    - Paper (6, 12): Top-center = Robot (0, 24) = Starting position

Transformations:
    Robot_X = Paper_X - 6
    Robot_Y = Paper_Y + 12

=== PHYSICAL SETUP ===

    θ1=0°, θ2=0° (Start Position)
    
              +Y (Robot)
               ↑
               │    End Effector: Robot(0,24) = Paper(6,12)
               ●────────────────────────○
               │         L1=12cm        │ L2=12cm
               │                        ○ Elbow
               │
               │    Paper is here:
               │    ┌─────────────┐
               │    │ (0,12)      │(12,12)
               │    │      ↑      │
               │    │   Paper +Y  │
               │    │      │      │
               │    │ (0,0)└──→   │(12,0)
               │    └────Paper +X─┘
    ───────────┼─────────────────────────→ +X (Robot)
         Robot │
          Base │
          (0,0)│
               ↓
              -Y (Robot)
"""

import math

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

L1 = 12.0  # Link 1 length in cm (base to elbow)
L2 = 12.0  # Link 2 length in cm (elbow to end-effector)

MAX_REACH = L1 + L2  # 24.0 cm - fully extended
MIN_REACH = abs(L1 - L2)  # 0.0 cm - fully folded (equal length links)

# Paper dimensions
PAPER_WIDTH = 12.0   # cm
PAPER_HEIGHT = 12.0  # cm

# Paper offset from robot base
# Paper (6, 12) = Robot (0, 24), so:
# CORRECT OFFSETS:
# Paper(6, 12) = Robot(0, 24) - top-middle of paper, max extension
# Paper(6, 6) = Robot(0, 18) - center of paper
# Paper(6, 0) = Robot(0, 12) - bottom-middle of paper
# Robot_X = Paper_X - 6
# Robot_Y = Paper_Y + 12
PAPER_OFFSET_X = -6.0   # Robot X = Paper X - 6
PAPER_OFFSET_Y = 12.0   # Robot Y = Paper Y + 12

# =============================================================================
# SAFETY LIMITS
# =============================================================================

# -----------------------------------------------------------------------------
# JOINT LIMITS (SAFETY)
# These should match your physical robot's actual limits!
# -----------------------------------------------------------------------------
THETA1_MIN = -120.0  # degrees - base motor
THETA1_MAX = 120.0   # degrees
THETA2_MIN = -135.0  # degrees - elbow motor (Pololu can likely go further)
THETA2_MAX = 135.0   # degrees

# =============================================================================
# COORDINATE TRANSFORMATIONS
# =============================================================================

def paper_to_robot(paper_x, paper_y):
    """
    Convert paper coordinates to robot coordinates.
    
    Paper(6, 12) → Robot(0, 24) - top-middle, max extension
    Paper(6, 6) → Robot(0, 18) - center of paper
    Paper(0, 0) → Robot(-6, 12) - bottom-left
    
    Parameters:
        paper_x, paper_y: Position on paper (0-12 range typical)
    
    Returns:
        robot_x, robot_y: Position in robot coordinate system
    """
    robot_x = paper_x + PAPER_OFFSET_X  # paper_x - 6
    robot_y = paper_y + PAPER_OFFSET_Y  # paper_y + 12
    return robot_x, robot_y


def robot_to_paper(robot_x, robot_y):
    """
    Convert robot coordinates to paper coordinates.
    
    Robot(0, 24) → Paper(6, 12) - top-middle
    Robot(0, 18) → Paper(6, 6) - center
    Robot(-6, 12) → Paper(0, 0) - bottom-left
    
    Parameters:
        robot_x, robot_y: Position in robot coordinate system
    
    Returns:
        paper_x, paper_y: Position on paper
    """
    paper_x = robot_x - PAPER_OFFSET_X  # robot_x + 6
    paper_y = robot_y - PAPER_OFFSET_Y  # robot_y - 12
    return paper_x, paper_y


# =============================================================================
# REACHABILITY CHECKS
# =============================================================================

def is_reachable_robot(robot_x, robot_y):
    """
    Check if robot coordinates are within the workspace.
    
    Parameters:
        robot_x, robot_y: Position in robot coordinate system
    
    Returns:
        bool: True if position is reachable
    """
    dist = math.sqrt(robot_x**2 + robot_y**2)
    return MIN_REACH <= dist <= MAX_REACH


def is_reachable_paper(paper_x, paper_y):
    """
    Check if paper coordinates are reachable by the robot.
    
    Parameters:
        paper_x, paper_y: Position on paper
    
    Returns:
        bool: True if position is reachable
    """
    robot_x, robot_y = paper_to_robot(paper_x, paper_y)
    return is_reachable_robot(robot_x, robot_y)


def get_reach_distance_robot(robot_x, robot_y):
    """
    Get the distance from robot base to a point.
    
    Parameters:
        robot_x, robot_y: Position in robot coordinate system
    
    Returns:
        float: Distance in cm
    """
    return math.sqrt(robot_x**2 + robot_y**2)


def get_reach_distance_paper(paper_x, paper_y):
    """
    Get the distance from robot base to a paper coordinate.
    
    Parameters:
        paper_x, paper_y: Position on paper
    
    Returns:
        float: Distance in cm
    """
    robot_x, robot_y = paper_to_robot(paper_x, paper_y)
    return get_reach_distance_robot(robot_x, robot_y)


# =============================================================================
# ANGLE LIMIT CHECKS
# =============================================================================

def angles_within_limits(theta1, theta2):
    """
    Check if joint angles are within safety limits.
    
    Parameters:
        theta1, theta2: Joint angles in degrees
    
    Returns:
        bool: True if both angles are within limits
    """
    return (THETA1_MIN <= theta1 <= THETA1_MAX and 
            THETA2_MIN <= theta2 <= THETA2_MAX)


def clamp_angles(theta1, theta2):
    """
    Clamp joint angles to safety limits.
    
    Parameters:
        theta1, theta2: Joint angles in degrees
    
    Returns:
        theta1_clamped, theta2_clamped: Clamped angles
    """
    theta1_clamped = max(THETA1_MIN, min(THETA1_MAX, theta1))
    theta2_clamped = max(THETA2_MIN, min(THETA2_MAX, theta2))
    return theta1_clamped, theta2_clamped


# =============================================================================
# HOME POSITION
# =============================================================================

# Home position: θ1=0°, θ2=0° → Robot (0, 24) → Paper (6, 12)
HOME_THETA1 = 0.0
HOME_THETA2 = 0.0
HOME_ROBOT_X = 0.0
HOME_ROBOT_Y = 24.0
HOME_PAPER_X = 6.0
HOME_PAPER_Y = 12.0


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Coordinate System Test")
    print("=" * 60)
    
    # Test transformations
    test_points = [
        ("Start/Home", 6, 12),
        ("Paper Origin", 0, 0),
        ("Paper Bottom-Right", 12, 0),
        ("Paper Top-Left", 0, 12),
        ("Paper Top-Right", 12, 12),
        ("Paper Center", 6, 6),
    ]
    
    print("\nPaper → Robot Transformations:")
    print("-" * 60)
    print(f"{'Name':<20} {'Paper':^15} {'Robot':^15} {'Dist':^8} {'Reach?':^8}")
    print("-" * 60)
    
    for name, px, py in test_points:
        rx, ry = paper_to_robot(px, py)
        dist = get_reach_distance_robot(rx, ry)
        reachable = "✓" if is_reachable_robot(rx, ry) else "✗"
        print(f"{name:<20} ({px:>5.1f}, {py:>5.1f}) ({rx:>5.1f}, {ry:>5.1f}) {dist:>6.1f}cm {reachable:^8}")
    
    print("\n" + "=" * 60)
    print("8x8 Center Area (Paper 2-10, 2-10) Reachability:")
    print("=" * 60)
    
    corners_8x8 = [(2, 2), (10, 2), (2, 10), (10, 10)]
    all_reachable = True
    for px, py in corners_8x8:
        rx, ry = paper_to_robot(px, py)
        dist = get_reach_distance_robot(rx, ry)
        reachable = is_reachable_robot(rx, ry)
        all_reachable = all_reachable and reachable
        status = "✓" if reachable else "✗"
        print(f"  Paper ({px}, {py}) → Robot ({rx:.1f}, {ry:.1f}) = {dist:.1f}cm {status}")
    
    print(f"\n8x8 center area fully reachable: {'YES ✓' if all_reachable else 'NO ✗'}")
    print("=" * 60)

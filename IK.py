import numpy as np
from FK import forward_kinematics 

def _wrap_to_pi(a):
    """Wrap angle to [-π, π]"""
    return (a + np.pi) % (2*np.pi) - np.pi


def inverse_kinematics_full(r1, r2, x, y, degrees=False, eps=1e-9):
    """
    Full IK solver - returns BOTH elbow-up and elbow-down solutions.
    
    Returns:
    --------
    down, up : tuples
        Two solutions: (theta1_down, theta2_down), (theta1_up, theta2_up)
    """
    d = np.hypot(x, y)
    if d > (r1 + r2) + eps or d < abs(r1 - r2) - eps:
        raise ValueError(f"target ({x:.3f}, {y:.3f}) out of range (r1={r1}, r2={r2})")

    cos_t2 = (d*d - r1*r1 - r2*r2) / (2.0 * r1 * r2)
    cos_t2 = np.clip(cos_t2, -1.0, 1.0)
    sin_t2 = np.sqrt(max(0.0, 1.0 - cos_t2*cos_t2))

    t2_pos = np.arctan2( sin_t2, cos_t2)   # up
    t2_neg = np.arctan2(-sin_t2, cos_t2)   # down

    def solve_t1(t2):
        k1 = r1 + r2*np.cos(t2)
        k2 = r2*np.sin(t2)
        return np.arctan2(y, x) - np.arctan2(k2, k1)

    t1_pos = solve_t1(t2_pos)
    t1_neg = solve_t1(t2_neg)

    t1_pos, t2_pos = _wrap_to_pi(t1_pos), _wrap_to_pi(t2_pos)
    t1_neg, t2_neg = _wrap_to_pi(t1_neg), _wrap_to_pi(t2_neg)

    if d < eps:
        elbow_pos_sign = np.sign(np.sin(t2_pos))
        elbow_neg_sign = np.sign(np.sin(t2_neg))
    else:
        ux, uy = x/d, y/d
        vx, vy = -uy, ux
        ex_pos, ey_pos = r1*np.cos(t1_pos), r1*np.sin(t1_pos)
        ex_neg, ey_neg = r1*np.cos(t1_neg), r1*np.sin(t1_neg)
        elbow_pos_sign = ex_pos*vx + ey_pos*vy
        elbow_neg_sign = ex_neg*vx + ey_neg*vy

    if elbow_pos_sign < elbow_neg_sign:
        down = (t1_pos, t2_pos)
        up   = (t1_neg, t2_neg)
    elif elbow_pos_sign > elbow_neg_sign:
        down = (t1_neg, t2_neg)
        up   = (t1_pos, t2_pos)
    else:
        down = (t1_neg, t2_neg)
        up   = (t1_pos, t2_pos)

    # Subtract 90° from θ1 to match FK convention where θ1=0° points +Y
    # (Standard IK computes angles where 0° points +X)
    down = (down[0] - np.pi/2, down[1])
    up = (up[0] - np.pi/2, up[1])

    if degrees:
        down = (np.rad2deg(down[0]), np.rad2deg(down[1]))
        up   = (np.rad2deg(up[0]),   np.rad2deg(up[1]))

    return down, up


def inverse_kinematics(r1, r2, x_target, y_target, elbow="down"):
    """
    Simple IK function for GUI - returns just ONE solution.
    
    Calculate joint angles (θ₁, θ₂) for a given end-effector position (x, y).
    
    Parameters:
    -----------
    r1, r2 : float
        Link lengths in cm
    x_target, y_target : float
        Target end-effector coordinates in cm
    elbow : str
        "up" or "down" configuration
    
    Returns:
    --------
    theta1_deg, theta2_deg : tuple of floats
        Joint angles in DEGREES
        Returns (None, None) if unreachable
    """
    try:
        # Call the full solver (returns both solutions)
        down, up = inverse_kinematics_full(r1, r2, -x_target, y_target, degrees=True)
        
        # Return the requested configuration
        if elbow.lower() == "up":
            return up[0], up[1]  # (theta1, theta2) for elbow-up
        else:
            return down[0], down[1]  # (theta1, theta2) for elbow-down
            
    except ValueError:
        # Target unreachable
        return None, None


def pose_from_xy(x, y, elbow='down', r1=0.08, r2=0.08, degrees=False):
    """
    Get joint angles AND transformation matrix for target position.
    
    Parameters:
    -----------
    x, y : float
        Target position (same units as r1, r2)
    elbow : str
        'up' or 'down' configuration
    r1, r2 : float
        Link lengths
    degrees : bool
        Return angles in degrees
    
    Returns:
    --------
    T : np.ndarray (4x4)
        Transformation matrix
    angles : tuple (theta1, theta2)
        Joint angles
    """
    down, up = inverse_kinematics_full(r1, r2, x, y, degrees=False)
    t1, t2 = (up if elbow == 'up' else down)
    T = forward_kinematics(r1, r2, t1, t2, degrees=False)
    
    if degrees:
        return T, (np.rad2deg(t1), np.rad2deg(t2))
    return T, (t1, t2)


# ============================================
# EXAMPLE USAGE & TESTING
# ============================================

if __name__ == "__main__":
    # Example robot parameters
    L1 = 8.0
    L2 = 8.0
    
    print("=" * 60)
    print("Inverse Kinematics Function Test")
    print("=" * 60)
    
    # Example target points (cm)
    test_points = [
        (16, 0),   # fully extended
        (8, 8),    # diagonal
        (0, 16),   # straight up
        (10, 5),   # random reachable
    ]
    
    for (x, y) in test_points:
        print(f"\nTarget: x={x:.2f} cm, y={y:.2f} cm")
        
        # Elbow-up
        t1_up, t2_up = inverse_kinematics(L1, L2, x, y, elbow="up")
        if t1_up is not None:
            print(f"  Elbow-Up  → θ₁={t1_up:.2f}°, θ₂={t2_up:.2f}°")
        
        # Elbow-down
        t1_dn, t2_dn = inverse_kinematics(L1, L2, x, y, elbow="down")
        if t1_dn is not None:
            print(f"  Elbow-Down→ θ₁={t1_dn:.2f}°, θ₂={t2_dn:.2f}°")
    
    print("\n" + "=" * 60)
    print("✓ IK test complete!")
    print("=" * 60)

def analytic_ik_trajectory(r1, r2, waypoints, elbow='down', prefer_closest=False, 
                           initial_theta1=None, initial_theta2=None):
    """
    Compute IK for a trajectory of waypoints.
    
    Parameters:
        r1, r2: Link lengths
        waypoints: List of (x, y) robot coordinates
        elbow: 'up' or 'down'
        prefer_closest: If True, try to minimize joint movement (not implemented)
        initial_theta1, initial_theta2: Starting angles (for prefer_closest)
    
    Returns:
        angles: List of (theta1, theta2) in degrees
        all_valid: True if all points reachable
        configs: List of 'down'/'up' for each point
    """
    angles = []
    configs = []
    all_valid = True
    
    for x, y in waypoints:
        t1, t2 = inverse_kinematics(r1, r2, x, y, elbow=elbow)
        if t1 is None:
            all_valid = False
            angles.append((None, None))
            configs.append(None)
        else:
            angles.append((t1, t2))
            configs.append(elbow)
    
    return angles, all_valid, configs
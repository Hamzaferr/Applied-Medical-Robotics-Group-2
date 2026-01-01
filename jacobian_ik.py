"""
Jacobian-based Inverse Kinematics for 2-Link Planar Robot Arm

Uses iterative pseudo-inverse Jacobian method for smooth trajectory tracking.

Convention:
    - θ1 = 0°, θ2 = 0° → End effector at (0, 24) - arm pointing along +Y
    - Uses 90° offset from standard mathematical convention
    - FK returns FLIPPED X (x = -T[0,3]), Jacobian accounts for this

FIXED: December 2025 - Jacobian signs corrected for X-flip in FK
"""

import numpy as np
from coords import L1, L2, angles_within_limits, clamp_angles

# Angle offset: User's θ1=0° corresponds to mathematical 90°
THETA1_OFFSET = 90.0  # degrees


def compute_jacobian(r1, r2, theta1, theta2):
    """
    Compute the Jacobian matrix for the 2-link arm.
    
    The Jacobian relates joint velocities to end-effector velocities:
        [dx/dt]   [J11  J12] [dθ1/dt]
        [dy/dt] = [J21  J22] [dθ2/dt]
    
    IMPORTANT: This Jacobian accounts for the X-flip in FK.py!
    FK returns: x = -T[0,3], y = T[1,3]
    
    Parameters:
    -----------
    r1, r2 : float
        Link lengths in cm
    theta1, theta2 : float
        Joint angles in DEGREES (user convention, with offset)
    
    Returns:
    --------
    J : numpy.ndarray (2x2)
        Jacobian matrix
    """
    # Convert to mathematical convention
    theta1_math = theta1 + THETA1_OFFSET
    
    # Convert to radians
    t1 = np.radians(theta1_math)
    t2 = np.radians(theta2)
    t12 = t1 + t2
    
    # Jacobian elements for our coordinate system
    # 
    # FK computes (internally):
    #   T[0,3] = r1*cos(t1) + r2*cos(t12)
    #   T[1,3] = r1*sin(t1) + r2*sin(t12)
    #
    # But get_end_effector_position returns:
    #   x = -T[0,3] = -(r1*cos(t1) + r2*cos(t12))
    #   y = T[1,3] = r1*sin(t1) + r2*sin(t12)
    #
    # So the Jacobian ∂[x,y]/∂[θ1,θ2] is:
    #   ∂x/∂θ1 = -(-r1*sin(t1) - r2*sin(t12)) = r1*sin(t1) + r2*sin(t12)
    #   ∂x/∂θ2 = -(-r2*sin(t12)) = r2*sin(t12)
    #   ∂y/∂θ1 = r1*cos(t1) + r2*cos(t12)
    #   ∂y/∂θ2 = r2*cos(t12)
    #
    J = np.array([
        [r1 * np.sin(t1) + r2 * np.sin(t12), r2 * np.sin(t12)],
        [r1 * np.cos(t1) + r2 * np.cos(t12), r2 * np.cos(t12)]
    ])
    
    return J


def jacobian_ik_step(r1, r2, theta1, theta2, target_x, target_y, gain=0.5):
    """
    Perform one step of Jacobian pseudo-inverse IK.
    
    Parameters:
    -----------
    r1, r2 : float
        Link lengths in cm
    theta1, theta2 : float
        Current joint angles in DEGREES
    target_x, target_y : float
        Target position in robot coordinates (cm)
    gain : float
        Step size multiplier (0 < gain <= 1)
    
    Returns:
    --------
    new_theta1, new_theta2 : float
        Updated joint angles in DEGREES
    error : float
        Position error (distance to target) in cm
    """
    from FK import get_end_effector_position
    
    # Current position (already accounts for X-flip)
    curr_x, curr_y = get_end_effector_position(r1, r2, theta1, theta2)
    
    # Position error
    dx = target_x - curr_x
    dy = target_y - curr_y
    error = np.sqrt(dx**2 + dy**2)
    
    if error < 1e-6:
        return theta1, theta2, error
    
    # Compute Jacobian (now correctly accounts for X-flip)
    J = compute_jacobian(r1, r2, theta1, theta2)
    
    # Check for singularity (determinant near zero)
    det = np.linalg.det(J)
    if abs(det) < 1e-6:
        # Near singularity - use damped least squares
        damping = 0.1
        JtJ = J.T @ J + damping**2 * np.eye(2)
        J_pinv = np.linalg.solve(JtJ, J.T)
    else:
        # Normal pseudo-inverse
        try:
            J_pinv = np.linalg.pinv(J)
        except np.linalg.LinAlgError:
            return theta1, theta2, error
    
    # Compute angle updates (error is in cm, Jacobian relates cm to radians)
    delta_theta = J_pinv @ np.array([dx, dy])
    
    # Apply with gain (delta_theta is in radians, convert to degrees)
    new_theta1 = theta1 + gain * np.degrees(delta_theta[0])
    new_theta2 = theta2 + gain * np.degrees(delta_theta[1])
    
    return new_theta1, new_theta2, error


def jacobian_ik(r1, r2, target_x, target_y, 
                initial_theta1=0.0, initial_theta2=0.0,
                max_iterations=100, tolerance=0.01, gain=0.5,
                enforce_limits=True):
    """
    Iterative Jacobian pseudo-inverse inverse kinematics.
    
    Parameters:
    -----------
    r1, r2 : float
        Link lengths in cm
    target_x, target_y : float
        Target position in robot coordinates (cm)
    initial_theta1, initial_theta2 : float
        Starting joint angles in DEGREES
    max_iterations : int
        Maximum number of iterations
    tolerance : float
        Position error tolerance in cm
    gain : float
        Step size multiplier
    enforce_limits : bool
        If True, clamp angles to safety limits each iteration
    
    Returns:
    --------
    theta1, theta2 : float
        Final joint angles in DEGREES
    converged : bool
        True if solution converged within tolerance
    iterations : int
        Number of iterations used
    final_error : float
        Final position error in cm
    """
    theta1 = initial_theta1
    theta2 = initial_theta2
    
    for i in range(max_iterations):
        theta1, theta2, error = jacobian_ik_step(
            r1, r2, theta1, theta2, target_x, target_y, gain
        )
        
        # Optionally enforce joint limits
        if enforce_limits:
            theta1, theta2 = clamp_angles(theta1, theta2)
        
        # Check convergence
        if error < tolerance:
            return theta1, theta2, True, i + 1, error
    
    return theta1, theta2, False, max_iterations, error


def jacobian_ik_trajectory(r1, r2, waypoints, 
                           initial_theta1=0.0, initial_theta2=0.0,
                           tolerance=0.01, gain=0.5):
    """
    Compute joint angles for a trajectory of waypoints.
    
    Uses the solution from each waypoint as the initial guess for the next,
    ensuring smooth motion.
    
    Parameters:
    -----------
    r1, r2 : float
        Link lengths in cm
    waypoints : list of (x, y) tuples
        Target positions in robot coordinates
    initial_theta1, initial_theta2 : float
        Starting angles for first waypoint
    tolerance : float
        Position error tolerance in cm
    gain : float
        Step size multiplier
    
    Returns:
    --------
    angles : list of (theta1, theta2) tuples
        Joint angles for each waypoint
    all_converged : bool
        True if all waypoints converged
    """
    angles = []
    theta1, theta2 = initial_theta1, initial_theta2
    all_converged = True
    
    for x, y in waypoints:
        theta1, theta2, converged, _, _ = jacobian_ik(
            r1, r2, x, y,
            initial_theta1=theta1, initial_theta2=theta2,
            tolerance=tolerance, gain=gain
        )
        angles.append((theta1, theta2))
        
        if not converged:
            all_converged = False
    
    return angles, all_converged


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    from FK import get_end_effector_position
    
    print("=" * 70)
    print("Jacobian IK Test (FIXED VERSION)")
    print(f"Link lengths: L1 = {L1} cm, L2 = {L2} cm")
    print("=" * 70)
    
    # Test positions
    test_targets = [
        (0, 24, "Home (fully extended)"),
        (0, 18, "Along +Y axis"),
        (6, 18, "Right side"),
        (-6, 18, "Left side"),
        (-2, 20, "Paper(4,8) target"),
        (0, 12, "Minimum useful reach"),
    ]
    
    print(f"\n{'Target':^15} {'Result θ1':>10} {'Result θ2':>10} {'FK_X':>8} {'FK_Y':>8} {'Error':>8} {'Conv':>6}")
    print("-" * 75)
    
    for tx, ty, desc in test_targets:
        t1, t2, conv, iters, err = jacobian_ik(L1, L2, tx, ty)
        
        # Verify with FK
        fx, fy = get_end_effector_position(L1, L2, t1, t2)
        
        conv_str = "✓" if conv else "✗"
        print(f"({tx:>5.1f}, {ty:>5.1f})  {t1:>9.2f}° {t2:>9.2f}° {fx:>7.2f} {fy:>7.2f} {err:>7.4f}cm {conv_str:>5}")
    
    print("\n" + "=" * 70)
    print("Verification: Target (-2, 20) = Paper(4, 8)")
    print("=" * 70)
    
    t1, t2, conv, iters, err = jacobian_ik(L1, L2, -2, 20)
    print(f"Result: θ1 = {t1:.4f}°, θ2 = {t2:.4f}°")
    print(f"Converged: {conv}, Iterations: {iters}, Error: {err:.6f} cm")
    
    # Verify with FK
    fx, fy = get_end_effector_position(L1, L2, t1, t2)
    print(f"FK Verification: ({fx:.4f}, {fy:.4f})")
    print(f"Target was: (-2, 20)")
    
    match = abs(fx - (-2)) < 0.1 and abs(fy - 20) < 0.1
    print(f"Match: {'✓ CORRECT' if match else '✗ WRONG'}")
    
    print("\n" + "=" * 70)
    print("Trajectory Test: Line from (0, 18) to (-6, 18)")
    print("=" * 70)
    
    # Generate simple line trajectory
    waypoints = [(x, 18) for x in np.linspace(0, -6, 5)]
    angles, all_conv = jacobian_ik_trajectory(L1, L2, waypoints)
    
    print(f"{'Waypoint':^15} {'θ1':>10} {'θ2':>10} {'FK Check':>15}")
    print("-" * 55)
    for (x, y), (t1, t2) in zip(waypoints, angles):
        fx, fy = get_end_effector_position(L1, L2, t1, t2)
        check = "✓" if abs(fx - x) < 0.1 and abs(fy - y) < 0.1 else "✗"
        print(f"({x:>5.1f}, {y:>5.1f})  {t1:>9.2f}° {t2:>9.2f}° ({fx:>5.2f}, {fy:>5.2f}) {check}")
    print(f"\nAll converged: {'✓ YES' if all_conv else '✗ NO'}")
import numpy as np



def forward_kinematics(r1, r2, theta1, theta2):
    # Calculate end-effector position and orientation for a 2-link planar robot.
    
    # θ1=0° means robot pointing UP (+Y direction toward paper)
    # Standard math: 0° is +X, so we add 90° to make 0° point to +Y
    theta1_adjusted = theta1 + 90.0
    
    # Convert degrees to radians
    theta1_rad = np.radians(theta1_adjusted)
    theta2_rad = np.radians(theta2)
    
    # Transformation matrix from base frame (0) to link 1 frame (1)
    A1_0 = np.array([
        [np.cos(theta1_rad), -np.sin(theta1_rad), 0, r1 * np.cos(theta1_rad)],
        [np.sin(theta1_rad),  np.cos(theta1_rad), 0, r1 * np.sin(theta1_rad)],
        [0,                   0,                   1, 0],
        [0,                   0,                   0, 1]
    ])
    
    # Transformation matrix from link 1 frame (1) to link 2 frame (2)
    A2_1 = np.array([
        [np.cos(theta2_rad), -np.sin(theta2_rad), 0, r2 * np.cos(theta2_rad)],
        [np.sin(theta2_rad),  np.cos(theta2_rad), 0, r2 * np.sin(theta2_rad)],
        [0,                   0,                   1, 0],
        [0,                   0,                   0, 1]
    ])
    
    # Multiply transformation matrices
    T = np.matmul(A1_0, A2_1)
    
    return T


def get_end_effector_position(r1, r2, theta1, theta2):
    """
    Simplified function to get just the (x, y) position of end-effector.
    
    Parameters:
    -----------
    r1, r2 : float
        Link lengths in cm
    theta1, theta2 : float
        Joint angles in DEGREES
    
    Returns:
    --------
    x, y : float
        End-effector position in Cartesian coordinates (cm)
    """
    T = forward_kinematics(r1, r2, theta1, theta2)
    x = -T[0, 3]
    y = T[1, 3]
    return x, y


# ============================================
# EXAMPLE USAGE & TESTING
# ============================================

if __name__ == "__main__":
    # Import plotting functions HERE (inside main block) to avoid circular import ✅
    from workspace_plotter import plot_workspace, plot_robot_configuration
    
    # Your robot parameters (adjust these to match your actual setup)
    L1 = 8.0  # Link 1 length in cm (Motor A to Motor B)
    L2 = 8.0  # Link 2 length in cm (Motor B to end-effector)
    
    print("=" * 60)
    print("Forward Kinematics Function Test")
    print("=" * 60)
    
    # # Test Case 1: Both joints at 0° (fully extended to the right)
    # print("\nTest 1: θ₁=0°, θ₂=0°")
    # T = forward_kinematics(L1, L2, 0, 0)
    # x, y = get_end_effector_position(L1, L2, 0, 0)
    # print(f"End-effector position: x={x:.2f} cm, y={y:.2f} cm")
    # print(f"Expected: x={L1+L2:.2f} cm, y=0.00 cm")
    
    # Test Case 2: First joint at 90° (pointing up)
    print("\nTest 2: θ₁=0°, θ₂=90°")
    x, y = get_end_effector_position(L1, L2, 0, 90)
    print(f"End-effector position: x={x:.2f} cm, y={y:.2f} cm")
    
    # # Test Case 3: Both joints at 90°
    # print("\nTest 3: θ₁=90°, θ₂=90°")
    # x, y = get_end_effector_position(L1, L2, 90, 90)
    # print(f"End-effector position: x={x:.2f} cm, y={y:.2f} cm")
    
    # # Test Case 4: Your current motor positions (example)
    # print("\nTest 4: θ₁=45°, θ₂=45°")
    # x, y = get_end_effector_position(L1, L2, 45, 45)
    # print(f"End-effector position: x={x:.2f} cm, y={y:.2f} cm")
    
    # Display full transformation matrix
    print("\n" + "=" * 60)
    print("Full Transformation Matrix T:")
    print("=" * 60)
    T = forward_kinematics(L1, L2, 0, 90)
    print(T)
    print(f"\nEnd-effector position: [{T[0,3]:.2f}, {T[1,3]:.2f}]")
    
    # ============================================
    # WORKSPACE PLOTTING
    # ============================================
    
    print("\n" + "=" * 60)
    print("WORKSPACE VISUALIZATION")
    print("=" * 60)
    
    # Plot 1: Full workspace
    print("\n[Plot 1] Generating full workspace...")
    fig1 = plot_workspace(L1, L2, theta1_range=(0, 180), theta2_range=(0, 180), resolution=0)
    fig1.show()
    
    # Plot 2: Example robot configurations
    print("\n[Plot 2] Visualizing robot configurations...")
    
    configs = [
        (0, 0, "Fully extended right"),
        (90, 0, "Pointing up"),
        (90, 90, "Elbow bent up"),
        (45, 45, "Mid-range pose"),
    ]
    
    for theta1, theta2, description in configs:
        print(f"  - θ₁={theta1}°, θ₂={theta2}° ({description})")
        fig = plot_robot_configuration(L1, L2, theta1, theta2)
        fig.show()
    
    print("\n" + "=" * 60)
    print("✓ All tests and plots complete!")
    print("=" * 60)

def get_elbow_position(r1, theta1):
    """
    Get (x, y) position of the elbow joint (end of link 1).
    
    Parameters:
    -----------
    r1 : float
        Length of link 1 in cm
    theta1 : float
        Joint 1 angle in DEGREES
    
    Returns:
    --------
    x, y : float
        Elbow position in robot coordinates (cm)
    """
    # Apply same 90° offset as FK - theta1=0 points +Y
    theta1_adjusted = theta1 + 90.0
    theta1_rad = np.radians(theta1_adjusted)
    
    x = -r1 * np.cos(theta1_rad)
    y = r1 * np.sin(theta1_rad)
    return x, y

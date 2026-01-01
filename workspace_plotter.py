"""
Workspace Plotting for 2-Link Planar Robot
Uses Plotly for interactive visualization
"""

import numpy as np
import plotly.graph_objects as go
from FK import forward_kinematics, get_end_effector_position


def plot_workspace(r1, r2, theta1_range=(0, 180), theta2_range=(0, 180), resolution=100):
    """
    Plot the reachable workspace of a 2-link planar robot using Plotly (interactive!)

    Parameters:
    -----------
    r1 : float
        Length of link 1 (cm)
    r2 : float
        Length of link 2 (cm)
    theta1_range : tuple
        (min, max) angles for joint 1 in degrees (default: 0° to 180°)
    theta2_range : tuple
        (min, max) angles for joint 2 in degrees (default: 0° to 180°)
    resolution : int
        Number of samples per joint (higher = more detailed but slower)

    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Interactive Plotly figure object
    """
    
    print("=" * 60)
    print("Generating Workspace...")
    print(f"Link 1: {r1} cm, Link 2: {r2} cm")
    print(f"θ₁ range: {theta1_range[0]}° to {theta1_range[1]}°")
    print(f"θ₂ range: {theta2_range[0]}° to {theta2_range[1]}°")
    print(f"Resolution: {resolution} samples per joint")
    print(f"Total configurations: {resolution * resolution}")
    print("=" * 60)
    
    # Generate angle samples
    theta1_samples = np.linspace(theta1_range[0], theta1_range[1], resolution)
    theta2_samples = np.linspace(theta2_range[0], theta2_range[1], resolution)

    # Compute workspace by sweeping through all joint angle combinations
    workspace_x = []
    workspace_y = []

    for th1 in theta1_samples:
        for th2 in theta2_samples:
            x, y = get_end_effector_position(r1, r2, th1, th2)
            workspace_x.append(x)
            workspace_y.append(y)

    print(f"✓ Computed {len(workspace_x)} reachable points")
    
    # Create figure
    fig = go.Figure()

    # Calculate theoretical reach limits
    max_reach = r1 + r2  # Fully extended
    min_reach = abs(r1 - r2)  # Fully folded

    # Add max reach circle (outer boundary)
    theta_circle = np.linspace(0, 2*np.pi, 100)
    x_max = max_reach * np.cos(theta_circle)
    y_max = max_reach * np.sin(theta_circle)

    fig.add_trace(go.Scatter(
        x=x_max,
        y=y_max,
        mode='lines',
        name=f'Max Reach ({max_reach:.1f} cm)',
        line=dict(color='green', width=3, dash='dash'),
        hoverinfo='skip'
    ))

    # Add min reach circle (inner boundary - only if r1 ≠ r2)
    if min_reach > 0:
        x_min = min_reach * np.cos(theta_circle)
        y_min = min_reach * np.sin(theta_circle)

        fig.add_trace(go.Scatter(
            x=x_min,
            y=y_min,
            mode='lines',
            name=f'Min Reach ({min_reach:.1f} cm)',
            line=dict(color='red', width=3, dash='dash'),
            hoverinfo='skip'
        ))

    # Plot workspace points (the actual reachable area)
    fig.add_trace(go.Scatter(
        x=workspace_x,
        y=workspace_y,
        mode='markers',
        name='Reachable Workspace',
        marker=dict(
            size=3,
            color='deeppink',
            opacity=0.3
        ),
        hovertemplate='Reachable Point<br>X: %{x:.2f} cm<br>Y: %{y:.2f} cm<extra></extra>'
    ))

    # Mark base position (origin)
    fig.add_trace(go.Scatter(
        x=[0],
        y=[0],
        mode='markers',
        name='Base (Motor A)',
        marker=dict(size=15, color='black', symbol='circle'),
        hovertemplate='Base<br>X: 0 cm<br>Y: 0 cm<extra></extra>'
    ))

    # Update layout with clean grid and proper scaling
    fig.update_layout(
        title=dict(
            text=f'<b>2-Link Planar Robot Workspace</b><br>'
                 f'Link 1: {r1} cm, Link 2: {r2} cm<br>'
                 f'θ₁ ∈ [{theta1_range[0]}°, {theta1_range[1]}°], '
                 f'θ₂ ∈ [{theta2_range[0]}°, {theta2_range[1]}°]',
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        xaxis=dict(
            title='X Position (cm)',
            range=[-max_reach*1.2, max_reach*1.2],
            scaleanchor='y',
            scaleratio=1,
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='Gray'
        ),
        yaxis=dict(
            title='Y Position (cm)',
            range=[-max_reach*1.2, max_reach*1.2],
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='Gray'
        ),
        hovermode='closest',
        plot_bgcolor='white',
        width=900,
        height=900,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="Black",
            borderwidth=1
        )
    )

    print("✓ Workspace plot generated successfully!")
    print("=" * 60)
    
    return fig


def plot_robot_configuration(r1, r2, theta1, theta2, show_workspace_boundary=True):
    """
    Plot the robot arm in a specific configuration using Plotly (interactive!)
    Shows both links, all joints (base, elbow, end-effector), and optional workspace boundary.

    Parameters:
    -----------
    r1 : float
        Length of link 1 in cm
    r2 : float
        Length of link 2 in cm
    theta1 : float
        Joint 1 angle in degrees
    theta2 : float
        Joint 2 angle in degrees
    show_workspace_boundary : bool
        If True, shows the maximum reach circle (default: True)

    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Interactive Plotly figure object
    """
    
    # Calculate joint positions
    theta1_rad = np.radians(theta1)
    theta2_rad = np.radians(theta2)
    
    # Base at origin
    base_x, base_y = 0, 0
    
    # Elbow (end of link 1)
    elbow_x = r1 * np.cos(theta1_rad)
    elbow_y = r1 * np.sin(theta1_rad)
    
    # End-effector (end of link 2)
    ee_x, ee_y = get_end_effector_position(r1, r2, theta1, theta2)
    
    # Create figure
    fig = go.Figure()

    # Plot Link 1 (base to elbow)
    fig.add_trace(go.Scatter(
        x=[base_x, elbow_x],
        y=[base_y, elbow_y],
        mode='lines+markers',
        name=f'Link 1 ({r1} cm)',
        line=dict(color='blue', width=8),
        marker=dict(size=12, color='blue'),
        hovertemplate='Link 1<br>X: %{x:.2f} cm<br>Y: %{y:.2f} cm<extra></extra>'
    ))

    # Plot Link 2 (elbow to end-effector)
    fig.add_trace(go.Scatter(
        x=[elbow_x, ee_x],
        y=[elbow_y, ee_y],
        mode='lines+markers',
        name=f'Link 2 ({r2} cm)',
        line=dict(color='red', width=8),
        marker=dict(size=12, color='red'),
        hovertemplate='Link 2<br>X: %{x:.2f} cm<br>Y: %{y:.2f} cm<extra></extra>'
    ))

    # Plot joints with different colors and sizes
    # Base (Motor A - EMG30)
    fig.add_trace(go.Scatter(
        x=[base_x],
        y=[base_y],
        mode='markers',
        name='Base (Motor A)',
        marker=dict(size=20, color='black', symbol='circle'),
        hovertemplate=f'Base<br>θ₁: {theta1:.1f}°<br>X: 0 cm<br>Y: 0 cm<extra></extra>'
    ))

    # Elbow (Motor B - Pololu)
    fig.add_trace(go.Scatter(
        x=[elbow_x],
        y=[elbow_y],
        mode='markers',
        name='Elbow (Motor B)',
        marker=dict(size=16, color='green', symbol='circle'),
        hovertemplate=f'Elbow<br>θ₂: {theta2:.1f}°<br>X: {elbow_x:.2f} cm<br>Y: {elbow_y:.2f} cm<extra></extra>'
    ))

    # End-Effector
    fig.add_trace(go.Scatter(
        x=[ee_x],
        y=[ee_y],
        mode='markers',
        name='End-Effector',
        marker=dict(size=16, color='orange', symbol='star'),
        hovertemplate=f'End-Effector<br>X: {ee_x:.2f} cm<br>Y: {ee_y:.2f} cm<extra></extra>'
    ))

    # Add workspace boundary circle (max reach)
    if show_workspace_boundary:
        max_reach = r1 + r2
        theta_circle = np.linspace(0, 2*np.pi, 100)
        x_circle = max_reach * np.cos(theta_circle)
        y_circle = max_reach * np.sin(theta_circle)

        fig.add_trace(go.Scatter(
            x=x_circle,
            y=y_circle,
            mode='lines',
            name=f'Max Reach ({max_reach:.1f} cm)',
            line=dict(color='gray', width=2, dash='dash'),
            hoverinfo='skip'
        ))

    # Update layout with clean grid
    max_reach = r1 + r2
    fig.update_layout(
        title=dict(
            text=f'<b>Robot Arm Configuration</b><br>'
                 f'θ₁={theta1:.1f}°, θ₂={theta2:.1f}°<br>'
                 f'End-Effector: ({ee_x:.2f}, {ee_y:.2f}) cm',
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        xaxis=dict(
            title='X Position (cm)',
            range=[-max_reach*1.2, max_reach*1.2],
            scaleanchor='y',
            scaleratio=1,
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='Gray'
        ),
        yaxis=dict(
            title='Y Position (cm)',
            range=[-max_reach*1.2, max_reach*1.2],
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='Gray'
        ),
        hovermode='closest',
        plot_bgcolor='white',
        width=800,
        height=800,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="Black",
            borderwidth=1
        )
    )

    return fig
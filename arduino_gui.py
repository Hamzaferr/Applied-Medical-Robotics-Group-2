"""
2-Link Robot Arm Controller GUI - TRAJECTORY BATCH VERSION

Combines:
- Old GUI design (5 tabs, all options)
- New serial_reader thread (no more graph freezing)
- Trajectory batch sending for smooth motion

Author: Yagmur
Date: December 2025
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import time
import math
import numpy as np

# Import robot modules
from coords import (L1, L2, THETA1_MIN, THETA1_MAX, THETA2_MIN, THETA2_MAX,
                    paper_to_robot, robot_to_paper, is_reachable_robot,
                    angles_within_limits, HOME_THETA1, HOME_THETA2)
from FK import get_end_effector_position, get_elbow_position
from IK import inverse_kinematics
from jacobian_ik import jacobian_ik
from shapes import TrajectoryGenerator
from serial_reader import get_reader, SerialReader
from evaluation import get_evaluator, RobotEvaluator, DEFAULT_JOINT_TEST_ANGLES, DEFAULT_CARTESIAN_TEST_POINTS


# =============================================================================
# GLOBAL STATE
# =============================================================================

# Serial reader (single owner of serial communication)
reader = get_reader()

# Evaluator
evaluator = get_evaluator()

# Trajectory generators
traj_gen = TrajectoryGenerator()

# Live position tracking (updated from serial_reader)
live_theta1 = 0.0
live_theta2 = 0.0

# Drawing state
drawing_active = False
drawing_trajectory = []
drawing_angles = []
drawing_delays = []
drawing_index = 0
drawing_last_send_time = 0
drawing_waypoints = []


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_arm_figure(theta1, theta2, target_robot=None, ghost_angles=None,
                      shape_waypoints=None, show_paper=True, mirror_x=False, 
                      title="", highlight_index=None):
    """Create a Plotly figure showing the robot arm configuration."""
    mx = -1 if mirror_x else 1
    
    base_x, base_y = 0, 0
    elbow_x, elbow_y = get_elbow_position(L1, theta1)
    ee_x, ee_y = get_end_effector_position(L1, L2, theta1, theta2)
    
    elbow_x *= mx
    ee_x *= mx
    
    fig = go.Figure()
    
    # Paper overlay
    if show_paper:
        paper_corners = [
            paper_to_robot(0, 0), paper_to_robot(12, 0),
            paper_to_robot(12, 12), paper_to_robot(0, 12), paper_to_robot(0, 0)
        ]
        px = [p[0] * mx for p in paper_corners]
        py = [p[1] for p in paper_corners]
        
        fig.add_trace(go.Scatter(
            x=px, y=py, mode='lines',
            line=dict(color='lightgray', width=2, dash='dash'),
            name='Paper (12x12)', hoverinfo='skip'
        ))
        
        center = paper_to_robot(6, 6)
        fig.add_trace(go.Scatter(
            x=[center[0] * mx], y=[center[1]],
            mode='markers', marker=dict(size=8, color='lightgray', symbol='x'),
            name='Paper Center', hoverinfo='skip'
        ))
    
    # Workspace circle
    theta_range = [i * 2 * math.pi / 100 for i in range(101)]
    ws_x = [(L1 + L2) * math.cos(t) * mx for t in theta_range]
    ws_y = [(L1 + L2) * math.sin(t) for t in theta_range]
    
    fig.add_trace(go.Scatter(
        x=ws_x, y=ws_y, mode='lines',
        line=dict(color='rgba(100,100,100,0.3)', width=1),
        name='Workspace', hoverinfo='skip'
    ))
    
    # Shape path
    if shape_waypoints and len(shape_waypoints) > 0:
        shape_x = [p[0] * mx for p in shape_waypoints]
        shape_y = [p[1] for p in shape_waypoints]
        
        fig.add_trace(go.Scatter(
            x=shape_x, y=shape_y, mode='lines+markers',
            line=dict(color='orange', width=2),
            marker=dict(size=4, color='orange'),
            name='Shape Path',
            hovertemplate='(%{x:.1f}, %{y:.1f})<extra></extra>'
        ))
        
        if highlight_index is not None and 0 <= highlight_index < len(shape_waypoints):
            hx, hy = shape_waypoints[highlight_index]
            fig.add_trace(go.Scatter(
                x=[hx * mx], y=[hy],
                mode='markers',
                marker=dict(size=12, color='red', symbol='circle'),
                name='Current', hoverinfo='skip'
            ))
    
    # Ghost/Preview arm
    if ghost_angles is not None:
        ghost_t1, ghost_t2 = ghost_angles
        ghost_elbow_x, ghost_elbow_y = get_elbow_position(L1, ghost_t1)
        ghost_ee_x, ghost_ee_y = get_end_effector_position(L1, L2, ghost_t1, ghost_t2)
        
        fig.add_trace(go.Scatter(
            x=[base_x, ghost_elbow_x * mx, ghost_ee_x * mx],
            y=[base_y, ghost_elbow_y, ghost_ee_y],
            mode='lines+markers',
            line=dict(color='rgba(0, 200, 0, 0.4)', width=6, dash='dot'),
            marker=dict(size=[10, 8, 12], color='rgba(0, 200, 0, 0.4)'),
            name='Preview', hoverinfo='skip'
        ))
    
    # LIVE Robot arm
    fig.add_trace(go.Scatter(
        x=[base_x, elbow_x, ee_x],
        y=[base_y, elbow_y, ee_y],
        mode='lines+markers',
        line=dict(color='blue', width=5),
        marker=dict(size=[14, 12, 16], color=['black', 'blue', 'red']),
        name='Robot Arm (Live)',
        hovertemplate='(%{x:.1f}, %{y:.1f})<extra></extra>'
    ))
    
    # Target marker
    if target_robot is not None:
        fig.add_trace(go.Scatter(
            x=[target_robot[0] * mx], y=[target_robot[1]],
            mode='markers',
            marker=dict(size=15, color='green', symbol='x', line=dict(width=3)),
            name='Target'
        ))
    
    fig.update_layout(
        xaxis=dict(range=[-30, 30], scaleanchor='y', scaleratio=1,
                   title='X (cm)' + (' [MIRRORED]' if mirror_x else ''),
                   gridcolor='lightgray', zeroline=True, zerolinecolor='gray'),
        yaxis=dict(range=[-5, 30], title='Y (cm)',
                   gridcolor='lightgray', zeroline=True, zerolinecolor='gray'),
        showlegend=True,
        legend=dict(x=0, y=1, bgcolor='rgba(255,255,255,0.8)'),
        margin=dict(l=60, r=20, t=40, b=60),
        plot_bgcolor='white',
        height=550,
        title=dict(text=title, x=0.5) if title else None
    )
    
    return fig


# =============================================================================
# DASH APP
# =============================================================================

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True)

# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------
header = dbc.Card([
    dbc.CardBody([
        dbc.Row([
            dbc.Col([html.H4("2-Link Robot Controller", className="mb-0")], width=2),
            dbc.Col([
                dbc.InputGroup([
                    dbc.Button("Connect", id='connect-btn', color="primary", size="sm"),
                ], size="sm")
            ], width=2),
            dbc.Col([
                dbc.Button("🏠 GO HOME", id='home-btn', color="success", className="w-100")
            ], width=1),
            dbc.Col([
                dbc.Button("⛔ STOP", id='stop-btn', color="danger", className="w-100")
            ], width=1),
            dbc.Col([
                dbc.Checklist(id='mirror-toggle',
                    options=[{'label': ' Mirror X', 'value': 'mirror'}],
                    value=[], inline=True, switch=True)
            ], width=1),
            dbc.Col([html.Div(id='status-display', className="text-muted small")], width=2),
            dbc.Col([html.Div(id='live-position-display', className="small font-monospace")], width=3)
        ], align="center")
    ])
], className="mb-3")


# -----------------------------------------------------------------------------
# TAB 1: Manual FK
# -----------------------------------------------------------------------------
tab_manual_fk = dbc.Card([dbc.CardBody([
    dbc.Row([
        dbc.Col([
            html.H5("Joint Angles (Manual)"),
            html.Label("θ1 (Base)"),
            dcc.Slider(id='theta1-slider', min=THETA1_MIN, max=THETA1_MAX, step=1, value=0,
                marks={int(THETA1_MIN): f'{int(THETA1_MIN)}°', 0: '0°', int(THETA1_MAX): f'{int(THETA1_MAX)}°'},
                tooltip={"placement": "bottom", "always_visible": True}),
            html.Br(),
            html.Label("θ2 (Elbow)"),
            dcc.Slider(id='theta2-slider', min=THETA2_MIN, max=THETA2_MAX, step=1, value=0,
                marks={int(THETA2_MIN): f'{int(THETA2_MIN)}°', 0: '0°', int(THETA2_MAX): f'{int(THETA2_MAX)}°'},
                tooltip={"placement": "bottom", "always_visible": True}),
            html.Br(),
            dbc.Button("Send to Robot", id='fk-send-btn', color="primary", className="w-100"),
            html.Hr(),
            html.Div(id='fk-result', className="mt-2")
        ], width=4),
        dbc.Col([dcc.Graph(id='fk-plot', config={'displayModeBar': False})], width=8)
    ])
])])


# -----------------------------------------------------------------------------
# TAB 2: Analytic IK
# -----------------------------------------------------------------------------
tab_analytic_ik = dbc.Card([dbc.CardBody([
    dbc.Row([
        dbc.Col([
            html.H5("Target Position (Paper Coords)"),
            html.P("Paper: 12×12 cm, (0,0) bottom-left", className="text-muted small"),
            dbc.Row([
                dbc.Col([html.Label("Paper X"), dbc.Input(id='aik-paper-x', type='number', value=6, min=0, max=12, step=0.5)]),
                dbc.Col([html.Label("Paper Y"), dbc.Input(id='aik-paper-y', type='number', value=12, min=0, max=12, step=0.5)])
            ]),
            html.Br(),
            html.Label("Elbow Configuration"),
            dbc.RadioItems(id='aik-elbow', options=[
                {'label': 'Elbow Down', 'value': 'down'},
                {'label': 'Elbow Up', 'value': 'up'}
            ], value='down', inline=True),
            html.Br(),
            dbc.ButtonGroup([
                dbc.Button("Calculate IK", id='aik-calc-btn', color="secondary"),
                dbc.Button("Send to Robot", id='aik-send-btn', color="primary"),
            ], className="w-100"),
            html.Hr(),
            html.Div(id='aik-result', className="mt-2")
        ], width=4),
        dbc.Col([dcc.Graph(id='aik-plot', config={'displayModeBar': False})], width=8)
    ])
])])


# -----------------------------------------------------------------------------
# TAB 3: Jacobian IK
# -----------------------------------------------------------------------------
tab_jacobian_ik = dbc.Card([dbc.CardBody([
    dbc.Row([
        dbc.Col([
            html.H5("Target Position (Paper Coords)"),
            html.P("Paper: 12×12 cm, (0,0) bottom-left", className="text-muted small"),
            dbc.Row([
                dbc.Col([html.Label("Paper X"), dbc.Input(id='jik-paper-x', type='number', value=6, min=0, max=12, step=0.5)]),
                dbc.Col([html.Label("Paper Y"), dbc.Input(id='jik-paper-y', type='number', value=12, min=0, max=12, step=0.5)])
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col([html.Label("Gain"), dbc.Input(id='jik-gain', type='number', value=0.5, min=0.1, max=1.0, step=0.1)]),
                dbc.Col([html.Label("Tolerance"), dbc.Input(id='jik-tol', type='number', value=0.01, min=0.001, max=1.0, step=0.01)])
            ]),
            html.Br(),
            dbc.ButtonGroup([
                dbc.Button("Calculate IK", id='jik-calc-btn', color="secondary"),
                dbc.Button("Send to Robot", id='jik-send-btn', color="primary"),
            ], className="w-100"),
            html.Hr(),
            html.Div(id='jik-result', className="mt-2")
        ], width=4),
        dbc.Col([dcc.Graph(id='jik-plot', config={'displayModeBar': False})], width=8)
    ])
])])


# -----------------------------------------------------------------------------
# TAB 4: Analytic Shape Drawing
# -----------------------------------------------------------------------------
tab_analytic_shapes = dbc.Card([dbc.CardBody([
    dbc.Row([
        dbc.Col([
            html.H5("Shape Drawing (Analytic IK)"),
            html.Label("Shape Type"),
            dbc.Select(id='ashape-type', options=[
                {'label': 'Square', 'value': 'square'},
                {'label': 'Circle', 'value': 'circle'},
                {'label': 'Triangle', 'value': 'triangle'},
                {'label': 'Star', 'value': 'star'},
                {'label': 'Rectangle', 'value': 'rectangle'},
                {'label': 'Line', 'value': 'line'},
            ], value='square'),
            html.Br(),
            dbc.Row([
                dbc.Col([html.Label("Start/Center X"), dbc.Input(id='ashape-cx', type='number', value=6, min=0, max=12, step=0.5)]),
                dbc.Col([html.Label("Start/Center Y"), dbc.Input(id='ashape-cy', type='number', value=6, min=0, max=12, step=0.5)])
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col([html.Label("End X (Line)"), dbc.Input(id='ashape-ex', type='number', value=10, min=0, max=12, step=0.5)]),
                dbc.Col([html.Label("End Y (Line)"), dbc.Input(id='ashape-ey', type='number', value=6, min=0, max=12, step=0.5)])
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col([html.Label("Size (cm)"), dbc.Input(id='ashape-size', type='number', value=3, min=0.5, max=10, step=0.5)]),
                dbc.Col([html.Label("Velocity"), dbc.Input(id='ashape-vel', type='number', value=2, min=0.5, max=10, step=0.5)])
            ]),
            html.Br(),
            html.Label("Quality Preset"),
            dbc.RadioItems(id='ashape-quality', options=[
                {'label': 'Fast', 'value': 'fast'},
                {'label': 'Normal', 'value': 'normal'},
                {'label': 'Smooth', 'value': 'smooth'},
            ], value='normal', inline=True),
            html.Br(),
            html.Label("Elbow Configuration"),
            dbc.RadioItems(id='ashape-elbow', options=[
                {'label': 'Down', 'value': 'down'}, {'label': 'Up', 'value': 'up'}
            ], value='down', inline=True),
            html.Br(),
            dbc.Checklist(id='ashape-options', 
                         options=[{"label": "Corner Slowdown", "value": "slowdown"}],
                         value=["slowdown"]),
            html.Br(),
            dbc.ButtonGroup([
                dbc.Button("Preview", id='ashape-preview-btn', color="secondary"),
                dbc.Button("▶ Draw", id='ashape-draw-btn', color="success"),
                dbc.Button("⏹ Stop", id='ashape-stop-btn', color="danger"),
            ], className="w-100"),
            html.Br(), html.Br(),
            dbc.Progress(id='ashape-progress', value=0, striped=True, animated=True, className="mb-2"),
            html.Div(id='ashape-result', className="mt-2")
        ], width=4),
        dbc.Col([dcc.Graph(id='ashape-plot', config={'displayModeBar': False})], width=8)
    ])
])])


# -----------------------------------------------------------------------------
# TAB 5: Jacobian Shape Drawing
# -----------------------------------------------------------------------------
tab_jacobian_shapes = dbc.Card([dbc.CardBody([
    dbc.Row([
        dbc.Col([
            html.H5("Shape Drawing (Jacobian IK)"),
            html.Label("Shape Type"),
            dbc.Select(id='jshape-type', options=[
                {'label': 'Square', 'value': 'square'},
                {'label': 'Circle', 'value': 'circle'},
                {'label': 'Triangle', 'value': 'triangle'},
                {'label': 'Star', 'value': 'star'},
                {'label': 'Rectangle', 'value': 'rectangle'},
                {'label': 'Line', 'value': 'line'},
            ], value='square'),
            html.Br(),
            dbc.Row([
                dbc.Col([html.Label("Start/Center X"), dbc.Input(id='jshape-cx', type='number', value=6, min=0, max=12, step=0.5)]),
                dbc.Col([html.Label("Start/Center Y"), dbc.Input(id='jshape-cy', type='number', value=6, min=0, max=12, step=0.5)])
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col([html.Label("End X (Line)"), dbc.Input(id='jshape-ex', type='number', value=10, min=0, max=12, step=0.5)]),
                dbc.Col([html.Label("End Y (Line)"), dbc.Input(id='jshape-ey', type='number', value=6, min=0, max=12, step=0.5)])
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col([html.Label("Size (cm)"), dbc.Input(id='jshape-size', type='number', value=3, min=0.5, max=10, step=0.5)]),
                dbc.Col([html.Label("Velocity"), dbc.Input(id='jshape-vel', type='number', value=2, min=0.5, max=10, step=0.5)])
            ]),
            html.Br(),
            html.Label("Quality Preset"),
            dbc.RadioItems(id='jshape-quality', options=[
                {'label': 'Fast', 'value': 'fast'},
                {'label': 'Normal', 'value': 'normal'},
                {'label': 'Smooth', 'value': 'smooth'},
            ], value='normal', inline=True),
            html.Br(),
            dbc.Checklist(id='jshape-options', 
                         options=[{"label": "Corner Slowdown", "value": "slowdown"}],
                         value=["slowdown"]),
            html.Br(),
            dbc.ButtonGroup([
                dbc.Button("Preview", id='jshape-preview-btn', color="secondary"),
                dbc.Button("▶ Draw", id='jshape-draw-btn', color="success"),
                dbc.Button("⏹ Stop", id='jshape-stop-btn', color="danger"),
            ], className="w-100"),
            html.Br(), html.Br(),
            dbc.Progress(id='jshape-progress', value=0, striped=True, animated=True, className="mb-2"),
            html.Div(id='jshape-result', className="mt-2")
        ], width=4),
        dbc.Col([dcc.Graph(id='jshape-plot', config={'displayModeBar': False})], width=8)
    ])
])])


# -----------------------------------------------------------------------------
# TAB 6: Evaluation
# -----------------------------------------------------------------------------
tab_evaluation = dbc.Card([dbc.CardBody([
    dbc.Row([
        # Left column: Test controls
        dbc.Col([
            html.H5("🔬 Robot Evaluation Suite"),
            html.P("Quantitative testing of accuracy and repeatability", className="text-muted small"),
            html.Hr(),
            
            # Test selection
            html.H6("1. Select Tests to Run"),
            dbc.Checklist(
                id='eval-test-selection',
                options=[
                    {'label': ' Joint Space Repeatability', 'value': 'joint'},
                    {'label': ' Cartesian Accuracy (IK)', 'value': 'cartesian'},
                    {'label': ' Line Trajectory (Batch Mode)', 'value': 'trajectory'},
                    {'label': ' Circle Trajectory (Batch Mode)', 'value': 'circle'},
                ],
                value=['joint', 'cartesian'],
                inline=False
            ),
            html.Br(),
            
            # Test parameters
            html.H6("2. Test Parameters"),
            dbc.Row([
                dbc.Col([
                    html.Label("Repetitions"),
                    dbc.Input(id='eval-repetitions', type='number', value=3, min=1, max=10, step=1)
                ]),
                dbc.Col([
                    html.Label("Settle Time (ms)"),
                    dbc.Input(id='eval-settle-time', type='number', value=1500, min=500, max=5000, step=100)
                ])
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    html.Label("Drawing Velocity (cm/s)"),
                    dbc.Input(id='eval-velocity', type='number', value=2.0, min=0.5, max=5.0, step=0.5)
                ]),
            ]),
            html.Br(),
            
            # Joint test angles
            html.Label("Joint Test Angles (comma-separated)"),
            dbc.Input(id='eval-joint-angles', type='text', value="30, 60, 90, -30, -60"),
            html.Br(),
            
            # Cartesian test points
            html.Label("Cartesian Test Points (x,y pairs)"),
            dbc.Textarea(id='eval-cartesian-points', 
                        value="4,4\n8,4\n4,8\n8,8\n6,6",
                        style={'height': '80px', 'fontFamily': 'monospace'}),
            html.Br(),
            
            # Trajectory test line
            html.Label("Line Test (start → end)"),
            dbc.Row([
                dbc.Col([dbc.Input(id='eval-traj-start-x', type='number', value=4, min=0, max=12, step=0.5)]),
                dbc.Col([dbc.Input(id='eval-traj-start-y', type='number', value=6, min=0, max=12, step=0.5)]),
                dbc.Col([html.Span("→", className="text-center d-block mt-2")], width=1),
                dbc.Col([dbc.Input(id='eval-traj-end-x', type='number', value=8, min=0, max=12, step=0.5)]),
                dbc.Col([dbc.Input(id='eval-traj-end-y', type='number', value=6, min=0, max=12, step=0.5)]),
            ]),
            html.Br(),
            
            # Circle test parameters
            html.Label("Circle Test (center, radius)"),
            dbc.Row([
                dbc.Col([html.Label("X", className="small"), dbc.Input(id='eval-circle-cx', type='number', value=6, min=0, max=12, step=0.5)]),
                dbc.Col([html.Label("Y", className="small"), dbc.Input(id='eval-circle-cy', type='number', value=6, min=0, max=12, step=0.5)]),
                dbc.Col([html.Label("R", className="small"), dbc.Input(id='eval-circle-radius', type='number', value=2, min=0.5, max=4, step=0.5)]),
            ]),
            html.Br(),
            
            # Run controls
            html.Hr(),
            dbc.ButtonGroup([
                dbc.Button("▶ Run Evaluation", id='eval-run-btn', color="success", className="me-2"),
                dbc.Button("⏹ Stop", id='eval-stop-btn', color="danger"),
            ], className="w-100"),
            html.Br(), html.Br(),
            
            dbc.Progress(id='eval-progress', value=0, striped=True, animated=True),
            html.Div(id='eval-status', className="mt-2 text-muted small"),
            
        ], width=4),
        
        # Right column: Results
        dbc.Col([
            dbc.Tabs([
                dbc.Tab([
                    html.Div(id='eval-results-summary', className="p-3")
                ], label="Summary"),
                dbc.Tab([
                    html.Div(id='eval-results-joint', className="p-3", 
                            style={'maxHeight': '500px', 'overflowY': 'auto'})
                ], label="Joint Tests"),
                dbc.Tab([
                    html.Div(id='eval-results-cartesian', className="p-3",
                            style={'maxHeight': '500px', 'overflowY': 'auto'})
                ], label="Cartesian Tests"),
                dbc.Tab([
                    html.Div(id='eval-results-trajectory', className="p-3")
                ], label="Trajectory Tests"),
                dbc.Tab([
                    html.Pre(id='eval-results-json', 
                            style={'maxHeight': '500px', 'overflowY': 'auto', 'fontSize': '11px'})
                ], label="Raw JSON"),
            ])
        ], width=8)
    ])
])])


# -----------------------------------------------------------------------------
# MAIN LAYOUT
# -----------------------------------------------------------------------------
app.layout = dbc.Container([
    header,
    dbc.Tabs([
        dbc.Tab(tab_manual_fk, label="Manual FK"),
        dbc.Tab(tab_analytic_ik, label="Analytic IK"),
        dbc.Tab(tab_jacobian_ik, label="Jacobian IK"),
        dbc.Tab(tab_analytic_shapes, label="Analytic Shapes"),
        dbc.Tab(tab_jacobian_shapes, label="Jacobian Shapes"),
        dbc.Tab(tab_evaluation, label="📊 Evaluation"),
    ], id='main-tabs'),
    
    # Hidden stores
    dcc.Store(id='aik-angles-store'),
    dcc.Store(id='jik-angles-store'),
    dcc.Store(id='live-angles-store', data={'theta1': 0, 'theta2': 0}),
    dcc.Store(id='ashape-data-store'),
    dcc.Store(id='jshape-data-store'),
    dcc.Store(id='drawing-state-store', data={'active': False, 'method': None, 'index': 0}),
    dcc.Store(id='eval-results-store', data=None),
    dcc.Store(id='eval-running-store', data=False),
    
    # Intervals
    dcc.Interval(id='live-interval', interval=200, n_intervals=0),
    dcc.Interval(id='drawing-interval', interval=100, n_intervals=0, disabled=True),
    dcc.Interval(id='eval-interval', interval=500, n_intervals=0, disabled=True),
    
], fluid=True, className="p-3")


# =============================================================================
# CALLBACKS - Connection & Live Updates
# =============================================================================

@app.callback(
    [Output('connect-btn', 'children'), Output('connect-btn', 'color'), Output('status-display', 'children')],
    Input('connect-btn', 'n_clicks'),
    prevent_initial_call=True
)
def toggle_connection(_):
    if reader.is_connected():
        success, msg = reader.disconnect()
        return "Connect", "primary", msg
    else:
        success, msg = reader.connect()
        return ("Disconnect", "danger", msg) if success else ("Connect", "primary", msg)


@app.callback(
    [Output('live-angles-store', 'data'), Output('live-position-display', 'children')],
    Input('live-interval', 'n_intervals'),
    prevent_initial_call=False
)
def update_live_display(_):
    global live_theta1, live_theta2
    
    if not reader.is_connected():
        return {'theta1': live_theta1, 'theta2': live_theta2}, html.Span(
            "Not connected", style={'color': 'gray'}
        )
    
    state = reader.get_state()
    live_theta1 = state.theta1_current
    live_theta2 = state.theta2_current
    
    robot_x, robot_y = get_end_effector_position(L1, L2, live_theta1, live_theta2)
    paper_x, paper_y = robot_to_paper(robot_x, robot_y)
    
    # Trajectory status
    traj_info = ""
    if state.trajectory_active:
        traj_info = f" | TRAJ:{state.trajectory_index}/{state.trajectory_total}"
    
    display = html.Span([
        f"θ1={live_theta1:.1f}° θ2={live_theta2:.1f}° | ",
        f"Paper({paper_x:.1f}, {paper_y:.1f})",
        traj_info
    ], style={'color': 'green'})
    
    return {'theta1': live_theta1, 'theta2': live_theta2}, display


@app.callback(
    [Output('status-display', 'children', allow_duplicate=True),
     Output('drawing-state-store', 'data', allow_duplicate=True),
     Output('drawing-interval', 'disabled', allow_duplicate=True),
     Output('ashape-progress', 'value', allow_duplicate=True),
     Output('jshape-progress', 'value', allow_duplicate=True)],
    Input('home-btn', 'n_clicks'),
    prevent_initial_call=True
)
def go_home(_):
    global drawing_active, drawing_index
    
    drawing_active = False
    drawing_index = 0
    
    if reader.is_connected():
        reader.send_home()
        return "HOME sent", {'active': False, 'method': None, 'index': 0}, True, 0, 0
    return "Not connected", {'active': False, 'method': None, 'index': 0}, True, 0, 0


@app.callback(
    Output('status-display', 'children', allow_duplicate=True),
    Input('stop-btn', 'n_clicks'),
    prevent_initial_call=True
)
def emergency_stop(_):
    global drawing_active
    drawing_active = False
    
    if reader.is_connected():
        reader.send_stop()
        return "STOP sent!"
    return "Not connected"


# =============================================================================
# CALLBACKS - Manual FK
# =============================================================================

@app.callback(
    [Output('fk-plot', 'figure'), Output('fk-result', 'children')],
    [Input('theta1-slider', 'value'), Input('theta2-slider', 'value'),
     Input('live-angles-store', 'data'), Input('mirror-toggle', 'value')]
)
def update_fk_plot(theta1, theta2, live_data, mirror_value):
    mirror_x = 'mirror' in (mirror_value or [])
    live_t1 = live_data.get('theta1', 0) if live_data else 0
    live_t2 = live_data.get('theta2', 0) if live_data else 0
    
    robot_x, robot_y = get_end_effector_position(L1, L2, theta1, theta2)
    paper_x, paper_y = robot_to_paper(robot_x, robot_y)
    within_limits = angles_within_limits(theta1, theta2)
    
    show_ghost = abs(theta1 - live_t1) > 2 or abs(theta2 - live_t2) > 2
    ghost = (theta1, theta2) if show_ghost else None
    
    fig = create_arm_figure(live_t1, live_t2,
        target_robot=(robot_x, robot_y) if show_ghost else None,
        ghost_angles=ghost, mirror_x=mirror_x, title="Manual FK")
    
    result = html.Div([
        html.P([html.Strong("Slider: "), f"θ1={theta1:.1f}°, θ2={theta2:.1f}°"]),
        html.P([html.Strong("Live: "), f"θ1={live_t1:.1f}°, θ2={live_t2:.1f}°"]),
        html.P([html.Strong("Target: "), f"Paper({paper_x:.1f}, {paper_y:.1f})"]),
        html.P([html.Strong("Limits: "), 
                html.Span("✓ OK" if within_limits else "⚠ Outside", 
                         style={'color': 'green' if within_limits else 'red'})])
    ])
    return fig, result


@app.callback(
    Output('status-display', 'children', allow_duplicate=True),
    Input('fk-send-btn', 'n_clicks'),
    [State('theta1-slider', 'value'), State('theta2-slider', 'value')],
    prevent_initial_call=True
)
def fk_send(_, theta1, theta2):
    if reader.is_connected():
        reader.send_angles(theta1, theta2)
        return f"Sent: θ1={theta1:.1f}°, θ2={theta2:.1f}°"
    return "Not connected"


# =============================================================================
# CALLBACKS - Analytic IK
# =============================================================================

@app.callback(
    [Output('aik-plot', 'figure'), Output('aik-result', 'children'), Output('aik-angles-store', 'data')],
    [Input('aik-calc-btn', 'n_clicks'), Input('live-angles-store', 'data'), Input('mirror-toggle', 'value')],
    [State('aik-paper-x', 'value'), State('aik-paper-y', 'value'), State('aik-elbow', 'value')],
    prevent_initial_call=False
)
def calc_analytic_ik(_, live_data, mirror_value, paper_x, paper_y, elbow):
    mirror_x = 'mirror' in (mirror_value or [])
    live_t1 = live_data.get('theta1', 0) if live_data else 0
    live_t2 = live_data.get('theta2', 0) if live_data else 0
    
    paper_x = paper_x if paper_x is not None else 6
    paper_y = paper_y if paper_y is not None else 12
    
    robot_x, robot_y = paper_to_robot(paper_x, paper_y)
    
    if not is_reachable_robot(robot_x, robot_y):
        fig = create_arm_figure(live_t1, live_t2, target_robot=(robot_x, robot_y),
                               mirror_x=mirror_x, title="Analytic IK")
        return fig, html.P("⚠ UNREACHABLE", style={'color': 'red'}), None
    
    theta1, theta2 = inverse_kinematics(L1, L2, robot_x, robot_y, elbow)
    
    if theta1 is None:
        fig = create_arm_figure(live_t1, live_t2, target_robot=(robot_x, robot_y),
                               mirror_x=mirror_x, title="Analytic IK")
        return fig, html.P("⚠ IK Failed", style={'color': 'red'}), None
    
    within_limits = angles_within_limits(theta1, theta2)
    fig = create_arm_figure(live_t1, live_t2, target_robot=(robot_x, robot_y),
                           ghost_angles=(theta1, theta2), mirror_x=mirror_x, title="Analytic IK")
    
    result = html.Div([
        html.P([html.Strong("Target: "), f"Paper({paper_x:.1f}, {paper_y:.1f})"]),
        html.P([html.Strong("Solution: "), f"θ1={theta1:.2f}°, θ2={theta2:.2f}°"]),
        html.P([html.Strong("Limits: "),
                html.Span("✓ OK" if within_limits else "⚠ Outside",
                         style={'color': 'green' if within_limits else 'orange'})])
    ])
    return fig, result, {'theta1': theta1, 'theta2': theta2}


@app.callback(
    Output('status-display', 'children', allow_duplicate=True),
    Input('aik-send-btn', 'n_clicks'),
    State('aik-angles-store', 'data'),
    prevent_initial_call=True
)
def aik_send(_, data):
    if not data:
        return "AIK: No solution"
    if reader.is_connected():
        reader.send_angles(data['theta1'], data['theta2'])
        return f"Sent: θ1={data['theta1']:.1f}°, θ2={data['theta2']:.1f}°"
    return "Not connected"


# =============================================================================
# CALLBACKS - Jacobian IK
# =============================================================================

@app.callback(
    [Output('jik-plot', 'figure'), Output('jik-result', 'children'), Output('jik-angles-store', 'data')],
    [Input('jik-calc-btn', 'n_clicks'), Input('live-angles-store', 'data'), Input('mirror-toggle', 'value')],
    [State('jik-paper-x', 'value'), State('jik-paper-y', 'value'),
     State('jik-gain', 'value'), State('jik-tol', 'value')],
    prevent_initial_call=False
)
def calc_jacobian_ik(_, live_data, mirror_value, paper_x, paper_y, gain, tol):
    mirror_x = 'mirror' in (mirror_value or [])
    live_t1 = live_data.get('theta1', 0) if live_data else 0
    live_t2 = live_data.get('theta2', 0) if live_data else 0
    
    paper_x = paper_x if paper_x is not None else 6
    paper_y = paper_y if paper_y is not None else 12
    gain = gain if gain is not None else 0.5
    tol = tol if tol is not None else 0.01
    
    robot_x, robot_y = paper_to_robot(paper_x, paper_y)
    
    if not is_reachable_robot(robot_x, robot_y):
        fig = create_arm_figure(live_t1, live_t2, target_robot=(robot_x, robot_y),
                               mirror_x=mirror_x, title="Jacobian IK")
        return fig, html.P("⚠ UNREACHABLE", style={'color': 'red'}), None
    
    theta1, theta2, converged, iters, error = jacobian_ik(
        L1, L2, robot_x, robot_y, initial_theta1=live_t1, initial_theta2=live_t2,
        gain=gain, tolerance=tol
    )
    
    within_limits = angles_within_limits(theta1, theta2)
    fig = create_arm_figure(live_t1, live_t2, target_robot=(robot_x, robot_y),
                           ghost_angles=(theta1, theta2), mirror_x=mirror_x, title="Jacobian IK")
    
    result = html.Div([
        html.P([html.Strong("Target: "), f"Paper({paper_x:.1f}, {paper_y:.1f})"]),
        html.P([html.Strong("Solution: "), f"θ1={theta1:.2f}°, θ2={theta2:.2f}°"]),
        html.P([html.Strong("Converged: "), f"{'✓' if converged else '✗'} ({iters} iters, {error:.3f}cm)"]),
    ])
    return fig, result, {'theta1': theta1, 'theta2': theta2}


@app.callback(
    Output('status-display', 'children', allow_duplicate=True),
    Input('jik-send-btn', 'n_clicks'),
    State('jik-angles-store', 'data'),
    prevent_initial_call=True
)
def jik_send(_, data):
    if not data:
        return "JIK: No solution"
    if reader.is_connected():
        reader.send_angles(data['theta1'], data['theta2'])
        return f"Sent: θ1={data['theta1']:.1f}°, θ2={data['theta2']:.1f}°"
    return "Not connected"


# =============================================================================
# CALLBACKS - Analytic Shape Drawing
# =============================================================================

@app.callback(
    [Output('ashape-plot', 'figure'), Output('ashape-result', 'children'), Output('ashape-data-store', 'data')],
    [Input('ashape-preview-btn', 'n_clicks'), Input('live-angles-store', 'data'), Input('mirror-toggle', 'value')],
    [State('ashape-type', 'value'), State('ashape-cx', 'value'), State('ashape-cy', 'value'),
     State('ashape-ex', 'value'), State('ashape-ey', 'value'),
     State('ashape-size', 'value'), State('ashape-vel', 'value'), State('ashape-elbow', 'value'),
     State('ashape-quality', 'value'), State('ashape-options', 'value')],
    prevent_initial_call=False
)
def preview_analytic_shape(_, live_data, mirror_value, shape_type, cx, cy, ex, ey, size, vel, elbow, quality, options):
    mirror_x = 'mirror' in (mirror_value or [])
    live_t1 = live_data.get('theta1', 0) if live_data else 0
    live_t2 = live_data.get('theta2', 0) if live_data else 0
    
    cx = cx if cx is not None else 6
    cy = cy if cy is not None else 6
    ex = ex if ex is not None else 10
    ey = ey if ey is not None else 6
    size = size if size is not None else 3
    vel = vel if vel is not None else 2
    shape_type = shape_type if shape_type else 'square'
    quality = quality if quality else 'normal'
    corner_slowdown = 'slowdown' in (options or [])
    
    # Generate trajectory
    if shape_type == 'square':
        result = traj_gen.generate_square(cx, cy, size, method='analytic', elbow=elbow,
                                          velocity=vel, quality=quality, corner_slowdown=corner_slowdown)
    elif shape_type == 'circle':
        result = traj_gen.generate_circle(cx, cy, size/2, method='analytic', elbow=elbow,
                                          velocity=vel, quality=quality)
    elif shape_type == 'triangle':
        result = traj_gen.generate_triangle(cx, cy, size, method='analytic', elbow=elbow,
                                            velocity=vel, quality=quality, corner_slowdown=corner_slowdown)
    elif shape_type == 'star':
        result = traj_gen.generate_star(cx, cy, size/2, method='analytic', elbow=elbow,
                                        velocity=vel, quality=quality, corner_slowdown=corner_slowdown)
    elif shape_type == 'rectangle':
        result = traj_gen.generate_rectangle(cx, cy, size, size*0.6, method='analytic', elbow=elbow,
                                             velocity=vel, quality=quality, corner_slowdown=corner_slowdown)
    elif shape_type == 'line':
        result = traj_gen.generate_line(cx, cy, ex, ey, method='analytic', elbow=elbow,
                                        velocity=vel, num_points=30)
    else:
        result = {'valid': False, 'errors': ['Unknown shape']}
    
    shape_waypoints = result.get('waypoints_robot', [])
    fig = create_arm_figure(live_t1, live_t2, shape_waypoints=shape_waypoints,
                           mirror_x=mirror_x, title=f"Analytic Shapes - {shape_type.title()}")
    
    if result['valid']:
        trajectory = result.get('trajectory', [])
        corners = result.get('corners', [])
        
        info = html.Div([
            html.P([html.Strong("Shape: "), f"{shape_type.title()} at ({cx}, {cy})"]),
            html.P([html.Strong("Quality: "), f"{quality.title()}"]),
            html.P([html.Strong("Waypoints: "), f"{len(trajectory)}"]),
            html.P([html.Strong("Corners: "), f"{len(corners)} detected"]),
            html.P([html.Strong("Path length: "), f"{result.get('path_length', 0):.1f} cm"]),
            html.P([html.Strong("Est. time: "), f"{result.get('total_time_ms', 0)/1000:.1f} s"]),
            html.P("✓ Ready (trajectory batch mode)", style={'color': 'green', 'fontWeight': 'bold'})
        ])
        
        store_data = {
            'valid': True,
            'trajectory': trajectory,
            'waypoints_robot': shape_waypoints
        }
    else:
        error_items = [html.Li(e) for e in result.get('errors', ['Unknown error'])]
        info = html.Div([
            html.P([html.Strong("Shape: "), f"{shape_type.title()}"]),
            html.P("⛔ CANNOT DRAW:", style={'color': 'red', 'fontWeight': 'bold'}),
            html.Ul(error_items)
        ])
        store_data = None
    
    return fig, info, store_data


@app.callback(
    [Output('drawing-state-store', 'data', allow_duplicate=True),
     Output('drawing-interval', 'disabled', allow_duplicate=True),
     Output('ashape-progress', 'value', allow_duplicate=True),
     Output('status-display', 'children', allow_duplicate=True)],
    Input('ashape-draw-btn', 'n_clicks'),
    State('ashape-data-store', 'data'),
    prevent_initial_call=True
)
def start_analytic_drawing(_, shape_data):
    global drawing_active
    
    if not shape_data or not shape_data['valid']:
        return {'active': False, 'method': None, 'index': 0}, True, 0, "No valid shape"
    
    if not reader.is_connected():
        return {'active': False, 'method': None, 'index': 0}, True, 0, "Not connected"
    
    trajectory = list(shape_data['trajectory'])  # Make a copy
    
    # Get FRESH current robot position directly from reader
    state = reader.get_state()
    current_t1 = state.theta1_current
    current_t2 = state.theta2_current
    
    # ALWAYS prepend move-to-start (even if close, for consistency)
    if len(trajectory) > 0:
        first_time, first_t1, first_t2 = trajectory[0]
        
        # Calculate distance to start (in degrees)
        dist_to_start = abs(first_t1 - current_t1) + abs(first_t2 - current_t2)
        
        # Calculate time to reach start
        # Use 40ms per degree - conservative to ensure robot can keep up
        # Minimum 500ms, NO MAXIMUM - let robot take as long as needed
        move_time = int(dist_to_start * 40)
        move_time = max(move_time, 500)  # At least 500ms
        
        # Build new trajectory: current -> start -> rest of shape
        # Point 0: Stay at current position (t=0)
        # Point 1: Reach start position (t=move_time)
        # Point 2+: Original shape shifted by move_time
        new_trajectory = [
            (0, current_t1, current_t2),           # At t=0, robot is HERE
            (move_time, first_t1, first_t2),       # At t=move_time, reach shape START
        ]
        # Add rest of shape, shifted by move_time
        for t, t1, t2 in trajectory[1:]:  # Skip first point (we already added it)
            new_trajectory.append((t + move_time, t1, t2))
        
        trajectory = new_trajectory
        status_msg = f"Moving to start ({dist_to_start:.0f}°), then {len(trajectory)-2} shape pts"
    else:
        status_msg = "Empty trajectory"
    
    # Send trajectory batch to Arduino
    success = reader.send_trajectory(trajectory)
    
    if success:
        drawing_active = True
        return ({'active': True, 'method': 'analytic', 'index': 0}, False, 0, status_msg)
    else:
        return {'active': False, 'method': None, 'index': 0}, True, 0, "Failed to send"


@app.callback(
    [Output('drawing-state-store', 'data', allow_duplicate=True),
     Output('drawing-interval', 'disabled', allow_duplicate=True),
     Output('ashape-progress', 'value', allow_duplicate=True)],
    Input('ashape-stop-btn', 'n_clicks'),
    prevent_initial_call=True
)
def stop_analytic_drawing(_):
    global drawing_active
    drawing_active = False
    if reader.is_connected():
        reader.send_stop()
    return {'active': False, 'method': None, 'index': 0}, True, 0


# =============================================================================
# CALLBACKS - Jacobian Shape Drawing
# =============================================================================

@app.callback(
    [Output('jshape-plot', 'figure'), Output('jshape-result', 'children'), Output('jshape-data-store', 'data')],
    [Input('jshape-preview-btn', 'n_clicks'), Input('live-angles-store', 'data'), Input('mirror-toggle', 'value')],
    [State('jshape-type', 'value'), State('jshape-cx', 'value'), State('jshape-cy', 'value'),
     State('jshape-ex', 'value'), State('jshape-ey', 'value'),
     State('jshape-size', 'value'), State('jshape-vel', 'value'), State('jshape-quality', 'value'),
     State('jshape-options', 'value')],
    prevent_initial_call=False
)
def preview_jacobian_shape(_, live_data, mirror_value, shape_type, cx, cy, ex, ey, size, vel, quality, options):
    mirror_x = 'mirror' in (mirror_value or [])
    live_t1 = live_data.get('theta1', 0) if live_data else 0
    live_t2 = live_data.get('theta2', 0) if live_data else 0
    
    cx = cx if cx is not None else 6
    cy = cy if cy is not None else 6
    ex = ex if ex is not None else 10
    ey = ey if ey is not None else 6
    size = size if size is not None else 3
    vel = vel if vel is not None else 2
    shape_type = shape_type if shape_type else 'square'
    quality = quality if quality else 'normal'
    corner_slowdown = 'slowdown' in (options or [])
    
    # Generate trajectory using Jacobian IK
    if shape_type == 'square':
        result = traj_gen.generate_square(cx, cy, size, method='jacobian',
                                          velocity=vel, quality=quality, corner_slowdown=corner_slowdown,
                                          initial_theta1=live_t1, initial_theta2=live_t2)
    elif shape_type == 'circle':
        result = traj_gen.generate_circle(cx, cy, size/2, method='jacobian',
                                          velocity=vel, quality=quality,
                                          initial_theta1=live_t1, initial_theta2=live_t2)
    elif shape_type == 'triangle':
        result = traj_gen.generate_triangle(cx, cy, size, method='jacobian',
                                            velocity=vel, quality=quality, corner_slowdown=corner_slowdown,
                                            initial_theta1=live_t1, initial_theta2=live_t2)
    elif shape_type == 'star':
        result = traj_gen.generate_star(cx, cy, size/2, method='jacobian',
                                        velocity=vel, quality=quality, corner_slowdown=corner_slowdown,
                                        initial_theta1=live_t1, initial_theta2=live_t2)
    elif shape_type == 'rectangle':
        result = traj_gen.generate_rectangle(cx, cy, size, size*0.6, method='jacobian',
                                             velocity=vel, quality=quality, corner_slowdown=corner_slowdown,
                                             initial_theta1=live_t1, initial_theta2=live_t2)
    elif shape_type == 'line':
        result = traj_gen.generate_line(cx, cy, ex, ey, method='jacobian',
                                        velocity=vel, num_points=30,
                                        initial_theta1=live_t1, initial_theta2=live_t2)
    else:
        result = {'valid': False, 'errors': ['Unknown shape']}
    
    shape_waypoints = result.get('waypoints_robot', [])
    fig = create_arm_figure(live_t1, live_t2, shape_waypoints=shape_waypoints,
                           mirror_x=mirror_x, title=f"Jacobian Shapes - {shape_type.title()}")
    
    if result['valid']:
        trajectory = result.get('trajectory', [])
        corners = result.get('corners', [])
        
        info = html.Div([
            html.P([html.Strong("Shape: "), f"{shape_type.title()} at ({cx}, {cy})"]),
            html.P([html.Strong("Quality: "), f"{quality.title()}"]),
            html.P([html.Strong("Waypoints: "), f"{len(trajectory)}"]),
            html.P([html.Strong("Corners: "), f"{len(corners)} detected"]),
            html.P([html.Strong("Path length: "), f"{result.get('path_length', 0):.1f} cm"]),
            html.P([html.Strong("Est. time: "), f"{result.get('total_time_ms', 0)/1000:.1f} s"]),
            html.P("✓ Ready (trajectory batch mode)", style={'color': 'green', 'fontWeight': 'bold'})
        ])
        
        store_data = {
            'valid': True,
            'trajectory': trajectory,
            'waypoints_robot': shape_waypoints
        }
    else:
        error_items = [html.Li(e) for e in result.get('errors', ['Unknown error'])]
        info = html.Div([
            html.P([html.Strong("Shape: "), f"{shape_type.title()}"]),
            html.P("⛔ CANNOT DRAW:", style={'color': 'red', 'fontWeight': 'bold'}),
            html.Ul(error_items)
        ])
        store_data = None
    
    return fig, info, store_data


@app.callback(
    [Output('drawing-state-store', 'data', allow_duplicate=True),
     Output('drawing-interval', 'disabled', allow_duplicate=True),
     Output('jshape-progress', 'value', allow_duplicate=True),
     Output('status-display', 'children', allow_duplicate=True)],
    Input('jshape-draw-btn', 'n_clicks'),
    State('jshape-data-store', 'data'),
    prevent_initial_call=True
)
def start_jacobian_drawing(_, shape_data):
    global drawing_active
    
    if not shape_data or not shape_data['valid']:
        return {'active': False, 'method': None, 'index': 0}, True, 0, "No valid shape"
    
    if not reader.is_connected():
        return {'active': False, 'method': None, 'index': 0}, True, 0, "Not connected"
    
    trajectory = list(shape_data['trajectory'])  # Make a copy
    
    # Get FRESH current robot position directly from reader
    state = reader.get_state()
    current_t1 = state.theta1_current
    current_t2 = state.theta2_current
    
    # ALWAYS prepend move-to-start (even if close, for consistency)
    if len(trajectory) > 0:
        first_time, first_t1, first_t2 = trajectory[0]
        
        # Calculate distance to start (in degrees)
        dist_to_start = abs(first_t1 - current_t1) + abs(first_t2 - current_t2)
        
        # Calculate time to reach start
        # Use 40ms per degree - conservative to ensure robot can keep up
        # Minimum 500ms, NO MAXIMUM - let robot take as long as needed
        move_time = int(dist_to_start * 40)
        move_time = max(move_time, 500)  # At least 500ms
        
        # Build new trajectory: current -> start -> rest of shape
        # Point 0: Stay at current position (t=0)
        # Point 1: Reach start position (t=move_time)
        # Point 2+: Original shape shifted by move_time
        new_trajectory = [
            (0, current_t1, current_t2),           # At t=0, robot is HERE
            (move_time, first_t1, first_t2),       # At t=move_time, reach shape START
        ]
        # Add rest of shape, shifted by move_time
        for t, t1, t2 in trajectory[1:]:  # Skip first point (we already added it)
            new_trajectory.append((t + move_time, t1, t2))
        
        trajectory = new_trajectory
        status_msg = f"Moving to start ({dist_to_start:.0f}°), then {len(trajectory)-2} shape pts"
    else:
        status_msg = "Empty trajectory"
    
    # Send trajectory batch to Arduino
    success = reader.send_trajectory(trajectory)
    
    if success:
        drawing_active = True
        return ({'active': True, 'method': 'jacobian', 'index': 0}, False, 0, status_msg)
    else:
        return {'active': False, 'method': None, 'index': 0}, True, 0, "Failed to send"


@app.callback(
    [Output('drawing-state-store', 'data', allow_duplicate=True),
     Output('drawing-interval', 'disabled', allow_duplicate=True),
     Output('jshape-progress', 'value', allow_duplicate=True)],
    Input('jshape-stop-btn', 'n_clicks'),
    prevent_initial_call=True
)
def stop_jacobian_drawing(_):
    global drawing_active
    drawing_active = False
    if reader.is_connected():
        reader.send_stop()
    return {'active': False, 'method': None, 'index': 0}, True, 0


# =============================================================================
# CALLBACK - Drawing Progress Update
# =============================================================================

@app.callback(
    [Output('drawing-state-store', 'data', allow_duplicate=True),
     Output('drawing-interval', 'disabled', allow_duplicate=True),
     Output('ashape-progress', 'value', allow_duplicate=True),
     Output('jshape-progress', 'value', allow_duplicate=True),
     Output('status-display', 'children', allow_duplicate=True)],
    Input('drawing-interval', 'n_intervals'),
    State('drawing-state-store', 'data'),
    prevent_initial_call=True
)
def update_drawing_progress(_, state):
    global drawing_active
    
    if not state or not state.get('active', False):
        return state, True, 0, 0, ""
    
    if not reader.is_connected():
        drawing_active = False
        return {'active': False, 'method': None, 'index': 0}, True, 0, 0, "Disconnected"
    
    # Get Arduino state
    arduino_state = reader.get_state()
    
    # Check if trajectory is still active
    if arduino_state.trajectory_active:
        progress = 0
        if arduino_state.trajectory_total > 0:
            progress = int(100 * arduino_state.trajectory_index / arduino_state.trajectory_total)
        
        status = f"Drawing: {arduino_state.trajectory_index}/{arduino_state.trajectory_total}"
        
        if state.get('method') == 'analytic':
            return {'active': True, 'method': 'analytic', 'index': arduino_state.trajectory_index}, False, progress, 0, status
        else:
            return {'active': True, 'method': 'jacobian', 'index': arduino_state.trajectory_index}, False, 0, progress, status
    else:
        # Trajectory complete
        drawing_active = False
        return {'active': False, 'method': None, 'index': 0}, True, 100, 100, "Drawing complete!"


# =============================================================================
# CALLBACKS - Evaluation
# =============================================================================

# Global variable to track evaluation thread
eval_thread = None
eval_results_cache = None

@app.callback(
    [Output('eval-running-store', 'data'),
     Output('eval-interval', 'disabled'),
     Output('eval-status', 'children', allow_duplicate=True),
     Output('eval-progress', 'value', allow_duplicate=True)],
    Input('eval-run-btn', 'n_clicks'),
    [State('eval-test-selection', 'value'),
     State('eval-repetitions', 'value'),
     State('eval-settle-time', 'value'),
     State('eval-velocity', 'value'),
     State('eval-joint-angles', 'value'),
     State('eval-cartesian-points', 'value'),
     State('eval-traj-start-x', 'value'),
     State('eval-traj-start-y', 'value'),
     State('eval-traj-end-x', 'value'),
     State('eval-traj-end-y', 'value'),
     State('eval-circle-cx', 'value'),
     State('eval-circle-cy', 'value'),
     State('eval-circle-radius', 'value')],
    prevent_initial_call=True
)
def start_evaluation(n_clicks, test_selection, repetitions, settle_time, velocity,
                    joint_angles_str, cartesian_points_str,
                    traj_start_x, traj_start_y, traj_end_x, traj_end_y,
                    circle_cx, circle_cy, circle_radius):
    """Start the evaluation tests in a background thread."""
    global eval_thread, eval_results_cache
    import threading
    
    if not reader.is_connected():
        return False, True, "Not connected! Connect to Arduino first.", 0
    
    if not test_selection:
        return False, True, "Select at least one test to run.", 0
    
    # Parse parameters
    try:
        joint_angles = [float(x.strip()) for x in joint_angles_str.split(',')]
    except:
        joint_angles = DEFAULT_JOINT_TEST_ANGLES
    
    try:
        cartesian_points = []
        for line in cartesian_points_str.strip().split('\n'):
            parts = line.split(',')
            if len(parts) >= 2:
                cartesian_points.append((float(parts[0].strip()), float(parts[1].strip())))
        if not cartesian_points:
            cartesian_points = DEFAULT_CARTESIAN_TEST_POINTS
    except:
        cartesian_points = DEFAULT_CARTESIAN_TEST_POINTS
    
    trajectory_line = None
    if 'trajectory' in test_selection:
        trajectory_line = ((traj_start_x, traj_start_y), (traj_end_x, traj_end_y))
    
    circle_params = None
    if 'circle' in test_selection:
        circle_params = ((circle_cx, circle_cy), circle_radius)
    
    # Clear previous results
    eval_results_cache = None
    
    # Define the evaluation function to run in thread
    def run_eval():
        global eval_results_cache
        try:
            results = evaluator.run_full_evaluation(
                reader,
                run_joint='joint' in test_selection,
                run_cartesian='cartesian' in test_selection,
                run_trajectory='trajectory' in test_selection,
                run_circle='circle' in test_selection,
                joint_angles=joint_angles,
                cartesian_points=cartesian_points,
                trajectory_line=trajectory_line,
                circle_params=circle_params,
                velocity=velocity,
                repetitions=repetitions,
                settle_time_ms=settle_time
            )
            eval_results_cache = results
        except Exception as e:
            print(f"Evaluation error: {e}")
            import traceback
            traceback.print_exc()
            evaluator.is_running = False
    
    # Start thread
    eval_thread = threading.Thread(target=run_eval, daemon=True)
    eval_thread.start()
    
    return True, False, "Starting evaluation...", 0


@app.callback(
    Output('eval-running-store', 'data', allow_duplicate=True),
    Input('eval-stop-btn', 'n_clicks'),
    prevent_initial_call=True
)
def stop_evaluation(_):
    """Stop the evaluation."""
    evaluator.stop()
    return False


@app.callback(
    [Output('eval-status', 'children'),
     Output('eval-progress', 'value'),
     Output('eval-interval', 'disabled', allow_duplicate=True),
     Output('eval-results-store', 'data'),
     Output('eval-results-summary', 'children'),
     Output('eval-results-joint', 'children'),
     Output('eval-results-cartesian', 'children'),
     Output('eval-results-trajectory', 'children'),
     Output('eval-results-json', 'children'),
     Output('eval-running-store', 'data', allow_duplicate=True)],
    Input('eval-interval', 'n_intervals'),
    State('eval-running-store', 'data'),
    prevent_initial_call=True
)
def update_evaluation_progress(_, is_running):
    """Update evaluation progress and results."""
    global eval_results_cache
    
    # Check if evaluation has completed (thread finished)
    if not evaluator.is_running:
        # Check if we have cached results to display
        if eval_results_cache is not None:
            results = eval_results_cache
            eval_results_cache = None  # Clear cache after displaying
            
            # Generate result displays
            summary = generate_summary_display(results)
            joint = generate_joint_results_display(results)
            cartesian = generate_cartesian_results_display(results)
            trajectory = generate_trajectory_results_display(results)
            json_str = results.to_json()
            
            return ("Evaluation complete!", 100, True, results.to_dict(),
                   summary, joint, cartesian, trajectory, json_str, False)
        
        # No cached results but not running - evaluation might have failed or stopped
        if is_running:
            # Was running but now stopped without results
            return ("Evaluation stopped or failed", 0, True, None,
                   html.P("No results available", className="text-muted"),
                   html.P("No results available", className="text-muted"),
                   html.P("No results available", className="text-muted"),
                   html.P("No results available", className="text-muted"),
                   "", False)
        
        # Not running and no store flag - just idle
        return (dash.no_update,) * 10
    
    # Still running - update progress
    return (evaluator.current_test, evaluator.progress, False, None,
            dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, True)


def generate_summary_display(results):
    """Generate summary statistics display."""
    return html.Div([
        html.H5("📊 Evaluation Summary"),
        html.P(f"Timestamp: {results.timestamp}"),
        html.Hr(),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Motor 1 (Base)"),
                    dbc.CardBody([
                        html.P([html.Strong("Mean Error: "), f"{results.joint_mean_error_m1:.2f}°"]),
                        html.P([html.Strong("Max Error: "), f"{results.joint_max_error_m1:.2f}°"]),
                        html.P([html.Strong("Repeatability (σ): "), f"{results.joint_std_m1:.3f}°"]),
                    ])
                ], color="primary", outline=True)
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Motor 2 (Elbow)"),
                    dbc.CardBody([
                        html.P([html.Strong("Mean Error: "), f"{results.joint_mean_error_m2:.2f}°"]),
                        html.P([html.Strong("Max Error: "), f"{results.joint_max_error_m2:.2f}°"]),
                        html.P([html.Strong("Repeatability (σ): "), f"{results.joint_std_m2:.3f}°"]),
                    ])
                ], color="info", outline=True)
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Cartesian (XY)"),
                    dbc.CardBody([
                        html.P([html.Strong("Mean Error: "), f"{results.cartesian_mean_error:.2f} cm"]),
                        html.P([html.Strong("Max Error: "), f"{results.cartesian_max_error:.2f} cm"]),
                        html.P([html.Strong("Std Dev: "), f"{results.cartesian_std:.3f} cm"]),
                    ])
                ], color="success", outline=True)
            ], width=4),
        ]),
        
        html.Hr(),
        html.H6("Test Counts"),
        html.P(f"Joint tests: {len(results.joint_tests)}"),
        html.P(f"Cartesian tests: {len(results.cartesian_tests)}"),
        html.P(f"Trajectory tests: {len(results.trajectory_tests)}"),
    ])


def generate_joint_results_display(results):
    """Generate joint test results table."""
    if not results.joint_tests:
        return html.P("No joint tests run.", className="text-muted")
    
    # Create table
    rows = []
    for t in results.joint_tests:
        color = "success" if abs(t.error) < 1.0 else ("warning" if abs(t.error) < 2.0 else "danger")
        rows.append(html.Tr([
            html.Td(f"M{t.motor}"),
            html.Td(f"{t.target_angle:.1f}°"),
            html.Td(f"{t.actual_angle:.1f}°"),
            html.Td(f"{t.error:+.2f}°", className=f"text-{color}"),
        ]))
    
    return html.Div([
        html.H6(f"Joint Test Results ({len(results.joint_tests)} tests)"),
        dbc.Table([
            html.Thead(html.Tr([
                html.Th("Motor"),
                html.Th("Target"),
                html.Th("Actual"),
                html.Th("Error"),
            ])),
            html.Tbody(rows)
        ], striped=True, bordered=True, hover=True, size="sm")
    ])


def generate_cartesian_results_display(results):
    """Generate Cartesian test results table."""
    if not results.cartesian_tests:
        return html.P("No Cartesian tests run.", className="text-muted")
    
    rows = []
    for t in results.cartesian_tests:
        color = "success" if t.error_euclidean < 0.5 else ("warning" if t.error_euclidean < 1.0 else "danger")
        rows.append(html.Tr([
            html.Td(f"({t.target_paper_x:.1f}, {t.target_paper_y:.1f})"),
            html.Td(f"({t.actual_paper_x:.1f}, {t.actual_paper_y:.1f})"),
            html.Td(f"{t.error_euclidean:.2f} cm", className=f"text-{color}"),
            html.Td(f"{t.theta1_error:+.1f}°"),
            html.Td(f"{t.theta2_error:+.1f}°"),
        ]))
    
    return html.Div([
        html.H6(f"Cartesian Test Results ({len(results.cartesian_tests)} tests)"),
        dbc.Table([
            html.Thead(html.Tr([
                html.Th("Target (X,Y)"),
                html.Th("Actual (X,Y)"),
                html.Th("XY Error"),
                html.Th("θ1 Error"),
                html.Th("θ2 Error"),
            ])),
            html.Tbody(rows)
        ], striped=True, bordered=True, hover=True, size="sm"),
        
        html.Hr(),
        html.H6("Error Breakdown by Joint"),
        html.P([
            "This shows which motor contributes more to positioning errors. ",
            "Larger θ1 errors suggest base motor issues, larger θ2 errors suggest elbow motor issues."
        ], className="text-muted small"),
    ])


def generate_trajectory_results_display(results):
    """Generate trajectory test results display."""
    if not results.trajectory_tests:
        return html.P("No trajectory tests run.", className="text-muted")
    
    displays = []
    
    for i, t in enumerate(results.trajectory_tests):
        # Determine if this is a line or circle test
        # Circle test stores (radius, num_points) in end_point
        is_circle = isinstance(t.end_point[0], (int, float)) and t.end_point[0] < 10 and t.end_point[1] > 20
        
        if is_circle:
            test_name = "Circle Trajectory Test"
            radius = t.end_point[0]
            description = html.P([
                f"Circle: center ({t.start_point[0]:.1f}, {t.start_point[1]:.1f}), radius {radius:.1f} cm"
            ])
        else:
            test_name = "Line Trajectory Test"
            description = html.P([
                f"Line: ({t.start_point[0]:.1f}, {t.start_point[1]:.1f}) → ",
                f"({t.end_point[0]:.1f}, {t.end_point[1]:.1f})"
            ])
        
        displays.append(html.Div([
            html.H6(f"{test_name} #{i+1}"),
            description,
            html.P(f"Samples collected: {t.num_samples}"),
            html.Hr(),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{t.max_deviation:.2f} cm", className="text-danger"),
                            html.P("Maximum Deviation", className="mb-0 text-muted")
                        ])
                    ])
                ]),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{t.mean_deviation:.2f} cm", className="text-warning"),
                            html.P("Mean Deviation", className="mb-0 text-muted")
                        ])
                    ])
                ]),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{t.rms_deviation:.3f} cm", className="text-info"),
                            html.P("RMS Deviation", className="mb-0 text-muted")
                        ])
                    ])
                ]),
            ]),
            
            html.Br(),
        ]))
    
    displays.append(html.Hr())
    displays.append(html.P([
        "For lines: perpendicular distance from ideal line. ",
        "For circles: radial distance from ideal radius. ",
        "Lower values = better path following."
    ], className="text-muted small"))
    
    return html.Div(displays)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("2-Link Robot Controller - Trajectory Batch Mode")
    print(f"Link lengths: L1={L1} cm, L2={L2} cm")
    print("=" * 60)
    print("\nFeatures:")
    print("  - Serial reader thread (no more graph freezing)")
    print("  - Trajectory batch sending (Arduino interpolates)")
    print("  - 6 tabs: Manual FK, Analytic IK, Jacobian IK, Shapes, Evaluation")
    print("  - NEW: Quantitative evaluation suite for accuracy testing")
    print("=" * 60)
    print("\nStarting server at http://127.0.0.1:8050")
    print("Press Ctrl+C to stop\n")
    
    app.run(debug=False, host='127.0.0.1', port=8050)
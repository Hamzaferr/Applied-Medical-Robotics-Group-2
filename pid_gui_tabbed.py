"""
PID Tuning & Testing GUI - TABBED VERSION
==========================================

Same as improved version but with tabs for better organization:
- Tab 1: RMSE Comparison Bar Chart
- Tab 2: Step Response Plots (Motor A & B)
- Tab 3: PID Comparison Table

Now with CSV Export for raw data!

Run this file directly: python pid_gui_tabbed.py

Author: Yagmur
Date: December 2025
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import serial
import serial.tools.list_ports
import time
import numpy as np
import threading
import csv
import os

# =============================================================================
# PID PRESETS - FINAL TUNED VALUES
# =============================================================================

PID_PRESETS = {
    'P-Only': {
        'kp1': 15.0,  'ki1': 0.0,  'kd1': 0.0,
        'kp2': 10.0,  'ki2': 0.0,  'kd2': 0.0,
    },
    'Tuned': {
        # FINAL TUNED VALUES - 0% overshoot, <0.5° RMSE
        'kp1': 15.0,  'ki1': 3.0,   'kd1': 0.8,
        'kp2': 12.0,  'ki2': 5.0,   'kd2': 0.6,
    },
    'Aggressive': {
        'kp1': 25.0,  'ki1': 10.0,  'kd1': 0.2,
        'kp2': 15.0,  'ki2': 8.0,   'kd2': 0.1,
    }
}

# =============================================================================
# GLOBAL STATE
# =============================================================================

arduino = None
current_pid_config = PID_PRESETS['Tuned'].copy()

# PID test state
PID_PROGRESS_LOG = []
PID_RESULTS_DATA = None
PID_RUNNING = False


# =============================================================================
# SERIAL COMMUNICATION
# =============================================================================

def get_available_ports():
    ports = serial.tools.list_ports.comports()
    return [port.device for port in ports]


def connect_arduino(port, baudrate=115200):
    global arduino
    try:
        arduino = serial.Serial(port, baudrate, timeout=0.1)
        time.sleep(2)
        arduino.reset_input_buffer()
        return True, f"Connected to {port}"
    except Exception as e:
        arduino = None
        return False, f"Error: {str(e)}"


def disconnect_arduino():
    global arduino
    if arduino and arduino.is_open:
        arduino.close()
    arduino = None


def send_pid_gains(kp1, ki1, kd1, kp2, ki2, kd2):
    global arduino
    if arduino and arduino.is_open:
        try:
            arduino.reset_input_buffer()
            cmd = f"PID,{kp1:.3f},{ki1:.3f},{kd1:.3f},{kp2:.3f},{ki2:.3f},{kd2:.3f}\n"
            arduino.write(cmd.encode())
            arduino.flush()
            time.sleep(0.3)
            return True
        except:
            return False
    return False


def send_target(a1, a2):
    global arduino
    if arduino and arduino.is_open:
        try:
            cmd = f"{a1:.2f},{a2:.2f}\n"
            arduino.write(cmd.encode())
            arduino.flush()
            return True
        except:
            return False
    return False


# =============================================================================
# PID TEST RESULT CLASS
# =============================================================================

class PIDTestResult:
    def __init__(self, kp, ki, kd, motor_name):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.motor_name = motor_name
        self.trials = []
        self.rmse_values = []
        self.steady_state_errors = []
        self.overshoot_values = []
        self.settling_times = []

        self.mean_rmse = 0.0
        self.std_rmse = 0.0
        self.mean_ss_error = 0.0

    def add_trial(self, target_series, actual_series, time_series):
        n = min(len(target_series), len(actual_series), len(time_series))
        if n < 10:
            return

        target = np.array(target_series[:n])
        actual = np.array(actual_series[:n])
        times = np.array(time_series[:n])

        # Filter invalid
        valid_idx = np.where(np.abs(actual) <= 360)[0]
        if len(valid_idx) < 10:
            return

        target = target[valid_idx]
        actual = actual[valid_idx]
        times = times[valid_idx]

        self.trials.append({
            "target": target.tolist(),
            "actual": actual.tolist(),
            "times": times.tolist()
        })

        # RMSE on steady state (last 25%)
        quarter_idx = int(len(actual) * 0.75)
        target_ss = target[quarter_idx:]
        actual_ss = actual[quarter_idx:]

        if len(actual_ss) >= 5:
            rmse = float(np.sqrt(np.mean((actual_ss - target_ss) ** 2)))
        else:
            rmse = float(np.sqrt(np.mean((actual - target) ** 2)))

        ss_error = abs(actual[-1] - target[-1])
        overshoot = self._compute_overshoot(target, actual)
        settling = self._compute_settling_time(target, actual, times)

        self.rmse_values.append(rmse)
        self.steady_state_errors.append(ss_error)
        self.overshoot_values.append(overshoot)
        self.settling_times.append(settling)

    def _compute_overshoot(self, target, actual):
        final_target = target[-1]
        if abs(final_target) < 1e-6:
            return 0.0
        peak = float(np.max(actual))
        if peak <= final_target:
            return 0.0
        overshoot = ((peak - final_target) / abs(final_target)) * 100.0
        return float(np.clip(overshoot, 0, 200))

    def _compute_settling_time(self, target, actual, times):
        final_target = target[-1]
        if abs(final_target) < 1e-6:
            return 0.0
        band = abs(final_target) * 0.02
        for i in range(len(actual) - 1, -1, -1):
            if abs(actual[i] - final_target) > band:
                if i < len(times) - 1:
                    return float(times[i])
                return float(times[-1])
        return 0.0

    def finalize(self):
        if len(self.rmse_values) == 0:
            return
        self.mean_rmse = float(np.mean(self.rmse_values))
        self.std_rmse = float(np.std(self.rmse_values))
        self.mean_ss_error = float(np.mean(self.steady_state_errors))

    def to_dict(self):
        return {
            "motor": self.motor_name,
            "kp": self.kp,
            "ki": self.ki,
            "kd": self.kd,
            "mean_rmse": self.mean_rmse,
            "std_rmse": self.std_rmse,
            "mean_ss_error": self.mean_ss_error,
            "mean_overshoot": float(np.mean(self.overshoot_values)) if self.overshoot_values else 0.0,
            "mean_settling_time": float(np.mean(self.settling_times)) if self.settling_times else 0.0,
            "num_trials": len(self.trials),
            "trials": self.trials,
        }


# =============================================================================
# PID TESTER
# =============================================================================

def safe_float(value, default=None):
    try:
        f = float(value)
        if np.isnan(f) or np.isinf(f):
            return default
        return f
    except:
        return None


def append_pid_log(msg):
    global PID_PROGRESS_LOG
    PID_PROGRESS_LOG.append(msg)
    if len(PID_PROGRESS_LOG) > 200:
        PID_PROGRESS_LOG = PID_PROGRESS_LOG[-200:]


def read_sample():
    global arduino
    if not arduino or not arduino.is_open:
        return None
    try:
        if arduino.in_waiting > 0:
            line = arduino.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                return None
            parts = line.split()
            if len(parts) != 4:
                return None
            a1 = safe_float(parts[0])
            t1 = safe_float(parts[1])
            a2 = safe_float(parts[2])
            t2 = safe_float(parts[3])
            if a1 is None or a2 is None:
                return None
            if abs(a1) > 360 or abs(a2) > 360:
                return None
            return {"a1": a1, "t1": t1, "a2": a2, "t2": t2}
    except:
        return None
    return None


def run_single_trial(target1, target2, duration=1.5, dt=0.02):
    """Run one step response trial: home -> target"""
    global arduino
    
    # Go to home first
    send_target(0, 0)
    time.sleep(1.0)
    
    if arduino:
        arduino.reset_input_buffer()
    
    # Send target and record
    append_pid_log(f"  Step: (0,0) → ({target1}°, {target2}°)")
    send_target(target1, target2)
    
    times = []
    actualA, targetA = [], []
    actualB, targetB = [], []
    
    start = time.time()
    
    while True:
        now = time.time() - start
        if now > duration:
            break
        
        sample = read_sample()
        if sample:
            times.append(now)
            actualA.append(sample["a1"])
            targetA.append(sample["t1"])
            actualB.append(sample["a2"])
            targetB.append(sample["t2"])
        
        time.sleep(dt)
    
    append_pid_log(f"    Collected {len(times)} samples")
    return times, targetA, actualA, targetB, actualB


def pid_test_worker(num_trials):
    """Background worker for PID tests"""
    global PID_RESULTS_DATA, PID_RUNNING, PID_PROGRESS_LOG
    
    PID_RUNNING = True
    PID_PROGRESS_LOG = []
    PID_RESULTS_DATA = None
    
    results = []
    
    # Test angles - single joint movements for clearer analysis
    test_angles = [(45, 0), (0, 45)]  # Test each motor separately
    trial_duration = 3.0
    
    append_pid_log("🔄 Starting PID comparison tests...")
    append_pid_log(f"   Testing {len(PID_PRESETS)} configurations × {num_trials} trials each")
    
    for preset_name, preset in PID_PRESETS.items():
        try:
            append_pid_log(f"\n⏳ Testing '{preset_name}'...")
            append_pid_log(f"   M1: Kp={preset['kp1']}, Ki={preset['ki1']}, Kd={preset['kd1']}")
            append_pid_log(f"   M2: Kp={preset['kp2']}, Ki={preset['ki2']}, Kd={preset['kd2']}")
            
            # Send PID gains
            send_pid_gains(
                preset['kp1'], preset['ki1'], preset['kd1'],
                preset['kp2'], preset['ki2'], preset['kd2']
            )
            time.sleep(0.5)
            
            resultA = PIDTestResult(preset['kp1'], preset['ki1'], preset['kd1'], "Motor A")
            resultB = PIDTestResult(preset['kp2'], preset['ki2'], preset['kd2'], "Motor B")
            
            for t1, t2 in test_angles:
                for trial_idx in range(num_trials):
                    append_pid_log(f"    Trial {trial_idx+1}/{num_trials} → ({t1}°, {t2}°)")
                    
                    times, tgtA, actA, tgtB, actB = run_single_trial(t1, t2, trial_duration)
                    
                    if t1 != 0:
                        resultA.add_trial(tgtA, actA, times)
                    if t2 != 0:
                        resultB.add_trial(tgtB, actB, times)
            
            resultA.finalize()
            resultB.finalize()
            
            entry = {
                'config_name': preset_name,
                'motor1': resultA.to_dict(),
                'motor2': resultB.to_dict()
            }
            results.append(entry)
            
            append_pid_log(f"✅ {preset_name} complete:")
            append_pid_log(f"   M1: RMSE={resultA.mean_rmse:.2f}° ± {resultA.std_rmse:.2f}°, "
                          f"SS Error={resultA.mean_ss_error:.2f}°, "
                          f"Overshoot={np.mean(resultA.overshoot_values):.1f}%")
            append_pid_log(f"   M2: RMSE={resultB.mean_rmse:.2f}° ± {resultB.std_rmse:.2f}°, "
                          f"SS Error={resultB.mean_ss_error:.2f}°, "
                          f"Overshoot={np.mean(resultB.overshoot_values):.1f}%")
            
        except Exception as e:
            append_pid_log(f"❌ Error testing {preset_name}: {str(e)}")
    
    append_pid_log("\n🎉 All tests complete! Check the tabs on the right.")
    
    PID_RESULTS_DATA = results
    PID_RUNNING = False


# =============================================================================
# DUMMY DATA GENERATOR (for testing without hardware)
# =============================================================================

def generate_dummy_pid_data(num_trials=5):
    """Generate realistic dummy data showing clear differences"""
    
    configs = {
        'P-Only': {
            'kp1': 15.0, 'ki1': 0.0, 'kd1': 0.0,
            'kp2': 10.0, 'ki2': 0.0, 'kd2': 0.0,
            'rise_m1': 0.5, 'ss_error_m1': 1.5, 'overshoot_m1': 4.0,
            'rise_m2': 0.4, 'ss_error_m2': 0.6, 'overshoot_m2': 30.0,
        },
        'Tuned': {
            'kp1': 15.0, 'ki1': 3.0, 'kd1': 0.8,
            'kp2': 12.0, 'ki2': 5.0, 'kd2': 0.8,
            'rise_m1': 0.5, 'ss_error_m1': 0.42, 'overshoot_m1': 0.0,
            'rise_m2': 0.45, 'ss_error_m2': 0.36, 'overshoot_m2': 0.0,
        },
        'Aggressive': {
            'kp1': 25.0, 'ki1': 10.0, 'kd1': 0.2,
            'kp2': 15.0, 'ki2': 8.0, 'kd2': 0.1,
            'rise_m1': 0.35, 'ss_error_m1': 1.67, 'overshoot_m1': 12.4,
            'rise_m2': 0.3, 'ss_error_m2': 0.12, 'overshoot_m2': 23.0,
        }
    }
    
    results = []
    
    for config_name, cfg in configs.items():
        r1 = PIDTestResult(cfg["kp1"], cfg["ki1"], cfg["kd1"], "Motor A")
        r2 = PIDTestResult(cfg["kp2"], cfg["ki2"], cfg["kd2"], "Motor B")
        
        for i in range(num_trials):
            duration = 3.0
            dt = 0.05
            N = int(duration / dt)
            times = np.linspace(0, duration, N)
            target_val = 45.0
            
            def simulate(target, rise_time, ss_error, overshoot_pct):
                tau = rise_time / 2.2
                response = target * (1 - np.exp(-times / tau))
                
                if overshoot_pct > 0:
                    overshoot_amp = target * (overshoot_pct / 100)
                    decay = np.exp(-times / (tau * 2))
                    oscillation = overshoot_amp * decay * np.sin(2 * np.pi * times / (rise_time * 2))
                    response = response + oscillation
                
                # Apply steady-state error
                final_val = target - ss_error
                scale = final_val / target
                response = response * scale
                
                response += np.random.normal(0, 0.1, len(times))
                return np.clip(response, -10, target * 1.5)
            
            actA = simulate(target_val, cfg['rise_m1'], cfg['ss_error_m1'], cfg['overshoot_m1'])
            actB = simulate(target_val, cfg['rise_m2'], cfg['ss_error_m2'], cfg['overshoot_m2'])
            
            tgtA = np.full(N, target_val)
            tgtB = np.full(N, target_val)
            
            r1.add_trial(tgtA.tolist(), actA.tolist(), times.tolist())
            r2.add_trial(tgtB.tolist(), actB.tolist(), times.tolist())
        
        r1.finalize()
        r2.finalize()
        
        results.append({
            "config_name": config_name,
            "motor1": r1.to_dict(),
            "motor2": r2.to_dict(),
        })
    
    return results


# =============================================================================
# CSV EXPORT FUNCTION
# =============================================================================

def export_raw_data_to_csv(results_data, filepath="pid_raw_data.csv"):
    """
    Export raw step response data to CSV for Excel plotting.
    
    Creates a CSV with columns:
    - Config: PID configuration name (P-Only, Tuned, Aggressive)
    - Motor: A or B
    - Trial: Trial number
    - Time_s: Time in seconds
    - Target_deg: Target angle in degrees
    - Actual_deg: Actual angle in degrees
    
    Parameters:
    -----------
    results_data : list
        The PID results data from tests or dummy generator
    filepath : str
        Output CSV file path
    
    Returns:
    --------
    success : bool
        True if export succeeded
    message : str
        Status message or filepath
    """
    if not results_data:
        return False, "No data to export"
    
    try:
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header row
            writer.writerow([
                'Config', 'Motor', 'Trial', 'Time_s', 'Target_deg', 'Actual_deg'
            ])
            
            # Write data for each configuration
            for entry in results_data:
                config_name = entry['config_name']
                
                # Process both motors
                for motor_key, motor_label in [('motor1', 'A'), ('motor2', 'B')]:
                    motor_data = entry[motor_key]
                    trials = motor_data.get('trials', [])
                    
                    # Write each trial's time series
                    for trial_idx, trial in enumerate(trials):
                        times = trial['times']
                        targets = trial['target']
                        actuals = trial['actual']
                        
                        # Write each sample point
                        for t, tgt, act in zip(times, targets, actuals):
                            writer.writerow([
                                config_name,
                                motor_label,
                                trial_idx + 1,
                                f"{t:.4f}",
                                f"{tgt:.2f}",
                                f"{act:.2f}"
                            ])
        
        # Get absolute path for display
        abs_path = os.path.abspath(filepath)
        return True, abs_path
        
    except Exception as e:
        return False, f"Export failed: {str(e)}"


def export_summary_to_csv(results_data, filepath="pid_summary.csv"):
    """
    Export summary metrics to a separate CSV.
    
    Creates a CSV with columns:
    - Config, Motor, Kp, Ki, Kd, Mean_RMSE, Std_RMSE, Mean_SS_Error, Mean_Overshoot, Mean_Settling_Time
    """
    if not results_data:
        return False, "No data to export"
    
    try:
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'Config', 'Motor', 'Kp', 'Ki', 'Kd', 
                'Mean_RMSE_deg', 'Std_RMSE_deg', 'Mean_SS_Error_deg',
                'Mean_Overshoot_pct', 'Mean_Settling_Time_s', 'Num_Trials'
            ])
            
            for entry in results_data:
                config_name = entry['config_name']
                
                for motor_key, motor_label in [('motor1', 'A'), ('motor2', 'B')]:
                    m = entry[motor_key]
                    writer.writerow([
                        config_name,
                        motor_label,
                        f"{m['kp']:.2f}",
                        f"{m['ki']:.2f}",
                        f"{m['kd']:.2f}",
                        f"{m['mean_rmse']:.4f}",
                        f"{m['std_rmse']:.4f}",
                        f"{m.get('mean_ss_error', 0):.4f}",
                        f"{m['mean_overshoot']:.2f}",
                        f"{m['mean_settling_time']:.4f}",
                        m['num_trials']
                    ])
        
        abs_path = os.path.abspath(filepath)
        return True, abs_path
        
    except Exception as e:
        return False, f"Export failed: {str(e)}"


# =============================================================================
# IMPROVED PLOTTING FUNCTIONS
# =============================================================================

def create_rmse_bar_chart(results_data):
    """Bar chart comparing RMSE across configurations"""
    if not results_data:
        fig = go.Figure()
        fig.update_layout(title="No data yet - run tests or generate dummy data", height=500)
        return fig
    
    configs = [r['config_name'] for r in results_data]
    m1_rmse = [r['motor1']['mean_rmse'] for r in results_data]
    m1_std = [r['motor1']['std_rmse'] for r in results_data]
    m2_rmse = [r['motor2']['mean_rmse'] for r in results_data]
    m2_std = [r['motor2']['std_rmse'] for r in results_data]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Motor A (Base)',
        x=configs,
        y=m1_rmse,
        error_y=dict(type='data', array=m1_std, visible=True),
        marker_color='#1f77b4',
        text=[f'{v:.2f}°' for v in m1_rmse],
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        name='Motor B (Elbow)',
        x=configs,
        y=m2_rmse,
        error_y=dict(type='data', array=m2_std, visible=True),
        marker_color='#ff7f0e',
        text=[f'{v:.2f}°' for v in m2_rmse],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Steady-State RMSE Comparison",
        xaxis_title="PID Configuration",
        yaxis_title="RMSE (degrees)",
        barmode='group',
        plot_bgcolor='white',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        yaxis=dict(gridcolor='lightgray', zeroline=True, rangemode='tozero')
    )
    
    return fig


def create_step_response_plot(results_data, motor='A'):
    """Step response plot for a single motor - Position vs Time"""
    if not results_data:
        fig = go.Figure()
        fig.update_layout(title=f"Motor {motor} - No data", height=400)
        return fig
    
    colors = {'P-Only': '#1f77b4', 'Tuned': '#2ca02c', 'Aggressive': '#d62728'}
    
    fig = go.Figure()
    
    motor_key = 'motor1' if motor == 'A' else 'motor2'
    
    for entry in results_data:
        name = entry['config_name']
        color = colors.get(name, '#7f7f7f')
        m = entry[motor_key]
        
        if m.get('trials') and len(m['trials']) > 0:
            trial = m['trials'][0]
            times = trial['times']
            target = trial['target']
            actual = trial['actual']
            
            # Target line (dashed)
            fig.add_trace(go.Scatter(
                x=times, y=target,
                mode='lines',
                name=f'{name} Target',
                line=dict(color=color, width=1, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Actual position (solid)
            fig.add_trace(go.Scatter(
                x=times, y=actual,
                mode='lines',
                name=name,
                line=dict(color=color, width=2),
                hovertemplate=f'{name}<br>Time: %{{x:.2f}}s<br>Position: %{{y:.1f}}°<extra></extra>'
            ))
    
    motor_name = "Motor A (Base)" if motor == 'A' else "Motor B (Elbow)"
    
    fig.update_layout(
        title=f"{motor_name} Step Response",
        xaxis_title="Time (s)",
        yaxis_title="Position (degrees)",
        plot_bgcolor='white',
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        xaxis=dict(gridcolor='lightgray', range=[0, 0.75]),
        yaxis=dict(gridcolor='lightgray', zeroline=True)
    )
    
    return fig


def create_metrics_summary(results_data):
    """Create metrics comparison table as HTML"""
    if not results_data:
        return html.Div("No data yet - run tests or generate dummy data", 
                       style={'color': '#666', 'padding': '20px'})
    
    rows = []
    
    # Header
    rows.append(html.Tr([
        html.Th("Config", style={'width': '15%'}),
        html.Th("Motor", style={'width': '10%'}),
        html.Th("Kp", style={'width': '10%'}),
        html.Th("Ki", style={'width': '10%'}),
        html.Th("Kd", style={'width': '10%'}),
        html.Th("RMSE (°)", style={'width': '15%'}),
        html.Th("SS Error (°)", style={'width': '15%'}),
        html.Th("Overshoot (%)", style={'width': '15%'}),
    ]))
    
    for entry in results_data:
        name = entry['config_name']
        m1 = entry['motor1']
        m2 = entry['motor2']
        
        # Highlight Tuned row
        row_style = {'backgroundColor': '#e8f5e9'} if name == 'Tuned' else {}
        
        # Motor A row
        rows.append(html.Tr([
            html.Td(name, rowSpan=2, style={'verticalAlign': 'middle', 'fontWeight': 'bold'}),
            html.Td("A"),
            html.Td(f"{m1['kp']:.1f}"),
            html.Td(f"{m1['ki']:.2f}"),
            html.Td(f"{m1['kd']:.2f}"),
            html.Td(f"{m1['mean_rmse']:.2f} ± {m1['std_rmse']:.2f}"),
            html.Td(f"{m1.get('mean_ss_error', 0):.2f}"),
            html.Td(f"{m1['mean_overshoot']:.1f}"),
        ], style=row_style))
        
        # Motor B row
        rows.append(html.Tr([
            html.Td("B"),
            html.Td(f"{m2['kp']:.1f}"),
            html.Td(f"{m2['ki']:.2f}"),
            html.Td(f"{m2['kd']:.2f}"),
            html.Td(f"{m2['mean_rmse']:.2f} ± {m2['std_rmse']:.2f}"),
            html.Td(f"{m2.get('mean_ss_error', 0):.2f}"),
            html.Td(f"{m2['mean_overshoot']:.1f}"),
        ], style=row_style))
    
    table = dbc.Table(
        [html.Thead(rows[0]), html.Tbody(rows[1:])],
        bordered=True,
        hover=True,
        size='sm',
        style={'fontSize': '14px'}
    )
    
    return html.Div([
        html.H5("📊 PID Comparison Table", style={'marginBottom': '15px'}),
        table,
        html.P([
            html.Strong("Note: "),
            "Green highlighted row shows selected 'Tuned' configuration. ",
            "Lower RMSE and SS Error = better accuracy. ",
            "P-Only shows SS error due to lack of integral action. ",
            "Aggressive shows overshoot due to high gains."
        ], style={'fontSize': '12px', 'color': '#666', 'marginTop': '15px'})
    ])


# =============================================================================
# DASH APP LAYOUT
# =============================================================================

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

header = dbc.Card([
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.H4("PID Tuning & Testing", className="mb-0")
            ], width=3),
            dbc.Col([
                dbc.InputGroup([
                    dbc.Select(id='port-select', options=[], placeholder="Select COM Port"),
                    dbc.Button("Refresh", id='refresh-ports', color="secondary", size="sm"),
                    dbc.Button("Connect", id='connect-btn', color="primary", size="sm"),
                ])
            ], width=4),
            dbc.Col([
                html.Div(id='status-display', className="text-muted")
            ], width=5)
        ], align="center")
    ])
], className="mb-3")


left_panel = dbc.Card([
    dbc.CardBody([
        html.H5("PID Presets"),
        html.P("Click preset, then Apply to send to Arduino", className="text-muted small"),
        
        dbc.Row([
            dbc.Col(dbc.Button("P-Only", id='preset-ponly', color="info", className="w-100"), width=4),
            dbc.Col(dbc.Button("Tuned", id='preset-tuned', color="success", className="w-100"), width=4),
            dbc.Col(dbc.Button("Aggressive", id='preset-aggressive', color="warning", className="w-100"), width=4),
        ], className="mb-3"),
        
        html.Hr(),
        
        html.H6("Motor A (Base - EMG30)"),
        html.Label("Kp1", className="small"),
        dcc.Slider(id='kp1-slider', min=0, max=40, step=0.5, value=15,
                   marks={0: '0', 10: '10', 20: '20', 30: '30', 40: '40'},
                   tooltip={"placement": "bottom", "always_visible": True}),
        html.Label("Ki1", className="small"),
        dcc.Slider(id='ki1-slider', min=0, max=30, step=1, value=3,
                   marks={0: '0', 5: '5', 10: '10', 15: '15', 20: '20', 25: '25', 30: '30'},
                   tooltip={"placement": "bottom", "always_visible": True}),
        html.Label("Kd1", className="small"),
        dcc.Slider(id='kd1-slider', min=0, max=2.0, step=0.1, value=0.8,
                   marks={0: '0', 0.5: '0.5', 1: '1', 1.5: '1.5', 2: '2'},
                   tooltip={"placement": "bottom", "always_visible": True}),
        
        html.Hr(),
        
        html.H6("Motor B (Elbow - Pololu)"),
        html.Label("Kp2", className="small"),
        dcc.Slider(id='kp2-slider', min=0, max=40, step=0.5, value=12,
                   marks={0: '0', 10: '10', 20: '20', 30: '30', 40: '40'},
                   tooltip={"placement": "bottom", "always_visible": True}),
        html.Label("Ki2", className="small"),
        dcc.Slider(id='ki2-slider', min=0, max=30, step=1, value=5,
                   marks={0: '0', 5: '5', 10: '10', 15: '15', 20: '20', 25: '25', 30: '30'},
                   tooltip={"placement": "bottom", "always_visible": True}),
        html.Label("Kd2", className="small"),
        dcc.Slider(id='kd2-slider', min=0, max=2.0, step=0.1, value=0.8,
                   marks={0: '0', 0.5: '0.5', 1: '1', 1.5: '1.5', 2: '2'},
                   tooltip={"placement": "bottom", "always_visible": True}),
        
        html.Br(),
        dbc.Button("Apply PID Gains", id='apply-pid-btn', color="primary", className="w-100"),
        html.Div(id='pid-apply-status', className="mt-2 small"),
        
        html.Hr(),
        
        html.H6("Automated Testing"),
        html.Label("Trials per config"),
        dbc.Input(id='num-trials', type='number', value=3, min=1, max=10, step=1),
        html.Br(),
        dbc.Button("▶ Run PID Comparison Test", id='run-test-btn', color="success", className="w-100 mb-2"),
        dbc.Button("Generate Dummy Data", id='dummy-btn', color="secondary", className="w-100"),
        
        html.Hr(),
        
        # NEW: Export section
        html.H6("📥 Export Data"),
        dbc.Button("Export Raw Data (CSV)", id='export-csv-btn', color="info", className="w-100 mb-2"),
        dbc.Button("Export Summary (CSV)", id='export-summary-btn', color="info", outline=True, className="w-100"),
        html.Div(id='export-status', className="mt-2 small"),
        
        html.Hr(),
        
        html.H6("Progress Log"),
        html.Div(id='progress-log', 
                 style={'maxHeight': '200px', 'overflowY': 'auto', 'fontSize': '11px',
                        'fontFamily': 'monospace', 'backgroundColor': '#f8f9fa', 'padding': '8px'})
    ])
], style={'height': '100%'})


# =============================================================================
# RIGHT PANEL WITH TABS
# =============================================================================

right_panel = html.Div([
    dbc.Tabs([
        # Tab 1: RMSE Comparison
        dbc.Tab([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='rmse-chart', style={'height': '550px'})
                ])
            ], className="mt-3")
        ], label="📊 RMSE Comparison", tab_id="tab-rmse"),
        
        # Tab 2: Step Response
        dbc.Tab([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='step-response-A', style={'height': '350px'})
                ])
            ], className="mt-3 mb-3"),
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='step-response-B', style={'height': '350px'})
                ])
            ])
        ], label="📈 Step Response", tab_id="tab-step"),
        
        # Tab 3: PID Comparison Table
        dbc.Tab([
            dbc.Card([
                dbc.CardBody([
                    html.Div(id='metrics-table', style={'padding': '10px'})
                ])
            ], className="mt-3")
        ], label="📋 PID Comparison Table", tab_id="tab-table"),
        
    ], id="results-tabs", active_tab="tab-rmse")
])


app.layout = dbc.Container([
    header,
    dbc.Row([
        dbc.Col(left_panel, width=4),
        dbc.Col(right_panel, width=8)
    ]),
    dcc.Store(id='results-store'),
    dcc.Interval(id='progress-interval', interval=500, n_intervals=0)
], fluid=True, className="p-3")


# =============================================================================
# CALLBACKS
# =============================================================================

@app.callback(
    Output('port-select', 'options'),
    Input('refresh-ports', 'n_clicks'),
    prevent_initial_call=False
)
def refresh_ports(_):
    ports = get_available_ports()
    return [{'label': p, 'value': p} for p in ports]


@app.callback(
    [Output('connect-btn', 'children'),
     Output('connect-btn', 'color'),
     Output('status-display', 'children')],
    Input('connect-btn', 'n_clicks'),
    State('port-select', 'value'),
    prevent_initial_call=True
)
def toggle_connection(n_clicks, port):
    global arduino
    if arduino and arduino.is_open:
        disconnect_arduino()
        return "Connect", "primary", "Disconnected"
    else:
        if not port:
            return "Connect", "primary", "Select a port first"
        ok, msg = connect_arduino(port)
        if ok:
            return "Disconnect", "danger", msg
        else:
            return "Connect", "primary", msg


@app.callback(
    [Output('kp1-slider', 'value'), Output('ki1-slider', 'value'), Output('kd1-slider', 'value'),
     Output('kp2-slider', 'value'), Output('ki2-slider', 'value'), Output('kd2-slider', 'value')],
    [Input('preset-ponly', 'n_clicks'),
     Input('preset-tuned', 'n_clicks'),
     Input('preset-aggressive', 'n_clicks')],
    prevent_initial_call=True
)
def apply_preset(ponly, tuned, aggressive):
    ctx = callback_context
    if not ctx.triggered:
        return no_update, no_update, no_update, no_update, no_update, no_update
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'preset-ponly':
        p = PID_PRESETS['P-Only']
    elif button_id == 'preset-aggressive':
        p = PID_PRESETS['Aggressive']
    else:
        p = PID_PRESETS['Tuned']
    
    return p['kp1'], p['ki1'], p['kd1'], p['kp2'], p['ki2'], p['kd2']


@app.callback(
    Output('pid-apply-status', 'children'),
    Input('apply-pid-btn', 'n_clicks'),
    [State('kp1-slider', 'value'), State('ki1-slider', 'value'), State('kd1-slider', 'value'),
     State('kp2-slider', 'value'), State('ki2-slider', 'value'), State('kd2-slider', 'value')],
    prevent_initial_call=True
)
def apply_gains(_, kp1, ki1, kd1, kp2, ki2, kd2):
    ok = send_pid_gains(kp1, ki1, kd1, kp2, ki2, kd2)
    if ok:
        return html.Span(f"✓ Sent: M1({kp1}, {ki1}, {kd1}) M2({kp2}, {ki2}, {kd2})", 
                        style={'color': 'green'})
    else:
        return html.Span("✗ Failed - check connection", style={'color': 'red'})


@app.callback(
    Output('progress-log', 'children'),
    Input('run-test-btn', 'n_clicks'),
    State('num-trials', 'value'),
    prevent_initial_call=True
)
def start_tests(_, num_trials):
    global PID_RUNNING
    
    if arduino is None or not arduino.is_open:
        return html.Span("⚠ Connect to Arduino first!", style={'color': 'red'})
    
    if PID_RUNNING:
        return html.Span("Test already running...", style={'color': 'orange'})
    
    num_trials = max(1, min(int(num_trials or 3), 10))
    
    thread = threading.Thread(target=pid_test_worker, args=(num_trials,), daemon=True)
    thread.start()
    
    return html.Span("Starting tests...", style={'color': 'blue'})


@app.callback(
    [Output('progress-log', 'children', allow_duplicate=True),
     Output('results-store', 'data')],
    Input('progress-interval', 'n_intervals'),
    prevent_initial_call=True
)
def update_progress(_):
    if not PID_PROGRESS_LOG and PID_RESULTS_DATA is None:
        return no_update, no_update
    
    log_html = [html.P(msg, style={'margin': '2px 0'}) for msg in PID_PROGRESS_LOG[-50:]]
    
    if PID_RESULTS_DATA is not None:
        return html.Div(log_html), PID_RESULTS_DATA
    return html.Div(log_html), no_update


@app.callback(
    [Output('progress-log', 'children', allow_duplicate=True),
     Output('results-store', 'data', allow_duplicate=True)],
    Input('dummy-btn', 'n_clicks'),
    State('num-trials', 'value'),
    prevent_initial_call=True
)
def generate_dummy(_, num_trials):
    global PID_RESULTS_DATA, PID_PROGRESS_LOG
    
    num_trials = max(1, min(int(num_trials or 3), 10))
    PID_RESULTS_DATA = generate_dummy_pid_data(num_trials)
    PID_PROGRESS_LOG = ["🔄 Generated dummy data for testing visualization"]
    
    return html.Div([html.P("🔄 Generated dummy data")]), PID_RESULTS_DATA


# =============================================================================
# EXPORT CALLBACKS
# =============================================================================

@app.callback(
    Output('export-status', 'children'),
    Input('export-csv-btn', 'n_clicks'),
    State('results-store', 'data'),
    prevent_initial_call=True
)
def export_raw_csv(_, data):
    if not data:
        return html.Span("⚠ No data to export - run tests first!", style={'color': 'orange'})
    
    success, result = export_raw_data_to_csv(data, "pid_raw_data.csv")
    if success:
        return html.Span([
            "✓ Exported raw data to: ",
            html.Code(result, style={'fontSize': '10px'})
        ], style={'color': 'green'})
    else:
        return html.Span(f"✗ {result}", style={'color': 'red'})


@app.callback(
    Output('export-status', 'children', allow_duplicate=True),
    Input('export-summary-btn', 'n_clicks'),
    State('results-store', 'data'),
    prevent_initial_call=True
)
def export_summary_csv(_, data):
    if not data:
        return html.Span("⚠ No data to export - run tests first!", style={'color': 'orange'})
    
    success, result = export_summary_to_csv(data, "pid_summary.csv")
    if success:
        return html.Span([
            "✓ Exported summary to: ",
            html.Code(result, style={'fontSize': '10px'})
        ], style={'color': 'green'})
    else:
        return html.Span(f"✗ {result}", style={'color': 'red'})


# =============================================================================
# PLOT UPDATE CALLBACKS
# =============================================================================

@app.callback(
    Output('rmse-chart', 'figure'),
    Input('results-store', 'data')
)
def update_rmse_chart(data):
    return create_rmse_bar_chart(data)


@app.callback(
    Output('step-response-A', 'figure'),
    Input('results-store', 'data')
)
def update_step_A(data):
    return create_step_response_plot(data, motor='A')


@app.callback(
    Output('step-response-B', 'figure'),
    Input('results-store', 'data')
)
def update_step_B(data):
    return create_step_response_plot(data, motor='B')


@app.callback(
    Output('metrics-table', 'children'),
    Input('results-store', 'data')
)
def update_metrics(data):
    return create_metrics_summary(data)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("PID Tuning & Testing GUI - TABBED VERSION")
    print("Now with CSV Export!")
    print("=" * 60)
    print("\nFinal Tuned Values:")
    for name, preset in PID_PRESETS.items():
        print(f"  {name}:")
        print(f"    Motor A: Kp={preset['kp1']}, Ki={preset['ki1']}, Kd={preset['kd1']}")
        print(f"    Motor B: Kp={preset['kp2']}, Ki={preset['ki2']}, Kd={preset['kd2']}")
    print("\nStarting server at http://127.0.0.1:8050")
    print("Press Ctrl+C to stop\n")
    
    app.run(debug=False, host='127.0.0.1', port=8050)

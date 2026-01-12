"""
PID Testing Module (Cleaned - NO SerialReader)
----------------------------------------------

Features:
- Safe serial parsing from raw serial port
- RMSE, overshoot, settling time computation
- Per-trial recording and sanitizing of data
- Config-by-config comparison
- Dummy data generator for testing GUI

Integration with Dash:
- PIDTester takes an optional `progress_cb(msg: str)` callback.
- All progress / debug text goes through `_log(msg)` so the GUI
  can capture and display it instead of (or as well as) printing.
"""

import numpy as np
import time
import json
from datetime import datetime


# ============================================================
# Utility: Safe helpers
# ============================================================

def safe_float(value, default=None):
    """Convert to float safely; return default if invalid."""
    try:
        f = float(value)
        if np.isnan(f) or np.isinf(f):
            return default
        return f
    except Exception:
        return default


def is_valid_angle(a):
    """Reject impossible angles > 360° or < -360°."""
    return a is not None and -360.0 <= a <= 360.0


# ============================================================
# Data container for each PID test result
# ============================================================

class PIDTestResult:
    def __init__(self, kp, ki, kd, motor_name):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.motor_name = motor_name

        self.trials = []          # raw data for plotting
        self.rmse_values = []
        self.overshoot_values = []
        self.settling_times = []

        self.mean_rmse = 0.0
        self.std_rmse = 0.0

    # --------------------------------------------------------
    def add_trial(self, target_series, actual_series, time_series):
        """Store trial and compute metrics.

        RMSE is computed on the *steady-state half* of the samples
        (last 50% of the trial), while overshoot and settling time
        still use the full time history.
        """
        n = min(len(target_series), len(actual_series), len(time_series))
        if n < 5:
            return  # Not enough data

        target = np.array(target_series[:n])
        actual = np.array(actual_series[:n])
        times = np.array(time_series[:n])

        # Filter invalid values
        valid_idx = np.where(np.abs(actual) <= 360)[0]
        if len(valid_idx) < 5:
            return

        target = target[valid_idx]
        actual = actual[valid_idx]
        times = times[valid_idx]

        # --- Save full arrays for overshoot/settling + plotting ---
        target_full = target
        actual_full = actual
        times_full = times

        # --- RMSE on LAST HALF (steady-state) ---
        half_idx = len(actual_full) // 2
        target_ss = target_full[half_idx:]
        actual_ss = actual_full[half_idx:]

        if len(actual_ss) < 5:
            # fallback: use full if too short
            rmse = float(np.sqrt(np.mean((actual_full - target_full) ** 2)))
        else:
            rmse = float(np.sqrt(np.mean((actual_ss - target_ss) ** 2)))

        # --- Overshoot / Settling are still computed on FULL response ---
        overshoot = self._compute_overshoot(target_full, actual_full)
        settling = self._compute_settling_time(target_full, actual_full, times_full)

        self.rmse_values.append(rmse)
        self.overshoot_values.append(overshoot)
        self.settling_times.append(settling)

        # Store full trial for plotting
        self.trials.append({
            "target": target_full.tolist(),
            "actual": actual_full.tolist(),
            "times": times_full.tolist()
        })

    # --------------------------------------------------------
    def _compute_overshoot(self, target, actual):
        final_target = target[-1]
        if abs(final_target) < 1e-6:
            return 0.0

        peak = float(np.max(actual))

        # If the response never reaches target (undershoot), report 0% overshoot
        if peak <= final_target:
            return 0.0

        overshoot = ((peak - final_target) / abs(final_target)) * 100.0
        overshoot = float(np.clip(overshoot, 0, 200))
        return overshoot

    # --------------------------------------------------------
    def _compute_settling_time(self, target, actual, times):
        final_target = target[-1]
        if abs(final_target) < 1e-6:
            return 0.0

        band = abs(final_target) * 0.02  # 2% band
        last_outside = None

        for i in range(len(actual) - 1, -1, -1):
            if abs(actual[i] - final_target) > band:
                last_outside = i
                break

        if last_outside is None:
            return 0.0
        return float(times[last_outside])

    # --------------------------------------------------------
    def finalize(self):
        if len(self.rmse_values) == 0:
            self.mean_rmse = 0.0
            self.std_rmse = 0.0
            return

        self.mean_rmse = float(np.mean(self.rmse_values))
        self.std_rmse = float(np.std(self.rmse_values))

    # --------------------------------------------------------
    def to_dict(self):
        return {
            "motor": self.motor_name,
            "kp": self.kp,
            "ki": self.ki,
            "kd": self.kd,
            "mean_rmse": self.mean_rmse,
            "std_rmse": self.std_rmse,
            "mean_overshoot": float(np.mean(self.overshoot_values)) if self.overshoot_values else 0.0,
            "mean_settling_time": float(np.mean(self.settling_times)) if self.settling_times else 0.0,
            "num_trials": len(self.trials),
            "trials": self.trials,
        }


# ============================================================
# PID Tester (NO SerialReader)
# ============================================================

class PIDTester:
    def __init__(self, ser, verbose=False, progress_cb=None):
        """
        ser         : pyserial Serial object
        verbose     : if True, also print to terminal
        progress_cb : optional callable(msg: str) for Dash to capture logs
        """
        self.ser = ser
        self.verbose = verbose
        self.progress_cb = progress_cb

    # --------------------------------------------------------
    def _log(self, msg: str):
        """Send debug/progress text to console and/or external callback."""
        if self.verbose:
            print(msg)
        if self.progress_cb is not None:
            try:
                self.progress_cb(msg)
            except Exception:
                # Don't crash testing if UI logging fails
                pass

    # --------------------------------------------------------
    def read_sample(self):
        """Reads one Arduino line: a1 t1 a2 t2 (actual1 target1 actual2 target2)."""
        if not self.ser or not self.ser.is_open:
            return None

        try:
            if self.ser.in_waiting > 0:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if not line:
                    return None

                parts = line.split()
                if len(parts) != 4:
                    return None

                a1 = safe_float(parts[0])
                t1 = safe_float(parts[1])
                a2 = safe_float(parts[2])
                t2 = safe_float(parts[3])

                if not (is_valid_angle(a1) and is_valid_angle(a2)):
                    return None

                # Optional: expose raw line to GUI
                self._log(f"RAW: {line}")

                return {"a1": a1, "t1": t1, "a2": a2, "t2": t2}

        except Exception as e:
            self._log(f"❌ read_sample error: {e}")
            return None

        return None

    # --------------------------------------------------------
    def send_pid(self, kp1, ki1, kd1, kp2, ki2, kd2):
        """Send PID gains to Arduino."""
        if not self.ser or not self.ser.is_open:
            self._log("❌ Cannot send PID: serial closed")
            return False

        self.ser.reset_input_buffer()

        cmd = f"PID,{kp1:.3f},{ki1:.3f},{kd1:.3f},{kp2:.3f},{ki2:.3f},{kd2:.3f}\n"
        self.ser.write(cmd.encode())
        self.ser.flush()
        self._log(f"→ Sent PID: {cmd.strip()}")

        time.sleep(0.3)
        return True

    # --------------------------------------------------------
    def send_target(self, a1, a2):
        """Send target joint angles to Arduino."""
        if not self.ser or not self.ser.is_open:
            self._log("❌ Cannot send target: serial closed")
            return False

        cmd = f"{a1:.2f},{a2:.2f}\n"
        self.ser.write(cmd.encode())
        self.ser.flush()
        self._log(f"→ Sent target: {cmd.strip()}")
        return True

    # --------------------------------------------------------
    def run_single_trial(self, target1, target2, duration=4.0, dt=0.05):
        """Runs one PID step-response trial."""
        self._log(f"  Trial → target=({target1}, {target2})")

        self.send_target(target1, target2)
        time.sleep(0.15)

        times = []
        actualA, targetA = [], []
        actualB, targetB = [], []

        start = time.time()
        count = 0

        while True:
            now = time.time() - start
            if now > duration:
                break

            sample = self.read_sample()
            if sample:
                a1, t1 = sample["a1"], sample["t1"]
                a2, t2 = sample["a2"], sample["t2"]

                times.append(now)
                actualA.append(a1)
                targetA.append(t1)
                actualB.append(a2)
                targetB.append(t2)

                count += 1

            time.sleep(dt)

        self._log(f"    Collected {count} samples")
        return times, targetA, actualA, targetB, actualB

    # --------------------------------------------------------
    def test_pid_configuration(
        self, kp1, ki1, kd1,
        kp2, ki2, kd2,
        test_angles,
        num_trials,
        trial_duration
    ):
        """Tests one PID configuration over several angles and trials."""
        self.send_pid(kp1, ki1, kd1, kp2, ki2, kd2)
        time.sleep(0.6)

        resultA = PIDTestResult(kp1, ki1, kd1, "Motor A")
        resultB = PIDTestResult(kp2, ki2, kd2, "Motor B")

        home = (0, 0)
        last = None

        for (t1, t2) in test_angles:

            # Return to home between distinct targets
            if last is not None and last != (t1, t2):
                self._log("  → Returning to home (0,0)")
                self.send_target(home[0], home[1])
                time.sleep(1.0)

            last = (t1, t2)

            for trial_idx in range(num_trials):
                self._log(f"    • Trial {trial_idx + 1}/{num_trials} at ({t1},{t2})")
                times, tgtA, actA, tgtB, actB = self.run_single_trial(
                    t1, t2,
                    duration=trial_duration
                )
                resultA.add_trial(tgtA, actA, times)
                resultB.add_trial(tgtB, actB, times)

        resultA.finalize()
        resultB.finalize()

        self._log(
            f"✔ PID config done | "
            f"M1 RMSE={resultA.mean_rmse:.3f}°, "
            f"M2 RMSE={resultB.mean_rmse:.3f}°"
        )

        return resultA, resultB


# ============================================================
# Plotting Data Helper
# ============================================================

def generate_comparison_plot_data(results):
    """
    Helper to convert a list of (config_name, resultA, resultB)
    into arrays that are easier to plot with Plotly.
    """
    out = {
        "motor1": {"config": [], "rmse": [], "std": [], "overshoot": [], "settling": []},
        "motor2": {"config": [], "rmse": [], "std": [], "overshoot": [], "settling": []},
    }

    for cfg, r1, r2 in results:
        out["motor1"]["config"].append(cfg)
        out["motor1"]["rmse"].append(r1.mean_rmse)
        out["motor1"]["std"].append(r1.std_rmse)
        out["motor1"]["overshoot"].append(float(np.mean(r1.overshoot_values)))
        out["motor1"]["settling"].append(float(np.mean(r1.settling_times)))

        out["motor2"]["config"].append(cfg)
        out["motor2"]["rmse"].append(r2.mean_rmse)
        out["motor2"]["std"].append(r2.std_rmse)
        out["motor2"]["overshoot"].append(float(np.mean(r2.overshoot_values)))
        out["motor2"]["settling"].append(float(np.mean(r2.settling_times)))

    return out


# ============================================================
# Dummy Data Generator
# ============================================================

def generate_dummy_pid_data(num_trials=10):
    """Returns 100% compatible dummy test data for GUI."""
    configs = {
        'Soft': {
            'kp1': 10.0, 'ki1': 0.1, 'kd1': 0.3,
            'kp2': 20.0, 'ki2': 0.05, 'kd2': 0.3,
            'settling_m1': 2.8, 'settling_m2': 3.2,
            'overshoot_m1': 5.0, 'overshoot_m2': 6.0
        },
        'Normal': {
            'kp1': 13.0, 'ki1': 0.15, 'kd1': 0.5,
            'kp2': 25.0, 'ki2': 0.1, 'kd2': 0.5,
            'settling_m1': 1.5, 'settling_m2': 1.8,
            'overshoot_m1': 8.0, 'overshoot_m2': 10.0
        },
        'Fast': {
            'kp1': 18.0, 'ki1': 0.2, 'kd1': 0.7,
            'kp2': 30.0, 'ki2': 0.15, 'kd2': 0.7,
            'settling_m1': 0.9, 'settling_m2': 1.1,
            'overshoot_m1': 15.0, 'overshoot_m2': 18.0
        }
    }

    results = []

    for config_name, cfg in configs.items():
        r1 = PIDTestResult(cfg["kp1"], cfg["ki1"], cfg["kd1"], "Motor A")
        r2 = PIDTestResult(cfg["kp2"], cfg["ki2"], cfg["kd2"], "Motor B")

        for i in range(num_trials):
            duration = 4.0
            dt = 0.05
            N = int(duration / dt)
            times = np.linspace(0, duration, N)

            target_val = 45 if i % 2 == 0 else 90

            def simulate(target, settling, overshoot):
                # artificially simulate response
                x = times
                tau = settling / 4
                response = target * (1 - np.exp(-x / tau))
                response *= 1 + (overshoot / 100) * np.sin(2 * np.pi * x / duration)
                response += np.random.normal(0, 0.5, len(x))
                return np.clip(response, 0, target * 1.5)

            actA = simulate(target_val, cfg['settling_m1'], cfg['overshoot_m1'])
            actB = simulate(target_val, cfg['settling_m2'], cfg['overshoot_m2'])

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


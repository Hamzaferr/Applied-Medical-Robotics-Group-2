"""
Author:Hamza
Serial Reader Thread for Robot Arm Controller

This module provides a dedicated thread that:
- Owns all serial port communication (read AND write)
- Parses Arduino output continuously
- Caches latest position for UI to read
- Provides thread-safe methods for sending commands

This fixes the graph freezing issue by ensuring only ONE thread touches serial.
"""

import serial
import serial.tools.list_ports
import threading
import time
import queue
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from collections import deque


@dataclass
class RobotState:
    """Current robot state from Arduino"""
    theta1_current: float = 0.0
    theta1_demand: float = 0.0
    theta2_current: float = 0.0
    theta2_demand: float = 0.0
    trajectory_active: bool = False
    trajectory_index: int = 0
    trajectory_total: int = 0
    timestamp: float = 0.0
    raw_line: str = ""


class SerialReader:
    """
    Dedicated serial reader/writer thread.
    
    Usage:
        reader = SerialReader()
        reader.connect('/dev/ttyUSB0')
        
        # Read latest state (non-blocking)
        state = reader.get_state()
        
        # Send commands (thread-safe)
        reader.send_angles(45.0, 30.0)
        reader.send_trajectory(points)
        reader.send_home()
        
        reader.disconnect()
    """
    
    def __init__(self, baudrate: int = 115200):
        self.baudrate = baudrate
        self.serial_port: Optional[serial.Serial] = None
        self.port_name: str = ""
        
        # Thread management
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        
        # Command queue (for sending to Arduino)
        self._command_queue: queue.Queue = queue.Queue()
        
        # Latest state (updated by reader thread)
        self._state = RobotState()
        self._state_lock = threading.Lock()
        
        # Position history for graphing
        self._history: deque = deque(maxlen=500)
        self._history_lock = threading.Lock()
        
        # Status messages from Arduino
        self._messages: deque = deque(maxlen=50)
        self._messages_lock = threading.Lock()
    
    # =========================================================================
    # CONNECTION MANAGEMENT
    # =========================================================================
    
    def find_arduino(self) -> Optional[str]:
        """Auto-detect Arduino port."""
        ports = serial.tools.list_ports.comports()
        
        for port in ports:
            desc = (port.description or "").lower()
            if any(x in desc for x in ['arduino', 'ch340', 'usb serial', 'acm']):
                return port.device
        
        # Fallback: try common ports
        common = ['/dev/ttyUSB0', '/dev/ttyACM0', 'COM3', 'COM4']
        for p in common:
            try:
                test = serial.Serial(p, self.baudrate, timeout=0.1)
                test.close()
                return p
            except:
                pass
        
        return None
    
    def connect(self, port: Optional[str] = None) -> Tuple[bool, str]:
        """Connect to Arduino and start reader thread."""
        if self.is_connected():
            return True, f"Already connected to {self.port_name}"
        
        # Find port
        if port is None:
            port = self.find_arduino()
        
        if port is None:
            return False, "No Arduino found"
        
        try:
            self.serial_port = serial.Serial(
                port=port,
                baudrate=self.baudrate,
                timeout=0.1,
                write_timeout=0.5
            )
            self.port_name = port
            
            # Clear buffers
            time.sleep(0.1)
            self.serial_port.reset_input_buffer()
            self.serial_port.reset_output_buffer()
            
            # Start reader thread
            self._running = True
            self._thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._thread.start()
            
            return True, f"Connected to {port}"
            
        except Exception as e:
            self.serial_port = None
            return False, f"Connection failed: {e}"
    
    def disconnect(self) -> Tuple[bool, str]:
        """Stop reader thread and disconnect."""
        self._running = False
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        
        if self.serial_port:
            try:
                self.serial_port.close()
            except:
                pass
            self.serial_port = None
        
        self.port_name = ""
        return True, "Disconnected"
    
    def is_connected(self) -> bool:
        """Check if connected and thread is running."""
        return (self.serial_port is not None and 
                self.serial_port.is_open and 
                self._running)
    
    # =========================================================================
    # STATE ACCESS (thread-safe)
    # =========================================================================
    
    def get_state(self) -> RobotState:
        """Get latest robot state (non-blocking, thread-safe)."""
        with self._state_lock:
            return RobotState(
                theta1_current=self._state.theta1_current,
                theta1_demand=self._state.theta1_demand,
                theta2_current=self._state.theta2_current,
                theta2_demand=self._state.theta2_demand,
                trajectory_active=self._state.trajectory_active,
                trajectory_index=self._state.trajectory_index,
                trajectory_total=self._state.trajectory_total,
                timestamp=self._state.timestamp,
                raw_line=self._state.raw_line
            )
    
    def get_angles(self) -> Tuple[float, float]:
        """Get current angles (convenience method)."""
        state = self.get_state()
        return state.theta1_current, state.theta2_current
    
    def get_history(self, n: int = 100) -> List[Dict]:
        """Get recent position history for graphing."""
        with self._history_lock:
            return list(self._history)[-n:]
    
    def get_messages(self) -> List[str]:
        """Get and clear status messages from Arduino."""
        with self._messages_lock:
            msgs = list(self._messages)
            self._messages.clear()
            return msgs
    
    # =========================================================================
    # COMMAND SENDING (thread-safe)
    # =========================================================================
    
    def send_angles(self, theta1: float, theta2: float) -> bool:
        """Send direct angle command (legacy mode)."""
        if not self.is_connected():
            return False
        
        cmd = f"{theta1:.2f},{theta2:.2f}\n"
        self._command_queue.put(cmd)
        return True
    
    def send_home(self) -> bool:
        """Send home command."""
        if not self.is_connected():
            return False
        
        self._command_queue.put("H\n")
        return True
    
    def send_stop(self) -> bool:
        """Send stop command."""
        if not self.is_connected():
            return False
        
        self._command_queue.put("S\n")
        return True
    
    def send_trajectory(self, points: List[Tuple[int, float, float]]) -> bool:
        """
        Send a trajectory batch to Arduino.
        
        Args:
            points: List of (time_ms, theta1, theta2) tuples
        
        Returns:
            True if sent successfully
        """
        if not self.is_connected():
            return False
        
        if len(points) == 0:
            return False
        
        if len(points) > 100:
            print(f"Warning: Trajectory truncated from {len(points)} to 100 points (Arduino buffer limit)")
            points = points[:100]
        
        # Queue the batch commands
        self._command_queue.put(f"B,{len(points)}\n")
        
        for t_ms, theta1, theta2 in points:
            self._command_queue.put(f"P,{int(t_ms)},{theta1:.2f},{theta2:.2f}\n")
        
        self._command_queue.put("E\n")
        
        return True
    
    def query_status(self) -> bool:
        """Query Arduino status."""
        if not self.is_connected():
            return False
        
        self._command_queue.put("?\n")
        return True
    
    # =========================================================================
    # READER THREAD
    # =========================================================================
    
    def _reader_loop(self):
        """Main reader thread loop - handles all serial I/O."""
        buffer = ""
        
        while self._running and self.serial_port:
            try:
                # ============================================
                # WRITE: Send any queued commands
                # ============================================
                while not self._command_queue.empty():
                    try:
                        cmd = self._command_queue.get_nowait()
                        self.serial_port.write(cmd.encode())
                    except queue.Empty:
                        break
                    except Exception as e:
                        print(f"Serial write error: {e}")
                
                # ============================================
                # READ: Process incoming data
                # ============================================
                if self.serial_port.in_waiting > 0:
                    try:
                        data = self.serial_port.read(self.serial_port.in_waiting)
                        buffer += data.decode('utf-8', errors='ignore')
                        
                        # Process complete lines
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            line = line.strip()
                            if line:
                                self._parse_line(line)
                    
                    except Exception as e:
                        print(f"Serial read error: {e}")
                
                # Small sleep to prevent CPU spinning
                time.sleep(0.005)  # 5ms
                
            except Exception as e:
                print(f"Reader thread error: {e}")
                time.sleep(0.1)
    
    def _parse_line(self, line: str):
        """Parse a line from Arduino output."""
        now = time.time()
        
        # Check for status messages
        if line.startswith(('READY', 'HOME', 'STOP', 'DONE', 'BATCH:', 'START:', 'STATUS:')):
            with self._messages_lock:
                self._messages.append(line)
            
            # Update trajectory state from messages
            if line == 'DONE':
                with self._state_lock:
                    self._state.trajectory_active = False
            elif line.startswith('START:'):
                with self._state_lock:
                    self._state.trajectory_active = True
                    try:
                        self._state.trajectory_total = int(line.split(':')[1])
                        self._state.trajectory_index = 0
                    except:
                        pass
            return
        
        # Parse position data: "current1 demand1 current2 demand2 [TRAJ:i/n]"
        parts = line.split()
        
        if len(parts) >= 4:
            try:
                c1 = float(parts[0])
                d1 = float(parts[1])
                c2 = float(parts[2])
                d2 = float(parts[3])
                
                # Validate (sanity check)
                if all(abs(v) < 400 for v in [c1, d1, c2, d2]):
                    with self._state_lock:
                        self._state.theta1_current = c1
                        self._state.theta1_demand = d1
                        self._state.theta2_current = c2
                        self._state.theta2_demand = d2
                        self._state.timestamp = now
                        self._state.raw_line = line
                        
                        # Check for trajectory info
                        if len(parts) >= 5 and parts[4].startswith('TRAJ:'):
                            self._state.trajectory_active = True
                            try:
                                traj_info = parts[4].split(':')[1]
                                idx, total = traj_info.split('/')
                                self._state.trajectory_index = int(idx)
                                self._state.trajectory_total = int(total)
                            except:
                                pass
                        else:
                            # No TRAJ tag means trajectory not active
                            self._state.trajectory_active = False
                    
                    # Add to history
                    with self._history_lock:
                        self._history.append({
                            'time': now,
                            'theta1': c1,
                            'theta2': c2,
                            'demand1': d1,
                            'demand2': d2
                        })
            
            except ValueError:
                pass  # Malformed line, ignore


# =============================================================================
# GLOBAL INSTANCE (for use by GUI)
# =============================================================================

_reader_instance: Optional[SerialReader] = None

def get_reader() -> SerialReader:
    """Get the global SerialReader instance."""
    global _reader_instance
    if _reader_instance is None:
        _reader_instance = SerialReader()
    return _reader_instance


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Serial Reader Test")
    print("=" * 60)
    
    reader = SerialReader()
    
    # Try to connect
    success, msg = reader.connect()
    print(f"Connect: {msg}")
    
    if success:
        print("\nReading for 5 seconds...")
        
        for i in range(50):
            time.sleep(0.1)
            state = reader.get_state()
            
            if i % 10 == 0:
                print(f"  θ1={state.theta1_current:.1f}° θ2={state.theta2_current:.1f}°")
        
        # Test sending home
        print("\nSending HOME...")
        reader.send_home()
        time.sleep(1)
        
        # Check messages
        msgs = reader.get_messages()
        print(f"Messages: {msgs}")
        
        # Disconnect
        reader.disconnect()
        print("\nDisconnected")
    
    print("\nDone!")

/*
   2-Link Robot Arm - SMOOTH VERSION V4 (Corrected Feedforward)
   
   Changes from v3:
   - FIXED: Feedforward sign (was wrong! now uses - Kv * velocity)
   - NEW: Low-pass filter on feedforward to prevent segment jumps
   - NEW: Smoothstep interpolation (cubic blend) for smoother motion
   
   Changes from v2:
   - Added velocity feedforward
   
   Changes from v1:
   - Removed integral reset during trajectory playback
   
   WHY THE SIGN FIX MATTERS:
   Our error = current - demand
   - Positive error → motor reduces position
   - Negative error → motor increases position
   
   So when desiredVelocity > 0 (demand increasing):
   - We want motor to INCREASE position
   - That needs NEGATIVE output contribution
   - Therefore: output += (-Kv) * desiredVelocity
   
   Commands:
     theta1,theta2  - Direct angle command
     B,n            - Begin batch of n points  
     P,t,θ1,θ2      - Trajectory point at time t (ms)
     E              - End batch, start playing
     H              - Home (go to 0,0)
     S              - Stop immediately
     ?              - Query status
   
   Author: Yagmur
   Date: December 2025
*/

// ============================================
// PIN DEFINITIONS
// ============================================

const int motorPin1_m1 = 6;
const int motorPin2_m1 = 5;
const int enablePin_m1 = 9;

const int motorPin1_m2 = 11;
const int motorPin2_m2 = 12;
const int enablePin_m2 = 10;

const int encoderPinA_m1 = 2;
const int encoderPinB_m1 = 4;

const int encoderPinA_m2 = 3;
const int encoderPinB_m2 = 8;

// ============================================
// MOTOR PARAMETERS
// ============================================

const float CPR_m1 = 180.0;
const float CYCLOIDAL_GR_m1 = 8.0;
const float CPR_m2 = 1500.0;

// ============================================
// TRAJECTORY BUFFER
// ============================================

const int MAX_TRAJECTORY_POINTS = 100;

struct TrajectoryPoint {
  uint16_t time_ms;
  int16_t theta1;    // Angle × 10
  int16_t theta2;    // Angle × 10
};

TrajectoryPoint trajectoryBuffer[MAX_TRAJECTORY_POINTS];
int trajectoryLength = 0;
int currentPointIndex = 0;
bool trajectoryActive = false;
unsigned long trajectoryStartTime = 0;

// Batch receiving state
bool receivingBatch = false;
int expectedPoints = 0;
int receivedPoints = 0;

// ============================================
// TIMING
// ============================================

unsigned long currentTime = 0;
unsigned long previousTime = 0;
float deltaT = 0.0;
const float SAMPLE_TIME = 0.01;  // 10ms = 100Hz

unsigned long lastSerialPrint = 0;
const unsigned long SERIAL_PRINT_INTERVAL = 50;  // 20Hz output

// ============================================
// ENCODER VARIABLES
// ============================================

volatile long counter_m1 = 0;
volatile long counter_m2 = 0;

float currentPositionInDegrees_m1 = 0.0;
float currentPositionInDegrees_m2 = 0.0;

// ============================================
// PID SETPOINTS
// ============================================

float demandPositionInDegrees_m1 = 0.0;
float demandPositionInDegrees_m2 = 0.0;

// ============================================
// PID VARIABLES
// ============================================

float error_m1 = 0.0;
float errorPrev_m1 = 0.0;
float errorSum_m1 = 0.0;
float errorDiff_m1 = 0.0;
float errorDiffFiltered_m1 = 0.0;
float controllerOutput_m1 = 0.0;

float error_m2 = 0.0;
float errorPrev_m2 = 0.0;
float errorSum_m2 = 0.0;
float errorDiff_m2 = 0.0;
float errorDiffFiltered_m2 = 0.0;
float controllerOutput_m2 = 0.0;

// Direction tracking (only for direct commands)
float prevDirection_m1 = 0;
float prevDirection_m2 = 0;

// ============================================
// VELOCITY FEEDFORWARD - V4 IMPROVEMENTS
// ============================================

// Raw velocity from trajectory slope
float rawVelocity_m1 = 0.0;
float rawVelocity_m2 = 0.0;

// Filtered velocity (smoothed to prevent jumps at segment boundaries)
float filteredVelocity_m1 = 0.0;
float filteredVelocity_m2 = 0.0;

// Filter coefficient: 0.3 = 30% new, 70% old (smooth but responsive)
const float FF_FILTER_COEFF = 0.3;

// ============================================
// PID TUNING PARAMETERS
// ============================================

float Kp_m1 = 15.0;
float Ki_m1 = 3.0;
float Kd_m1 = 0.8;
const int MAX_PWM_M1 = 180;
const int MIN_PWM_M1 = 25;

float Kp_m2 = 12.0;
float Ki_m2 = 5.0;
float Kd_m2 = 0.6;
const int MAX_PWM_M2 = 200;
const int MIN_PWM_M2 = 30;

// ============================================
// FEEDFORWARD GAINS - V4 (ACTIVE BY DEFAULT)
// ============================================
// These are POSITIVE values. The NEGATIVE sign is applied in the control law
// because our error convention is (current - demand).
//
// If circles get WORSE with feedforward, try:
// 1. Reduce Kv values (try 0.3, 0.2)
// 2. Or set to 0 to disable

const float Kv_m1 = 0.5;  // Feedforward gain motor 1
const float Kv_m2 = 0.4;  // Feedforward gain motor 2

// ============================================
// PID HELPER CONSTANTS
// ============================================

const float DEAD_ZONE = 0.3;
const float INTEGRAL_MAX = 150.0;
const float DERIVATIVE_FILTER = 0.3;
const float REVERSAL_THRESHOLD = 0.5;

// ============================================
// SETUP
// ============================================

void setup() {
  pinMode(motorPin1_m1, OUTPUT);
  pinMode(motorPin2_m1, OUTPUT);
  pinMode(enablePin_m1, OUTPUT);
  
  pinMode(motorPin1_m2, OUTPUT);
  pinMode(motorPin2_m2, OUTPUT);
  pinMode(enablePin_m2, OUTPUT);
  
  pinMode(encoderPinA_m1, INPUT_PULLUP);
  pinMode(encoderPinB_m1, INPUT_PULLUP);
  pinMode(encoderPinA_m2, INPUT_PULLUP);
  pinMode(encoderPinB_m2, INPUT_PULLUP);
  
  attachInterrupt(digitalPinToInterrupt(encoderPinA_m1), encoderISR_m1, CHANGE);
  attachInterrupt(digitalPinToInterrupt(encoderPinA_m2), encoderISR_m2, CHANGE);
  
  Serial.begin(115200);
  previousTime = micros();
  
  Serial.println("READY-V4-FF-CORRECTED");
}

// ============================================
// ENCODER ISRs
// ============================================

void encoderISR_m1() {
  int aState = digitalRead(encoderPinA_m1);
  int bState = digitalRead(encoderPinB_m1);
  if (aState != bState) {
    counter_m1++;
  } else {
    counter_m1--;
  }
}

void encoderISR_m2() {
  int aState = digitalRead(encoderPinA_m2);
  int bState = digitalRead(encoderPinB_m2);
  if (aState != bState) {
    counter_m2++;
  } else {
    counter_m2--;
  }
}

// ============================================
// MOTOR DRIVER
// ============================================

void driveMotor(int motorNum, float output, int minPWM) {
  int pwm = abs((int)output);
  
  if (pwm > 0 && pwm < minPWM) {
    pwm = minPWM;
  }
  
  if (motorNum == 1) {
    if (output > 0) {
      digitalWrite(motorPin1_m1, LOW);
      digitalWrite(motorPin2_m1, HIGH);
    } else if (output < 0) {
      digitalWrite(motorPin1_m1, HIGH);
      digitalWrite(motorPin2_m1, LOW);
    } else {
      digitalWrite(motorPin1_m1, LOW);
      digitalWrite(motorPin2_m1, LOW);
    }
    analogWrite(enablePin_m1, pwm);
  } else {
    if (output > 0) {
      digitalWrite(motorPin1_m2, LOW);
      digitalWrite(motorPin2_m2, HIGH);
    } else if (output < 0) {
      digitalWrite(motorPin1_m2, HIGH);
      digitalWrite(motorPin2_m2, LOW);
    } else {
      digitalWrite(motorPin1_m2, LOW);
      digitalWrite(motorPin2_m2, LOW);
    }
    analogWrite(enablePin_m2, pwm);
  }
}

void stopMotors() {
  analogWrite(enablePin_m1, 0);
  analogWrite(enablePin_m2, 0);
  digitalWrite(motorPin1_m1, LOW);
  digitalWrite(motorPin2_m1, LOW);
  digitalWrite(motorPin1_m2, LOW);
  digitalWrite(motorPin2_m2, LOW);
  
  rawVelocity_m1 = 0;
  rawVelocity_m2 = 0;
  filteredVelocity_m1 = 0;
  filteredVelocity_m2 = 0;
}

// ============================================
// SMOOTHSTEP FUNCTION - V4 NEW
// ============================================
// Cubic interpolation: s = 3t² - 2t³
// Properties: s(0)=0, s(1)=1, s'(0)=0, s'(1)=0
// This makes motion "glide" through waypoints instead of 
// constant-velocity-then-jump

float smoothstep(float t) {
  t = constrain(t, 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

// Derivative of smoothstep: ds/dt = 6t - 6t²
// Used to compute velocity from smoothstep position
float smoothstepDerivative(float t) {
  t = constrain(t, 0.0, 1.0);
  return 6.0 * t * (1.0 - t);
}

// ============================================
// TRAJECTORY INTERPOLATION - V4 WITH SMOOTHSTEP
// ============================================

void updateTrajectoryTarget() {
  if (!trajectoryActive || trajectoryLength < 2) {
    rawVelocity_m1 = 0;
    rawVelocity_m2 = 0;
    return;
  }
  
  unsigned long elapsed = millis() - trajectoryStartTime;
  
  // Find current segment
  while (currentPointIndex < trajectoryLength - 1 && 
         trajectoryBuffer[currentPointIndex + 1].time_ms <= elapsed) {
    currentPointIndex++;
  }
  
  // Check if complete
  if (currentPointIndex >= trajectoryLength - 1) {
    demandPositionInDegrees_m1 = trajectoryBuffer[trajectoryLength - 1].theta1 / 10.0;
    demandPositionInDegrees_m2 = trajectoryBuffer[trajectoryLength - 1].theta2 / 10.0;
    rawVelocity_m1 = 0;
    rawVelocity_m2 = 0;
    
    if (elapsed > (unsigned long)trajectoryBuffer[trajectoryLength - 1].time_ms + 500) {
      trajectoryActive = false;
      Serial.println("DONE");
    }
    return;
  }
  
  // Get segment endpoints
  TrajectoryPoint* p0 = &trajectoryBuffer[currentPointIndex];
  TrajectoryPoint* p1 = &trajectoryBuffer[currentPointIndex + 1];
  
  float t0_theta1 = p0->theta1 / 10.0;
  float t0_theta2 = p0->theta2 / 10.0;
  float t1_theta1 = p1->theta1 / 10.0;
  float t1_theta2 = p1->theta2 / 10.0;
  
  unsigned long dt_ms = p1->time_ms - p0->time_ms;
  if (dt_ms == 0) {
    demandPositionInDegrees_m1 = t1_theta1;
    demandPositionInDegrees_m2 = t1_theta2;
    rawVelocity_m1 = 0;
    rawVelocity_m2 = 0;
    return;
  }
  
  // Calculate normalized time in segment [0, 1]
  float t_linear = (float)(elapsed - p0->time_ms) / (float)dt_ms;
  t_linear = constrain(t_linear, 0.0, 1.0);
  
  // ============================================
  // V4: SMOOTHSTEP INTERPOLATION
  // ============================================
  // Instead of linear: demand = t0 + (t1-t0) * t
  // Use cubic smoothstep: demand = t0 + (t1-t0) * smoothstep(t)
  // This creates zero velocity at segment boundaries = smoother motion
  
  float s = smoothstep(t_linear);
  
  demandPositionInDegrees_m1 = t0_theta1 + (t1_theta1 - t0_theta1) * s;
  demandPositionInDegrees_m2 = t0_theta2 + (t1_theta2 - t0_theta2) * s;
  
  // ============================================
  // V4: VELOCITY FROM SMOOTHSTEP DERIVATIVE
  // ============================================
  // velocity = d(demand)/dt = (t1-t0) * smoothstep'(t) / dt
  // smoothstep'(t) = 6t(1-t), peaks at t=0.5
  
  float dt_seconds = dt_ms / 1000.0;
  
  if (dt_seconds > 0.001) {
    float ds_dt = smoothstepDerivative(t_linear);
    
    // Raw velocity in deg/s
    rawVelocity_m1 = (t1_theta1 - t0_theta1) * ds_dt / dt_seconds;
    rawVelocity_m2 = (t1_theta2 - t0_theta2) * ds_dt / dt_seconds;
  } else {
    rawVelocity_m1 = 0;
    rawVelocity_m2 = 0;
  }
  
  // ============================================
  // V4: LOW-PASS FILTER ON FEEDFORWARD
  // ============================================
  // Prevents abrupt jumps at segment boundaries
  // filtered = (1-α)*filtered + α*raw
  
  filteredVelocity_m1 = (1.0 - FF_FILTER_COEFF) * filteredVelocity_m1 + 
                         FF_FILTER_COEFF * rawVelocity_m1;
  filteredVelocity_m2 = (1.0 - FF_FILTER_COEFF) * filteredVelocity_m2 + 
                         FF_FILTER_COEFF * rawVelocity_m2;
}

// ============================================
// SERIAL COMMAND HANDLER
// ============================================

void processSerialCommand(String& cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;
  
  char cmdType = cmd.charAt(0);
  
  switch (cmdType) {
    case 'B':
    case 'b': {
      int comma = cmd.indexOf(',');
      if (comma > 0) {
        expectedPoints = cmd.substring(comma + 1).toInt();
        expectedPoints = constrain(expectedPoints, 1, MAX_TRAJECTORY_POINTS);
        receivedPoints = 0;
        receivingBatch = true;
        trajectoryActive = false;
        trajectoryLength = 0;
        Serial.print("BATCH:");
        Serial.println(expectedPoints);
      }
      return;
    }
    
    case 'P':
    case 'p': {
      if (receivingBatch) {
        int c1 = cmd.indexOf(',');
        int c2 = cmd.indexOf(',', c1 + 1);
        int c3 = cmd.indexOf(',', c2 + 1);
        
        if (c1 > 0 && c2 > 0 && c3 > 0 && receivedPoints < MAX_TRAJECTORY_POINTS) {
          unsigned long t = cmd.substring(c1 + 1, c2).toInt();
          float theta1 = cmd.substring(c2 + 1, c3).toFloat();
          float theta2 = cmd.substring(c3 + 1).toFloat();
          
          trajectoryBuffer[receivedPoints].time_ms = (uint16_t)min(t, 65535UL);
          trajectoryBuffer[receivedPoints].theta1 = (int16_t)(theta1 * 10.0);
          trajectoryBuffer[receivedPoints].theta2 = (int16_t)(theta2 * 10.0);
          receivedPoints++;
        }
      }
      return;
    }
    
    case 'E':
    case 'e': {
      if (receivingBatch && receivedPoints > 0) {
        trajectoryLength = receivedPoints;
        currentPointIndex = 0;
        trajectoryStartTime = millis();
        trajectoryActive = true;
        receivingBatch = false;
        errorSum_m1 = 0;
        errorSum_m2 = 0;
        prevDirection_m1 = 0;
        prevDirection_m2 = 0;
        rawVelocity_m1 = 0;
        rawVelocity_m2 = 0;
        filteredVelocity_m1 = 0;
        filteredVelocity_m2 = 0;
        Serial.print("START:");
        Serial.println(trajectoryLength);
      }
      return;
    }
    
    case 'H':
    case 'h': {
      trajectoryActive = false;
      receivingBatch = false;
      demandPositionInDegrees_m1 = 0.0;
      demandPositionInDegrees_m2 = 0.0;
      errorSum_m1 = 0;
      errorSum_m2 = 0;
      filteredVelocity_m1 = 0;
      filteredVelocity_m2 = 0;
      Serial.println("HOME");
      return;
    }
    
    case 'S':
    case 's': {
      trajectoryActive = false;
      receivingBatch = false;
      stopMotors();
      demandPositionInDegrees_m1 = currentPositionInDegrees_m1;
      demandPositionInDegrees_m2 = currentPositionInDegrees_m2;
      errorSum_m1 = 0;
      errorSum_m2 = 0;
      Serial.println("STOP");
      return;
    }
    
    case '?': {
      Serial.print("STATUS:");
      Serial.print(trajectoryActive ? "PLAYING" : "IDLE");
      Serial.print(",");
      Serial.print(currentPointIndex);
      Serial.print("/");
      Serial.println(trajectoryLength);
      return;
    }
  }
  
  // Direct angle command (feedforward not used)
  int comma = cmd.indexOf(',');
  if (comma > 0) {
    float t1 = cmd.substring(0, comma).toFloat();
    float t2 = cmd.substring(comma + 1).toFloat();
    
    if (t1 >= -180 && t1 <= 180 && t2 >= -180 && t2 <= 180) {
      trajectoryActive = false;
      receivingBatch = false;
      
      filteredVelocity_m1 = 0;
      filteredVelocity_m2 = 0;
      
      float newDir_m1 = t1 - demandPositionInDegrees_m1;
      float newDir_m2 = t2 - demandPositionInDegrees_m2;
      
      if (abs(newDir_m1) > REVERSAL_THRESHOLD) {
        float sign_new = (newDir_m1 > 0) ? 1.0 : -1.0;
        if (prevDirection_m1 != 0 && sign_new != prevDirection_m1) {
          errorSum_m1 = 0.0;
        }
        prevDirection_m1 = sign_new;
      }
      
      if (abs(newDir_m2) > REVERSAL_THRESHOLD) {
        float sign_new = (newDir_m2 > 0) ? 1.0 : -1.0;
        if (prevDirection_m2 != 0 && sign_new != prevDirection_m2) {
          errorSum_m2 = 0.0;
        }
        prevDirection_m2 = sign_new;
      }
      
      demandPositionInDegrees_m1 = t1;
      demandPositionInDegrees_m2 = t2;
      
      Serial.print("TARGET:");
      Serial.print(t1, 1);
      Serial.print(",");
      Serial.println(t2, 1);
    }
  }
}

void checkSerialInput() {
  static String inputBuffer = "";
  
  while (Serial.available() > 0) {
    char c = Serial.read();
    
    if (c == '\n' || c == '\r') {
      if (inputBuffer.length() > 0) {
        processSerialCommand(inputBuffer);
        inputBuffer = "";
      }
    } else {
      inputBuffer += c;
      if (inputBuffer.length() > 100) {
        inputBuffer = "";
      }
    }
  }
}

// ============================================
// MAIN LOOP
// ============================================

void loop() {
  checkSerialInput();
  
  currentTime = micros();
  deltaT = (currentTime - previousTime) / 1000000.0;

  if (deltaT >= SAMPLE_TIME) {
    previousTime = currentTime;
    
    if (trajectoryActive) {
      updateTrajectoryTarget();
    }
    
    // Encoder to degrees
    currentPositionInDegrees_m1 = (counter_m1 * 360.0) / (CPR_m1 * CYCLOIDAL_GR_m1);
    currentPositionInDegrees_m2 = (counter_m2 * 360.0) / CPR_m2;

    // Wrap around
    if (currentPositionInDegrees_m1 >= 360.0 || currentPositionInDegrees_m1 <= -360.0) {
      int fullRotations = (int)(currentPositionInDegrees_m1 / 360.0);
      counter_m1 -= fullRotations * CPR_m1 * CYCLOIDAL_GR_m1;
    }
    if (currentPositionInDegrees_m2 >= 360.0 || currentPositionInDegrees_m2 <= -360.0) {
      int fullRotations = (int)(currentPositionInDegrees_m2 / 360.0);
      counter_m2 -= fullRotations * CPR_m2;
    }

    // ========================================
    // PID + CORRECTED FEEDFORWARD for Motor 1
    // ========================================
    
    error_m1 = currentPositionInDegrees_m1 - demandPositionInDegrees_m1;
    
    errorDiff_m1 = (error_m1 - errorPrev_m1) / deltaT;
    errorDiffFiltered_m1 = DERIVATIVE_FILTER * errorDiff_m1 + 
                           (1.0 - DERIVATIVE_FILTER) * errorDiffFiltered_m1;
    
    if (abs(error_m1) > DEAD_ZONE) {
      errorSum_m1 += error_m1 * deltaT;
      errorSum_m1 = constrain(errorSum_m1, -INTEGRAL_MAX, INTEGRAL_MAX);
    }
    
    errorPrev_m1 = error_m1;
    
    // V4: CORRECTED FEEDFORWARD WITH NEGATIVE SIGN
    // error = current - demand, so positive error → reduce position
    // When desiredVelocity > 0, we want to INCREASE position
    // That requires NEGATIVE output → hence MINUS sign on feedforward
    controllerOutput_m1 = (Kp_m1 * error_m1) + 
                          (Ki_m1 * errorSum_m1) + 
                          (Kd_m1 * errorDiffFiltered_m1) -
                          (Kv_m1 * filteredVelocity_m1);  // NOTE: MINUS SIGN!
    
    if (abs(error_m1) <= DEAD_ZONE && abs(filteredVelocity_m1) < 5.0) {
      controllerOutput_m1 = 0;
      errorSum_m1 *= 0.95;
    }
    
    controllerOutput_m1 = constrain(controllerOutput_m1, -MAX_PWM_M1, MAX_PWM_M1);

    // ========================================
    // PID + CORRECTED FEEDFORWARD for Motor 2
    // ========================================
    
    error_m2 = currentPositionInDegrees_m2 - demandPositionInDegrees_m2;
    
    errorDiff_m2 = (error_m2 - errorPrev_m2) / deltaT;
    errorDiffFiltered_m2 = DERIVATIVE_FILTER * errorDiff_m2 + 
                           (1.0 - DERIVATIVE_FILTER) * errorDiffFiltered_m2;
    
    if (abs(error_m2) > DEAD_ZONE) {
      errorSum_m2 += error_m2 * deltaT;
      errorSum_m2 = constrain(errorSum_m2, -INTEGRAL_MAX, INTEGRAL_MAX);
    }
    
    errorPrev_m2 = error_m2;
    
    // V4: CORRECTED FEEDFORWARD WITH NEGATIVE SIGN
    controllerOutput_m2 = (Kp_m2 * error_m2) + 
                          (Ki_m2 * errorSum_m2) + 
                          (Kd_m2 * errorDiffFiltered_m2) -
                          (Kv_m2 * filteredVelocity_m2);  // NOTE: MINUS SIGN!
    
    if (abs(error_m2) <= DEAD_ZONE && abs(filteredVelocity_m2) < 5.0) {
      controllerOutput_m2 = 0;
      errorSum_m2 *= 0.95;
    }
    
    controllerOutput_m2 = constrain(controllerOutput_m2, -MAX_PWM_M2, MAX_PWM_M2);

    // ========================================
    // Drive motors
    // ========================================
    
    driveMotor(1, controllerOutput_m1, MIN_PWM_M1);
    driveMotor(2, controllerOutput_m2, MIN_PWM_M2);

    // ========================================
    // Serial output at 20Hz
    // ========================================
    
    if (millis() - lastSerialPrint >= SERIAL_PRINT_INTERVAL) {
      lastSerialPrint = millis();
      
      Serial.print(currentPositionInDegrees_m1, 1);
      Serial.print(" ");
      Serial.print(demandPositionInDegrees_m1, 1);
      Serial.print(" ");
      Serial.print(currentPositionInDegrees_m2, 1);
      Serial.print(" ");
      Serial.print(demandPositionInDegrees_m2, 1);
      
      if (trajectoryActive) {
        Serial.print(" TRAJ:");
        Serial.print(currentPointIndex);
        Serial.print("/");
        Serial.print(trajectoryLength);
        Serial.print(" V:");
        Serial.print(filteredVelocity_m1, 0);
        Serial.print(",");
        Serial.print(filteredVelocity_m2, 0);
      }
      
      Serial.println();
    }
  }
}

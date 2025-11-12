import cv2
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import mediapipe as mp
from sklearn.preprocessing import StandardScaler
import joblib
import pygame
import time

# ============================================================================
# MediaPipe Setup
# ============================================================================
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# ============================================================================
# Utility Functions from MediaPipe Script
# ============================================================================
def angle_3pt(a, b, c):
    """Calculate angle between three points"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-9)
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

# ============================================================================
# BenchCounter Class for Stage Detection
# ============================================================================
class BenchCounter:
    def __init__(self, elbow_low_thresh=80, elbow_high_thresh=165, 
                 bar_down_delta=0.06, smooth=5, refractory_frames=6):
        self.elbow_low_thresh = elbow_low_thresh
        self.elbow_high_thresh = elbow_high_thresh
        self.bar_down_delta = bar_down_delta
        self.stage = "top"
        self.count = 0
        self.elbow_angles = deque(maxlen=smooth)
        self.bar_y_vals = deque(maxlen=smooth)
        self.personal_top_bar_y = None
        self.frames_since_top = refractory_frames
        self.refractory_frames = refractory_frames

    def update(self, left_elbow_angle, right_elbow_angle, bar_y, left_vis, right_vis, bar_valid):
        # Average elbow angles
        if left_vis and right_vis:
            elbow_angle = 0.5*(left_elbow_angle + right_elbow_angle)
        elif left_vis:
            elbow_angle = left_elbow_angle
        elif right_vis:
            elbow_angle = right_elbow_angle
        else:
            elbow_angle = 180.0

        self.elbow_angles.append(elbow_angle)

        if bar_valid:
            self.bar_y_vals.append(bar_y)
        elif len(self.bar_y_vals) == 0:
            self.bar_y_vals.append(0.0)

        sm_elbow = float(np.mean(self.elbow_angles))
        sm_bar_y = float(np.mean(self.bar_y_vals))

        # Update personal top bar position
        if sm_elbow >= self.elbow_high_thresh and bar_valid:
            if self.personal_top_bar_y is None:
                self.personal_top_bar_y = sm_bar_y
            else:
                self.personal_top_bar_y = min(self.personal_top_bar_y, sm_bar_y)

        self.frames_since_top += 1

        # Stage detection
        if self.stage == "top":
            down_ok = False
            if (self.personal_top_bar_y is not None) and bar_valid:
                down_ok = (sm_bar_y - self.personal_top_bar_y) >= self.bar_down_delta
            else:
                down_ok = True

            if (sm_elbow <= self.elbow_low_thresh) and down_ok:
                self.stage = "bottom"
                self.frames_since_top = 0

        elif self.stage == "bottom":
            if (sm_elbow >= self.elbow_high_thresh) and (self.frames_since_top >= self.refractory_frames):
                self.stage = "top"
                self.count += 1
                self.frames_since_top = 0

        return sm_elbow, sm_bar_y, self.count, self.stage, (self.personal_top_bar_y or 0.0)

# ============================================================================
# LSTM Model Definition
# ============================================================================
class BenchPressLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes=2, dropout=0.2):
        super(BenchPressLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM encoder
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Dense layers to map to class logits
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use the last hidden state
        last_hidden = h_n[-1]  # Take the last layer's hidden state

        # Apply dropout
        out = self.dropout(last_hidden)

        # Dense layers
        out = self.relu(self.fc1(out))
        out = self.fc2(out)

        return out

# ============================================================================
# Main Real-Time Detection System
# ============================================================================
def main():
    # Model parameters
    SIGNAL_MAX_LENGTH = 400  # 400 samples
    FAILURE_THRESHOLD = 12
    input_size = 2  # 2 features (elbow angle, wrist y position)
    hidden_size = 128
    num_layers = 4
    num_classes = 2  # Binary classification: 0=non-failure, 1=failure
    dropout = 0.2
    failure_frame_count = 0
    

    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize pygame mixer for beep sound
    pygame.mixer.init()
    try:
        beep_sound = pygame.mixer.Sound('lightweightbaby.mp3')
    except:
        print("WARNING: beep not found. Will skip sound alerts.")
        beep_sound = None

    # Load scaler

    # Initialize and load LSTM model
    model = BenchPressLSTM(input_size, hidden_size, num_layers, num_classes, dropout)
    model.load_state_dict(torch.load('best_bench_press_lstm.pth', map_location=device))
    model.to(device)
    model.eval()
    print("LSTM model loaded successfully")
    

    try:
        scaler = joblib.load('scaler_X.pkl')
        print("Scaler loaded successfully")
    except:
        print("WARNING: Scaler not found. Using StandardScaler without fitting.")
        scaler = StandardScaler()

    # Initialize MediaPipe Pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=False,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    # Initialize BenchCounter for stage detection
    bench = BenchCounter()

    # Signal buffer for LSTM input (stores last 400 frames)
    signal_buffer = []

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    print("Starting real-time detection... Press 'q' to quit")

    while True:
        loop_start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame")
            break

        H, W = frame.shape[:2]

        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        # Default values
        l_ang = r_ang = 180.0
        bar_y = 0.0
        l_vis = r_vis = False
        bar_valid = False

        # Process pose landmarks
        if res.pose_landmarks:
            # Draw landmarks on frame
            mp_draw.draw_landmarks(
                frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
            )

            lm = res.pose_landmarks.landmark

            # Extract relevant landmarks
            def get_landmark(idx):
                return (lm[idx].x, lm[idx].y), lm[idx].visibility

            (LSH, vLSH) = get_landmark(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
            (LEL, vLEL) = get_landmark(mp_pose.PoseLandmark.LEFT_ELBOW.value)
            (LWR, vLWR) = get_landmark(mp_pose.PoseLandmark.LEFT_WRIST.value)
            (RSH, vRSH) = get_landmark(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
            (REL, vREL) = get_landmark(mp_pose.PoseLandmark.RIGHT_ELBOW.value)
            (RWR, vRWR) = get_landmark(mp_pose.PoseLandmark.RIGHT_WRIST.value)

            # Check visibility
            l_vis = (vLSH > 0.5) and (vLEL > 0.5) and (vLWR > 0.5)
            r_vis = (vRSH > 0.5) and (vREL > 0.5) and (vRWR > 0.5)

            # Calculate elbow angles
            if l_vis:
                l_ang = angle_3pt(LSH, LEL, LWR)
            if r_vis:
                r_ang = angle_3pt(RSH, REL, RWR)

            # Calculate bar y position (wrist y position)
            if vLWR > 0.5 and vRWR > 0.5:
                bar_y = 0.5 * (LWR[1] + RWR[1])
                bar_valid = True
            elif vLWR > 0.5:
                bar_y = LWR[1]
                bar_valid = True
            elif vRWR > 0.5:
                bar_y = RWR[1]
                bar_valid = True

        # Update bench counter and get stage
        sm_elbow, sm_bar_y, rep_count, stage, personal_top_y = bench.update(
            l_ang, r_ang, bar_y, l_vis, r_vis, bar_valid
        )

        # Add current frame data to signal buffer
        # Signal format: [elbow_angle, wrist_y_position]
        signal_buffer.append([sm_elbow, sm_bar_y])

        # Keep only last SIGNAL_MAX_LENGTH frames
        if len(signal_buffer) > SIGNAL_MAX_LENGTH:
            signal_buffer = signal_buffer[-SIGNAL_MAX_LENGTH:]

        # Prepare input for LSTM (left-padded to 400 frames)
        current_signal = np.array(signal_buffer)

        # Left pad if necessary
        if len(current_signal) < SIGNAL_MAX_LENGTH:
            padding_length = SIGNAL_MAX_LENGTH - len(current_signal)
            padding = np.zeros((padding_length, 2))
            padded_signal = np.vstack([padding, current_signal])
        else:
            padded_signal = current_signal

        # Normalize using scaler
        try:
            # Reshape for scaling
            signal_flat = padded_signal.reshape(-1, 2)
            signal_scaled = scaler.transform(signal_flat).reshape(1, SIGNAL_MAX_LENGTH, 2)
        except:
            # If scaler fails, use raw signal
            signal_scaled = padded_signal.reshape(1, SIGNAL_MAX_LENGTH, 2)

        # Convert to tensor and get prediction
        signal_tensor = torch.FloatTensor(signal_scaled).to(device)

        with torch.no_grad():
            output = model(signal_tensor)
            probs = torch.softmax(output, dim=1)
            failure_prob = probs[0, 1].item()  # Probability of failure
            prediction = 1 if failure_prob > 0.75 else 0

        
        
        if prediction == 0:
            failure_frame_count = 0
        else:
            failure_frame_count += 1
        
        # Determine prediction label
        prediction_label = "FAILURE" if prediction == 1 and failure_frame_count > FAILURE_THRESHOLD and rep_count > 2 else "NORMAL"
        prediction_color = (0, 0, 255) if prediction == 1 and failure_frame_count > FAILURE_THRESHOLD and rep_count > 2 else (0, 255, 0)
        
        # Beep if failure detected
        if prediction == 1 and beep_sound is not None and failure_frame_count > FAILURE_THRESHOLD and rep_count > 2:
            beep_sound.play()

        # Draw HUD on frame
        # Background box
        cv2.rectangle(frame, (10, 10), (400, 180), (0, 0, 0), -1)

        # Title
        cv2.putText(frame, 'VIRTUAL SPOTTER', (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # Stage and rep count
        cv2.putText(frame, f'Stage: {stage.upper()}', (20, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f'Reps: {rep_count}', (20, 95), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # LSTM Prediction
        cv2.putText(frame, f'Prediction: {prediction_label}', (20, 125), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, prediction_color, 2)
        cv2.putText(frame, f'Failure Risk: {failure_prob*100:.1f}%', (20, 155), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, prediction_color, 2)

        # Buffer fill indicator
        buffer_fill = len(signal_buffer)
        cv2.putText(frame, f'Buffer: {buffer_fill}/{SIGNAL_MAX_LENGTH}', (20, 175), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Display frame
        cv2.imshow('Virtual Spotter - Real-time Rep Failure Detection', frame)
        
        loop_end_time = time.time()
        loop_latency_ms = (loop_end_time - loop_start_time) * 1000
        fps = 1.0 / (loop_end_time - loop_start_time)
        
        # Log to console
        print(f"Loop latency: {loop_latency_ms:.2f} ms | FPS: {fps:.1f}")
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    print("Detection stopped")

if __name__ == "__main__":
    main()

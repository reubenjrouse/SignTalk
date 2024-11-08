import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import torch
import time
from collections import deque
from models.model import SignLanguageModel

# Add drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
# Constants
BUFFER_SIZE = 10
MOTION_THRESHOLD = 0.05
STATIC_THRESHOLD = 0.05
STATIC_FRAME_COUNT = 20
MAX_GESTURE_FRAMES = 200
MIN_GESTURE_FRAMES = 35
REQUIRED_FRAMES = 65

# Label mapping
label_mapping = {'Alright': 0,
 'Doctor': 1,
 'Expensive': 2,
 'Female': 3,
 'Good Morning': 4,
 'Goodnight': 5,
 'He': 6,
 'Hello': 7,
 'How are you': 8,
 'I': 9,
 'It': 10,
 'Lawyer': 11,
 'Male': 12,
 'Mean': 13,
 'Nice': 14,
 'Patient ': 15,
 'Pleased': 16,
 'Poor': 17,
 'She': 18,
 'Teacher': 19,
 'Thank you': 20,
 'Thick': 21,
 'Thin': 22,
 'You': 23}

class SignLanguageDetector:
    def __init__(self, model_path, device='cpu'):
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.label_mapping = label_mapping
        self.idx_to_label = {v: k for k, v in self.label_mapping.items()}
        
        # Initialize MediaPipe
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7
        )
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.7,
            model_complexity=1
        )
        
        # Load model
        self.device = torch.device(device)
        checkpoint = torch.load(model_path, map_location=device)
        self.model = SignLanguageModel(
            input_size=108,
            hidden_size=256,
            num_layers=2,
            num_classes=len(self.label_mapping),
            dropout=0.5
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Initialize buffers
        self.frame_buffer = deque(maxlen=BUFFER_SIZE)
        self.gesture_frames = []
        self.static_frame_counter = 0
        self.gesture_in_progress = False
        self.last_prediction = ""
        self.last_confidence = 0.0
        
    def normalize_features(self, features):
        """Normalize features to [-1, 1] range"""
        if np.all(features == 0):
            return features
            
        features_reshaped = features.reshape(-1, 2)
        
        x_min, x_max = features_reshaped[:, 0].min(), features_reshaped[:, 0].max()
        if x_min != x_max:
            features_reshaped[:, 0] = 2 * (features_reshaped[:, 0] - x_min) / (x_max - x_min) - 1
            
        y_min, y_max = features_reshaped[:, 1].min(), features_reshaped[:, 1].max()
        if y_min != y_max:
            features_reshaped[:, 1] = 2 * (features_reshaped[:, 1] - y_min) / (y_max - y_min) - 1
            
        return features_reshaped.reshape(features.shape)

    def draw_landmarks(self, frame, hand_results, pose_results):
        """Draw MediaPipe landmarks on frame"""
        annotated_frame = frame.copy()
        
        # Draw hand landmarks
        if hand_results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                # Get handedness label
                hand_label = handedness.classification[0].label
                
                # Set color based on handedness (green for right, blue for left)
                hand_color = (0, 255, 0) if hand_label == "Right" else (255, 0, 0)
                
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=hand_color, thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=hand_color, thickness=1)
                )
                
                # Add hand label
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * annotated_frame.shape[1])
                    y = int(landmark.y * annotated_frame.shape[0])
                    cv2.putText(annotated_frame, hand_label, (x-20, y-20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 1)
                    break  # Only add label once per hand
        
        # Draw pose landmarks (only upper body 11-22)
        if pose_results.pose_landmarks:
            # Draw selected pose landmarks
            pose_connections = [conn for conn in mp_pose.POSE_CONNECTIONS 
                              if all(11 <= idx <= 22 for idx in conn)]
            
            # Create a subset of landmarks for upper body
            upper_body_landmarks = type('obj', (object,), {'landmark': pose_results.pose_landmarks.landmark[11:23]})
            
            # Draw landmarks
            mp_drawing.draw_landmarks(
                annotated_frame,
                pose_results.pose_landmarks,
                pose_connections,  # Custom connections
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(255, 0, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(255, 0, 0), thickness=1)
            )
            
            # Add joint labels for pose landmarks
            h, w = annotated_frame.shape[:2]
            for idx, landmark in enumerate(pose_results.pose_landmarks.landmark[11:23], start=11):
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.putText(annotated_frame, f'{idx}', (x-10, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        
        return annotated_frame

    def process_frame(self, frame):
        """Process frame and extract features"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get results
        hand_results = self.hands.process(frame_rgb)
        pose_results = self.pose.process(frame_rgb)
        
        # Draw landmarks
        annotated_frame = self.draw_landmarks(frame_rgb, hand_results, pose_results)
        
        # Extract features
        left_hand = np.zeros((21, 2))
        right_hand = np.zeros((21, 2))
        pose_points = np.zeros((12, 2))
        
        # Extract pose landmarks (11-22)
        if pose_results.pose_landmarks:
            for idx, landmark_idx in enumerate(range(11, 23)):
                landmark = pose_results.pose_landmarks.landmark[landmark_idx]
                pose_points[idx] = [landmark.x, landmark.y]
        
        # Extract hand landmarks with handedness
        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            for hand_landmarks, handedness in zip(
                hand_results.multi_hand_landmarks,
                hand_results.multi_handedness
            ):
                hand_type = "Left" if handedness.classification[0].label == "Right" else "Right"
                coords = np.array([[landmark.x, landmark.y] 
                                 for landmark in hand_landmarks.landmark])
                
                if coords.shape[0] == 21:
                    if hand_type == "Left":
                        left_hand = coords
                    else:
                        right_hand = coords
        
        # Normalize features
        pose_points = self.normalize_features(pose_points)
        left_hand = self.normalize_features(left_hand)
        right_hand = self.normalize_features(right_hand)
        
        # Combine features
        features = np.concatenate([
            pose_points.flatten(),
            left_hand.flatten(),
            right_hand.flatten()
        ])
        
        return features, annotated_frame

    def adjust_frames(self, frames):
        """Adjust frames to required length"""
        frames = np.array(frames)
        if len(frames) == REQUIRED_FRAMES:
            return frames
        elif len(frames) > REQUIRED_FRAMES:
            indices = np.linspace(0, len(frames) - 1, REQUIRED_FRAMES, dtype=int)
            return frames[indices]
        else:
            indices = np.linspace(0, len(frames) - 1, REQUIRED_FRAMES)
            new_frames = []
            for idx in indices:
                i = int(idx)
                if i >= len(frames) - 1:
                    new_frames.append(frames[-1])
                    continue
                alpha = idx - i
                frame1 = frames[i]
                frame2 = frames[i + 1]
                interpolated = frame1 * (1 - alpha) + frame2 * alpha
                new_frames.append(interpolated)
            return np.array(new_frames)

    def process_gesture(self, features):
        adjusted_features = self.adjust_frames(self.gesture_frames)
        model_input = torch.FloatTensor(adjusted_features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs, _ = self.model(model_input)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = torch.softmax(outputs, dim=1)[0][predicted_class].item()
            
        self.last_prediction = self.idx_to_label[predicted_class]
        self.last_confidence = confidence

def main():
    st.title("Sign Language Detector")
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Add a placeholder for the video feed
        frame_placeholder = st.empty()
    
    with col2:
        # Add placeholders for text display
        st.markdown("### Detection Status")
        prediction_text = st.empty()
        confidence_text = st.empty()
        frame_count_text = st.empty()
        
        # Add legend
        st.markdown("### Legend")
        st.markdown("""
        - 🟢 Right Hand Landmarks
        - 🔵 Left Hand Landmarks
        - 🔴 Pose Landmarks (11-22)
        """)
    
    # Initialize detector
    model_path = "models/best_model_24.pth"  # Update this path
    detector = SignLanguageDetector(model_path)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame")
            break
            
        # Process frame and get features and annotated frame
        features, annotated_frame = detector.process_frame(frame)
        
        # Update frame buffer and check for motion
        detector.frame_buffer.append(features)
        
        if len(detector.frame_buffer) > 1:
            motion = np.mean(np.abs(detector.frame_buffer[-1] - detector.frame_buffer[-2]))
            
            if not detector.gesture_in_progress and motion > MOTION_THRESHOLD:
                detector.gesture_in_progress = True
                detector.gesture_frames = []
                detector.static_frame_counter = 0
            
            elif detector.gesture_in_progress:
                detector.gesture_frames.append(features)
                
                if motion < STATIC_THRESHOLD:
                    detector.static_frame_counter += 1
                else:
                    detector.static_frame_counter = 0
                
                if (detector.static_frame_counter >= STATIC_FRAME_COUNT or 
                    len(detector.gesture_frames) >= MAX_GESTURE_FRAMES):
                    
                    if len(detector.gesture_frames) >= MIN_GESTURE_FRAMES:
                        detector.process_gesture(features)
                    
                    detector.gesture_in_progress = False
                    detector.gesture_frames = []
                    detector.static_frame_counter = 0
        
        # Display frame with landmarks
        frame_placeholder.image(annotated_frame, channels="RGB")
        
        # Update text displays
        if detector.last_prediction:
            prediction_text.markdown(f"**Prediction:** {detector.last_prediction}")
            confidence_text.markdown(f"**Confidence:** {detector.last_confidence:.2%}")
        frame_count_text.markdown(f"**Frames:** {len(detector.gesture_frames)}")
        
    cap.release()

if __name__ == "__main__":
    main()
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import torch
import time
from collections import deque
from models.model import SignLanguageModel
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import torch
import time
from collections import deque
from models.model import SignLanguageModel
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Add drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
# Constants
SENTENCE_END_TIMEOUT = 5.0
BUFFER_SIZE = 10
MOTION_THRESHOLD = 0.05
STATIC_THRESHOLD = 0.05
STATIC_FRAME_COUNT = 20
MAX_GESTURE_FRAMES = 200
MIN_GESTURE_FRAMES = 20
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

        self.current_sentence = []
        self.last_prediction_time = time.time()
        self.sentence_in_progress = False
        
        # Initialize Groq
        self.llm = ChatGroq(
            model="gemma2-9b-it",
            groq_api_key=GROQ_API_KEY
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that takes a list of sign language words "
                      "and converts them into a natural, grammatically correct sentence. "
                      "Focus on maintaining the original meaning while making it flow naturally."),
            MessagesPlaceholder(variable_name="messages")
        ])
        self.chain = self.prompt | self.llm

    def process_sentence(self):
        """Process collected words through LLM to form a sentence"""
        if not self.current_sentence:
            return ""
                
        words = " ".join(self.current_sentence)
        try:
            response = self.chain.invoke({
                "messages": [
                    HumanMessage(content=f"Convert to natural sentence, be brief and direct: {words}")
                ]
            })
            # Clean up the response - remove quotes and extra text
            sentence = response.content
            sentence = sentence.strip('"').strip("'")  # Remove quotes
            sentence = sentence.split('\n')[0]  # Take only first line
            # Remove any extra commentary
            if '.' in sentence:
                sentence = sentence.split('.')[0] + '.'
            return sentence
        except Exception as e:
            return " ".join(self.current_sentence)

        
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
        """Process gesture and get prediction"""
        adjusted_features = self.adjust_frames(self.gesture_frames)
        model_input = torch.FloatTensor(adjusted_features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs, _ = self.model(model_input)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = torch.softmax(outputs, dim=1)[0][predicted_class].item()
            
        self.last_prediction = self.idx_to_label[predicted_class]
        self.last_confidence = confidence
        
        # Start sentence if needed
        if self.last_confidence > 0.5:
            if not self.sentence_in_progress:
                self.sentence_in_progress = True
                self.current_sentence = []
            
            # Add word if it's new
            if not self.current_sentence or self.last_prediction != self.current_sentence[-1]:
                self.current_sentence.append(self.last_prediction)
                self.last_prediction_time = time.time()
        
        return self.last_prediction, self.last_confidence

    def update_sentence(self):
        """Handle sentence building and completion"""
        current_time = time.time()
        time_since_last = current_time - self.last_prediction_time
        
        # Handle word collection
        if self.last_confidence > 0.5:
            # Start collecting words
            if not self.sentence_in_progress:
                self.sentence_in_progress = True
                self.current_sentence = []
            
            # Reset timer when new word is added
            if (not self.current_sentence or 
                (self.last_prediction != self.current_sentence[-1] and 
                time_since_last < SENTENCE_END_TIMEOUT)):
                self.current_sentence.append(self.last_prediction)
                self.last_prediction_time = current_time
                return False, None  # Continue collecting words
        
        # Only process sentence if:
        # 1. We have words collected
        # 2. Timer has expired
        # 3. Sentence is in progress
        if (self.sentence_in_progress and 
            len(self.current_sentence) > 0 and 
            time_since_last >= SENTENCE_END_TIMEOUT):
            
            refined_sentence = self.process_sentence()
            self.sentence_in_progress = False
            temp_sentence = self.current_sentence.copy()  # Store for returning
            self.current_sentence = []
            
            return True, refined_sentence
        
        return False, None

def process_tutorial_video(input_path):
    """Process video to ensure compatibility with Streamlit"""
    output_path = input_path.replace('.mp4', '_processed.mp4')
    if not os.path.exists(output_path):
        try:
            os.system(f'ffmpeg -i {input_path} -vcodec libx264 {output_path}')
            return output_path
        except Exception as e:
            st.error(f"Error processing video: {e}")
            return None
    return output_path

@st.cache_resource
def get_cap():
    return cv2.VideoCapture(0)

@st.cache_resource
def initialize_detector(model_path):
    return SignLanguageDetector(model_path)

@st.cache_data
def load_video(video_path):
    return open(video_path, 'rb').read()

def toggle_camera():
    st.session_state.webcam_on = not st.session_state.webcam_on
    if not st.session_state.webcam_on:
        st.session_state.detection_active = False

def toggle_detection():
    if st.session_state.webcam_on:
        st.session_state.detection_active = not st.session_state.detection_active

def main():
    # Custom CSS for consistent button sizes
    st.markdown("""
        <style>
        .stButton > button {
            width: 150px;
            height: 50px;
        }
        .stRadio > label {
            font-size: 1.2rem;
            padding: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("Sign Language Learning and Detection")
    
    # Initialize states
    if 'detection_active' not in st.session_state:
        st.session_state.detection_active = False
    if 'webcam_on' not in st.session_state:
        st.session_state.webcam_on = False
    if 'processed_sentences' not in st.session_state:
        st.session_state.processed_sentences = []
    if 'mode' not in st.session_state:
        st.session_state.mode = "Word Detection"
    
    # Get cached resources
    cap = get_cap()
    detector = initialize_detector("models/best_model_24.pth")
    
    # Main interface
    tab1, tab2, tab3 = st.tabs(["📹 Live Detection", "🎥 Tutorial Signs", "ℹ️ Instructions"])
    
    with tab1:
        st.markdown("### Detection Mode")
        mode = st.radio("Select Mode", 
               ["Word Detection", "Sentence Formation"],
               horizontal=True,
               key='mode')  # Remove the on_change parameter
        
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            st.button("Toggle Camera", 
                     on_click=toggle_camera,
                     key='camera_toggle')
        
        with col2:
            st.button("Stop" if st.session_state.detection_active else "Start",
                     on_click=toggle_detection,
                     disabled=not st.session_state.webcam_on,
                     key='detection_toggle')
        
        video_col, info_col = st.columns([2, 1])
        
        with video_col:
            frame_placeholder = st.empty()
            if not st.session_state.webcam_on:
                frame_placeholder.markdown("""
                    <div style='background-color: black; color: white; padding: 20px; 
                         text-align: center; border-radius: 10px; height: 400px; 
                         display: flex; align-items: center; justify-content: center;'>
                        <div>
                            Camera is turned off<br>
                            Click 'Toggle Camera' to start
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        with info_col:
            st.markdown("### Live Detection")
            prediction_text = st.empty()
            confidence_text = st.empty()
            frame_count_text = st.empty()
            
            if st.session_state.mode == "Sentence Formation":
                st.markdown("### Current Sentence")
                words_text = st.empty()
                sentence_status = st.empty()
                
                st.markdown("### Processed Sentences")
                sentences_container = st.empty()

    with tab2:
        st.markdown("### Learn Sign Language")
        st.markdown("Click on any word to see its tutorial video")
        
        # Load cached videos
        sign_tutorials = {
            "Hello": {
                "video": load_video("Doctor_13_features.mp4"),
                "description": "Wave your hand in a friendly greeting motion."
            },
            "Thank you": {
                "video": load_video("Doctor_13_features.mp4"),
                "description": "Touch your chin with your fingertips and move your hand forward."
            },
            "Good Morning": {
                "video": load_video("Doctor_13_features.mp4"),
                "description": "Combine the signs for 'good' and 'morning'."
            }
        }
        
        cols = st.columns(3)
        for idx, (sign, info) in enumerate(sign_tutorials.items()):
            with cols[idx % 3]:
                st.markdown(f"#### {sign}")
                if st.button(f"Watch Tutorial", key=f"btn_{sign}"):
                    st.session_state.selected_tutorial = sign
                st.markdown(info["description"])
                st.markdown("---")
        
        if 'selected_tutorial' in st.session_state:
            sign = st.session_state.selected_tutorial
            if sign in sign_tutorials:
                st.markdown(f"### Tutorial for: {sign}")
                st.video(sign_tutorials[sign]["video"])
                st.markdown(sign_tutorials[sign]["description"])

    with tab3:
        st.markdown("""
        ### How to Use the App
        
        #### Detection Modes
        1. **Word Detection Mode:**
           - Detects individual signs as you make them
           - Shows real-time confidence scores
           - Perfect for practice and learning
        
        2. **Sentence Formation Mode:**
           - Collects multiple signs into sentences
           - Wait 3 seconds to complete a sentence
           - Automatically processes and displays sentences
        
        #### Using the App
        1. Toggle the camera on
        2. Select your preferred mode
        3. Click Start to begin detection
        4. Make signs clearly in front of the camera
        5. Keep your hands within the camera frame
        
        #### Tips for Better Detection
        - Ensure good lighting
        - Keep a steady position
        - Make clear, deliberate signs
        - Pause briefly between signs
        - Stay within camera view
        """)
    
    # Main detection loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame")
            break
        
        if st.session_state.webcam_on:
            features, annotated_frame = detector.process_frame(frame)
            frame_placeholder.image(annotated_frame, channels="RGB")
            
            if st.session_state.detection_active:
                detector.frame_buffer.append(features)
                
                if len(detector.frame_buffer) > 1:
                    motion = np.mean(np.abs(detector.frame_buffer[-1] - detector.frame_buffer[-2]))
                    
                    if motion > MOTION_THRESHOLD:
                        if not detector.gesture_in_progress:
                            detector.gesture_in_progress = True
                            detector.gesture_frames = []
                            detector.static_frame_counter = 0
                            detector.last_prediction_time = time.time()
                    
                    if detector.gesture_in_progress:
                        detector.gesture_frames.append(features)
                        
                        if motion < STATIC_THRESHOLD:
                            detector.static_frame_counter += 1
                        else:
                            detector.static_frame_counter = 0
                        
                        if (detector.static_frame_counter >= STATIC_FRAME_COUNT or 
                            len(detector.gesture_frames) >= MAX_GESTURE_FRAMES):
                            
                            if len(detector.gesture_frames) >= MIN_GESTURE_FRAMES:
                                prediction, confidence = detector.process_gesture(features)
                                
                                if st.session_state.mode == "Word Detection":
                                    prediction_text.markdown(f"**Detected Sign:** {detector.last_prediction}")
                                    confidence_text.markdown(f"**Confidence:** {detector.last_confidence:.2%}")
                            
                            detector.gesture_in_progress = False
                            detector.gesture_frames = []
                            detector.static_frame_counter = 0
                
                # Handle sentence mode
                if st.session_state.mode == "Sentence Formation":
                    current_time = time.time()
                    if detector.sentence_in_progress:
                        time_since_last = current_time - detector.last_prediction_time
                        if time_since_last >= SENTENCE_END_TIMEOUT and detector.current_sentence:
                            refined_sentence = detector.process_sentence()
                            if refined_sentence:
                                st.session_state.processed_sentences.append({
                                    'refined': refined_sentence
                                })
                            detector.sentence_in_progress = False
                            detector.current_sentence = []
                    
                    # Update sentence displays
                    if detector.sentence_in_progress and detector.current_sentence:
                        words_text.markdown(f"**Words:** {' '.join(detector.current_sentence)}")
                        time_since_last = current_time - detector.last_prediction_time
                        time_left = max(0, SENTENCE_END_TIMEOUT - time_since_last)
                        sentence_status.markdown(f"⏱️ *Collecting words... ({time_left:.1f}s until sentence end)*")
                    else:
                        words_text.markdown("*Waiting for signs...*")
                        sentence_status.markdown("")
                    
                    # Display processed sentences
                    sentences_text = ""
                    if st.session_state.processed_sentences:
                        for sentence in reversed(st.session_state.processed_sentences[-5:]):
                            sentences_text += f"• {sentence['refined']}\n\n"
                    else:
                        sentences_text = "*No sentences yet...*"
                    sentences_container.markdown(sentences_text)
            
            if not st.session_state.detection_active:
                prediction_text.markdown("*Detection paused*")
                confidence_text.empty()
                frame_count_text.empty()
                if st.session_state.mode == "Sentence Formation":
                    words_text.markdown("*Detection paused*")
                    sentence_status.empty()
        
        time.sleep(0.01)
    
    cap.release()

if __name__ == "__main__":
    main()
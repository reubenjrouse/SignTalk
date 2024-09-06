from flask import Flask, request, jsonify,  render_template, redirect, url_for
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
import os
import time
# import tensorflow as tf
from collections import deque
import base64
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121,44,350), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))


def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

@app.route('/')
@app.route('/index.html')
def index():
    room_id = request.args.get('room')  # Get the 'room' query parameter
    if room_id:
        # Handle the case where a room ID is present
        return render_template('index.html', room_id=room_id)
    else:
        # Redirect to lobby.html if no room ID is present
        return redirect(url_for('lobby'))

@app.route('/lobby.html')
def lobby():
    return render_template('lobby.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    # Get the frame data from the request
    frame_data = request.json['frame']
    
    # Decode the base64 image
    frame_bytes = base64.b64decode(frame_data.split(',')[1])
    frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(frame_array, flags=cv2.IMREAD_COLOR)
    
    # Process the frame with MediaPipe
    image, results = mediapipe_detection(frame, holistic)
    draw_landmarks(image, results)
    keypoints = extract_keypoints(results)
    
    # Encode the processed image back to base64
    _, buffer = cv2.imencode('.jpg', image)
    processed_frame = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        'processed_frame': f'data:image/jpeg;base64,{processed_frame}',
        'keypoints': keypoints.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
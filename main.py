from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import base64
import numpy as np
import mediapipe as mp

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# Initialize the pose estimation model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load the overlay image
overlay = cv2.imread("shirt.png", cv2.IMREAD_UNCHANGED)

def process_frame(frame):
    # Convert the frame to RGB format
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with the pose estimation model
    results = pose.process(image)

    # Check if the pose was detected
    if results.pose_landmarks:
        # Get the keypoints for the shoulders and the torso
        shoulder_points = [
            (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y),
            (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)
        ]
        torso_points = [
            (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y),
            (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y)
        ]

        # Calculate the width and height of the overlay based on the keypoints
        width = int(np.linalg.norm(np.array(shoulder_points[0]) - np.array(shoulder_points[1])) * 2)
        height = int(np.linalg.norm(np.array(torso_points[0]) - np.array(torso_points[1])) * 2)

        # Calculate the transformation matrix to warp the overlay
        src_points = np.array(shoulder_points + torso_points, dtype=np.float32).reshape((-1, 1, 2))
        dst_points = np.array([[0, 0], [width, 0], [0, height], [width, height]], dtype=np.float32).reshape((-1, 1, 2))
        M = cv2.getPerspectiveTransform(src_points, dst_points)

        # Warp the overlay to fit the person's body shape
        overlay_warped = cv2.warpPerspective(overlay, M, (width, height))

        # Blend the overlay with the original frame
        alpha = 0.5
        mask = overlay_warped[:, :, 3] / 255.0
        for c in range(0, 3):
            frame[:, :, c] = (1 - mask) * frame[:, :, c] + mask * overlay_warped[:, :, c]

    return frame

def capture_frame():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    processed_frame = process_frame(frame)
    _, buffer = cv2.imencode('.jpg', processed_frame)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    return frame_base64

@socketio.on('connect')
def handle_connect():
    emit('connected', {'data': 'Connected to the server'})

@socketio.on('capture_frame')
def handle_capture_frame():
    frame_base64 = capture_frame()
    if frame_base64:
        emit('frame', {'frame': frame_base64})
    else:
        emit('frame', {'frame': None})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=80)

import cv2
import numpy as np
import mediapipe as mp
import base64
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app)

# Initialize the pose estimation model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load the overlay image
overlay = cv2.imread("shirt.png", cv2.IMREAD_UNCHANGED)

# Define the keypoints for the shoulders and the torso
shoulder_keypoints = [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER]
torso_keypoints = [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP]

@socketio.on('frame')
def handle_frame(data):
    # Decode the base64 image
    frame = base64.b64decode(data)
    frame = np.frombuffer(frame, dtype=np.uint8)
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

    # Convert the frame to RGB format
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with the pose estimation model
    results = pose.process(image)

    if results.pose_landmarks:
        # Get the keypoints for the shoulders and the torso
        shoulder_points = [(results.pose_landmarks.landmark[k].x, results.pose_landmarks.landmark[k].y) for k in shoulder_keypoints]
        torso_points = [(results.pose_landmarks.landmark[k].x, results.pose_landmarks.landmark[k].y) for k in torso_keypoints]

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
        for c in range(3):
            frame[:, :, c] = frame[:, :, c] * (1 - mask) + overlay_warped[:, :, c] * mask

    # Encode the frame back to base64
    _, buffer = cv2.imencode('.jpg', frame)
    frame = base64.b64encode(buffer).decode('utf-8')

    emit('response_frame', {'frame': frame})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=80)

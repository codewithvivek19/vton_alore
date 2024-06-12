import cv2
import numpy as np
import mediapipe as mp

# Initialize the pose estimation model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load the overlay image
overlay = cv2.imread("shirt.png", cv2.IMREAD_UNCHANGED)

# Define the keypoints for the shoulders and the torso
shoulder_keypoints = [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER]
torso_keypoints = [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP]

# Define the width and height of the overlay based on the keypoints
width = 0
height = 0

# Open the camera
cap = cv2.VideoCapture(0)

# Process each frame of the video
while True:
    # Read the frame from the camera
    ret, frame = cap.read()

    # Convert the frame to RGB format
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with the pose estimation model
    results = pose.process(image)

    # Check if the pose was detected
    if results.pose_landmarks:
        # Get the keypoints for the shoulders and the torso
        shoulder_points = []
        torso_points = []
        for keypoint in shoulder_keypoints:
            shoulder_points.append((results.pose_landmarks.landmark[keypoint].x, results.pose_landmarks.landmark[keypoint].y))
        for keypoint in torso_keypoints:
            torso_points.append((results.pose_landmarks.landmark[keypoint].x, results.pose_landmarks.landmark[keypoint].y))

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
        frame = cv2.addWeighted(frame, 1 - alpha, overlay_warped[:, :, :3], alpha, 0)

        # Draw the pose keypoints on the frame
        for landmark in results.pose_landmarks.landmark:
            x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # Display the final frame with the overlay
    cv2.imshow("Frame", frame)

    # Break the loop if the user presses the 'q' key
     if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()

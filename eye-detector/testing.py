import cv2
import pyglet.media
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.FaceDetectionModule import FaceDetector
import csv
from datetime import datetime
import numpy as np
from tensorflow.keras.models import load_model
import os

# Check if model and sound file exist
if not os.path.exists("StayAwake.keras"):
    print("Error: Model file 'StayAwake.keras' not found.")
    exit()

if not os.path.exists("alarm.wav"):
    print("Error: Alarm sound file 'alarm.wav' not found.")
    exit()

# Set up the camera
cap = cv2.VideoCapture(0)
ws, hs = 1280, 720
cap.set(3, ws)
cap.set(4, hs)

if not cap.isOpened():
    print("Camera couldn't access!")
    exit()

# Initialize face detector and face mesh detector
face_detector = FaceDetector()
face_mesh_detector = FaceMeshDetector(maxFaces=1)

# Load the eye detection model
best_model = load_model('StayAwake.keras')

# Initialize variables
eye_closed_frames_left = 0
eye_closed_frames_right = 0
threshold_closed_frames = 60  # Adjust based on desired sensitivity

breakcount_y = 0
counter_y = 0
state_y = False

breakcount_c = 0
counter_c = 0
state_c = False

sound = pyglet.media.load("alarm.wav", streaming=False)

def alert():
    sound.play()
    cv2.rectangle(img, (700, 20), (1250, 80), (0, 0, 255), cv2.FILLED)
    cv2.putText(img, "DROWSINESS ALERT!!!", (710, 60), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)

def recordData(condition):
    with open("database.csv", "a", newline="") as file:
        now = datetime.now()
        dtString = now.strftime("%d-%m-%Y %H:%M:%S")
        writer = csv.writer(file)
        writer.writerow((dtString, condition))

def process_eye(face, eye_indices):
    eye_points = [face[i] for i in eye_indices]
    x_coords = [p[0] for p in eye_points]
    y_coords = [p[1] for p in eye_points]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    padding = int(0.1 * (x_max - x_min))
    x_min, x_max = max(x_min - padding, 0), min(x_max + padding, img.shape[1])
    y_min, y_max = max(y_min - padding, 0), min(y_max + padding, img.shape[0])
    eye_crop = img[y_min:y_max, x_min:x_max]
    gray_eye = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
    resized_eye = cv2.resize(gray_eye, (64, 64))
    normalized_eye = resized_eye / 255.0
    model_input = np.expand_dims(normalized_eye, axis=(0, -1))
    result = best_model.predict(model_input)
    return result, (x_min, y_min, x_max, y_max)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # Face tracking/head tracking
    img, bboxs = face_detector.findFaces(img, draw=False)
    if bboxs:
        fx, fy = bboxs[0]["center"][0], bboxs[0]["center"][1]
        pos = [fx, fy]

        # Draw target indicators
        cv2.putText(img, f'Position (x,y): {pos}', (50, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        cv2.putText(img, "TARGET LOCKED", (850, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    else:
        cv2.putText(img, "NO TARGET", (880, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
        cv2.circle(img, (640, 360), 80, (0, 0, 255), 2)
        cv2.circle(img, (640, 360), 15, (0, 0, 255), cv2.FILLED)
        cv2.line(img, (0, 360), (ws, 360), (0, 0, 0), 2)  # x line
        cv2.line(img, (640, hs), (640, 0), (0, 0, 0), 2)  # y line

    img, faces = face_mesh_detector.findFaceMesh(img, draw=False)
    if faces:
        face = faces[0]
        faceId = [11, 16, 57, 287]

        # Mouth detection
        mouth_indices = [11, 16, 57, 287]  # up, down, left, right
        mouth_ver, _ = face_mesh_detector.findDistance(face[mouth_indices[0]], face[mouth_indices[1]])
        mouth_hor, _ = face_mesh_detector.findDistance(face[mouth_indices[2]], face[mouth_indices[3]])
        mouth_ratio = int((mouth_ver / mouth_hor) * 100)

        cv2.putText(img, f'Mouth Ratio: {mouth_ratio}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        cv2.putText(img, f'Yawn Count: {counter_y}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        if mouth_ratio > 40:
            breakcount_y += 1
            if breakcount_y >= 30:
                alert() 
                if not state_y:
                    counter_y += 1
                    sound.play()
                    recordData("Yawn")
                    state_y = not state_y
        else:
            breakcount_y = 0
            if state_y:
                state_y = not state_y
        
        for id in faceId:
            cv2.circle(img, face[id], 5, (0, 0, 255), cv2.FILLED)

        # Left eye detection
        left_eye_indices = [145, 159, 153, 154, 155, 133, 144, 163, 7, 33]
        left_result, left_coords = process_eye(face, left_eye_indices)
        left_label = "Open" if left_result > 0.5 else "Closed"
        if left_label == "Closed":
            eye_closed_frames_left += 1
        else:
            eye_closed_frames_left = 0

        cv2.rectangle(img, (left_coords[0], left_coords[1]), (left_coords[2], left_coords[3]), (255, 0, 0), 2)
        cv2.putText(img, left_label, (left_coords[0], left_coords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Right eye detection
        right_eye_indices = [374, 386, 373, 380, 381, 362, 385, 387, 263, 249]
        right_result, right_coords = process_eye(face, right_eye_indices)
        right_label = "Open" if right_result > 0.5 else "Closed"
        if right_label == "Closed":
            eye_closed_frames_right += 1
        else:
            eye_closed_frames_right = 0

        cv2.rectangle(img, (right_coords[0], right_coords[1]), (right_coords[2], right_coords[3]), (255, 0, 0), 2)
        cv2.putText(img, right_label, (right_coords[0], right_coords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        print(left_result,right_result)

        # Alert if both eyes are closed
        if eye_closed_frames_left >= threshold_closed_frames and eye_closed_frames_right >= threshold_closed_frames:
            alert()
            recordData("Drowsiness detected")
            eye_closed_frames_left = 0
            eye_closed_frames_right = 0
        
        if right_label == "Closed" and left_label == "Closed":
            breakcount_c += 1
            if breakcount_c >= 30:
                alert()
                if not state_c:
                    counter_c += 1
                    sound.play()
                    recordData("Yawn")
                    state_c = not state_c
        else:
            breakcount_c = 0
            if state_c:
                state_c = not state_c

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

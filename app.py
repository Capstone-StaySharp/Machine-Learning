from flask import Flask, request, jsonify
import os
import cv2
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

# import pyglet.media
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.FaceDetectionModule import FaceDetector
import csv
from datetime import datetime
import numpy as np
import keras

print(keras.__version__)

# Initialize Flask app
app = Flask(__name__)

face_detector = FaceDetector()
face_mesh_detector = FaceMeshDetector(maxFaces=1)

# Configure upload folder
UPLOAD_FOLDER = "uploaded_images"
MODEL_PATH = "StayAwake.keras"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    raise FileNotFoundError(f"Model file {MODEL_PATH} not found.")

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Hello, World!"}), 200

@app.route("/upload", methods=["POST"])
def upload_folder():
    """Upload a folder containing images."""
    if "folder" not in request.files:
        return jsonify({"error": "No folder uploaded"}), 400

    folder = request.files.getlist("folder")
    for file in folder:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

    return jsonify({"message": "Folder uploaded successfully"}), 200


# @app.route('/process', methods=['GET'])
def process_eye(face, eye_indices, img):
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
    result = model.predict(model_input)
    return result, (x_min, y_min, x_max, y_max)


@app.route("/proces", methods=["GET"])
def process_images():
    # Initialize variables
    eye_closed_frames_left = 0
    eye_closed_frames_right = 0
    threshold_closed_frames = 60  # Adjust based on desired sensitivity

    breakcount_y = 0
    counter_y = 0
    # state_y = False

    breakcount_c = 0
    counter_c = 0
    # state_c = False
    """Process uploaded images for drowsiness detection."""
    yawn_detected = False
    both_eyes_closed_detected = False

    for filename in os.listdir(UPLOAD_FOLDER):
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        img = cv2.imread(filepath)

        if img is None:
            print(f"Error: Could not read {filename}")
            continue

        img = cv2.flip(img, 1)  # If you still want to flip the image

        # Face tracking/head tracking
        img, bboxs = face_detector.findFaces(img, draw=False)
        if bboxs:
            fx, fy = bboxs[0]["center"][0], bboxs[0]["center"][1]
            pos = [fx, fy]

            # Draw target indicators
            cv2.putText(
                img,
                f"Position (x,y): {pos}",
                (50, 150),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                img,
                "TARGET LOCKED",
                (850, 50),
                cv2.FONT_HERSHEY_PLAIN,
                3,
                (255, 0, 255),
                3,
            )
        else:
            cv2.putText(
                img, "NO TARGET", (880, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3
            )
            cv2.circle(img, (640, 360), 80, (0, 0, 255), 2)
            cv2.circle(img, (640, 360), 15, (0, 0, 255), cv2.FILLED)
            cv2.line(img, (0, 360), (img.shape[1], 360), (0, 0, 0), 2)  # x line
            cv2.line(img, (640, 0), (640, img.shape[0]), (0, 0, 0), 2)  # y line

        img, faces = face_mesh_detector.findFaceMesh(img, draw=False)
        if faces:
            face = faces[0]
            faceId = [11, 16, 57, 287]

            # Mouth detection
            mouth_indices = [11, 16, 57, 287]  # up, down, left, right
            mouth_ver, _ = face_mesh_detector.findDistance(
                face[mouth_indices[0]], face[mouth_indices[1]]
            )
            mouth_hor, _ = face_mesh_detector.findDistance(
                face[mouth_indices[2]], face[mouth_indices[3]]
            )
            mouth_ratio = int((mouth_ver / mouth_hor) * 100)

            cv2.putText(
                img,
                f"Mouth Ratio: {mouth_ratio}",
                (50, 50),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                img,
                f"Yawn Count: {counter_y}",
                (50, 100),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 0, 255),
                2,
            )

            if mouth_ratio > 40:
                yawn_detected = True
                # breakcount_y += 1
                # if breakcount_y >= 24:
                #     yawn_detected = True
                    # alert()
                    # if not state_y:
                    #     counter_y += 1
                    #     sound.play()
                    #     recordData("Yawn")
                    #     state_y = not state_y
            else:
                breakcount_y = 0
                # if state_y:
                #     state_y = not state_y

            for id in faceId:
                cv2.circle(img, face[id], 5, (0, 0, 255), cv2.FILLED)

            # Left eye detection
            left_eye_indices = [145, 159, 153, 154, 155, 133, 144, 163, 7, 33]
            left_result, left_coords = process_eye(face, left_eye_indices, img)
            left_label = "Open" if left_result > 0.5 else "Closed"
            if left_label == "Closed":
                eye_closed_frames_left += 1
            else:
                eye_closed_frames_left = 0

            cv2.rectangle(
                img,
                (left_coords[0], left_coords[1]),
                (left_coords[2], left_coords[3]),
                (255, 0, 0),
                2,
            )
            cv2.putText(
                img,
                left_label,
                (left_coords[0], left_coords[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # Right eye detection
            right_eye_indices = [374, 386, 373, 380, 381, 362, 385, 387, 263, 249]
            right_result, right_coords = process_eye(face, right_eye_indices, img)
            right_label = "Open" if right_result > 0.5 else "Closed"
            if right_label == "Closed":
                eye_closed_frames_right += 1
            else:
                eye_closed_frames_right = 0

            cv2.rectangle(
                img,
                (right_coords[0], right_coords[1]),
                (right_coords[2], right_coords[3]),
                (255, 0, 0),
                2,
            )
            cv2.putText(
                img,
                right_label,
                (right_coords[0], right_coords[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # print(f"{img_path} result : {left_label},{right_label}")

            # Alert if both eyes are closed
            if (
                eye_closed_frames_left >= threshold_closed_frames
                and eye_closed_frames_right >= threshold_closed_frames
            ):
                # alert()
                # recordData("Drowsiness detected")
                eye_closed_frames_left = 0
                eye_closed_frames_right = 0

            if right_label == "Closed" or left_label == "Closed":
                both_eyes_closed_detected = True
                # breakcount_c += 1
                # if breakcount_c >= 24:
                #     both_eyes_closed_detected = True
                    # alert()
                    # if not state_c:
                    #     counter_c += 1
                    #     sound.play()
                    #     # recordData("Yawn")
                    #     state_c = not state_c
            else:
                breakcount_c = 0
                # if state_c:
                #     state_c = not state_c
    
    for filename in os.listdir(UPLOAD_FOLDER):
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        os.remove(filepath)
    
    # Return the overall results
    return (
        jsonify({"yawn": yawn_detected, "both_eye_closed": both_eyes_closed_detected}),
        200,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

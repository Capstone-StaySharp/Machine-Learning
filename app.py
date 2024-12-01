import base64
import numpy as np
from io import BytesIO
from flask import Flask, request, jsonify
import cv2
from tensorflow.keras.models import load_model
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.FaceDetectionModule import FaceDetector
import json

# Initialize Flask app
app = Flask(__name__)

# Load the models and face detectors
face_detector = FaceDetector()
face_mesh_detector = FaceMeshDetector(maxFaces=1)

# Load the eye detection model
print("Loading eye detection model...")
best_model = load_model('StayAwake.keras')

# Initialize variables
threshold_closed_frames = 60  # Adjust based on desired sensitivity
threshold_yawn_frames = 30
minimum_images = 30

# Process single eye for closed/open status
def process_eye(img, face, eye_indices):
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

# Endpoint to process image batch
@app.route('/predict', methods=['POST'])
def process_images():
    data = request.get_json()
    if not data or 'images' not in data:
        return jsonify({"error": "No images provided"}), 400
    
    images = data['images']

    if len(images) < minimum_images:
        return jsonify({"error": "At least 30 images must be provided"}), 400

    left_eye_closed_count_in_frame = 0
    right_eye_closed_count_in_frame = 0
    yawn_count_in_frame = 0
    
    for img_data in images:
        # Decode the base64 image data
        img_bytes = base64.b64decode(img_data)
        img_array = np.array(bytearray(img_bytes), dtype=np.uint8)
        img = cv2.imdecode(img_array, -1)

        # Flip image to simulate the original code behavior
        img = cv2.flip(img, 1)

        # Face tracking
        img, bboxs = face_detector.findFaces(img, draw=False)
        is_face_detected = bool(bboxs)
        
        if is_face_detected:
            face = face_mesh_detector.findFaceMesh(img, draw=False)[0]
            
            # Mouth detection for yawning
            mouth_indices = [11, 16, 57, 287]  # up, down, left, right
            mouth_ver, _ = face_mesh_detector.findDistance(face[mouth_indices[0]], face[mouth_indices[1]])
            mouth_hor, _ = face_mesh_detector.findDistance(face[mouth_indices[2]], face[mouth_indices[3]])
            mouth_ratio = int((mouth_ver / mouth_hor) * 100)

            # Yawning detection
            is_yawning = mouth_ratio > 60  # Define threshold as needed

            # Left eye detection
            left_eye_indices = [145, 159, 153, 154, 155, 133, 144, 163, 7, 33]
            left_result, _ = process_eye(img, face, left_eye_indices)
            is_left_eye_closed = left_result <= 0.5

            # Right eye detection
            right_eye_indices = [374, 386, 373, 380, 381, 362, 385, 387, 263, 249]
            right_result, _ = process_eye(img, face, right_eye_indices)
            is_right_eye_closed = right_result <= 0.5

            # Alert if both eyes are closed
            if is_left_eye_closed:
                left_eye_closed_count_in_frame += 1

            if is_right_eye_closed:
                right_eye_closed_count_in_frame += 1

            if is_yawning:
                yawn_count_in_frame += 1

    return jsonify({
        'is_yawning': yawn_count_in_frame >= threshold_yawn_frames,
        'is_left_eye_closed': left_eye_closed_count_in_frame >= threshold_closed_frames,
        'is_right_eye_closed': right_eye_closed_count_in_frame >= threshold_closed_frames,
    })

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=5000)

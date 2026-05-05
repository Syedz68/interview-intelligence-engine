import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

# Download model once if not present
MODEL_PATH = "face_landmarker.task"
if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        MODEL_PATH
    )

# Initialize once
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=2
)
face_landmarker = vision.FaceLandmarker.create_from_options(options)


def extract_face_landmarks(frames):
    processed_frames = []

    for frame_data in frames:
        frame_path = frame_data["frame_path"]
        timestamp = frame_data["timestamp"]

        image = mp.Image.create_from_file(frame_path)
        results = face_landmarker.detect(image)

        frame_result = {
            "timestamp": timestamp,
            "face_detected": False,
            "face_count": len(results.face_landmarks),
            "landmarks": []
        }

        if results.face_landmarks:
            frame_result["face_detected"] = True

            # Pick largest face (POC assumption)
            face_landmarks = results.face_landmarks[0]

            landmark_points = [
                {"x": lm.x, "y": lm.y, "z": lm.z}
                for lm in face_landmarks
            ]

            frame_result["landmarks"] = landmark_points

        processed_frames.append(frame_result)

    return processed_frames
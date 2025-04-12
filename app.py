from flask import Flask, render_template, request, jsonify, Response
import os
import cv2
import threading
import base64
import logging
import numpy as np
import time
import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['REALTIME_EMOTIONS_FOLDER'] = 'realtime_emotions'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs(app.config['REALTIME_EMOTIONS_FOLDER'], exist_ok=True)

logging.basicConfig(level=logging.INFO)

detected_data = {"frames": {}, "emotions": {}, "processing": False, "progress": 0}
realtime_emotions = []

# Load the hybrid model
try:
    hybrid_model = load_model('hybrid_fer_model.h5')  # Replace with your actual model file
except Exception as e:
    print(f"Error loading model: {e}")
    hybrid_model = None

# Emotion labels (match your training data)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def preprocess_image(image):
    """
    Preprocesses the input image for the FER model.

    Args:
        image: A PIL Image object or a NumPy array.

    Returns:
        A preprocessed NumPy array.
    """
    if not isinstance(image, np.ndarray):
        image = image.resize((224, 224))  # Resize if it's a PIL Image
        image = np.array(image)
    image = image.astype('float32') / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


def process_video(video_path):
    try:
        logging.info(f"Processing video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        detected_data["processing"] = True
        detected_data["frames"].clear()
        detected_data["emotions"].clear()
        detected_data["progress"] = 0

        highest_prob_frames = {}
        highest_probs = {}
        processed_frames = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = RetinaFace.detect_faces(frame_rgb)

                for face_id in faces:
                    face = faces[face_id]['facial_area']
                    x1, y1, x2, y2 = face
                    face_crop = frame_rgb[y1:y2, x1:x2]
                    face_crop_pil = Image.fromarray(face_crop)  # Convert to PIL Image

                    try:
                        # Preprocess the face crop for the hybrid model
                        processed_face = preprocess_image(face_crop_pil)

                        # Make prediction using the hybrid model
                        predictions = hybrid_model.predict([processed_face, processed_face])

                        # Process the predictions
                        emotion_index = np.argmax(predictions)
                        emotion = emotion_labels[emotion_index]
                        probability = float(predictions[0][emotion_index])

                        if probability > 0.75:  # Adjust threshold as needed
                            if emotion not in highest_probs or probability > highest_probs[emotion]:
                                highest_probs[emotion] = probability
                                _, buffer = cv2.imencode('.jpg', frame)
                                highest_prob_frames[emotion] = base64.b64encode(buffer).decode('utf-8')
                                detected_data["frames"] = highest_prob_frames.copy()
                                detected_data["emotions"] = {
                                    e: {"emotion": e, "probability": float(p)} for e, p in
                                    highest_probs.items()}.copy()

                    except Exception as e:
                        logging.error(f"Emotion analysis error: {e}")

            except Exception as e:
                logging.error(f"Frame processing error: {e}")

            processed_frames += 1
            if total_frames > 0:
                detected_data["progress"] = int((processed_frames / total_frames) * 100)
            else:
                detected_data["progress"] = 0

        detected_data["frames"] = highest_prob_frames
        detected_data["emotions"] = {e: {"emotion": e, "probability": float(p)} for e, p in highest_probs.items()}

    except Exception as e:
        logging.error(f"Video processing failed: {e}")
    finally:
        detected_data["processing"] = False
        detected_data["progress"] = 100
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        logging.info(f"Video processing complete.")


def generate_frames():
    cap = cv2.VideoCapture(0)
    global realtime_emotions
    realtime_emotions = []
    while True:
        try:
            success, frame = cap.read()
            if not success:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                faces = RetinaFace.detect_faces(frame_rgb)
            except Exception as e:
                logging.error(f"Error detecting faces in real-time: {e}")
                faces = {}

            emotions = []

            for key in faces:
                face = faces[key]['facial_area']
                x1, y1, x2, y2 = face
                face_crop = frame_rgb[y1:y2, x1:x2]
                face_crop_pil = Image.fromarray(face_crop)  # Convert to PIL Image

                try:
                    # Preprocess the face crop for the hybrid model
                    processed_face = preprocess_image(face_crop_pil)

                    # Make prediction using the hybrid model
                    predictions = hybrid_model.predict([processed_face, processed_face])

                    # Process the predictions
                    emotion_index = np.argmax(predictions)
                    emotion = emotion_labels[emotion_index]
                    probability = float(predictions[0][emotion_index])

                    emotions.append({"emotion": emotion, "probability": probability})

                except Exception as e:
                    logging.error(f"Hybrid model error in real-time detection: {e}")
            realtime_emotions = emotions
            save_realtime_emotions(emotions)  # save emotions
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as e:
            logging.error(f"Error generating frames: {e}")
    if 'cap' in locals() and cap.isOpened():
        cap.release()


def save_realtime_emotions(emotions):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"realtime_emotions_{timestamp}.txt"
    filepath = os.path.join(app.config['REALTIME_EMOTIONS_FOLDER'], filename)

    with open(filepath, "w") as f:
        for emotion in emotions:
            f.write(f"{emotion['emotion']}: {emotion['probability']:.2f}%\n")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No file uploaded"})
        file = request.files['video']
        if file.filename == '':
            return jsonify({"error": "No selected file"})

        allowed_extensions = ['mp4', 'avi', 'mov', 'mkv']
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension not in allowed_extensions:
            return jsonify(
                {"error": f"Unsupported file format! Please upload a video file ({', '.join(allowed_extensions)})."})

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        threading.Thread(target=process_video, args=(file_path,)).start()
        return render_template('results.html')
    except Exception as e:
        logging.error(f"Error handling upload: {e}")
        return jsonify({"error": "Failed to upload file."})


@app.route('/get_detected_data')
def get_detected_data():
    try:
        return jsonify(detected_data)
    except Exception as e:
        logging.error(f"Error fetching detected data: {e}")
        return jsonify({"error": "Failed to fetch data."})


@app.route('/realtime')
def realtime():
    return render_template('realtime.html')


@app.route('/clear', methods=['POST'])
def clear():
    try:
        detected_data["frames"].clear()
        detected_data["emotions"].clear()
        detected_data["processing"] = False
        detected_data["progress"] = 0
        return render_template('index.html')
    except Exception as e:
        logging.error(f"Error clearing data: {e}")
        return jsonify({"error": "Failed to clear data."})


@app.route('/get_realtime_emotions')
def get_realtime_emotions():
    global realtime_emotions
    return jsonify(realtime_emotions)


if __name__ == '__main__':
    app.run(debug=True)
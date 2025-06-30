from flask import Flask, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Load model and cascade once
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/emotion_model.h5')
model = load_model(MODEL_PATH)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return jsonify({'error': 'No face detected'}), 200
    x, y, w, h = faces[0]
    roi_gray = gray[y:y+h, x:x+w]
    roi = cv2.resize(roi_gray, (48, 48))
    roi = roi.astype('float32') / 255.0
    roi = np.reshape(roi, (1, 48, 48, 1))
    pred = model.predict(roi)
    label = emotion_labels[int(np.argmax(pred))]
    confidence = float(np.max(pred))
    return jsonify({'emotion': label, 'confidence': confidence})

@app.route('/detect-video', methods=['POST'])
def detect_video():
    # Accepts a video file, returns a list of detected emotions per frame with faces
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400
    file = request.files['video']
    temp_path = 'temp_video.mp4'
    file.save(temp_path)
    cap = cv2.VideoCapture(temp_path)
    results = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        frame_result = []
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi_gray, (48, 48))
            roi = roi.astype('float32') / 255.0
            roi = np.reshape(roi, (1, 48, 48, 1))
            pred = model.predict(roi)
            label = emotion_labels[int(np.argmax(pred))]
            confidence = float(np.max(pred))
            frame_result.append({'emotion': label, 'confidence': confidence, 'box': [int(x), int(y), int(w), int(h)]})
        results.append({'frame': frame_count, 'faces': frame_result})
        frame_count += 1
    cap.release()
    os.remove(temp_path)
    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

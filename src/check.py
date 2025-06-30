# src/check.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model('models/emotion_model.h5')  # Or .keras if you saved that way

# Emotion categories (same order as training)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load and preprocess image
img = cv2.imread('sample.jpg')  # Change to your image path
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face = cv2.resize(gray, (48, 48))
face = face.reshape(1, 48, 48, 1).astype('float32') / 255.0

# Predict
pred = model.predict(face)
emotion = emotion_labels[np.argmax(pred)]

# Output result
print(f"Detected Emotion: {emotion}")
cv2.putText(img, emotion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
from keras.models import model_from_json
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Load Model
with open("emotiondetector.json", "r") as json_file:  # Update this line
    model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")  # Update this line


# Load Haar Cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to preprocess image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)  # Reshape for model input
    return feature / 255.0  # Normalize pixel values

# Open Webcam
webcam = cv2.VideoCapture(0)

# Emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    ret, im = webcam.read()
    if not ret:
        print("Error: Could not capture frame.")
        break

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (p, q, r, s) in faces:
        image = gray[q:q + s, p:p + r]
        cv2.rectangle(im, (p, q), (p + r, q + s), (255, 0, 0), 2)

        try:
            image = cv2.resize(image, (48, 48))  # Resize to model input size
            img = extract_features(image)
            pred = model.predict(img, verbose=0)  # Suppress unnecessary output
            prediction_label = labels[pred.argmax()]

            # Ensure text position is within frame bounds
            text_x = max(10, p)  # Prevent going out of bounds
            text_y = max(10, q - 10)
            
            cv2.putText(im, prediction_label, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        except Exception as e:
            print(f"Error processing face: {e}")

    cv2.imshow("Output", im)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()

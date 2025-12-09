import cv2
import numpy as np
from model_architecture import create_emotion_model

# --- 1. Load the Brain ---
# We load the architecture
model = create_emotion_model()
# We load the trained weights (Make sure this matches the filename in train_model.py)
# If your training is still running, this line will fail until the file is created.
try:
    model.load_weights('emotion_model.weights.h5')
    print("Loaded model weights successfully.")
except:
    print("Error: Could not load weights. Make sure train_model.py finished successfully.")
    exit()

# Dictionary to convert numbers (0-6) to words
emotion_dict = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised"
}

# --- 2. Setup Camera & Face Detection ---
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for the model
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw the box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # --- 3. The Critical Step: Preprocessing ---
        # The model expects a 48x48 pixel image. We must crop and resize the face to match.
        
        # Crop the face out of the gray image
        roi_gray = gray[y:y+h, x:x+w]
        
        # Resize to 48x48
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        
        # Normalize (divide by 255, just like in training)
        prediction_input = cropped_img / 255.0

        # --- 4. Prediction ---
        # Ask the model to predict
        prediction = model.predict(prediction_input)
        
        # The model returns a list of 7 probabilities. We want the highest one.
        # argmax returns the *index* of the highest number.
        maxindex = int(np.argmax(prediction))
        
        # Get the emotion word
        detected_emotion = emotion_dict[maxindex]

        # Draw the text above the box
        cv2.putText(frame, detected_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2  # Import the OpenCV library

# 1. Load the "Face Detector"
# OpenCV comes with pre-trained XML files that know what a face looks like.
# We are loading the "Frontal Face" detector.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 2. Open the Webcam
# The '0' usually refers to your default laptop webcam.
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Camera is active! Press 'q' to quit.")

while True:
    # 3. Capture frame-by-frame
    # 'ret' is a boolean (True/False) if the frame was read correctly
    # 'frame' is the actual image (array of pixels)
    ret, frame = cap.read()
    
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # 4. Convert to Grayscale
    # Face detection works faster and better in black & white (grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 5. Detect Faces
    # detectMultiScale scans the image to find faces.
    # scaleFactor=1.1: Reduces image size by 10% each pass to find big and small faces.
    # minNeighbors=5: How picky the algorithm is. Higher = fewer errors but might miss faces.
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 6. Draw a box around the face
    # The 'faces' variable is a list of coordinates (x, y, width, height)
    for (x, y, w, h) in faces:
        # Draw a rectangle: (image, start_point, end_point, color_BGR, thickness)
        cv2.rectangle(frame, (x+w, y+h), (x+w+w, y+h+h), (0, 255, 0), 2)

    # 7. Display the resulting frame
    cv2.imshow('Face Detection (Press q to quit)', frame)

    # 8. Quit logic
    # Wait 1ms for a key press. If 'q' is pressed, break the loop.
    if cv2.waitKey(1) == ord('q'):
        break

# 9. Clean up
cap.release()
cv2.destroyAllWindows()
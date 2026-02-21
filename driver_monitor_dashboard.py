import cv2
import dlib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from scipy.spatial import distance as dist
import time
import winsound

# -----------------------------
# Load trained model
# -----------------------------
model = load_model("driver_drowsiness_model.h5")
IMG_SIZE = 64

# -----------------------------
# Dlib face and landmarks for blink + head tilt
# -----------------------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # download from dlib

# Eye aspect ratio function
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# -----------------------------
# Helper for head tilt angle
# -----------------------------
def get_head_tilt(shape):
    left_eye = shape[36:42]
    right_eye = shape[42:48]
    left_center = np.mean(left_eye, axis=0)
    right_center = np.mean(right_eye, axis=0)
    dx = right_center[0] - left_center[0]
    dy = right_center[1] - left_center[1]
    angle = np.degrees(np.arctan2(dy, dx))
    return angle

# -----------------------------
# Real-time monitoring
# -----------------------------
cap = cv2.VideoCapture(0)

drowsy_counter = 0
threshold_frames = 10
blink_counter = 0
total_blinks = 0
ear_threshold = 0.25  # typical threshold for closed eyes

PERCLOS_window = []
window_size = 60  # number of frames to calculate PERCLOS

alert_history = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    # Default values
    ear = 0
    tilt_angle = 0

    for face in faces:
        shape = predictor(gray, face)
        shape = np.array([[p.x, p.y] for p in shape.parts()])

        left_eye = shape[36:42]
        right_eye = shape[42:48]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        tilt_angle = get_head_tilt(shape)

        # Blink detection
        if ear < ear_threshold:
            blink_counter += 1
        else:
            if blink_counter >= 2:  # minimal frames for blink
                total_blinks += 1
            blink_counter = 0

    # PERCLOS calculation
    PERCLOS_window.append(ear < ear_threshold)
    if len(PERCLOS_window) > window_size:
        PERCLOS_window.pop(0)
    perclos = sum(PERCLOS_window) / len(PERCLOS_window) * 100

    # Model prediction (drowsy/alert)
    roi = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    roi = roi.astype("float") / 255.0
    roi = np.expand_dims(roi, axis=0)
    pred = model.predict(roi, verbose=0)
    label = np.argmax(pred)

    # Drowsiness counter for alarm
    if label == 1:
        drowsy_counter += 1
    else:
        drowsy_counter = 0

    if drowsy_counter >= threshold_frames:
        alert_history.append(time.strftime("%H:%M:%S"))
        cv2.putText(frame, "DROWSY ALERT!", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
        winsound.Beep(2500, 500)

    # -----------------------------
    # Draw Dashboard Overlay
    # -----------------------------
    cv2.rectangle(frame, (0,0), (350,200), (50,50,50), -1)  # background
    cv2.putText(frame, f"Blink Count: {total_blinks}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.putText(frame, f"PERCLOS: {perclos:.1f}%", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.putText(frame, f"Head Tilt: {tilt_angle:.1f} deg", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.putText(frame, "Alert History:", (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    for i, t in enumerate(alert_history[-3:]):
        cv2.putText(frame, t, (10,150 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # Display predicted label
    text = "Drowsy" if label == 1 else "Alert"
    color = (0,0,255) if label == 1 else (0,255,0)
    cv2.putText(frame, text, (50,450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow("Driver Monitoring Dashboard", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

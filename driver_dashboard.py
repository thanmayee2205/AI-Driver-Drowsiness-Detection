import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import winsound
from datetime import datetime
import csv
import os

# ----------------------
# Load Drowsiness Model
# ----------------------
model = load_model("driver_drowsiness_model.h5")

# Drowsiness Threshold
DROWSY_THRESHOLD = 0.6  # 60% probability considered drowsy

# Log file setup
LOG_FILE = "drowsiness_log.csv"
if not os.path.isfile(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Drowsy_Probability"])

# Initialize Webcam
cap = cv2.VideoCapture(0)
IMG_SIZE = 64

print("Driver Monitoring System Started...")
print("Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip frame if needed

    # ----------------------
    # Preprocess for Model
    # ----------------------
    face = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = img_to_array(face) / 255.0
    face = np.expand_dims(face, axis=0)

    # Predict drowsiness
    prediction = model.predict(face, verbose=0)[0]
    drowsy_prob = prediction[1]  # Assuming index 1 = drowsy

    # ----------------------
    # Drowsiness Meter
    # ----------------------
    meter_height = 200
    meter_x, meter_y = 50, 50
    meter_filled = int(drowsy_prob * meter_height)

    if drowsy_prob < 0.4:
        color = (0, 255, 0)  # Green
    elif drowsy_prob < 0.7:
        color = (0, 255, 255)  # Yellow
    else:
        color = (0, 0, 255)  # Red

    cv2.rectangle(frame, (meter_x, meter_y), (meter_x + 50, meter_y + meter_height), (50, 50, 50), 2)
    cv2.rectangle(frame, (meter_x, meter_y + meter_height - meter_filled),
                  (meter_x + 50, meter_y + meter_height), color, -1)
    cv2.putText(frame, f"{int(drowsy_prob*100)}%", (meter_x, meter_y + meter_height + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # ----------------------
    # Beep Alarm + Logging
    # ----------------------
    if drowsy_prob >= DROWSY_THRESHOLD:
        winsound.Beep(2000, 400)  # Frequency: 2000Hz, Duration: 400ms

        # Log timestamp and drowsiness probability
        with open(LOG_FILE, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f"{drowsy_prob:.2f}"])

    # Show frame
    cv2.imshow("Driver Monitoring Dashboard", frame)

    key = cv2.waitKey(1)
    if key == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
print("System stopped. Log saved as drowsiness_log.csv")

# AI-Based Driver Drowsiness Detection System

A real-time Driver Drowsiness Detection System built using Deep Learning and Computer Vision.  
This project detects driver fatigue using a Convolutional Neural Network (CNN) and advanced behavioral analysis techniques such as blink detection, PERCLOS, and head tilt estimation.

---

## Project Overview

Driver drowsiness is a major cause of road accidents.  
This system uses a trained CNN model to classify drivers as **Drowsy** or **Alert** using real-time webcam input.

Two monitoring modes are implemented:

1. CNN-Based Probability Dashboard  
2. Behavioral Monitoring Dashboard (Facial Landmark-Based)

---

## Features

### 1. CNN-Based Dashboard (`driver_dashboard.py`)
- Real-time webcam monitoring
- Drowsiness probability prediction
- Color-coded alert system:
  - 🟢 Green → Alert
  - 🟡 Yellow → Mild Drowsiness
  - 🔴 Red → Drowsy
- Sound alarm when threshold exceeded
- CSV logging of drowsy events
- Visual probability percentage display

---

### 2. Behavioral Monitoring Dashboard (`driver_monitor_dashboard.py`)
- Blink detection using Eye Aspect Ratio (EAR)
- PERCLOS (Percentage of Eye Closure) calculation
- Head tilt angle estimation
- Alert history tracking
- Real-time dashboard overlay
- Continuous drowsiness alarm detection

---

## Project Structure

AI-Driver-Drowsiness-Detection/

├── preprocess_dataset.py  
├── train_model.py  
├── driver_dashboard.py  
├── driver_monitor_dashboard.py  
├── requirements.txt  
├── README.md  
└── .gitignore  

---

## Technologies Used

- Python  
- OpenCV  
- TensorFlow / Keras  
- NumPy  
- Scikit-learn  
- Dlib  
- SciPy  

---

## Dataset

The dataset contains two categories:
- Drowsy
- Non-Drowsy

Images are:
- Resized to 64x64
- Normalized
- One-hot encoded before training

Dataset images and trained model files are excluded from the repository due to size constraints.

---

## How to Run the Project

### Step 1 — Preprocess Dataset
python preprocess_dataset.py

### Step 2 — Train Model
python train_model.py

This generates:
driver_drowsiness_model.h5

### Step 3 — Run CNN-Based Dashboard
python driver_dashboard.py

OR

### Step 4 — Run Behavioral Monitoring Dashboard
python driver_monitor_dashboard.py

---

## Model Architecture

- 3 Convolutional Layers  
- MaxPooling Layers  
- Dropout Regularization  
- Dense Fully Connected Layer  
- Softmax Output Layer (2 Classes)  
- Optimizer: Adam  
- Loss Function: Categorical Crossentropy  

---

## Alert Logic

- Alarm triggers when:
  - CNN drowsiness probability exceeds threshold
  - Continuous eye closure detected via EAR
- Alarm stops automatically when driver returns to alert state.

---

## CSV Logging

In CNN Dashboard mode:
- Drowsy events are recorded in:
drowsiness_log.csv
- Each entry contains:
  - Timestamp
  - Drowsiness probability


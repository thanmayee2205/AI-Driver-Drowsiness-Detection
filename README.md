# AI-Based Driver Drowsiness Detection System

A real-time Driver Drowsiness Detection System developed using Deep Learning and Computer Vision techniques.  
The system detects driver fatigue using a Convolutional Neural Network (CNN) combined with behavioral analysis methods such as blink detection, PERCLOS, and head tilt estimation.

This project demonstrates model development, real-time deployment, visualization, alert systems, and logging integration.

---

## Abstract

Driver fatigue is a major contributor to road accidents worldwide.  
This system uses a trained CNN model to classify drivers as **Drowsy** or **Alert** using live webcam input.  

In addition to deep learning-based classification, a second behavioral monitoring mode is implemented using facial landmarks to measure eye closure rate and head orientation.

The project provides both probabilistic classification and physiological fatigue metrics.

---

## System Modes

### 1. CNN-Based Probability Dashboard (`driver_dashboard.py`)

- Real-time webcam monitoring
- Drowsiness probability prediction
- Color-coded alert system:
  - 🟢 Green → Alert
  - 🟡 Yellow → Mild Drowsiness
  - 🔴 Red → Drowsy
- Threshold-based alarm activation
- Automatic alarm stop when alert
- CSV logging of drowsy events
- Live probability percentage visualization

This mode demonstrates model deployment and user-interface integration.

---

### 2. Behavioral Monitoring Dashboard (`driver_monitor_dashboard.py`)

- Blink detection using Eye Aspect Ratio (EAR)
- PERCLOS (Percentage of Eye Closure) calculation
- Head tilt angle estimation
- Consecutive frame-based drowsiness detection
- Alert history tracking
- Real-time dashboard overlay

This mode demonstrates facial landmark analysis and fatigue metric computation.

---

## Key Contributions

- Designed and trained a CNN for binary fatigue classification.
- Implemented real-time inference using OpenCV.
- Developed probability-based visual alert system.
- Integrated threshold-controlled alarm logic.
- Implemented CSV logging for fatigue event tracking.
- Developed blink detection using Eye Aspect Ratio (EAR).
- Calculated PERCLOS for behavioral fatigue estimation.
- Integrated head tilt detection using facial landmarks.

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

The dataset consists of two classes:

- Drowsy
- Non-Drowsy

Preprocessing steps:
- Resize to 64×64 pixels
- Normalize pixel values
- One-hot encode labels
- Train-test split (80:20)

Dataset images and trained model files are excluded from the repository due to size constraints.

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

## Installation

Install required dependencies:

pip install -r requirements.txt

---

## How to Run

Step 1 — Preprocess Dataset  
python preprocess_dataset.py

Step 2 — Train Model  
python train_model.py  

This generates:
driver_drowsiness_model.h5

Step 3 — Run CNN-Based Dashboard  
python driver_dashboard.py

OR

Step 4 — Run Behavioral Monitoring Dashboard  
python driver_monitor_dashboard.py

---

## Alert Logic

Alarm triggers when:

- CNN drowsiness probability exceeds defined threshold  
OR  
- Continuous eye closure is detected over consecutive frames  

Alarm stops automatically when the driver returns to alert state.

---

## CSV Logging

In CNN Dashboard mode:

Drowsy events are recorded in:
drowsiness_log.csv

Each entry contains:
- Timestamp
- Drowsiness probability

This enables fatigue trend analysis and system auditing.



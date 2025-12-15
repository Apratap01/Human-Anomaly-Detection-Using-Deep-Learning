# 🧠 Human Anomaly Detection Using Deep Learning

## 📌 Overview

This project implements a **Human Anomaly Detection system** using **Deep Learning-based video classification**. It analyzes human activities from video data and identifies **abnormal or unusual behaviors** by learning both **spatial** and **temporal patterns** in videos.

The system uses:

* **CNNs** for spatial feature extraction
* **LSTM networks** for temporal modeling
* **Streamlit** for deployment as a web application

---

## 1️⃣ Introduction

Human Anomaly Detection is an important problem in computer vision and video analytics, especially in areas such as **surveillance systems, public safety, healthcare monitoring, and smart environments**. The goal is to automatically detect unusual or abnormal human activities from video streams that deviate from normal behavior patterns.

Manual monitoring of surveillance videos is inefficient, time-consuming, and error-prone. With the rapid growth of video data, **automated intelligent systems** are required to analyze human actions in real time and identify anomalies such as violent actions, unusual movements, or unexpected activities.

In this project, a **deep learning–based video classification approach** is used. The system combines:

* **Convolutional Neural Networks (CNNs)** for spatial feature extraction
* **Long Short-Term Memory (LSTM)** networks for temporal sequence learning

The final system is deployed as an **interactive Streamlit web application** for real-time anomaly detection from uploaded videos.

---

## 2️⃣ Problem Statement

* Surveillance cameras generate huge volumes of video data
* Manual observation cannot scale effectively
* Abnormal activities often have temporal dependencies
* Frame-based methods fail to capture motion patterns

### Requirements:

* Understand human actions over time
* Detect deviations from learned normal patterns
* Provide an easy-to-use interface for real-world usage

---

## 3️⃣ How This System Solves the Problem

This project solves the problem by:

* Extracting spatial features using a pretrained CNN (MobileNetV2 / EfficientNet)
* Modeling temporal behavior using a Bidirectional LSTM
* Learning normal and abnormal patterns from labeled video data
* Classifying unseen videos based on learned behavior
* Detecting anomalies using confidence-based thresholds
* Providing real-time predictions through a web interface

This **spatio-temporal learning** approach captures **what happens** and **how it happens**, which is crucial for anomaly detection.

---

## 4️⃣ Proposed Solution Architecture

### 4.1 Dataset Structure

```
dataset/
│── train/
│   ├── Normal/
│   ├── Running/
│   ├── Fighting/
│   └── ...
│── test/
│   ├── Normal/
│   ├── Running/
│   ├── Fighting/
│   └── ...
```

---

### 4.2 Feature Extraction

* Videos are sampled into a fixed number of frames
* Frames are resized and center-cropped
* A pretrained CNN extracts high-level spatial features
* Features are cached to disk for faster training

---

### 4.3 Temporal Modeling

* Extracted frame features are fed into a **Bidirectional LSTM**
* Temporal dependencies and motion patterns are learned
* Masking handles variable-length videos

---

### 4.4 Classification & Anomaly Detection

* Dense layers process LSTM outputs
* Softmax layer predicts class probabilities
* If max probability < threshold → **Anomaly Detected**

---

### 4.5 Web Application (Streamlit)

* Users upload a video file
* Model runs inference
* Top predictions are displayed
* Anomaly warning is shown if confidence is low

---

## 5️⃣ Technologies Used

| Component            | Technology                   |
| -------------------- | ---------------------------- |
| Programming Language | Python                       |
| Deep Learning        | TensorFlow, Keras            |
| CNN Backbone         | MobileNetV2 / EfficientNetB0 |
| Sequence Modeling    | Bidirectional LSTM           |
| Video Processing     | OpenCV                       |
| Data Handling        | NumPy, Pandas                |
| Evaluation           | Scikit-learn                 |
| Visualization        | Matplotlib                   |
| Deployment           | Streamlit                    |

---

## 6️⃣ Dataset Used

**UCF101 – Action Recognition Dataset**

* 101 human action classes
* Real-world videos
* Widely used benchmark dataset

🔗 **Dataset Link:**
[https://www.crcv.ucf.edu/research/data-sets/ucf101/](https://www.crcv.ucf.edu/research/data-sets/ucf101/)

---

## 7️⃣ Experimental Results

* Model learns temporal human action patterns effectively
* Confusion matrix and classification reports generated
* Good generalization on unseen test videos
* Confidence-based anomaly detection performs reliably

---

## 8️⃣ GitHub Repository

🔗 **Project Repository:**
[https://github.com/Aditya-Partap/Human-Anomaly-Detection-Using-Deep-Learning](https://github.com/Apratap01/Human-Anomaly-Detection-Using-Deep-Learning)

---

## 9️⃣ Steps to Run the Code

### Step 1: Clone the Repository

```bash
git clone https://github.com/Aditya-Partap/Human-Anomaly-Detection-Using-Deep-Learning.git
cd Human-Anomaly-Detection-Using-Deep-Learning
```

---

### Step 2: Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

---

### Step 3: Install Dependencies

```bash
pip install tensorflow opencv-python numpy pandas scikit-learn matplotlib streamlit
```

---

### Step 4: Prepare Dataset

* Download UCF101 dataset
* Extract videos into:

```
dataset/train/
dataset/test/
```

---

### Step 5: Train the Model

* Open notebook:

```
VideoClassification.ipynb
```

* Select virtual environment as kernel
* Restart kernel → Run All cells
* Model weights are saved automatically

---

### Step 6: Run Streamlit App

```bash
streamlit run streamlit_app.py
```

Open in browser:

```
http://localhost:8501
```

---

## 🔚 10️⃣ Conclusion

This project demonstrates an effective deep learning–based approach for **human anomaly detection in videos**. By integrating CNN-based feature extraction with LSTM-based temporal modeling, the system accurately identifies abnormal human activities. The Streamlit interface makes the solution practical and deployable for real-world applications.

---

## 🔮 11️⃣ Future Scope

* Real-time CCTV stream integration
* Unsupervised anomaly detection
* Transformer-based video models
* Edge deployment (Jetson / Mobile devices)
* Multi-camera anomaly correlation

---

### ⭐ If you like this project, don’t forget to give it a star on Git

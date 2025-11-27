# **DeepScreen – AI-Powered Early Autism Detection from Videos**


DeepScreen is an AI-powered video-based system designed to detect early signs of Autism Spectrum Disorder (ASD) using **action recognition**, **pose estimation**, and **spatio-temporal deep learning**.
This project implements and compares multiple architectures—**LSTM, BiLSTM, GRU, CNN-LSTM, and 3D-CNN**—to learn behavioral patterns from children’s movement sequences.


---

## 🚀 Key Features

### Computer Vision (CV)

* **Pose Estimation:** MediaPipe Pose
* **Keypoint Extraction:** 2D skeletal joint detection
* **Motion Feature Processing:** Temporal keypoint sequence analysis
* **Video Preprocessing:** Frame sampling, normalization, resizing

### Deep Learning (DL) & Sequential Modeling

* *Recurrent Neural Networks*
* *Hybrid Architectures and Spatio-Temporal Model*

### Research & Evaluation

* Action recognition evaluation metrics: Accuracy, F1-Score, Precision, Recall
* Comparative model benchmarking

--

# 📁 **Dataset – SSBD2**

This project uses the **Self-Stimulatory Behavior Dataset 2 (SSBD2)**, a real-world video dataset containing children performing behaviors such as:

* Hand-flapping
* Rocking
* Spinning
* Head-banging
* Finger-tapping

Limitations of SSBD2 include only ASD-related actions. To improve generalization, **additional non-ASD videos** were recorded and added, creating two broad categories:

* **Non-ASD:** no_action, neutral behavior
* **ASD:** spinning, head banging, arm flapping

---

# 🛠️ **Preprocessing Pipeline**

The preprocessing framework follows the detailed methodology from the research study :

### **1️⃣ Pose Extraction (MediaPipe Pose)**

For each video:

* 33 skeletal keypoints per frame
* Extracted `(x, y)` coordinates
* Stored as time-series sequences

---

# 🧠 **Model Architectures**

DeepScreen implements a **comparative analysis** of 5 deep learning models, exactly as evaluated in the paper .

---

### **1️⃣ LSTM**

* 128 hidden units
* Learns long-term temporal dependencies
* Accuracy: **85.77%**

---

### **2️⃣ Bi-LSTM (Best Model)**

* Bidirectional learning of past + future context
* 128 + 64 stacked layers
* Accuracy: **95.69%**

---

### **3️⃣ GRU**

* Faster, parameter-efficient RNN variant
* 128 + 64 units
* Accuracy: **90.52%**

---

### **4️⃣ CNN-LSTM (Hybrid)**

* 1D CNN layers extract local temporal cues
* LSTM layers process extended sequences
* Accuracy: **92.67%**

---

### **5️⃣ 3D-CNN**

* 3D convolutions model spatio-temporal features jointly
* 4 convolutional blocks
* Accuracy: **86.20%**

---

# 📊 **Results Summary**

| Model       | Accuracy   | Precision | Recall | F1-Score |
| ----------- | ---------- | --------- | ------ | -------- |
| **Bi-LSTM** | **0.9569** | 0.9566    | 0.9569 | 0.9560   |
| GRU         | 0.9051     | 0.9051    | 0.9051 | 0.9035   |
| CNN-LSTM    | 0.9267     | 0.9266    | 0.9267 | 0.9266   |
| 3D-CNN      | 0.8621     | 0.8898    | 0.8621 | 0.8624   |
| LSTM        | 0.8577     | 0.8549    | 0.8577 | 0.8546   |

📌 **Bi-LSTM outperformed all architectures**, proving that bidirectional temporal context is critical for ASD behavior recognition.

---

# 🧪 **How to Run**

## **1️⃣ Install Dependencies**

```bash
pip install -r requirements.txt
```

## **2️⃣ Extract Pose Sequences**

```bash
python preprocess.py --input raw_videos/ --output skeleton_data/
```

## **3️⃣ Train Models**

Example:

```bash
python train_bilstm.py
```

## **4️⃣ Evaluate**

```bash
python evaluate.py --model bilstm
```

---

# 🌍 **Applications**

* Early ASD screening
* Home-based behavioral analysis
* Low-cost clinical decision support
* Non-invasive monitoring

---

# 🔮 **Future Improvements**

As suggested in the paper’s conclusion :

* 🎙️ **Multimodal data** (audio, eye-gaze, facial affect)
* 🧩 **Transformer-based video models (ViT, TimeSformer)**
* 📱 **Mobile-friendly deployment**
* 🧠 **Explainable AI for behavioral justification**
* 📈 **Larger, diverse datasets for generalization**

---

# 📜 **Citation**

If you use this work, please cite:

```
Sharma, A. (2025).
AI-Powered Early Autism Detection from Videos: A Comparative Approach.
Bennett University.
```

---

# 👩‍💻 **Author**

**Anika Sharma, Satyam**
Bennett University

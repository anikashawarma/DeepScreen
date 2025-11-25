# app.py
"""
Flask app for video-based ASD behavior classification using a multiclass .h5 model.
This version converts the model’s 5 classes into a binary label:
 - "Not an ASD behavior"
 - "Could be an indication of ASD"
Includes motion filtering to avoid false positives on still videos.
"""

import os
import uuid
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers as keras_layers
import mediapipe as mp

# -------------------------
# Configuration
# -------------------------
UPLOAD_DIR = "uploads"
MODEL_PATH = "models/CNN_LSTM_asd_behavior_model.h5"
ALLOWED_EXT = {"mp4", "mov", "avi", "mkv"}
TARGET_FRAMES = 150
MAX_CONTENT_MB = 500

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_MB * 1024 * 1024

# -------------------------
# Class mappings (IMPORTANT)
# -------------------------
# Must match EXACT training order
MULTI_CLASS_NAMES = {
    0: "action",
    1: "arm flapping",
    2: "head banging",
    3: "spinning",
    4: "still"
}

# ASD-related stereotypical behaviors
ASD_CLASS_SET = {1, 2, 3}  # arm flapping, head banging, spinning

# human-friendly labels
CLASS_MAP_BINARY = {
    0: "Not an ASD behavior",
    1: "Could be an indication of ASD"
}

# -------------------------
# Compatibility wrappers for legacy LSTMs
# -------------------------
@tf.keras.utils.register_keras_serializable(package="compat", name="LSTM")
class LSTM_Compat(keras_layers.LSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop("time_major", None)
        kwargs.pop("unit_forget_bias", None)
        kwargs.pop("zero_output_for_mask", None)
        kwargs.pop("seed", None)
        super().__init__(*args, **kwargs)

@tf.keras.utils.register_keras_serializable(package="compat", name="Bidirectional")
class Bidirectional_Compat(keras_layers.Bidirectional):
    def __init__(self, layer, *args, **kwargs):
        super().__init__(layer, *args, **kwargs)

_CUSTOM_OBJECTS = {
    "LSTM": LSTM_Compat,
    "Bidirectional": Bidirectional_Compat
}

# -------------------------
# Load model
# -------------------------
print("Loading model:", MODEL_PATH)
try:
    model = load_model(MODEL_PATH, compile=False)
    print("Loaded without custom objects.")
except:
    model = load_model(MODEL_PATH, compile=False, custom_objects=_CUSTOM_OBJECTS)
    print("Loaded with custom objects.")

print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)

# -------------------------
# MediaPipe
# -------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# -------------------------
# Helper functions
# -------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def extract_skeleton_numpy(video_path, target_frames=150):
    cap = cv2.VideoCapture(video_path)
    keypoints = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if results.pose_landmarks:
            kp = [[lm.x, lm.y] for lm in results.pose_landmarks.landmark]
            keypoints.append(kp)
    cap.release()

    if len(keypoints) == 0:
        return None

    data = np.array(keypoints, dtype=np.float32)

    mean = np.mean(data, axis=(0, 1))
    std = np.std(data, axis=(0, 1)) + 1e-6
    data = (data - mean) / std

    if len(data) > target_frames:
        data = data[:target_frames]
    else:
        pad = np.zeros((target_frames - len(data), 33, 2), dtype=np.float32)
        data = np.concatenate([data, pad], axis=0)

    return data


def compute_motion_score(skel):
    diffs = np.linalg.norm(np.diff(skel, axis=0), axis=2)
    per_frame = np.mean(diffs, axis=1)
    return float(np.median(per_frame))


def model_prepare_input(skeleton):
    inp_shape = model.input_shape

    flat = skeleton.reshape(1, skeleton.shape[0], -1).astype(np.float32)

    # (batch, T, features)
    if len(inp_shape) == 3:
        return flat

    # (batch, T, features, 1)
    if len(inp_shape) == 4:
        return flat.reshape(flat.shape[0], flat.shape[1], flat.shape[2], 1)

    return flat


def predict_from_video(path):
    skeleton = extract_skeleton_numpy(path)
    if skeleton is None:
        return {"success": False, "error": "No pose detected."}

    # Motion filter to avoid misclassifying still videos
    motion = compute_motion_score(skeleton)
    if motion < 1e-3:
        return {
            "success": True,
            "binary_index": 0,
            "label": CLASS_MAP_BINARY[0],
            "confidence": 0.50,
            "note": "Low motion detected — treated as non-ASD."
        }

    x = model_prepare_input(skeleton)
    probs = np.array(model.predict(x))

    pred_idx = int(np.argmax(probs))
    confidence = float(np.max(probs))
    pred_action = MULTI_CLASS_NAMES[pred_idx]

    # Multiclass → binary
    # --- CUSTOM FINAL LABEL LOGIC ---
    # action (0) or still (4) → unlikely ASD
    # arm flapping (1), head banging (2), spinning (3) → likely ASD

    if pred_idx in {0, 4}:   # action or still
        final_label = f"{pred_action} – unlikely ASD"
        binary_idx = 0

    elif pred_idx in {1, 2, 3}:  # ASD-related behaviors
        final_label = "Likely ASD"
        binary_idx = 1

    else:
        final_label = pred_action  # fallback
        binary_idx = 0

    return {
        "success": True,
        "pred_class_index": pred_idx,
        "pred_class_name": pred_action,
        "binary_index": binary_idx,
        "label": final_label,
        "confidence": confidence,
        "raw": probs.tolist()
    }



# -------------------------
# Routes
# -------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded."})

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"success": False, "error": "No file selected."})
    if not allowed_file(f.filename):
        return jsonify({"success": False, "error": "Invalid file type."})

    name = secure_filename(f.filename)
    unique = f"{uuid.uuid4().hex}_{name}"
    path = os.path.join(UPLOAD_DIR, unique)
    f.save(path)

    result = predict_from_video(path)
    os.remove(path)

    return jsonify(result)


@app.route("/static/<path:p>")
def static_file(p):
    return send_from_directory("static", p)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

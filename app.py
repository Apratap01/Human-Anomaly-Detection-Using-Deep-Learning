# streamlit_app.py
#
# Streamlit app to load your trained MobileNetV2 + (Bi)LSTM video classifier,
# accept a video upload, and predict its class. If the top predicted probability
# is below a user-set threshold the app will declare "Anomaly detected".
#
# Save this file in the same folder as your project (where dataset/, features_cache/
# and best_model_mobilenetv2_lstm.weights.h5 live), then run:
#    streamlit run streamlit_app.py

import streamlit as st
from pathlib import Path
import tempfile
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import os

# ---------------- CONFIG (tweak if needed) ----------------
ROOT = Path.cwd()
DATASET_DIR = ROOT / "dataset"
WEIGHT_FILE = ROOT / "best_model_mobilenetv2_lstm.weights.h5"
CACHE_DIR = ROOT / "features_cache"   # optional use
IMG_SIZE = 224
MAX_SEQ_LENGTH = 60
NUM_FEATURES = 1280

st.set_page_config(page_title="Video Classifier + Anomaly Detector", layout="wide")

# ---------------- Helpers (frame loading + cropping) ----------------
def crop_center_square(frame):
    y, x = frame.shape[:2]
    min_dim = min(y, x)
    start_x = (x - min_dim) // 2
    start_y = (y - min_dim) // 2
    return frame[start_y:start_y+min_dim, start_x:start_x+min_dim]

def load_video_frames_from_path(path, max_frames=MAX_SEQ_LENGTH, resize=(IMG_SIZE, IMG_SIZE)):
    """
    Read frames from a video file path (supports local file path).
    Uniformly sample up to max_frames frames. Returns numpy array
    shape (num_selected, IMG_SIZE, IMG_SIZE, 3) in RGB uint8.
    """
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            # fallback sequential read
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
        else:
            if total <= max_frames:
                idxs = list(range(total))
            else:
                idxs = np.linspace(0, total - 1, num=max_frames, dtype=int)
            fno = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if fno in idxs:
                    frames.append(frame)
                fno += 1
    finally:
        cap.release()

    out = []
    for f in frames:
        if f is None:
            continue
        f = crop_center_square(f)
        f = cv2.resize(f, resize)
        f = f[:, :, ::-1]  # BGR -> RGB
        out.append(f)
    if len(out) == 0:
        return np.zeros((0, resize[0], resize[1], 3), dtype=np.uint8)
    return np.array(out, dtype=np.uint8)

# ---------------- Build / load MobileNet feature extractor (cached) ----------------
@st.cache_resource
def build_feature_extractor():
    base = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False, weights="imagenet", pooling="avg"
    )
    base.trainable = False
    preprocess = keras.applications.mobilenet_v2.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = preprocess(inputs)
    outputs = base(x)
    model = keras.Model(inputs, outputs, name="feature_extractor")
    return model

feature_extractor = build_feature_extractor()

# ---------------- Build classifier architecture (must match training) ----------------
def get_sequence_model(max_seq=MAX_SEQ_LENGTH, num_features=NUM_FEATURES, num_classes=None):
    frame_features_input = keras.Input((max_seq, num_features), name="frame_features")
    mask_input = keras.Input((max_seq,), dtype="bool", name="mask")

    x = keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True))(frame_features_input, mask=mask_input)
    x = keras.layers.Bidirectional(keras.layers.LSTM(128))(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model([frame_features_input, mask_input], outputs)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.Adam(1e-4), metrics=["accuracy"])
    return model

# ---------------- Utility: read class list from dataset/train ----------------
def get_class_vocab_from_dataset(train_dir: Path):
    """
    Reads class folders from dataset/train and returns a sorted list of class names.
    This must match the vocabulary used when training the model.
    """
    train_dir = Path(train_dir)
    if not train_dir.exists():
        st.warning(f"train dataset directory not found at {train_dir}. Using empty class list.")
        return []
    classes = [p.name for p in train_dir.iterdir() if p.is_dir()]
    classes = sorted(classes)
    return classes

# ---------------- FEATURE PIPELINE: frames -> features + mask ----------------
def frames_to_feats_and_mask(frames, max_seq=MAX_SEQ_LENGTH):
    """
    frames: np.array shape (num_frames, H, W, 3) dtype uint8 or float
    Returns:
        feats: (max_seq, NUM_FEATURES) float32
        mask: (max_seq,) bool
    """
    if frames is None or frames.shape[0] == 0:
        feats = np.zeros((max_seq, NUM_FEATURES), dtype=np.float32)
        mask = np.zeros((max_seq,), dtype=bool)
        return feats, mask

    frames_f = frames.astype("float32")
    # MobileNet preprocess handled inside the feature_extractor model wrapper
    feats_batch = feature_extractor.predict(frames_f, verbose=0)
    length = feats_batch.shape[0]
    feats = np.zeros((max_seq, NUM_FEATURES), dtype=np.float32)
    feats[:length] = feats_batch[:max_seq]
    mask = np.zeros((max_seq,), dtype=bool)
    mask[:min(length, max_seq)] = True
    return feats, mask

# ---------------- App UI ----------------
st.title("Video Classification + Anomaly Detection")
st.write("Upload a video. If the model is not confident (max probability < threshold), it will report 'Anomaly detected'.")

# load class names from dataset/train
class_vocab = get_class_vocab_from_dataset(DATASET_DIR / "train")
num_classes = len(class_vocab)
if num_classes == 0:
    st.error("No classes found in dataset/train. Put class subfolders with videos under dataset/train and restart the app.")
    st.stop()

st.sidebar.header("Settings")
threshold = st.sidebar.slider("Anomaly threshold (max probability below this → anomaly)", min_value=0.01, max_value=0.99, value=0.50, step=0.01)
top_k = st.sidebar.slider("Number of top predictions to show", min_value=1, max_value=min(10, num_classes), value=5)

# load the trained model and weights (cached)
@st.cache_resource
def load_trained_model():
    model = get_sequence_model(max_seq=MAX_SEQ_LENGTH, num_features=NUM_FEATURES, num_classes=num_classes)
    if not WEIGHT_FILE.exists():
        st.warning(f"Weight file not found at {WEIGHT_FILE}. Model will be uninitialized.")
    else:
        model.load_weights(str(WEIGHT_FILE))
    return model

model = load_trained_model()

# Uploader widget
uploaded_file = st.file_uploader("Upload a video file (mp4 / avi / mov / mkv)", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    # Save uploaded video to a temporary file so OpenCV can read it
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix)
    try:
        tfile.write(uploaded_file.read())
        tfile.flush()
        local_video_path = Path(tfile.name)

        st.video(str(local_video_path))  # display the uploaded video

        with st.spinner("Extracting frames and computing features..."):
            frames = load_video_frames_from_path(local_video_path, max_frames=MAX_SEQ_LENGTH, resize=(IMG_SIZE, IMG_SIZE))
            feats, mask = frames_to_feats_and_mask(frames, max_seq=MAX_SEQ_LENGTH)

        # prepare batch dims
        feats_b = feats[np.newaxis, ...]   # (1, max_seq, NUM_FEATURES)
        mask_b = mask[np.newaxis, ...]     # (1, max_seq)

        with st.spinner("Running prediction..."):
            preds = model.predict([feats_b, mask_b], verbose=0)[0]  # (num_classes,)

        top_idxs = np.argsort(preds)[::-1][:top_k]
        max_prob = float(np.max(preds))
        best_idx = int(np.argmax(preds))
        best_label = class_vocab[best_idx]
        best_prob = float(preds[best_idx])

        st.subheader("Prediction results")
        if max_prob < threshold:
            st.error(f"Anomaly detected — model not confident (max prob {max_prob:.3f} < threshold {threshold:.3f})")
        else:
            st.success(f"Predicted class: **{best_label}** with probability {best_prob*100:.2f}%")

        st.write("Top predictions:")
        for rank, idx in enumerate(top_idxs, start=1):
            label = class_vocab[idx]
            prob = preds[idx]
            st.write(f"{rank}. {label:<30} — {prob*100:.2f}%")

        st.write("---")
        st.write("Raw prediction vector (first 20 values):")
        st.write(np.round(preds[:min(20, len(preds))], 4))

    finally:
        # clean up temp file
        try:
            tfile.close()
            os.unlink(tfile.name)
        except Exception:
            pass

else:
    st.info("Upload a video to get started. The app reads frames from the video, uses MobileNetV2 to extract per-frame features, then runs the (Bi)LSTM classifier to predict the class.")

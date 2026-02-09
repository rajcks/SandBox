# DL.py
# Deep-learning features (YAMNet embeddings) + simple classifier (Logistic Regression)
# Folders:
#   data/good/*.wav (or mp3/flac/ogg/m4a)
#   data/bad/*.wav
#
# Train:
#   python AI.py --train --data data --model sound_classifier_DL.joblib
#
# Predict:
#   python AI.py --predict path\to\new.wav --data data --model sound_classifier_DL.joblib --ignore_threshold 0.85 --autosort
#
# Notes:
# - Uses YAMNet from TF Hub as feature extractor (deep learning).
# - Trains a lightweight classifier on top (works with small datasets).
# - If confidence < ignore_threshold -> "ignore_unknown" (good for later "nature" filtering).
# - With --autosort: copies confident predictions into data/good or data/bad (self-collecting).

import os
import shutil
from pathlib import Path
from collections import Counter

import numpy as np
import librosa
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# ---- YAMNet deps ----
import tensorflow as tf
import tensorflow_hub as hub

# ----------------- CONFIG -----------------
SR = 16000
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

# You can switch to "copy" or "move" behavior below
DEFAULT_SORT_MODE = "copy"  # "copy" is safer than "move"

# ----------------- YAMNET LOADING -----------------
# Load once at startup. First run may download from TF Hub.
_YAMNET = hub.load("https://tfhub.dev/google/yamnet/1")


def load_audio_16k_mono(path: str, sr: int = SR) -> np.ndarray:
    """Load audio, convert to mono, resample to 16kHz float32 in [-1,1]."""
    y, orig_sr = librosa.load(path, sr=None, mono=True)
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
    # librosa already returns float in [-1, 1] typically
    return y.astype(np.float32)


def yamnet_embedding(path: str) -> np.ndarray:
    """
    Extract a single fixed-length (1024-d) embedding for a whole file by
    mean-pooling YAMNet's frame embeddings.
    """
    wav = load_audio_16k_mono(path, sr=SR)
    waveform = tf.convert_to_tensor(wav, dtype=tf.float32)

    # YAMNet returns: scores (frames x classes), embeddings (frames x 1024), spectrogram
    scores, embeddings, spectrogram = _YAMNET(waveform)

    emb = tf.reduce_mean(embeddings, axis=0).numpy().astype(np.float32)  # (1024,)

    # L2 normalize
    emb /= (np.linalg.norm(emb) + 1e-12)
    return emb


def load_dataset(root="data"):
    """
    Reads data/good and data/bad, returns X (Nx1024), y (N,), and paths list.
    y: good=1, bad=0
    """
    root = Path(root)
    X, y, paths = [], [], []

    for label_name, label in [("good", 1), ("bad", 0)]:
        folder = root / label_name
        if not folder.exists():
            continue
        for p in sorted(folder.rglob("*")):
            if p.suffix.lower() not in AUDIO_EXTS:
                continue
            try:
                X.append(yamnet_embedding(str(p)))
                y.append(label)
                paths.append(str(p))
            except Exception as e:
                print(f"Skipping {p} بسبب error: {e}")

    if not X:
        raise ValueError(
            f"No audio files found. Expected folders: {root / 'good'} and {root / 'bad'}"
        )

    X = np.vstack(X)
    y = np.array(y, dtype=int)
    return X, y, paths


def train(data_root="data", out_model="sound_classifier_DL.joblib"):
    X, y, _ = load_dataset(data_root)

    counts = Counter(y)
    pretty_counts = {("bad" if k == 0 else "good"): v for k, v in counts.items()}
    print("Class counts:", pretty_counts)

    if len(np.unique(y)) < 2:
        raise ValueError("You need BOTH classes (good and bad) to train.")

    # If very small data, we skip train/test split
    min_count = min(counts.values())
    do_split = min_count >= 2 and len(y) >= 10  # you can tune this rule

    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=4000, class_weight="balanced")),
        ]
    )

    if not do_split:
        print("\n Small dataset: training on ALL data (no evaluation).\n")
        clf.fit(X, y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
        print(
            "\nReport:\n",
            classification_report(y_test, y_pred, target_names=["bad", "good"]),
        )

    joblib.dump({"model": clf, "sr": SR, "feature": "yamnet_mean_embedding"}, out_model)
    print(f"\n Saved model to: {out_model}")


def predict(audio_path, model_path="sound_classifier_DL.joblib", ignore_threshold=0.85):
    pack = joblib.load(model_path)
    clf = pack["model"]

    x = yamnet_embedding(audio_path).reshape(1, -1)

    # predict_proba -> [P(bad), P(good)]
    proba = clf.predict_proba(x)[0]
    p_bad, p_good = float(proba[0]), float(proba[1])

    best_label = "good" if p_good >= p_bad else "bad"
    best_conf = max(p_good, p_bad)

    if best_conf < ignore_threshold:
        return {
            "decision": "ignore_unknown",
            "p_good": p_good,
            "p_bad": p_bad,
            "confidence": best_conf,
        }

    return {"decision": best_label, "p_good": p_good, "p_bad": p_bad, "confidence": best_conf}


def sort_into_folder(
    audio_path: str, decision: str, data_root="data", mode: str = DEFAULT_SORT_MODE
):
    """
    decision must be 'good' or 'bad'
    mode: 'copy' (safe) or 'move'
    """
    assert decision in ("good", "bad")
    src = Path(audio_path)
    dst_dir = Path(data_root) / decision
    dst_dir.mkdir(parents=True, exist_ok=True)

    dst = dst_dir / src.name

    # If same name exists, add suffix
    if dst.exists():
        stem, suffix = src.stem, src.suffix
        i = 1
        while True:
            candidate = dst_dir / f"{stem}_{i}{suffix}"
            if not candidate.exists():
                dst = candidate
                break
            i += 1

    if mode == "move":
        shutil.move(str(src), str(dst))
        print(f"Moved to: {dst}")
    else:
        shutil.copy2(str(src), str(dst))
        print(f"Copied to: {dst}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="Train model from data folders")
    ap.add_argument("--data", default="data", help="Root data folder with good/ bad/")
    ap.add_argument("--model", default="sound_classifier_DL.joblib", help="Model output/input path")
    ap.add_argument("--predict", default=None, help="Audio file to predict")
    ap.add_argument("--ignore_threshold", type=float, default=0.85, help="Below this -> ignore_unknown")
    ap.add_argument("--autosort", action="store_true", help="Copy predicted files into data/good or data/bad")
    ap.add_argument("--sort_mode", choices=["copy", "move"], default=DEFAULT_SORT_MODE, help="copy is safer than move")
    ap.add_argument("--top", type=int, default=1, help="(unused, reserved)")

    args = ap.parse_args()

    if args.train:
        train(args.data, args.model)

    if args.predict:
        res = predict(args.predict, args.model, args.ignore_threshold)
        print(res)
        # if args.autosort and res["decision"] in ("good", "bad"):
        #     sort_into_folder(args.predict, res["decision"], data_root=args.data, mode=args.sort_mode)

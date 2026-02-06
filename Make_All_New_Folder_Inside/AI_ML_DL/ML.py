# Train:
#   python ML.py --train --data data --model sound_classifier_ML.joblib
#
# Predict:
#   python ML.py --predict path\to\new.wav --data data --model sound_classifier_ML.joblib --ignore_threshold 0.85

from pathlib import Path
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import shutil

SR = 8000
DURATION = 6.0     # seconds (tune to your typical clip length)
N_MFCC = 20

def load_audio(path, sr=SR, duration=DURATION):
    y, _ = librosa.load(path, sr=sr, mono=True)
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y

def featurize(path: str) -> np.ndarray:
    """
    MFCC + deltas + summary stats → fixed-length vector.
    Good baseline for good/bad classification.
    """
    y = load_audio(path)

    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)

    feats = np.vstack([mfcc, d1, d2])  # (N, T)

    mean = feats.mean(axis=1)
    std = feats.std(axis=1)
    vec = np.hstack([mean, std]).astype(np.float32)
    return vec

def load_dataset(root="data"):
    root = Path(root)
    X, y, paths = [], [], []

    for label_name, label in [("good", 1), ("bad", 0)]:
        for p in sorted((root / label_name).rglob("*")):
            if p.suffix.lower() not in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]:
                continue
            try:
                X.append(featurize(str(p)))
                y.append(label)
                paths.append(str(p))
            except Exception as e:
                print(f"Skipping {p}: {e}")

    X = np.vstack(X)
    y = np.array(y, dtype=int)
    return X, y, paths

def train(root="data", out_model="sound_classifier_ML.joblib"):
    X, y, _ = load_dataset(root)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Model: scaler + logistic regression
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nReport:\n", classification_report(y_test, y_pred, target_names=["bad", "good"]))

    joblib.dump({"model": clf, "sr": SR, "duration": DURATION, "n_mfcc": N_MFCC}, out_model)
    print(f"\nSaved model to {out_model}")

def predict(audio_path, model_path="sound_classifier_ML.joblib", ignore_threshold=0.75):
    pack = joblib.load(model_path)
    clf = pack["model"]

    x = featurize(audio_path).reshape(1, -1)
    proba = clf.predict_proba(x)[0]  # [P(bad), P(good)]
    p_bad, p_good = float(proba[0]), float(proba[1])

    best_label = "good" if p_good >= p_bad else "bad"
    best_conf = max(p_good, p_bad)

    # This is how you "ignore nature/other" later without having nature data yet:
    if best_conf < ignore_threshold:
        return {"decision": "ignore_unknown", "p_good": p_good, "p_bad": p_bad, "confidence": best_conf}

    return {"decision": best_label, "p_good": p_good, "p_bad": p_bad, "confidence": best_conf}

def move_to_class(audio_path, decision, data_root="data"):
    src = Path(audio_path)
    dst_dir = Path(data_root) / decision
    dst_dir.mkdir(parents=True, exist_ok=True)

    dst = dst_dir / src.name
    shutil.copy2(src, dst)   # copy instead of move (safer)
    print(f"Copied to: {dst}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--data", default="data")
    ap.add_argument("--model", default="sound_classifier_ML.joblib")
    ap.add_argument("--predict", default=None)
    ap.add_argument("--ignore_threshold", type=float, default=0.75)
    args = ap.parse_args()

    if args.train:
        train(args.data, args.model)

    if args.predict:
      res = predict(args.predict, args.model, args.ignore_threshold)
      print(res)
      # Auto-sort only if confident and not ignored
      # if res["decision"] in ["good", "bad"]:
      #     move_to_class(args.predict, res["decision"], data_root=args.data)

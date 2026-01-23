import tensorflow as tf
import tensorflow_hub as hub
import librosa
import urllib.request

# 1. Load YAMNet model from TensorFlow Hub
MODEL_HANDLE = "https://tfhub.dev/google/yamnet/1"
print("Loading model...")
model = hub.load(MODEL_HANDLE)
print("Model loaded.")

# 2. Download class labels file (once)
LABELS_URL = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
LABELS_FILE = "yamnet_class_map.csv"

try:
    open(LABELS_FILE, "r").close()
except FileNotFoundError:
    print("Downloading labels...")
    urllib.request.urlretrieve(LABELS_URL, LABELS_FILE)

# 3. Read labels into a list
labels = []
with open(LABELS_FILE, "r", encoding="utf-8") as f:
    next(f)  # skip header
    for line in f:
        _, _, display_name = line.strip().split(",")
        labels.append(display_name.strip('"'))

# 4. Load an audio file (change this path)
audio_path = "test.wav"  # <- put your audio file here
print(f"Loading audio: {audio_path}")
waveform, sr = librosa.load(audio_path, sr=16000, mono=True)  # YAMNet expects 16 kHz mono

# 5. Run the model
scores, embeddings, spectrogram = model(waveform)

# 6. Average scores over time and get top predictions
mean_scores = tf.reduce_mean(scores, axis=0)
top_n = 5
top_indices = tf.argsort(mean_scores, direction="DESCENDING")[:top_n]

print("\nTop predictions:")
for i in top_indices.numpy():
    print(f"- {labels[i]} ({mean_scores[i].numpy():.3f})")

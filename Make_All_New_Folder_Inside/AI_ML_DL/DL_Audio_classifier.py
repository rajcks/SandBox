import os
import numpy as np
import librosa
import tensorflow as tf

class AudioClassifier:
    def __init__(self, base_dir, target_sr=16000, clip_seconds=3, frame_length=320, frame_step=32):
        self.model = None

        self.base_dir = base_dir
        self.target_sr = target_sr
        self.clip_seconds = clip_seconds
        self.target_samples = target_sr * clip_seconds
        self.frame_length = frame_length
        self.frame_step = frame_step

        self.pos_glob = os.path.join(self.base_dir, "data/good", "*.wav")
        self.neg_glob = os.path.join(self.base_dir, "data/bad", "*.wav")
        self.Input_dir = os.path.join(self.base_dir, "data/Input_Data")  # adjust if different

        self.model_path = os.path.join(self.base_dir, "Models", "audio_classifier_model.keras")


    # ---------- audio loading (wav/mp3) ----------
    def _load_audio_np(self, path):
        if isinstance(path, np.ndarray):
            path = path.item()
        if isinstance(path, bytes):
            path = path.decode("utf-8")

        y, _ = librosa.load(path, sr=self.target_sr, mono=True)
        return y.astype(np.float32)

    def load_audio_tf(self, path):
        audio = tf.numpy_function(self._load_audio_np, [path], tf.float32)
        audio.set_shape([None])
        return audio
    
    # ---------- preprocessing ----------
    def file_to_spectrogram(self, file_path, label):
        wav = self.load_audio_tf(file_path)
        wav = wav[:self.target_samples]                 # Truncate or pad to 3 seconds (assuming 16kHz sample rate)
        pad_len = self.target_samples - tf.shape(wav)
        zero_padding = tf.zeros(pad_len, dtype=tf.float32)
        wav = tf.concat([zero_padding, wav], axis=0)
        # Convert waveform to spectrogram
        spectrogram = tf.signal.stft(wav, frame_length=self.frame_length, frame_step=self.frame_step)
        spectrogram = tf.abs(spectrogram)
        spectrogram = tf.expand_dims(spectrogram, axis=2)
        return spectrogram, label
    
    def slice_to_spectrogram(self, wav_slice):
        wav_slice = tf.squeeze(wav_slice, axis=0)  # (1, N) -> (N,)
        spec = tf.signal.stft(wav_slice, frame_length=self.frame_length, frame_step=self.frame_step)
        spec = tf.abs(spec)
        spec = tf.expand_dims(spec, axis=2)
        return spec
    
    # ---------- dataset ----------
    def build_dataset(self, batch_size=16, shuffle=1000):
        pos_files = tf.io.gfile.glob(self.pos_glob)
        neg_files = tf.io.gfile.glob(self.neg_glob)

        pos = tf.data.Dataset.from_tensor_slices(pos_files)
        neg = tf.data.Dataset.from_tensor_slices(neg_files)

        positives = tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos_files)))))
        negatives = tf.data.Dataset.zip((neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg_files)))))

        data = positives.concatenate(negatives)

        # ------- TensorFlow Data Pipeline --------
        data = data.map(self.file_to_spectrogram)
        data = data.cache()
        data = data.shuffle(buffer_size=shuffle)
        data = data.batch(batch_size)
        data = data.prefetch(tf.data.AUTOTUNE)

        return data

    # ---------- model ----------
    def build_model(self, input_shape):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape))
        self.model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        self.model.compile(
            optimizer='Adam',
            loss='BinaryCrossentropy',
            metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
        )
        return self.model

    def train(self, epochs=4, save_path=None):
        save_path = self.model_path
        data = self.build_dataset()

        # get real spectrogram shape automatically (no guessing 1491/257)
        x0, _ = next(iter(data.take(1)))
        input_shape = tuple(x0.shape[1:])  # (time, freq, 1)

        self.build_model(input_shape)

        train = data.take(36)
        test = data.skip(36).take(15)

        self.model.fit(train, epochs=epochs, validation_data=test)

        self.model.save(save_path)
        print(f"Model saved to {save_path}")
        return train, test
    
    def load_model(self, model_path=None):
        model_path = self.model_path
        self.model = tf.keras.models.load_model(model_path)
        print(f"Loaded model from {model_path}")


    # ---------- predict file ----------
    def predict_file(self, filename, threshold=0.99):
        filepath = os.path.join(self.Input_dir, filename)

        wav = self.load_audio_tf(tf.constant(filepath))
        slices = tf.keras.utils.timeseries_dataset_from_array(
            wav, None,
            sequence_length=self.target_samples,
            sequence_stride=self.target_samples,
            batch_size=1
        )

        slices = slices.map(self.slice_to_spectrogram).batch(64)
        probs = self.model.predict(slices).reshape(-1)
        preds = (probs > threshold).astype(int)

        return probs, preds

if __name__ == "__main__":
    BASE_DIR = "Make_ALL_New_Folder_Inside/AI_ML_DL"

    clf = AudioClassifier(BASE_DIR)
    clf.train(epochs=4)
    # clf.load_model("audio_classifier_model.keras") # Uncomment if you want to load a pre-trained model instead of training
    file = "your.wav"  # Replace with your actual file name in the Input_dir
    probs, preds = clf.predict_one_file(file, threshold=0.99)

    print("File:", file)
    print("Probabilities:", probs)
    print("Predictions:", preds)
    print("OutPut:", bool(preds.sum() > 0))

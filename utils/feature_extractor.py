import librosa
import numpy as np

SAMPLE_RATE = 22050
DURATION = 5  # seconds

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)

        features = np.concatenate((
            np.mean(mfccs, axis=1),
            np.mean(chroma, axis=1),
            np.mean(spectral_contrast, axis=1)
        ))
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

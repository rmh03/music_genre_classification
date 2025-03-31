import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

def extract_features_from_audio(audio_path, label):
    """
    Extract features from an audio file.

    Parameters:
        audio_path (str): Path to the audio file.
        label (str): Label of the audio file (e.g., genre).

    Returns:
        dict: A dictionary containing the extracted features.
    """
    try:
        # Load the audio file (at least 3 seconds)
        audio, sr = librosa.load(audio_path, duration=3)

        # Extract features
        features = {
            "filename": os.path.basename(audio_path),
            "length": len(audio),
            "chroma_stft_mean": np.mean(librosa.feature.chroma_stft(y=audio, sr=sr)),
            "chroma_stft_var": np.var(librosa.feature.chroma_stft(y=audio, sr=sr)),
            "rms_mean": np.mean(librosa.feature.rms(y=audio)),
            "rms_var": np.var(librosa.feature.rms(y=audio)),
            "spectral_centroid_mean": np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)),
            "spectral_centroid_var": np.var(librosa.feature.spectral_centroid(y=audio, sr=sr)),
            "spectral_bandwidth_mean": np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)),
            "spectral_bandwidth_var": np.var(librosa.feature.spectral_bandwidth(y=audio, sr=sr)),
            "rolloff_mean": np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr)),
            "rolloff_var": np.var(librosa.feature.spectral_rolloff(y=audio, sr=sr)),
            "zero_crossing_rate_mean": np.mean(librosa.feature.zero_crossing_rate(y=audio)),
            "zero_crossing_rate_var": np.var(librosa.feature.zero_crossing_rate(y=audio)),
            "harmony_mean": np.mean(librosa.effects.harmonic(y=audio)),
            "harmony_var": np.var(librosa.effects.harmonic(y=audio)),
            "perceptr_mean": np.mean(librosa.effects.percussive(y=audio)),
            "perceptr_var": np.var(librosa.effects.percussive(y=audio)),
            "tempo": librosa.feature.rhythm.tempo(y=audio, sr=sr)[0],
        }

        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        for i in range(1, 21):
            features[f"mfcc{i}_mean"] = np.mean(mfccs[i - 1])
            features[f"mfcc{i}_var"] = np.var(mfccs[i - 1])

        # Add the label
        features["label"] = label

        return features
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def process_audio_directory(input_dir, output_csv):
    """
    Process all audio files in a directory and save the extracted features to a CSV file.

    Parameters:
        input_dir (str): Path to the directory containing audio files.
        output_csv (str): Path to save the output CSV file.
    """
    features_list = []

    # Iterate through all subdirectories and files
    for root, _, files in os.walk(input_dir):
        for file in tqdm(files, desc="Processing audio files"):
            if file.endswith(".wav"):
                audio_path = os.path.join(root, file)
                label = os.path.basename(root)  # Use the folder name as the label
                features = extract_features_from_audio(audio_path, label)
                if features:
                    features_list.append(features)

    # Save features to a CSV file
    df = pd.DataFrame(features_list)
    df.to_csv(output_csv, index=False)
    print(f"Features saved to {output_csv}")

if __name__ == "__main__":
    # Define input directory and output CSV file
    input_directory = "h:/ML project/Project/data/raw/genres_original"
    output_csv_file = "h:/ML project/Project/data/processed/audio_features.csv"

    # Process the audio files and save features
    process_audio_directory(input_directory, output_csv_file)
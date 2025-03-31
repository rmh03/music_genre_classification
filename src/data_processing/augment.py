import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

def augment_audio(audio, sr):
    """
    Applies basic data augmentation techniques to an audio signal.
    
    Parameters:
        audio (numpy.ndarray): The audio signal.
        sr (int): The sampling rate of the audio signal.
    
    Returns:
        list: A list of augmented audio signals.
    """
    return [
        librosa.effects.pitch_shift(audio, sr=sr, n_steps=2),  # Pitch up
        librosa.effects.pitch_shift(audio, sr=sr, n_steps=-2),  # Pitch down
        librosa.effects.time_stretch(audio, rate=0.8),  # Slow down
        librosa.effects.time_stretch(audio, rate=1.2),  # Speed up
        audio + np.random.randn(len(audio)) * 0.005  # Add noise
    ]

def extract_features(audio, sr, label, augmentation_type):
    """
    Extract features from an audio signal.
    
    Parameters:
        audio (numpy.ndarray): The audio signal.
        sr (int): The sampling rate of the audio signal.
        label (str): The label of the audio file (e.g., genre).
        augmentation_type (str): The type of augmentation applied (e.g., 'original', 'pitch_up').
    
    Returns:
        dict: A dictionary containing the extracted features.
    """
    try:
        if len(audio) == 0:
            raise ValueError("Audio data is empty or invalid.")

        features = {
            "filename": label,  # Use the label as the filename for simplicity
            "augmentation": augmentation_type,
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
            "tempo": librosa.beat.tempo(y=audio, sr=sr)[0],  # Use librosa.beat.tempo for compatibility
            "label": label
        }

        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        for i in range(1, 21):
            features[f"mfcc{i}_mean"] = np.mean(mfccs[i - 1])
            features[f"mfcc{i}_var"] = np.var(mfccs[i - 1])

        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def process_and_save_augmented_features(input_dir, output_csv):
    """
    Process audio files, apply augmentations, extract features, and save them to a CSV file.
    
    Parameters:
        input_dir (str): The directory containing the audio files.
        output_csv (str): The path to save the output CSV file.
    """
    features_list = []

    for root, _, files in os.walk(input_dir):
        for file in tqdm(files, desc="Processing audio files"):
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                label = os.path.basename(root)  # Use the folder name as the label

                try:
                    # Load the original audio
                    audio, sr = librosa.load(file_path, sr=None, duration=3)

                    # Extract features for the original audio
                    original_features = extract_features(audio, sr, label, "original")
                    if original_features:
                        features_list.append(original_features)

                    # Apply augmentations and extract features
                    augmentations = augment_audio(audio, sr)
                    augmentation_types = ["pitch_up", "pitch_down", "slow_down", "speed_up", "noise"]
                    for aug_audio, aug_type in zip(augmentations, augmentation_types):
                        augmented_features = extract_features(aug_audio, sr, label, aug_type)
                        if augmented_features:
                            features_list.append(augmented_features)

                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

    # Save features to a CSV file
    df = pd.DataFrame(features_list)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)  # Ensure the directory exists
    df.to_csv(output_csv, index=False)
    print(f"Augmented features saved to {output_csv}")

if __name__ == "__main__":
    # Define input directory and output CSV file
    input_directory = "h:/ML project/Project/data/raw/genres_original"
    output_csv_file = "h:/ML project/Project/data/augmented/augmented_features.csv"

    # Process audio files and save augmented features
    process_and_save_augmented_features(input_directory, output_csv_file)
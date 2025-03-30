import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from src.utils.config import Config

def extract_and_plot_features(audio_path):
    """
    Extract audio features and save visualizations to the figures directory.

    Parameters:
        audio_path (str): Path to the audio file.
    """
    try:
        # Initialize Config to get the figures directory
        config = Config()
        output_dir = config.figures_dir

        # Load audio file
        try:
            audio, sr = librosa.load(audio_path, sr=None)
        except Exception as e:
            print(f"Error processing file {audio_path}: {e}")
            return

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Chroma STFT
        chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sr)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(chroma_stft, x_axis='time', y_axis='chroma', cmap='coolwarm', sr=sr)
        plt.colorbar()
        plt.title('Chroma STFT')
        plt.savefig(os.path.join(output_dir, 'chroma_stft.png'))
        plt.close()

        # RMS Energy
        rms = librosa.feature.rms(y=audio)
        plt.figure(figsize=(10, 4))
        plt.semilogy(rms.T, label='RMS Energy', color='b')
        plt.xlabel('Frames')
        plt.ylabel('RMS Energy')
        plt.title('RMS Energy')
        plt.savefig(os.path.join(output_dir, 'rms_energy.png'))
        plt.close()

        # Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        plt.figure(figsize=(10, 4))
        plt.semilogy(spectral_centroid.T, label='Spectral Centroid', color='g')
        plt.xlabel('Frames')
        plt.ylabel('Hz')
        plt.title('Spectral Centroid')
        plt.savefig(os.path.join(output_dir, 'spectral_centroid.png'))
        plt.close()

    except Exception as e:
        print(f"Error extracting features for {audio_path}: {e}")
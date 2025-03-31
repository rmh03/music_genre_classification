import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from src.utils.config import Config

def extract_and_plot_features(audio_path, genre):
    """
    Extract audio features and save visualizations to genre-specific folders in the figures directory.

    Parameters:
        audio_path (str): Path to the audio file.
        genre (str): Genre of the audio file.
    """
    try:
        # Initialize Config to get the figures directory
        config = Config()
        output_dir = os.path.join(config.figures_dir, genre)  # Create a subdirectory for the genre
        os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

        # Load audio file
        audio, sr = librosa.load(audio_path, sr=None)

        # Plot Chroma STFT
        chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sr)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(chroma_stft, x_axis='time', y_axis='chroma', cmap='coolwarm', sr=sr)
        plt.colorbar()
        plt.title(f'Chroma STFT - {genre}')
        plt.savefig(os.path.join(output_dir, f'chroma_stft_{genre}.png'))
        plt.close()

        # Plot RMS Energy
        rms = librosa.feature.rms(y=audio)
        plt.figure(figsize=(10, 4))
        plt.semilogy(rms.T, label='RMS Energy', color='b')
        plt.xlabel('Frames')
        plt.ylabel('RMS Energy')
        plt.title(f'RMS Energy - {genre}')
        plt.savefig(os.path.join(output_dir, f'rms_energy_{genre}.png'))
        plt.close()

        # Plot Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        plt.figure(figsize=(10, 4))
        plt.semilogy(spectral_centroid.T, label='Spectral Centroid', color='g')
        plt.xlabel('Frames')
        plt.ylabel('Hz')
        plt.title(f'Spectral Centroid - {genre}')
        plt.savefig(os.path.join(output_dir, f'spectral_centroid_{genre}.png'))
        plt.close()

        # Plot MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfccs, x_axis='time', sr=sr)
        plt.colorbar()
        plt.title(f'MFCCs - {genre}')
        plt.savefig(os.path.join(output_dir, f'mfccs_{genre}.png'))
        plt.close()

        print(f"Plots saved for genre: {genre}")

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")

def main():
    """
    Main function to process example audio files for each genre.
    """
    # Define the directory containing raw audio files
    input_dir = "h:/ML project/Project/data/raw/genres_original"

    # Example audio files for each genre
    example_files = {
        "blues": "blues.00000.wav",
        "classical": "classical.00000.wav",
        "country": "country.00000.wav",
        "disco": "disco.00000.wav",
        "hiphop": "hiphop.00000.wav",
        "jazz": "jazz.00000.wav",
        "metal": "metal.00000.wav",
        "pop": "pop.00000.wav",
        "reggae": "reggae.00000.wav",
        "rock": "rock.00000.wav"
    }

    # Process each genre
    for genre, filename in example_files.items():
        audio_path = os.path.join(input_dir, genre, filename)
        extract_and_plot_features(audio_path, genre)

if __name__ == "__main__":
    main()
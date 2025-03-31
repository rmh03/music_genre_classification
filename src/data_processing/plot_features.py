import os
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.config import Config

def plot_features_for_genre(df, genre, output_dir):
    """
    Plot all features for a single audio file of a specific genre.

    Parameters:
        df (DataFrame): DataFrame containing the features.
        genre (str): The genre to plot features for.
        output_dir (str): Directory to save the plots.
    """
    # Filter the DataFrame for the selected genre
    genre_df = df[df['label'] == genre]

    # Select the first audio file for the genre
    if genre_df.empty:
        print(f"No data found for genre: {genre}")
        return
    audio_data = genre_df.iloc[0]

    # Create output directory for the genre
    genre_output_dir = os.path.join(output_dir, genre)
    os.makedirs(genre_output_dir, exist_ok=True)

    # Plot all features
    features_to_plot = [
        'chroma_stft_mean', 'chroma_stft_var', 'rms_mean', 'rms_var',
        'spectral_centroid_mean', 'spectral_centroid_var', 'spectral_bandwidth_mean', 'spectral_bandwidth_var',
        'rolloff_mean', 'rolloff_var', 'zero_crossing_rate_mean', 'zero_crossing_rate_var', 'tempo'
    ]
    mfcc_features = [f'mfcc{i}_mean' for i in range(1, 21)] + [f'mfcc{i}_var' for i in range(1, 21)]

    # Plot spectral features
    plt.figure(figsize=(10, 6))
    for feature in features_to_plot:
        plt.bar(feature, audio_data[feature])
    plt.title(f"Spectral Features - {genre}")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(genre_output_dir, f"{genre}_spectral_features.png"))
    plt.close()

    # Plot MFCC features
    plt.figure(figsize=(12, 6))
    mfcc_means = [audio_data[f'mfcc{i}_mean'] for i in range(1, 21)]
    mfcc_vars = [audio_data[f'mfcc{i}_var'] for i in range(1, 21)]
    plt.bar(range(1, 21), mfcc_means, label='MFCC Means', alpha=0.7)
    plt.bar(range(1, 21), mfcc_vars, label='MFCC Variances', alpha=0.7)
    plt.title(f"MFCC Features - {genre}")
    plt.xlabel("MFCC Index")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(genre_output_dir, f"{genre}_mfcc_features.png"))
    plt.close()

    print(f"Plots saved for genre: {genre}")

def main():
    """
    Main function to plot features for one audio file per genre.
    """
    # Initialize Config and paths
    config = Config()
    input_csv = os.path.join(config.data_dir, "augmented", "augmented_features.csv")
    output_dir = os.path.join(config.docs_dir, "figures")

    # Load the augmented features CSV
    df = pd.read_csv(input_csv)

    # Get the list of unique genres
    genres = df['label'].unique()

    # Plot features for one audio file per genre
    for genre in genres:
        plot_features_for_genre(df, genre, output_dir)

if __name__ == "__main__":
    main()
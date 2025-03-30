import os
import pandas as pd
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.data_processing.augment import augment_audio

def extract_features(audio, sr, n_mfcc=13):
    """
    Extract MFCC features from an audio signal.
    
    Parameters:
        audio (numpy.ndarray): The audio signal.
        sr (int): The sampling rate of the audio signal.
        n_mfcc (int): The number of MFCC features to extract.
    
    Returns:
        numpy.ndarray: The extracted MFCC features.
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)

def process_audio(file_path, genre, n_mfcc=13, augment=False):
    """
    Process a single audio file and optionally apply augmentation.
    
    Parameters:
        file_path (str): The path to the audio file.
        genre (str): The genre label of the audio file.
        n_mfcc (int): The number of MFCC features to extract.
        augment (bool): Whether to apply data augmentation.
    
    Returns:
        list: A list of tuples containing features and labels.
    """
    try:
        audio, sr = librosa.load(file_path, sr=None, res_type='kaiser_fast')
        features = [(extract_features(audio, sr, n_mfcc), genre)]
        
        if augment:
            for aug_audio in augment_audio(audio, sr):
                features.append((extract_features(aug_audio, sr, n_mfcc), genre))
        
        return features
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return []

def load_audio_files(dataset_path, max_files_per_genre=100, n_mfcc=13, augment=False):
    """
    Load audio files from the dataset directory and extract features.

    Parameters:
        dataset_path (str): Path to the dataset directory.
        max_files_per_genre (int): Maximum number of files to load per genre.
        n_mfcc (int): Number of MFCC features to extract.
        augment (bool): Whether to apply data augmentation.

    Returns:
        tuple: Features (X) and labels (y) as numpy arrays.
    """
    X, y = [], []
    genres = os.listdir(dataset_path)
    for genre in genres:
        genre_path = os.path.join(dataset_path, genre)
        if not os.path.isdir(genre_path):
            continue

        files = os.listdir(genre_path)[:max_files_per_genre]
        for file in files:
            file_path = os.path.join(genre_path, file)
            try:
                audio, sr = librosa.load(file_path, sr=None)
                mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
                X.append(np.mean(mfcc.T, axis=0))
                y.append(genre)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
    
    return np.array(X), np.array(y)

def encode_labels(labels):
    """
    Encode string labels into integers.

    Parameters:
        labels (list): List of string labels.

    Returns:
        tuple: Encoded labels (as integers) and the LabelEncoder instance.
    """
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    return np.array(encoded_labels, dtype=int), label_encoder
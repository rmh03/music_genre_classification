import librosa
import numpy as np

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
        librosa.effects.pitch_shift(audio, sr, n_steps=2),  # Pitch up
        librosa.effects.pitch_shift(audio, sr, n_steps=-2),  # Pitch down
        librosa.effects.time_stretch(audio, rate=0.8),  # Slow down
        librosa.effects.time_stretch(audio, rate=1.2),  # Speed up
        audio + np.random.randn(len(audio)) * 0.005  # Add noise
    ]
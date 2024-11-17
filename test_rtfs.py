import torch
import torchaudio
import numpy as np

from src.model import RTFSNetwork
from src.encoders import STFT

def load_audio(path):
    audio_tensor, sr = torchaudio.load(path)
    audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
    target_sr = 16000
    if sr != target_sr:
        audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
    return audio_tensor

def load_object(path):
    """
    Load object from disk.

    Args:
        path (str): path to the object.
    Returns:
        data_object (Tensor):
    """
    if path.endswith('.npy'):
        data_object = torch.from_numpy(np.load(path))
    elif path.endswith('.npz'):
        with np.load(path) as data:
            data_object = torch.from_numpy(data[next(iter(data))])
    elif path.endswith('.pt') or path.endswith('.pth'):
        data_object = torch.load(path)
    
    return data_object.unsqueeze(0)

# sample paths
mix_path = "/Users/andreypetukhov/Documents/Машинное-обучение/AVSS/dla_dataset/audio/val/mix/00000083568_00411722726.wav"
embedding1_path = "/Users/andreypetukhov/Documents/Машинное-обучение/AVSS/dla_dataset/embeddings/00000083568.npz"
embedding2_path = "/Users/andreypetukhov/Documents/Машинное-обучение/AVSS/dla_dataset/embeddings/00411722726.npz"

# loading inputs
mix_audio = load_audio(mix_path)
embed1 = load_object(embedding1_path)
embed2 = load_object(embedding2_path)

encoder = STFT()
model = RTFSNetwork()

with torch.no_grad():
    complex_spectrogram = encoder.stft(mix_audio)
    output = model(complex_spectrogram, embed1, embed2)
import torch
import torch.nn as nn


class BaseEncoder:
    "To make beautiful typing in datasets"
    def __init__(self, *args, **kwargs):
        print("Encoder initialized")

    def stft(self, *args, **kwargs):
        raise NotImplementedError


class STFT(BaseEncoder):   
    def __init__(
        self,
        n_fft: int = 256,
        hop_length: int = 128
    ):
        super(STFT, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(self.n_fft)

    def stft(self, mix_audio: torch.Tensor):
        alpha = torch.stft(
            mix_audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True)

        complex_spec = torch.stack(
            [alpha.real, alpha.imag],
            dim = 1
        ).transpose(2, 3).contiguous() # [B, 2, T, F]

        return complex_spec
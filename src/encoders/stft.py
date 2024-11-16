import torch
import torch.nn as nn


class STFT:   
    def __init__(
            self,
            n_fft: int = 256,
            hop_length: int = 128
    ):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(self.n_fft)

    def stft(self, mix: torch.Tensor):
        alpha = torch.stft(
            mix,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True)

        complex_domain = torch.stack(
            [alpha.real, alpha.imag],
            dim = 1
        ).transpose(2, 3).contiguous() # [B, 2, T, F]

        return complex_domain
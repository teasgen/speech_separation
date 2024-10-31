import torch

class VoiceFilter:
    # TODO: add model
    def __init__(self, ...):
        self.embedder = ...

    def forward(self, mix_spectrogram, s1_video, s2_video):
        s1_embedding = self.embedder(s1_video, mix_spectrogram)
        s2_embedding = self.embedder(s2_video, mix_spectrogram)

        s1_pred = s1_embedding * mix_spectrogram
        s2_pred = s2_embedding * mix_spectrogram

        return {"s1_pred": s1_pred, "s2_pred": s2_pred}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
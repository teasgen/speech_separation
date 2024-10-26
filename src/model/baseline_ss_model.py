from torch import nn
from torch.nn import Sequential


class SSBaselineModel(nn.Module):
    """
    Simple MLP
    """

    def __init__(self, n_feats, fc_hidden=512, max_audio_len=32000):
        """
        Args:
            n_feats (int): number of input features.
            fc_hidden (int): number of hidden features.
        """
        super().__init__()

        self.common_model = Sequential(
            nn.Linear(in_features=n_feats, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=fc_hidden),
            nn.ReLU(),
        )
        self.mask_1 = Sequential(
            nn.Linear(in_features=fc_hidden, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=max_audio_len),
        )
        self.mask_2 = Sequential(
            nn.Linear(in_features=fc_hidden, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=max_audio_len),
        )

    def forward(self, mix_spectrogram, mix, **batch):
        common_features = self.common_model(mix_spectrogram.transpose(1, 2)) # B, T, C
        common_features = common_features.mean(1) # B, C
        mask_1 = self.mask_1(common_features)
        mask_2 = self.mask_2(common_features)
        s1_pred = mix * mask_1
        s2_pred = mix * mask_2
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

import numpy as np
import torch
import torch.nn as nn

from lipreader.lipreading.utils import load_model, load_json
from lipreader.lipreading.model import Lipreading
from lipreader.lipreading.dataloaders import get_preprocessing_pipelines

from src.utils.init_utils import init_lipreader


class VoiceFilter(nn.Module):
    def __init__(
            self,
            lipreader_path: str = "lrw_snv1x_tcn1x.pth",
            lipreader_config: str = "configs/lrw_snv1x_tcn1x.json",
            max_audio_len: int = 32000
    ):
        super(VoiceFilter, self).__init__()

        self.lipreader = init_lipreader(
            lipreader_config,
            lipreader_path
        )
        self.lipreader.eval()
        self.preprocessing_func = get_preprocessing_pipelines(modality="video")["test"]

        # gru layer for lip embeddings
        self.gru = nn.GRU(input_size=1024, hidden_size=256, num_layers=2, bidirectional=True, batch_first=True)
        
        # fc to obtain dvector
        self.fc_dvector = nn.Linear(256 * 2, 256)  # *2 bidirectional gru

        # cnn for mix spec
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 64, (1, 7), dilation=(1, 1)), nn.ReLU(),
            nn.Conv2d(64, 64, (7, 1), dilation=(1, 1)), nn.ReLU(),
            nn.Conv2d(64, 64, (5, 5), dilation=(1, 1)), nn.ReLU(),
            nn.Conv2d(64, 64, (5, 5), dilation=(2, 1)), nn.ReLU(),
            nn.Conv2d(64, 64, (5, 5), dilation=(4, 1)), nn.ReLU(),
            nn.Conv2d(64, 64, (5, 5), dilation=(8, 1)), nn.ReLU(),
            nn.Conv2d(64, 64, (5, 5), dilation=(16, 1)), nn.ReLU(),
            nn.Conv2d(64, 8, (1, 1), dilation=(1, 1)), nn.ReLU()
        )

        # lstm for mask
        self.lstm = nn.LSTM(input_size=8 + 256, hidden_size=400, batch_first=True)

        # fc for output
        self.fc1 = nn.Linear(400, 600)
        self.fc2 = nn.Linear(600, max_audio_len) 

    def forward(self, mix_spectrogram, s1_video: np.ndarray, s2_video: np.ndarray):
        s1_data = self.preprocessing_func(s1_video)
        s2_data = self.preprocessing_func(s2_video)

        s1_embedding = self.lipreader(torch.FloatTensor(s1_data)[None, None, :, :, :], lengths=[s1_data.shape[0]])  # 1 x T x C = (50, 1024)
        s2_embedding = self.lipreader(torch.FloatTensor(s2_data)[None, None, :, :, :], lengths=[s2_data.shape[0]])  # 1 x T x C = (50, 1024)

        # d-vectors
        s1_gru, _ = self.gru(s1_embedding)
        s2_gru, _ = self.gru(s2_embedding)
        
        s1_dvector = self.fc_dvector(s1_gru[:, -1, :])
        s2_dvector = self.fc_dvector(s2_gru[:, -1, :])

        mix_spectrogram = mix_spectrogram.unsqueeze(1)
        cnn_out = self.cnn_layers(mix_spectrogram)

        # concatenate dvectors to cnn out at each time frame
        cnn_out = cnn_out.squeeze(2).permute(0, 2, 1)  # (B, T, 8)
        s1_cnn_lstm_input = torch.cat((cnn_out, s1_dvector.expand_as(cnn_out)), dim=-1)
        s2_cnn_lstm_input = torch.cat((cnn_out, s2_dvector.expand_as(cnn_out)), dim=-1)

        # making masks
        s1_lstm_out, _ = self.lstm(s1_cnn_lstm_input)
        s2_lstm_out, _ = self.lstm(s2_cnn_lstm_input)

        s1_fc_out = self.fc2(nn.ReLU(self.fc1(s1_lstm_out)))
        s2_fc_out = self.fc2(nn.ReLU(self.fc1(s2_lstm_out)))

        # apply masks
        s1_pred = s1_fc_out * mix_spectrogram
        s2_pred = s2_fc_out * mix_spectrogram

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

import numpy as np
import torch
import torch.nn as nn

from src.lipreader.lipreading.dataloaders import get_preprocessing_pipelines
from src.lipreader.lipreading.model import Lipreading
from src.lipreader.lipreading.utils import load_json, load_model
from src.utils.init_utils import init_lipreader


class VoiceFilter(nn.Module):
    def __init__(
            self,
            input_size: int,
            lipreader_path: str,
            lipreader_config: str
    ):
        super(VoiceFilter, self).__init__()

        self.lipreader = init_lipreader(lipreader_config, lipreader_path)
        for param in self.lipreader.parameters():
            param.requires_grad = False # to use a frozen lipreader
        self.lipreader.eval()

        self.preprocessing_func = get_preprocessing_pipelines(modality="video")["test"]

        # gru layer for lip embeddings
        self.gru = nn.GRU(
            input_size=1024,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )

        # fc to obtain dvector
        self.fc_dvector = nn.Linear(256 * 2, 256)  # *2 bidirectional gru

        # cnn for mix spec, parameters from paper
        # 8 cnn layers
        self.cnn_layers = nn.Sequential(
            nn.ZeroPad2d((3, 3, 0, 0)),
            nn.Conv2d(1, 64, (1, 7), dilation=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ZeroPad2d((0, 0, 3, 3)),
            nn.Conv2d(64, 64, (7, 1), dilation=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ZeroPad2d((2, 2, 2, 2)),
            nn.Conv2d(64, 64, (5, 5), dilation=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ZeroPad2d((2, 2, 4, 4)),
            nn.Conv2d(64, 64, (5, 5), dilation=(2, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ZeroPad2d((2, 2, 8, 8)),
            nn.Conv2d(64, 64, (5, 5), dilation=(4, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ZeroPad2d((2, 2, 16, 16)),
            nn.Conv2d(64, 64, (5, 5), dilation=(8, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ZeroPad2d((2, 2, 32, 32)),
            nn.Conv2d(64, 64, (5, 5), dilation=(16, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 8, (1, 1), dilation=(1, 1)),
            nn.BatchNorm2d(8),
        )
        # padding for size consistency

        # lstm for mask
        self.lstm = nn.LSTM(input_size=8 * input_size + 256, hidden_size=400, batch_first=True)

        # fc for output
        self.fc1 = nn.Linear(400, 600)
        self.fc2 = nn.Linear(
            600, input_size
        )

    def forward(
        self, mix_magnitude, s1_video: torch.tensor, s2_video: torch.tensor, **batch
    ):
        # s1_video.size() = [B, T, H, W] = (10, 50, 96, 96)
        s1_data = torch.stack(
            [self.preprocessing_func(video) for video in s1_video], dim=0
        )
        s2_data = torch.stack(
            [self.preprocessing_func(video) for video in s2_video], dim=0
        )
        # preprocessing_func просто обрезает H и W и нормирует (см. код)
        # s1_data.size() = [B, T, H', W'] = [10, 50, 88, 88]

        # s1_data.unsqueeze(1).size() = [B, 1, T, H', W'] = [10, 1, 50, 88, 88]
        # TODO: is torch.no_grad() necessary?
        s1_embedding = self.lipreader(s1_data.unsqueeze(1), lengths=[50])
        s2_embedding = self.lipreader(s2_data.unsqueeze(1), lengths=[50])
        # TODO: merge two vectors, is it possible?..
        # считать lipreader от конкатенации быстрее?
        # s1_embedding.size() = [B, T, 1024] = [10, 50, 1024]

        # d-vectors
        s1_gru, _ = self.gru(s1_embedding)
        s2_gru, _ = self.gru(s2_embedding)
        # s1_gru.size() = [B, T, 512]

        s1_dvector = self.fc_dvector(s1_gru[:, -1, :])
        s2_dvector = self.fc_dvector(s2_gru[:, -1, :])
        # s1_dvector.size() = [B, 256] = [10, 256]

        # mix_magnitude.size() = [B, H, W] = [10, 201, 321]
        x = mix_magnitude.unsqueeze(1)
        x = self.cnn_layers(x)
        B, C, H, W = x.shape

        x = x.permute(0, 3, 2, 1).contiguous()  # [B, W, H, C] = [10, 321, 201, 8]
        x = x.view(B, W, -1)  # [B, W, H * C]

        # making masks
        outputs = {}
        for i, dvector in enumerate([s1_dvector, s2_dvector], start=1):
            dvector_expanded = dvector.unsqueeze(1).expand(
                -1, x.size(1), -1
            )  # [B, W, 256] = [10, 321, 256]
            concat = torch.cat(
                (x, dvector_expanded), dim=2
            )  # [B, W, H*C + 256] = [10, 321, 1280]
            lstm_out, _ = self.lstm(concat)  # [B, W, 400] = [10, 321, 400]
            mask = self.fc2(self.fc1(lstm_out))  # [B, W, H] = [10, 321, 201]
            mask = mask.permute(0, 2, 1)  # [B, H, W] = [10, 201, 321]
            outputs[f"s{i}_magnitude_pred"] = mask * mix_magnitude  # [B, H, W] = [10, 201, 321]
            # outputs[f"s{i}_mask"] = mask # TODO: maybe log mask (?)

        return outputs
    
    def train(self, mode=True):
        # keeping lipreader in eval mode
        super(VoiceFilter, self).train(mode)
        self.lipreader.eval()

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

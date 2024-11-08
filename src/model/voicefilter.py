import numpy as np
import torch
import torch.nn as nn

from src.lipreader.lipreading.dataloaders import get_preprocessing_pipelines
from src.lipreader.lipreading.model import Lipreading
from src.lipreader.lipreading.utils import load_json, load_model
from src.utils.init_utils import init_lipreader


class VoiceFilter(nn.Module):
    def __init__(self, input_size: int, lipreader_path: str, lipreader_config: str):
        super(VoiceFilter, self).__init__()

        self.lipreader = init_lipreader(lipreader_config, lipreader_path)
        for param in self.lipreader.parameters():
            param.requires_grad = False  # to use a frozen lipreader
        self.lipreader.eval()

        self.preprocessing_func = get_preprocessing_pipelines(modality="video")["test"]

        # gru layer for lip embeddings
        # TODO: refactor 1024 to arbitrary embed_size from lipreader
        self.gru = nn.GRU(
            input_size=1024,  # video embedding size per frame, needs adjustment for different lipreaders
            hidden_size=input_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )

        # fc to obtain dvector
        self.fc_dvector = nn.Linear(input_size * 2, input_size)  # *2 bidirectional gru

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
        self.lstm = nn.LSTM(
            input_size=9 * input_size, hidden_size=400, batch_first=True
        )

        # fc for output
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(400, 600),
            nn.Dropout2d(p=0.35),  # add for regularization
            nn.ReLU(),
            nn.Linear(600, input_size),
            nn.Sigmoid(),
        )  # changed this

    def forward(
        self, mix_magnitude, s1_video: torch.tensor, s2_video: torch.tensor, **batch
    ):
        # TODO: CHANGE TO PROPER SIZING AND CONCATENATING!!!
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
        # s1_dvector.size() = [B, input_size] = [10, input_size]

        # mix_magnitude.size() = [B, H, W] = [10, 201, 321]
        x = mix_magnitude.unsqueeze(1)
        x = self.cnn_layers(x)  # [B, C, H, W]
        B, C, H, W = x.shape

        x = x.permute(0, 3, 2, 1).contiguous()  # [B, W, H, C] = [10, 321, 201, 8]
        # x = x.view(B, W, -1)  # [B, W, H * C]
        x = x.view(B, C * H, W)  # [B, C * H, W]

        # making masks
        outputs = {}
        for i, dvector in enumerate([s1_dvector, s2_dvector], start=1):
            dvector_expanded = dvector.unsqueeze(-1).expand(
                -1, -1, W
            )  # [B, input_size, W]
            concat = torch.cat(
                (x, dvector_expanded), dim=1
            )  # [B, C*H + input_size, W] = [10, 1280, 321]
            lstm_out, _ = self.lstm(
                concat.transpose(1, 2)
            )  # [B, W, 400] = [10, 321, 400]
            mask = self.fc(lstm_out)  # [B, W, H] = [10, 321, 201]
            mask = mask.transpose(1, 2)  # [B, H, W] = [10, 201, 321]
            outputs[f"s{i}_spec_pred"] = (
                mask * mix_magnitude
            )  # [B, H, W] = [10, 201, 321]
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


class AttentionVoiceFilter(nn.Module):
    def __init__(self, input_size: int, embed_size: int, lipreader_path: str, lipreader_config: str):
        super(AttentionVoiceFilter, self).__init__()

        self.lipreader = init_lipreader(lipreader_config, lipreader_path)
        for param in self.lipreader.parameters():
            param.requires_grad = False
        self.lipreader.eval()

        self.preprocessing_func = get_preprocessing_pipelines(modality="video")["test"]

        self.initial_linear = nn.Linear(
            embed_size, input_size * 2
        )
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_size * 2, nhead=8),
            num_layers=2,
        )

        # for dvector
        self.fc_dvector = nn.Linear(input_size * 2, input_size)  # TODO: watch out for dims

        # cnn for spec
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

        # change lstm to transformer
        self.input_linear = nn.Linear(9 * input_size, 400)  # TODO: maybe not linear?
        self.transformer_encoder2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=400, nhead=8),
            num_layers=2,
        )

        # fc for mask
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(400, 600),
            nn.Dropout2d(p=0.35),  # regularization
            nn.ReLU(),
            nn.Linear(600, input_size),
            nn.Sigmoid(),
        )

    def forward(
        self, mix_magnitude, s1_video: torch.tensor, s2_video: torch.tensor, **batch
    ):
        s1_data = torch.stack(
            [self.preprocessing_func(video) for video in s1_video], dim=0
        )  # s1_data: [B, T, H', W'] = [10, 50, 88, 88]
        s2_data = torch.stack(
            [self.preprocessing_func(video) for video in s2_video], dim=0
        )

        s1_embedding = self.lipreader(s1_data.unsqueeze(1), lengths=[50])
        s2_embedding = self.lipreader(s2_data.unsqueeze(1), lengths=[50])
        # [B, T, 1024] = [10, 50, 1024]

        s1_embed = self.initial_linear(s1_embedding)  # [B, T, input_size * 2]
        s2_embed = self.initial_linear(s2_embedding)

        # TODO: check shuffling
        s1_attn = self.transformer_encoder(
            s1_embed.permute(1, 0, 2)
        )  # [T, B, input_size * 2]
        s1_attn = s1_attn.permute(1, 0, 2)  # [B, T, input_size * 2]

        s2_attn = self.transformer_encoder(
            s2_embed.permute(1, 0, 2)
        )
        s2_attn = s2_attn.permute(1, 0, 2)

        s1_attn = s1_attn.mean(dim=1)  # [B, input_size * 2]
        s2_attn = s2_attn.mean(dim=1)

        s1_dvector = self.fc_dvector(s1_attn)  # [B, input_size]
        s2_dvector = self.fc_dvector(s2_attn)

        # mix_magnitude: [B, H, W]
        x = mix_magnitude.unsqueeze(1)  # [B, 1, H, W] = [10, 1, 201, 321]
        x = self.cnn_layers(x)  # [B, 8, H, W]
        B, C, H, W = x.shape  # C=8

        x = x.permute(0, 3, 2, 1).contiguous()  # [B, W, H, C]
        x = x.view(B, W, -1)  # [B, W, C * H] = [B, W, 8 * H]

        # making masks
        outputs = {}
        for i, dvector in enumerate([s1_dvector, s2_dvector], start=1):
            dvector_expanded = dvector.unsqueeze(1).expand(
                -1, W, -1
            )  # [B, W, input_size]

            concat = torch.cat(
                (x, dvector_expanded), dim=2
            )  # [B, W, 8*H + input_size]

            # Преобразуем размерность и применяем Transformer Encoder
            concat = self.input_linear(concat)  # [B, W, 400]

            concat = concat.permute(1, 0, 2)  # [W, B, 400]
            transformer_out = self.transformer_encoder2(concat)
            transformer_out = transformer_out.permute(1, 0, 2)  # [B, W, 400]

            mask = self.fc(transformer_out)  # [B, W, input_size]
            mask = mask.transpose(1, 2)  # [B, input_size, W] -> [B, H, W]
            outputs[f"s{i}_spec_pred"] = (
                mask * mix_magnitude
            )  # [B, H, W] = [10, 201, 321]

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

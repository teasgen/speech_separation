import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .dprnn import OverlapAdd, SplitToFolds


class TransformerDPRNN(nn.Module):
    """
    https://arxiv.org/pdf/1910.06379
    """

    def __init__(self, num_features, hidden_dim, num_heads, dropout, bidir=True) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=num_features,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln1 = nn.LayerNorm(num_features)
        self.rnn = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            dropout=1,
            bidirectional=bidir,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim * (bidir + 1), num_features),
        )
        self.ln2 = nn.LayerNorm(num_features)

    def forward(self, data):
        """
        Args:
            data (batch_size, ts, num_features)
        Returns:
            output (batch_size, ts, num_features)
        """
        self.rnn.flatten_parameters()

        residual = data
        x = self.mha(data, data, data, need_weights=False)[0] + residual  # saves shape
        x = self.ln1(x)
        residual = x
        x = self.rnn(x)[0]  # saves shape
        x = self.ffn(x) + residual
        x = self.ln2(x)
        return x


class DPTNBlock(nn.Module):
    def __init__(self, num_features, hidden_dim, num_heads, dropout, bidir):
        super().__init__()

        self.intra_chunk_block = TransformerDPRNN(num_features, hidden_dim, num_heads, dropout=dropout)
        self.inter_chunk_block = TransformerDPRNN(num_features, hidden_dim, num_heads, dropout=dropout, bidir=bidir)

    def forward(self, data):
        """
        Args:
            data (batch_size, num_features, S, chunk_size)
        Returns:
            output (batch_size, num_features, S, chunk_size)
        """
        _, num_features, S, chunk_size = data.shape
        # in chunk
        x = rearrange(data, "b n s chunk_size -> (b s) chunk_size n", n=num_features, chunk_size=chunk_size, s=S)
        x = self.intra_chunk_block(x)
        # across chunks
        x = rearrange(x, "(b s) chunk_size n -> (b chunk_size) s n", n=num_features, chunk_size=chunk_size, s=S)
        output = self.inter_chunk_block(x)

        output = rearrange(output, "(b chunk_size) s n -> b n s chunk_size", n=num_features, chunk_size=chunk_size, s=S)

        return output


class DPTN(nn.Module):
    def __init__(
        self, num_features, hidden_dim, num_blocks=6, chunk_size=10, step_size=5, num_heads=4, dropout=0.1, bidir=False
    ):
        super().__init__()

        self.chunk_size = chunk_size

        self.segmenter = SplitToFolds(chunk_size, step_size)
        self.overladd = OverlapAdd(chunk_size, step_size)

        self.model = nn.Sequential()

        for _ in range(num_blocks):
            self.model.append(DPTNBlock(num_features, hidden_dim, num_heads, dropout, bidir=bidir))

        self.speakers_separation = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(num_features, 2 * num_features, 1),
        )

        self.output_gate = nn.Sequential(
            nn.Conv1d(num_features, num_features, 1),
            nn.Sigmoid(),
        )

        self.output = nn.Sequential(
            nn.Conv1d(num_features, num_features, 1),
            nn.Tanh(),
        )

        self.postprocessing = nn.Sequential(
            nn.ReLU(),
        )

    def forward(self, data):
        """
        Args:
            data (batch_size, num_features, ts)
        Returns:
            output [(batch_size, num_features, ts), (batch_size, num_features, ts)]
        """
        bs, num_features, ts = data.shape
        x = self.segmenter(data)  # -> (batch_size, num_features, S, chunk_size)

        output = self.model(x)  # -> (batch_size, num_features, S, chunk_size)

        output = self.speakers_separation(output)  # -> (batch_size, 2 * num_features, S, chunk_size)

        output = self.overladd(output)  # -> (batch_size, 2 * num_features, ts)

        # dirty hack
        padding_needed = ts - output.shape[-1]
        pad_left = padding_needed // 2
        pad_right = padding_needed - pad_left
        output = torch.nn.functional.pad(output, (pad_left, pad_right))

        output = output.view(bs, 2, num_features, ts).transpose(0, 1)

        output = [self.postprocessing(self.output(x) * self.output_gate(x)) for x in output]

        return output


class DPTNEncDec(nn.Module):
    """
    Arch:
    1. Compressing wav using conv1d
    2. Encodes latent wav representation & separate for 2 speakers
    3. Decompressing to original length using transpose conv1d
    """

    def __init__(
        self,
        num_features=64,
        kernel_size_enc=2,
        hidden_dim=32,
        num_blocks=6,
        chunk_size=10,
        step_size=5,
        num_heads=4,
        dropout=0.1,
        bidir=True,
    ):
        super().__init__()
        # 50% overlap
        self.encoder = nn.Conv1d(1, num_features, kernel_size=kernel_size_enc, stride=kernel_size_enc // 2, bias=False)
        self.dprnn = DPTN(
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            chunk_size=chunk_size,
            step_size=step_size,
            num_heads=num_heads,
            dropout=dropout,
            bidir=bidir,
        )
        self.decoder = nn.ConvTranspose1d(
            num_features, 1, kernel_size=kernel_size_enc, stride=kernel_size_enc // 2, bias=False
        )

    def forward(self, mix, **batch):
        mix = mix.unsqueeze(1)  # (batch_size, ts) -> (batch_size, 1, ts) for correct channels dimention
        encoded = self.encoder(mix)
        hidden = self.dprnn(encoded)  # list of 2
        preds = []
        for x in hidden:
            x = self.decoder(x * encoded)
            padding_needed = mix.shape[-1] - x.shape[-1]
            pad_left = padding_needed // 2
            pad_right = padding_needed - pad_left
            x = torch.nn.functional.pad(x, (pad_left, pad_right))
            preds.append(x.squeeze(1))
        return {"s1_pred": preds[0], "s2_pred": preds[1]}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum([p.numel() for p in self.parameters() if p.requires_grad])

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class IntraChunkRNN(nn.Module):
    """
    The intra-chunk bi-directional RNN is first applied to individual chunks in parallel to process local information.
    """
    def __init__(self, num_features, hidden_channels, norm=True):
        super().__init__()

        self.num_features, self.hidden_channels = num_features, hidden_channels
        self.norm = norm

        # always bidirectional
        num_directions = 2
        self.rnn = nn.LSTM(input_size=num_features, hidden_size=hidden_channels, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(num_directions * hidden_channels, num_features)

        if self.norm:
            self.norm1d = nn.LayerNorm(num_features) # TODO: change to global?

    def forward(self, input):
        """
        Args:
            input (batch_size, num_features, S, chunk_size)
        Returns:
            output (batch_size, num_features, S, chunk_size)
        """
        num_features = self.num_features
        batch_size, _, S, chunk_size = input.size()

        self.rnn.flatten_parameters()

        residual = input  # (batch_size, num_features, S, chunk_size)
        x = input.permute(0, 2, 3, 1).contiguous()  # -> (batch_size, S, chunk_size, num_features)
        x = rearrange(x, "b s chunk_size n -> (b s) chunk_size n", chunk_size=chunk_size, s=S)
        x, _ = self.rnn(x)
        x = self.fc(x)
        x = rearrange(x, "(b s) chunk_size n -> b (s chunk_size) n", chunk_size=chunk_size, s=S)
        if self.norm:
            x = self.norm1d(x)
        x = rearrange(x, "b (s chunk_size) n -> b n s chunk_size", chunk_size=chunk_size, s=S)
        output = x + residual

        return output


class InterChunkRNN(nn.Module):
    """
    The inter-chunk RNN is then applied across the chunks to capture global dependency.
    """
    def __init__(self, num_features, hidden_channels, bidir, norm=True):
        super().__init__()

        self.num_features, self.hidden_channels = num_features, hidden_channels
        self.norm = norm

        num_directions = 2 if bidir else 1

        self.rnn = nn.LSTM(input_size=num_features, hidden_size=hidden_channels, batch_first=True, bidirectional=bidir)
        self.fc = nn.Linear(num_directions * hidden_channels, num_features)

        if self.norm:
            self.norm1d = nn.LayerNorm(num_features) # TODO: change to global?

    def forward(self, input):
        """
        Args:
            input (batch_size, num_features, S, chunk_size)
        Returns:
            output (batch_size, num_features, S, chunk_size)
        """
        num_features = self.num_features
        _, _, S, chunk_size = input.size()

        self.rnn.flatten_parameters()

        residual = input  # (batch_size, num_features, S, chunk_size)
        x = input.permute(0, 3, 2, 1).contiguous()  # -> (batch_size, chunk_size, S, num_features)
        x = rearrange(x, "b chunk_size s n -> (b chunk_size) s n", chunk_size=chunk_size, s=S)
        x, _ = self.rnn(x)
        x = self.fc(x)
        x = rearrange(x, "(b chunk_size) s n -> b (chunk_size s) n", chunk_size=chunk_size, s=S)
        if self.norm:
            x = self.norm1d(x)
        x = rearrange(x, "b (chunk_size s) n -> b n s chunk_size", chunk_size=chunk_size, s=S)

        output = x + residual

        return output


class DPRNNBlock(nn.Module):
    """
    Read https://arxiv.org/pdf/1910.06379 2.1. Model Design for more information
    """
    def __init__(self, num_features, hidden_channels, bidir, norm=True):
        super().__init__()

        self.intra_chunk_block = IntraChunkRNN(num_features, hidden_channels, norm=norm)
        self.inter_chunk_block = InterChunkRNN(num_features, hidden_channels, norm=norm, bidir=bidir)

    def forward(self, input):
        """
        Args:
            input (batch_size, num_features, S, chunk_size)
        Returns:
            output (batch_size, num_features, S, chunk_size)
        """
        x = self.intra_chunk_block(input)
        output = self.inter_chunk_block(x)

        return output


class SplitToFolds(nn.Module):
    def __init__(self, chunk_size, step_size):
        super().__init__()

        self.chunk_size, self.step_size = chunk_size, step_size

    def forward(self, input):
        """
        Args:
            input (batch_size, num_features, n_frames)
        Returns:
            output (batch_size, num_features, S, chunk_size): S is length of global output, where S = (n_frames-chunk_size)//hop_size + 1
        """
        chunk_size, hop_size = self.chunk_size, self.step_size
        batch_size, num_features, n_frames = input.size()

        input = input.view(batch_size, num_features, n_frames, 1)
        x = F.unfold(input, kernel_size=(chunk_size, 1), stride=(hop_size, 1))
        x = x.view(batch_size, num_features, chunk_size, -1)
        output = x.permute(0, 1, 3, 2).contiguous()
        return output


class OverlapAdd(nn.Module):
    def __init__(self, chunk_size, step_size):
        super().__init__()

        self.chunk_size, self.step_size = chunk_size, step_size

    def forward(self, input):
        """
        Args:
            input: (batch_size, num_features, S, chunk_size)
        Returns:
            output: (batch_size, num_features, n_frames)
        """
        chunk_size, hop_size = self.chunk_size, self.step_size
        batch_size, num_features, S, chunk_size = input.size()
        n_frames = (S - 1) * hop_size + chunk_size

        x = input.permute(0, 1, 3, 2).contiguous()  # -> (batch_size, num_features, chunk_size, S)
        x = x.view(batch_size, num_features * chunk_size, S)  # -> (batch_size, num_features*chunk_size, S)
        output = F.fold(
            x, kernel_size=(chunk_size, 1), stride=(hop_size, 1), output_size=(n_frames, 1)
        )  # -> (batch_size, num_features, n_frames, 1)
        output = output.squeeze(dim=3)

        return output


class DPRNN(nn.Module):
    def __init__(self, num_features, hidden_channels, num_blocks=6, chunk_size=10, step_size=5, norm=True, bidir=False):
        super().__init__()

        self.chunk_size = chunk_size

        self.segmenter = SplitToFolds(chunk_size, step_size)
        self.overladd = OverlapAdd(chunk_size, step_size)

        self.model = nn.Sequential()

        for _ in range(num_blocks):
            self.model.append(DPRNNBlock(num_features, hidden_channels, norm=norm, bidir=bidir))

        self.speakers_separation = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(num_features, 2 * num_features, 1),
        )

        self.postprocessing = nn.Sequential(
            nn.Conv1d(num_features, num_features, 1),
            nn.ReLU(),
        )

    def forward(self, input):
        """
        Args:
            input (batch_size, num_features, ts)
        Returns:
            output [(batch_size, num_features, ts), (batch_size, num_features, ts)]
        """
        bs, num_features, ts = input.shape
        x = self.segmenter(input)  # -> (batch_size, num_features, S, chunk_size)

        output = self.model(x)  # -> (batch_size, num_features, S, chunk_size)

        output = self.speakers_separation(output)  # -> (batch_size, 2 * num_features, S, chunk_size)

        output = self.overladd(output)  # -> (batch_size, 2 * num_features, ts)

        # dirty hack
        padding_needed = ts - output.shape[-1]
        pad_left = padding_needed // 2
        pad_right = padding_needed - pad_left
        output = torch.nn.functional.pad(output, (pad_left, pad_right))

        output = output.view(bs, 2, num_features, ts).transpose(0, 1)

        output = [self.postprocessing(x) for x in output]

        return output


class DPRNNEncDec(nn.Module):
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
        hidden_channels=32,
        num_blocks=6,
        chunk_size=10,
        step_size=5,
        norm=True,
        bidir=True,
    ):
        super().__init__()
        # 50% overlap
        self.encoder = nn.Conv1d(1, num_features, kernel_size=kernel_size_enc, stride=kernel_size_enc // 2, bias=False)
        self.dprnn = DPRNN(
            num_features=num_features,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            chunk_size=chunk_size,
            step_size=step_size,
            norm=norm,
            bidir=bidir,
        )
        self.decoder = nn.ConvTranspose1d(
            num_features, 1, kernel_size=kernel_size_enc, stride=kernel_size_enc // 2, bias=False
        )

    def forward(self, mix, **batch):
        mix = mix.unsqueeze(1)
        encoded = self.encoder(mix)
        hidden = self.dprnn(encoded)  # list of 2
        preds = []
        for x in hidden:
            x = self.decoder(x)
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
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info

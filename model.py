"""CNN-SE + BiLSTM + Multi-Head Self-Attention model for Kannada word OCR (CTC)."""
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ImprovedCNNBackbone(nn.Module):
    """CNN with SE blocks. Turns a 1x64x256 image into a 64-step feature sequence."""

    def __init__(self, in_channels=1, out_planes=512):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 64x256 -> 32x128
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            SEBlock(64),
            nn.MaxPool2d(2, 2),

            # Block 2: 32x128 -> 16x64
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            SEBlock(128),
            nn.MaxPool2d(2, 2),

            # Block 3: 16x64 -> 8x64 (pool height only, keep width = time steps)
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            SEBlock(256),
            nn.MaxPool2d((2, 1), (2, 1)),

            # Block 4: 8x64 -> 4x64
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            SEBlock(512),
            nn.MaxPool2d((2, 1), (2, 1)),

            # Final: 4x64 -> 1x64
            nn.Conv2d(512, out_planes, 3, 1, 1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, None)),
        )

    def forward(self, x):
        out = self.features(x)      # (B, C, 1, W)
        out = out.squeeze(2)        # (B, C, W)
        out = out.permute(2, 0, 1)  # (T=W, B, C)
        return out


class BidirectionalLSTMBlock(nn.Module):
    """Single-layer BiLSTM -> linear projection -> LayerNorm."""

    def __init__(self, input_size, hidden_size, dropout=0.3):
        super().__init__()
        # Note: nn.LSTM only applies `dropout` between stacked layers, so on a
        # single-layer LSTM it has no effect. It is left out here to avoid the
        # PyTorch warning; behaviour and checkpoints are unchanged.
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=False)
        self.proj = nn.Linear(hidden_size * 2, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        T, B, H = recurrent.size()
        out = self.proj(recurrent.reshape(T * B, H))
        out = out.reshape(T, B, -1)
        out = self.layer_norm(out)
        return out


class MultiHeadSelfAttention(nn.Module):
    """Transformer-style multi-head self-attention block with residuals."""

    def __init__(self, hidden_size, num_heads=8, dropout=0.2):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads,
                                         dropout=dropout, batch_first=False)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.mha(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class EnhancedKannadaOCRModel(nn.Module):
    """CNN-SE backbone -> stacked BiLSTM -> multi-head self-attention -> CTC classifier."""

    def __init__(self, num_classes, cnn_out=512, rnn_hidden=384,
                 n_rnn_layers=3, num_heads=8, dropout=0.3):
        super().__init__()
        self.cnn = ImprovedCNNBackbone(in_channels=1, out_planes=cnn_out)

        rnn_blocks = []
        in_size = cnn_out
        for _ in range(n_rnn_layers):
            rnn_blocks.append(BidirectionalLSTMBlock(in_size, rnn_hidden, dropout))
            in_size = rnn_hidden
        self.rnn_stack = nn.Sequential(*rnn_blocks)

        self.attention = MultiHeadSelfAttention(rnn_hidden, num_heads, dropout)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(rnn_hidden, num_classes + 1)  # +1 for the CTC blank

    def forward(self, x):
        conv_feats = self.cnn(x)
        rnn_out = self.rnn_stack(conv_feats)
        attn_out = self.attention(rnn_out)

        T, B, H = attn_out.size()
        out = self.dropout(attn_out)
        out = out.reshape(T * B, H)
        out = self.fc(out)
        out = out.reshape(T, B, -1)
        return out.log_softmax(2)  # (T, B, num_classes + 1)

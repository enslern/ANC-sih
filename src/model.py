import torch
import torch.nn as nn


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        pad = (kernel_size - 1) * dilation
        super().__init__(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.trim = pad

    def forward(self, x):
        x = super().forward(x)
        if self.trim > 0:
            x = x[..., :-self.trim]
        return x


class MaskNet(nn.Module):
    """Predicts a real-valued spectral mask (0-1) per frequency bin per frame.
    Input: magnitude spectrogram (B, F, T). Output: mask (B, F, T).
    """
    def __init__(self, n_freq=257, hidden=128, gru_layers=1):
        super().__init__()
        self.enc = nn.Sequential(
            CausalConv1d(n_freq, hidden, kernel_size=5, dilation=1),
            nn.BatchNorm1d(hidden),
            nn.PReLU(),
            CausalConv1d(hidden, hidden, kernel_size=5, dilation=2),
            nn.BatchNorm1d(hidden),
            nn.PReLU(),
        )
        self.gru = nn.GRU(hidden, hidden, num_layers=gru_layers, batch_first=True)
        self.dec = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.PReLU(),
            nn.Linear(hidden, n_freq),
            nn.Sigmoid(),
        )

    def forward(self, mag):
        # mag: (B, F, T)
        x = self.enc(mag)              # (B, hidden, T)
        x = x.transpose(1, 2)          # (B, T, hidden)
        x, _ = self.gru(x)             # (B, T, hidden)
        mask = self.dec(x)             # (B, T, F)
        mask = mask.transpose(1, 2)    # (B, F, T)
        return mask


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    m = MaskNet()
    print("params:", count_params(m))
    x = torch.randn(2, 257, 100)
    y = m(x)
    print(y.shape)
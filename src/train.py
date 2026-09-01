import os
import glob
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import SAMPLE_RATE, GEN_DIR
from src.model import MaskNet, count_params
from src.metrics import evaluate

N_FFT = 512
HOP = 160  # 10ms hop at 16kHz


class NoisyCleanDataset(Dataset):
    def __init__(self, split="train", val_ratio=0.1):
        noisy_files = sorted(glob.glob(os.path.join(GEN_DIR, "noisy", "*.wav")))
        clean_files = sorted(glob.glob(os.path.join(GEN_DIR, "clean", "*.wav")))
        assert len(noisy_files) == len(clean_files) and len(noisy_files) > 0, \
            "Run dataset_gen.py first."

        n_val = max(1, int(len(noisy_files) * val_ratio))
        if split == "train":
            self.noisy = noisy_files[:-n_val]
            self.clean = clean_files[:-n_val]
        else:
            self.noisy = noisy_files[-n_val:]
            self.clean = clean_files[-n_val:]

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, idx):
        noisy, _ = sf.read(self.noisy[idx], dtype="float32")
        clean, _ = sf.read(self.clean[idx], dtype="float32")
        return torch.from_numpy(noisy), torch.from_numpy(clean)


def stft(x):
    window = torch.hann_window(N_FFT, device=x.device)
    return torch.stft(x, n_fft=N_FFT, hop_length=HOP, window=window, return_complex=True)


def istft(X, length):
    window = torch.hann_window(N_FFT, device=X.device)
    return torch.istft(X, n_fft=N_FFT, hop_length=HOP, window=window, length=length)


def si_snr_loss(est, target, eps=1e-8):
    target = target - target.mean(dim=-1, keepdim=True)
    est = est - est.mean(dim=-1, keepdim=True)
    s_target = (torch.sum(est * target, dim=-1, keepdim=True) /
                (torch.sum(target ** 2, dim=-1, keepdim=True) + eps)) * target
    e_noise = est - s_target
    ratio = torch.sum(s_target ** 2, dim=-1) / (torch.sum(e_noise ** 2, dim=-1) + eps)
    return -10 * torch.log10(ratio + eps).mean()


def stft_mag_loss(est, target):
    Est = stft(est)
    Tgt = stft(target)
    return F.l1_loss(torch.abs(Est), torch.abs(Tgt))


def train(epochs=20, batch_size=8, lr=1e-3, device=None, ckpt_path="checkpoints/masknet.pt"):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    train_ds = NoisyCleanDataset("train")
    val_ds = NoisyCleanDataset("val")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)

    model = MaskNet(n_freq=N_FFT // 2 + 1).to(device)
    print("Model params:", count_params(model))
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_stoi = -1
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for noisy, clean in tqdm(train_loader, desc=f"epoch {epoch}"):
            noisy, clean = noisy.to(device), clean.to(device)
            Noisy = stft(noisy)
            mag = torch.abs(Noisy)
            mask = model(mag)
            Enh = Noisy * mask
            enhanced = istft(Enh, length=noisy.shape[-1])

            loss = si_snr_loss(enhanced, clean) + 0.5 * stft_mag_loss(enhanced, clean)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # validation
        model.eval()
        stois = []
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                Noisy = stft(noisy)
                mask = model(torch.abs(Noisy))
                enhanced = istft(Noisy * mask, length=noisy.shape[-1])
                for b in range(enhanced.shape[0]):
                    e = enhanced[b].cpu().numpy()
                    c = clean[b].cpu().numpy()
                    try:
                        m = evaluate(c, e, SAMPLE_RATE)
                        stois.append(m["STOI"])
                    except Exception:
                        pass
        mean_stoi = float(np.mean(stois)) if stois else 0.0
        print(f"epoch {epoch}: loss={avg_loss:.4f} val_STOI={mean_stoi:.3f}")

        if mean_stoi > best_stoi:
            best_stoi = mean_stoi
            torch.save(model.state_dict(), ckpt_path)
            print("  saved best checkpoint ->", ckpt_path)

    print("Training done. Best val STOI:", best_stoi)


if __name__ == "__main__":
    train()
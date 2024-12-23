import os
import random
from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass
class ModelConfig:
    n_input: int = 250
    n_feat: int = 1
    n_ext: int = 100
    learning_rate: float = 3e-4
    batch_size: int = 32
    epochs: int = 5000
    seed: int = 42
    physics_loss_weight: float = 10.0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class PinnModel(nn.Module):
    def __init__(self, n_input, n_feat=1, n_ext=100):
        super().__init__()
        self.n_input = n_input
        self.n_feat = n_feat
        self.n_ext = n_ext

        self.cnn_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1),
            nn.Flatten(),
        )

        cnn_output_size = self._get_cnn_output_size(n_input)
        self.feat_ext = nn.Linear(cnn_output_size, n_ext)

        self.output_layers = nn.Sequential(
            nn.Linear(n_feat * 3 + n_ext, 60), nn.ReLU(), nn.Linear(60, n_feat)
        )

    def _get_cnn_output_size(self, n_input):
        x = torch.randn(1, 1, n_input)
        x = self.cnn_layers(x)
        return x.shape[1]

    def forward(self, beat, feat1, feat2, feat3):
        x = beat.unsqueeze(1)  # (batch_size, 1, n_input)
        x = self.cnn_layers(x)
        x = self.feat_ext(x)

        combined = torch.cat([x, feat1, feat2, feat3], dim=1)
        return self.output_layers(combined)


class PinnDataset(Dataset):
    def __init__(self, df, beat_key, feat_keys, out_key, transform=True):
        self.beat_data = df[beat_key].values
        self.features = [df[key].values for key in feat_keys]
        self.targets = df[out_key].values

        if transform:
            self.beat_scaler = StandardScaler()
            self.feat_scalers = [StandardScaler() for _ in feat_keys]
            self.target_scaler = StandardScaler()

            self.beat_data = np.vstack(self.beat_data)
            self.beat_data = self.beat_scaler.fit_transform(self.beat_data).reshape(
                len(df), -1
            )

            for i, feature in enumerate(self.features):
                self.features[i] = self.feat_scalers[i].fit_transform(
                    feature.reshape(-1, 1)
                )

            self.targets = self.target_scaler.fit_transform(self.targets.reshape(-1, 1))

    def __len__(self):
        return len(self.beat_data)

    def __getitem__(self, idx):
        beat = torch.FloatTensor(self.beat_data[idx])
        feats = [torch.FloatTensor(feat[idx]) for feat in self.features]
        target = torch.FloatTensor(self.targets[idx])
        return beat, feats[0], feats[1], feats[2], target


class PinnTrainer:
    def __init__(self, model, device, config):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    def compute_physics_loss(self, pred, feat1, feat2, feat3):
        grad_feat1 = torch.autograd.grad(pred.sum(), feat1, create_graph=True)[0]
        grad_feat2 = torch.autograd.grad(pred.sum(), feat2, create_graph=True)[0]
        grad_feat3 = torch.autograd.grad(pred.sum(), feat3, create_graph=True)[0]

        physics_pred = (
            pred[:-1]
            + grad_feat1[:-1] * (feat1[1:] - feat1[:-1])
            + grad_feat2[:-1] * (feat2[1:] - feat2[:-1])
            + grad_feat3[:-1] * (feat3[1:] - feat3[:-1])
        )

        return F.mse_loss(physics_pred, pred[1:])

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0

        for batch in train_loader:
            beat, feat1, feat2, feat3, target = [b.to(self.device) for b in batch]

            feat1.requires_grad = True
            feat2.requires_grad = True
            feat3.requires_grad = True

            self.optimizer.zero_grad()

            pred = self.model(beat, feat1, feat2, feat3)

            mse_loss = F.mse_loss(pred, target)
            physics_loss = self.compute_physics_loss(pred, feat1, feat2, feat3)
            loss = mse_loss + self.config.physics_loss_weight * physics_loss

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)


def main():
    config = ModelConfig()
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_pickle("data_demo_pinn_bioz_bp", compression="gzip")
    dataset = PinnDataset(
        df=df, beat_key="bioz_beats", feat_keys=["phys_feat_1", "phys_feat_2", "phys_feat_3"], out_key="sys"
    )
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    model = PinnModel(
        n_input=dataset.beat_data.shape[1], n_feat=config.n_feat, n_ext=config.n_ext
    )
    trainer = PinnTrainer(model, device, config)

    for epoch in range(config.epochs):
        train_loss = trainer.train_epoch(train_loader)
        print(f"Epoch {epoch + 1}/{config.epochs}, Train Loss: {train_loss:.4f}")


if __name__ == "__main__":
    main()

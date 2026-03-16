import torch
import torch.nn as nn


class BioEncoder(nn.Module):
    def __init__(self, out_features=32):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),

            nn.Flatten(),
            nn.Linear(32, out_features)
        )

    def forward(self, x):

        if x.dim() == 2:
            x = x.unsqueeze(1)

        return self.net(x)


class FedBioXAI(nn.Module):

    def __init__(self):
        super().__init__()

        self.ecg_enc = BioEncoder(32)
        self.eda_enc = BioEncoder(32)
        self.emg_enc = BioEncoder(32)

        # Attention layer to weight modalities
        self.attention = nn.Sequential(
            nn.Linear(32*3, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(32 + 12, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3)
        )

    def forward(self, inputs):

        e_ecg = self.ecg_enc(inputs['ecg'])
        e_eda = self.eda_enc(inputs['eda'])
        e_emg = self.emg_enc(inputs['emg'])

        context = inputs['context']

        # Stack embeddings
        modalities = torch.stack([e_ecg, e_eda, e_emg], dim=1)

        # Compute attention weights
        concat = torch.cat([e_ecg, e_eda, e_emg], dim=1)
        weights = self.attention(concat)

        weights = weights.unsqueeze(-1)

        # Weighted modality fusion
        fused = torch.sum(modalities * weights, dim=1)

        combined = torch.cat([fused, context], dim=1)

        return self.classifier(combined)
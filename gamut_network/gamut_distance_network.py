"""Differentiable color-gamut distance networks."""

import torch
import torch.nn as nn


class ConditionalGamutNetwork(nn.Module):

    def __init__(self, input_dim=39, p_vector_dim=12):
        super().__init__()

        assert input_dim == 39, f"Expected input_dim=39 (3 + 36), got {input_dim}"
        assert p_vector_dim == 12, f"Expected p_vector_dim=12, got {p_vector_dim}"

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
        )

        self.gamut_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

        self.p_vector_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, p_vector_dim),
            nn.Sigmoid()
        )

        self.distance_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Softplus()
        )

    def forward(self, x):
        features = self.encoder(x)

        gamut_score = self.gamut_head(features)
        p_vector = self.p_vector_head(features)
        distance = self.distance_head(features)

        return gamut_score, p_vector, distance


class ColorMixingControlNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(256, 768)

    def forward(self, x):
        """
        Args:
            x:  (N, 3, 64, 64), encoded color-mixing image.

        Returns:
            embedding: (N, 768)
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)  # 32×32

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)  # 16×16

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)  # 8×8

        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)  # 4×4

        x = self.global_pool(x)  # (N, 256, 1, 1)
        x = x.view(x.size(0), -1)  # (N, 256)

        embedding = self.fc(x)  # (N, 768)

        return embedding

if __name__ == "__main__":
    print("=" * 80)
    print("Model smoke test")
    print("=" * 80)


    model = ConditionalGamutNetwork(input_dim=39, p_vector_dim=12)

    batch_size = 32
    x = torch.randn(batch_size, 39)  # [C(3), T(36)]

    gamut_score, p_vector, distance = model(x)

    print(f"  input:       {x.shape}")
    print(f"  gamut_score:  {gamut_score.shape}")
    print(f"  p_vector:   {p_vector.shape}")
    print(f"  distance:    {distance.shape}")
    print("  passed")

    print("\n[2] ColorMixingControlNet")
    cm_model = ColorMixingControlNet()

    x_img = torch.randn(batch_size, 3, 64, 64)
    embedding = cm_model(x_img)

    print(f"  input:      {x_img.shape}")
    print(f"  embedding:  {embedding.shape}")
    print("  passed")

    print("\n" + "=" * 80)
    print("All checks passed.")
    print("=" * 80)

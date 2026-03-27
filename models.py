# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 11:01:36 2026

"""

# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from utils import pick_nhead


class CrossAttention(nn.Module):
    def __init__(self, dim, nhead=4, dropout=0.1):
        super().__init__()
        nhead = pick_nhead(dim, nhead)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=nhead,
            batch_first=True,
            dropout=dropout
        )

    def forward(self, q, k, v):
        out, _ = self.attn(q, k, v, need_weights=False)
        return out


class GateNetwork(nn.Module):
    def __init__(self, input_dim, num_experts, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts)
        )

    def forward(self, x):
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)


class SharedCNNBackbone(nn.Module):
    """
    Input  : (B, T, N)
    Output : (B, L, H)
    """
    def __init__(self, n_sensors=12, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.n_sensors = int(n_sensors)

        self.features = nn.Sequential(
            nn.Conv1d(self.n_sensors, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Conv1d(128, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f"Expected x dim=3 (B,T,N), got {tuple(x.shape)}")

        _, _, n_channels = x.shape
        if n_channels != self.n_sensors:
            raise ValueError(f"Expected N={self.n_sensors}, got N={n_channels}")

        x = x.permute(0, 2, 1)   # (B,N,T)
        h = self.features(x)     # (B,H,L)
        h = h.permute(0, 2, 1)   # (B,L,H)
        return h


class CNNOnly(nn.Module):
    def __init__(self, n_sensors=12, hidden_dim=128, num_classes=6, dropout=0.2):
        super().__init__()
        self.backbone = SharedCNNBackbone(
            n_sensors=n_sensors,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h = self.backbone(x)
        feat = h.mean(dim=1)
        logits = self.fc(feat)
        return feat, logits, None


class CNN1Transformer(nn.Module):
    def __init__(
        self,
        n_sensors=12,
        cnn_emb=128,
        hidden_dim=128,
        temporal_dim=10000,
        kernel_size=15,
        stride=50,
        padding=7,
        dilation=1,
        num_classes=6,
        nhead=4,
        dropout=0.2,
        pooling="mean"
    ):
        super().__init__()
        self.pooling = pooling
        self.hidden_dim = int(hidden_dim)

        self.backbone = SharedCNNBackbone(
            n_sensors=n_sensors,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        self.ln = nn.LayerNorm(self.hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=pick_nhead(self.hidden_dim, nhead),
                batch_first=True,
                dropout=dropout
            ),
            num_layers=1
        )

        self.fc = nn.Linear(self.hidden_dim, num_classes)

    def pool_tokens(self, x):
        return x.max(dim=1).values if self.pooling == "max" else x.mean(dim=1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.ln(x)
        x = self.transformer(x)
        feat = self.pool_tokens(x)
        logits = self.fc(feat)
        return feat, logits, None


class TemporalExpert(nn.Module):
    def __init__(self, hidden_dim, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        nhead = pick_nhead(hidden_dim, nhead)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                batch_first=True,
                dropout=dropout
            ),
            num_layers=num_layers
        )
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x_tokens):
        return self.ln(self.encoder(x_tokens))


class SpatialExpert(nn.Module):
    def __init__(self, hidden_dim, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        nhead = pick_nhead(hidden_dim, nhead)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                batch_first=True,
                dropout=dropout
            ),
            num_layers=num_layers
        )
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x_tokens):
        return self.ln(self.encoder(x_tokens))


class FusionExpert(nn.Module):
    def __init__(self, hidden_dim, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.cross_t2s = CrossAttention(hidden_dim, nhead=nhead, dropout=dropout)
        self.cross_s2t = CrossAttention(hidden_dim, nhead=nhead, dropout=dropout)

        self.shared = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=pick_nhead(hidden_dim, nhead),
                batch_first=True,
                dropout=dropout
            ),
            num_layers=num_layers
        )
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, temp_tokens, spat_tokens):
        temp_tokens = temp_tokens + self.cross_t2s(temp_tokens, spat_tokens, spat_tokens)
        spat_tokens = spat_tokens + self.cross_s2t(spat_tokens, temp_tokens, temp_tokens)
        fused = torch.cat([temp_tokens, spat_tokens], dim=1)
        return self.ln(self.shared(fused))


class SpatioTemporalTrueMoE(nn.Module):
    def __init__(
        self,
        n_sensors=12,
        cnn_emb=128,
        hidden_dim=128,
        temporal_dim=10000,
        kernel_size=15,
        stride=50,
        padding=7,
        dilation=1,
        num_classes=6,
        nhead=4,
        num_layers_expert=2,
        num_layers_shared=2,
        dropout=0.2,
        pooling="mean"
    ):
        super().__init__()
        self.n_sensors = int(n_sensors)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.pooling = pooling
        self.num_experts = 3

        self.backbone = SharedCNNBackbone(
            n_sensors=n_sensors,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        nhead_hidden = pick_nhead(self.hidden_dim, nhead)

        self.temporal_expert = TemporalExpert(
            hidden_dim=self.hidden_dim,
            nhead=nhead_hidden,
            num_layers=num_layers_expert,
            dropout=dropout
        )

        self.spatial_expert = SpatialExpert(
            hidden_dim=self.hidden_dim,
            nhead=nhead_hidden,
            num_layers=num_layers_expert,
            dropout=dropout
        )

        self.fusion_expert = FusionExpert(
            hidden_dim=self.hidden_dim,
            nhead=nhead_hidden,
            num_layers=num_layers_shared,
            dropout=dropout
        )

        self.gate = GateNetwork(
            input_dim=self.hidden_dim,
            num_experts=self.num_experts,
            hidden_dim=max(64, self.hidden_dim // 2),
            dropout=dropout
        )

        self.fc = nn.Linear(self.hidden_dim, self.num_classes)

    def pool_tokens(self, x):
        return x.max(dim=1).values if self.pooling == "max" else x.mean(dim=1)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f"Expected x dim=3 (B,T,N), got {tuple(x.shape)}")

        _, _, n_channels = x.shape
        if n_channels != self.n_sensors:
            raise ValueError(f"Expected N={self.n_sensors}, got N={n_channels}")

        x_tokens = self.backbone(x)

        gate_input = x_tokens.mean(dim=1)
        gate_weights = self.gate(gate_input)

        temp_tokens = self.temporal_expert(x_tokens)
        spat_tokens = self.spatial_expert(x_tokens)
        fusion_tokens = self.fusion_expert(temp_tokens, spat_tokens)

        temp_vec = self.pool_tokens(temp_tokens)
        spat_vec = self.pool_tokens(spat_tokens)
        fusion_vec = self.pool_tokens(fusion_tokens)

        expert_stack = torch.stack([temp_vec, spat_vec, fusion_vec], dim=1)
        moe_feature = (expert_stack * gate_weights.unsqueeze(-1)).sum(dim=1)

        logits = self.fc(moe_feature)
        return moe_feature, logits, gate_weights
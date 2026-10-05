import copy
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        # x: (batch, seq_len, in_dim) -> pool over time -> (batch, out_dim)
        h = self.net(x)
        return h.mean(dim=1)


class Predictor(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class DenseJEPA(nn.Module):
    """Standard dense JEPA: context encoder predicts the target encoder's
    representation. Target encoder is an EMA copy (stop-gradient), same
    trick BYOL/I-JEPA use to avoid representational collapse."""

    def __init__(self, in_dim, hidden_dim=128, repr_dim=64, ema_decay=0.99):
        super().__init__()
        self.context_encoder = Encoder(in_dim, hidden_dim, repr_dim)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.predictor = Predictor(repr_dim, hidden_dim)
        self.ema_decay = ema_decay

    @torch.no_grad()
    def update_target(self):
        for p_t, p_c in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            p_t.data.mul_(self.ema_decay).add_(p_c.data, alpha=1 - self.ema_decay)

    def forward(self, context, target):
        ctx_repr = self.context_encoder(context)
        pred_repr = self.predictor(ctx_repr)
        with torch.no_grad():
            target_repr = self.target_encoder(target)
        loss = torch.mean((pred_repr - target_repr) ** 2)
        return loss, pred_repr, target_repr

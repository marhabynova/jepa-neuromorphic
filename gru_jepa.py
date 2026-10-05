import copy
import torch
import torch.nn as nn


class GRUEncoder(nn.Module):
    """Control condition: has temporal integration (hidden state accumulates
    across timesteps, same as LIF's membrane potential) but is FULLY DENSE —
    no spiking, no sparsity. If this matches spiking's advantage over the
    plain MLP, the advantage was from temporal integration, not sparsity."""

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden_dim, batch_first=True)
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        _, h_n = self.gru(x)  # h_n: (1, batch, hidden_dim)
        h_n = h_n.squeeze(0)
        return self.out_proj(h_n)


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


class GRUJepa(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, repr_dim=64, ema_decay=0.99):
        super().__init__()
        self.context_encoder = GRUEncoder(in_dim, hidden_dim, repr_dim)
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

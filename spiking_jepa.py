import copy
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class SpikingEncoder(nn.Module):
    """Event-driven encoder: each timestep's input only propagates onward if
    the LIF neuron's membrane potential crosses threshold (a spike). This is
    the software analogue of Loihi 2's address-event routing -- neurons that
    don't spike don't cost downstream compute."""

    def __init__(self, in_dim, hidden_dim, out_dim, beta=0.9):
        super().__init__()
        # 3 linear layers, matching DenseJEPA's Encoder (in->hid, hid->hid,
        # hid->out) layer-for-layer so parameter counts are comparable.
        spike_grad = surrogate.fast_sigmoid()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):
        # x: (batch, seq_len, in_dim) -- each seq step fed as one SNN timestep
        batch, seq_len, _ = x.shape
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        spk3_accum = torch.zeros(batch, self.fc3.out_features, device=x.device)
        n_spikes = x.new_zeros(())
        n_possible = x.new_zeros(())

        for t in range(seq_len):
            cur1 = self.fc1(x[:, t, :])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3_accum = spk3_accum + spk3
            n_spikes = n_spikes + spk1.sum() + spk2.sum() + spk3.sum()
            n_possible = n_possible + spk1.numel() + spk2.numel() + spk3.numel()

        repr_out = spk3_accum / seq_len
        spike_rate = n_spikes / n_possible  # efficiency proxy: lower = sparser = cheaper
        return repr_out, spike_rate


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


class SpikingJEPA(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, repr_dim=64, ema_decay=0.99, beta=0.9):
        super().__init__()
        self.context_encoder = SpikingEncoder(in_dim, hidden_dim, repr_dim, beta=beta)
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
        ctx_repr, ctx_spike_rate = self.context_encoder(context)
        pred_repr = self.predictor(ctx_repr)
        with torch.no_grad():
            target_repr, _ = self.target_encoder(target)
        loss = torch.mean((pred_repr - target_repr) ** 2)
        return loss, pred_repr, target_repr, ctx_spike_rate

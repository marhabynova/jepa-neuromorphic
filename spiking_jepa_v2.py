"""
Redesign of spiking_jepa.py per §29 (Opus research pass, 2026-10-02):
- snn.Leaky -> snn.Synaptic: snntorch's NIR export only maps Synaptic to
  nir.CubaLIF (the Loihi-native neuron model); Leaky doesn't map cleanly.
  This makes the architecture portable (NIR -> Lava-DL/SpiNNaker2/
  BrainScaleS-2) even though Lava itself is dead (§29a).
- Targets SHD (Spiking Heidelberg Digits) instead of N-MNIST: N-MNIST is
  solvable from spatial features alone (§29f), SHD requires genuine temporal
  structure and its raw frame representation is already ~24% sparse natively
  (measured, real_data_shd.py), unlike N-MNIST's fully-dense frames that made
  layer 1 92% of total compute with zero sparsity (§27-28).
- No BatchNorm anywhere (consistent with spiking_jepa.py v1, confirmed safe
  from the §28 BN-confound finding, and NIR's own export map has no
  BatchNorm mapping either).
"""
import copy
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class SpikingEncoderV2(nn.Module):
    """Same 3-layer structure as v1 (in->hid, hid->hid, hid->out), but using
    snn.Synaptic (CUBA-LIF, Loihi/NIR-portable) instead of snn.Leaky."""

    def __init__(self, in_dim, hidden_dim, out_dim, alpha=0.8, beta=0.8):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.lif1 = snn.Synaptic(alpha=alpha, beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.lif2 = snn.Synaptic(alpha=alpha, beta=beta, spike_grad=spike_grad)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.lif3 = snn.Synaptic(alpha=alpha, beta=beta, spike_grad=spike_grad)
        self.out_features = out_dim

    def forward(self, x):
        # x: (batch, seq_len, in_dim)
        batch, seq_len, _ = x.shape
        syn1, mem1 = self.lif1.init_synaptic()
        syn2, mem2 = self.lif2.init_synaptic()
        syn3, mem3 = self.lif3.init_synaptic()
        spk3_accum = torch.zeros(batch, self.out_features, device=x.device)
        n_spikes = x.new_zeros(())
        n_possible = x.new_zeros(())
        # Per-layer spike counts, needed for an honest per-layer SynOps model
        # (layer 1 must not be silently assumed dense OR silently assumed
        # sparse -- measure it directly, §27-28's lesson).
        layer_spikes = [x.new_zeros(()) for _ in range(3)]
        layer_possible = [x.new_zeros(()) for _ in range(3)]

        for t in range(seq_len):
            cur1 = self.fc1(x[:, t, :])
            spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)
            cur2 = self.fc2(spk1)
            spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)
            cur3 = self.fc3(spk2)
            spk3, syn3, mem3 = self.lif3(cur3, syn3, mem3)
            spk3_accum = spk3_accum + spk3
            for i, spk in enumerate((spk1, spk2, spk3)):
                layer_spikes[i] = layer_spikes[i] + spk.sum()
                layer_possible[i] = layer_possible[i] + spk.numel()
            n_spikes = n_spikes + spk1.sum() + spk2.sum() + spk3.sum()
            n_possible = n_possible + spk1.numel() + spk2.numel() + spk3.numel()

        repr_out = spk3_accum / seq_len
        spike_rate = n_spikes / n_possible
        layer_spike_rates = [ls / lp for ls, lp in zip(layer_spikes, layer_possible)]
        return repr_out, spike_rate, layer_spike_rates


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


class SpikingJEPA_V2(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, repr_dim=64, ema_decay=0.99, alpha=0.8, beta=0.8):
        super().__init__()
        self.context_encoder = SpikingEncoderV2(in_dim, hidden_dim, repr_dim, alpha=alpha, beta=beta)
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
        ctx_repr, ctx_spike_rate, ctx_layer_rates = self.context_encoder(context)
        pred_repr = self.predictor(ctx_repr)
        with torch.no_grad():
            target_repr, _, _ = self.target_encoder(target)
        loss = torch.mean((pred_repr - target_repr) ** 2)
        return loss, pred_repr, target_repr, ctx_spike_rate, ctx_layer_rates

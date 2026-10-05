"""
Pure-PyTorch reimplementation of hxtorch's CUBA-LIF integration formula
(from `F.CuBaStepCode`, see RESEARCH_NOTES.md §62), with the 1-dt synaptic
delay toggleable. This is the FIRST check (§63) — a second, independent
check against Brian2 lives in ../brian2_crossvalidation.py.

Formula (leaky=True, fire=True, refractory=False, exponential=False, no
adaptation — our configuration):
    dv = dt/c * (i + g_l*(leak - v))   # delay=True: uses OLD i
    i  = i*(1 - dt/tau_syn) + input[ts]
    v  = v + dv
    z  = 1 if v > threshold else 0
    v  = reset if z == 1 else v

delay=False swaps the first two lines (i updated before dv is computed),
matching snnTorch's `Synaptic` neuron's same-timestep current contribution.
"""
import json
import torch

GT_PATH = '/home/aby/Projects/jepa-neuromorphic/trained_encoder_groundtruth.json'
HXTORCH_MEASURED = [0.3160807192325592, 0.2578125, 0.2024739533662796]  # §60, ws=500 + bias

DT = 1e-6
TAU_SYN = 4.48e-6
C_MEM = 1e-5
G_L = 1.0
LEAK = 80.0
THRESHOLD = 125.0
RESET = 80.0
WEIGHT_SCALE = 500
DECAY_I = 1 - DT / TAU_SYN
DV_COEFF = DT / C_MEM


def load_ground_truth():
    gt = json.load(open(GT_PATH))
    return {
        'x': torch.tensor(gt['x']),
        'fc1_w': torch.tensor(gt['fc1_weight']), 'fc1_b': torch.tensor(gt['fc1_bias']),
        'fc2_w': torch.tensor(gt['fc2_weight']), 'fc2_b': torch.tensor(gt['fc2_bias']),
        'fc3_w': torch.tensor(gt['fc3_weight']), 'fc3_b': torch.tensor(gt['fc3_bias']),
        'expected_layer_spike_rates': gt['expected_layer_spike_rates'],
    }


def run(gt, delay: bool):
    x = gt['x']
    batch, T, in_dim = x.shape
    hid = gt['fc1_w'].shape[0]
    out = gt['fc3_w'].shape[0]

    def make_state(n):
        return {'i': torch.zeros(batch, n), 'v': torch.full((batch, n), LEAK)}

    s1, s2, s3 = make_state(hid), make_state(hid), make_state(out)
    layer_spikes = [0.0, 0.0, 0.0]
    layer_possible = [0.0, 0.0, 0.0]
    spike_trains = [[], [], []]  # per-layer list of (batch, n) tensors, one per timestep

    def step(state, inp):
        i_old, v_old = state['i'], state['v']
        if delay:
            dv = DV_COEFF * (i_old + G_L * (LEAK - v_old))
            i_new = i_old * DECAY_I + inp
        else:
            i_new = i_old * DECAY_I + inp
            dv = DV_COEFF * (i_new + G_L * (LEAK - v_old))
        v_new = v_old + dv
        z = (v_new > THRESHOLD).float()
        v_final = (1 - z) * v_new + z * RESET
        state['i'], state['v'] = i_new, v_final
        return z

    for t in range(T):
        inp1 = x[:, t, :] @ (gt['fc1_w'] * WEIGHT_SCALE).T + gt['fc1_b'] * WEIGHT_SCALE
        z1 = step(s1, inp1)
        inp2 = z1 @ (gt['fc2_w'] * WEIGHT_SCALE).T + gt['fc2_b'] * WEIGHT_SCALE
        z2 = step(s2, inp2)
        inp3 = z2 @ (gt['fc3_w'] * WEIGHT_SCALE).T + gt['fc3_b'] * WEIGHT_SCALE
        z3 = step(s3, inp3)
        for idx, z in enumerate((z1, z2, z3)):
            layer_spikes[idx] += z.sum().item()
            layer_possible[idx] += z.numel()
            spike_trains[idx].append(z)

    rates = [a / b for a, b in zip(layer_spikes, layer_possible)]
    trains = [torch.stack(s) for s in spike_trains]  # each (T, batch, n)
    return rates, trains


if __name__ == '__main__':
    gt = load_ground_truth()
    r_delay, _ = run(gt, delay=True)
    r_nodelay, _ = run(gt, delay=False)
    expected = gt['expected_layer_spike_rates']

    print(f"delay=True  (hxtorch's actual formula): {[f'{v:.4f}' for v in r_delay]}")
    print(f"delay=False (snnTorch-style, no delay):  {[f'{v:.4f}' for v in r_nodelay]}")
    print(f"hxtorch measured (Collab, §60):          {[f'{v:.4f}' for v in HXTORCH_MEASURED]}")
    print(f"local 'subtract' ground truth:           {[f'{v:.4f}' for v in expected]}")
    err_delay = sum((a - b) ** 2 for a, b in zip(r_delay, HXTORCH_MEASURED)) ** 0.5
    err_nodelay = sum((a - b) ** 2 for a, b in zip(r_nodelay, HXTORCH_MEASURED)) ** 0.5
    print(f"\nerr vs hxtorch measured: delay=True -> {err_delay:.4f}   delay=False -> {err_nodelay:.4f}")

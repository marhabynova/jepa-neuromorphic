"""
Independent cross-check of §62/§63's delay hypothesis using Brian2 — a
separately engineered spiking-network simulator, not another hand-rolled
script sharing the same author's bugs as `tests/manual_reimplementation.py`.

Each layer is a Brian2 NeuronGroup (size = batch * layer_width) driven by a
TimedArray of externally-computed per-timestep injected current (this
architecture is rate/graded-current-coded and feedforward, not sparse
event-driven, so there is no native `Synapses` object between layers — the
previous layer's spike train is turned into the next layer's current
exactly as `fc(x)` would in the original PyTorch model, same as in
`tests/manual_reimplementation.py`). Layers run sequentially: finish layer
N's Brian2 run, read its spike train, compute layer N+1's TimedArray, run
layer N+1.

The 1-dt delay is toggled via Brian2's own `run_regularly(..., order=...)`
scheduling (not by changing simulation-loop code), which controls whether
the membrane update this tick sees last tick's or this tick's synaptic
current — exactly the structural difference identified in §62.
"""
import json
import sys
import numpy as np
from brian2 import NeuronGroup, TimedArray, SpikeMonitor, Network, second, start_scope, defaultclock

sys.path.insert(0, '/home/aby/Projects/jepa-neuromorphic/validation/delay_hypothesis/tests')
from manual_reimplementation import (  # noqa: E402
    load_ground_truth, run as manual_run, HXTORCH_MEASURED,
    DT, TAU_SYN, C_MEM, G_L, LEAK, THRESHOLD, RESET, WEIGHT_SCALE, DECAY_I, DV_COEFF,
)


def simulate_layer(inj, delay: bool):
    """inj: (T, n_total) numpy array of per-timestep injected current.
    Returns spikes: (T, n_total) binary numpy array."""
    start_scope()
    defaultclock.dt = DT * second
    T, n_total = inj.shape

    I_ext = TimedArray(inj, dt=DT * second)

    G = NeuronGroup(
        n_total,
        'v : 1\ni_curr : 1',
        threshold='v > THRESHOLD_', reset='v = RESET_',
        method='exact',
        namespace={'THRESHOLD_': THRESHOLD, 'RESET_': RESET},
    )
    G.v = LEAK
    G.i_curr = 0

    ns = {'DV_COEFF_': DV_COEFF, 'G_L_': G_L, 'LEAK_': LEAK, 'DECAY_I_': DECAY_I}

    if delay:
        # order=0: dv uses i_curr from BEFORE this tick's injection (delay)
        G.run_regularly('v += DV_COEFF_ * (i_curr + G_L_ * (LEAK_ - v))',
                         dt=DT * second, order=0)
        G.run_regularly('i_curr = i_curr * DECAY_I_ + I_ext(t, i)',
                         dt=DT * second, order=1)
    else:
        # order=0: i_curr updated first, dv uses the FRESH value (no delay)
        G.run_regularly('i_curr = i_curr * DECAY_I_ + I_ext(t, i)',
                         dt=DT * second, order=0)
        G.run_regularly('v += DV_COEFF_ * (i_curr + G_L_ * (LEAK_ - v))',
                         dt=DT * second, order=1)

    G.namespace.update(ns)

    mon = SpikeMonitor(G)
    net = Network(G, mon)
    net.run(T * DT * second)

    spikes = np.zeros((T, n_total), dtype=np.float32)
    idx = np.round(np.array(mon.t / second) / DT).astype(int)
    idx = np.clip(idx, 0, T - 1)
    spikes[idx, mon.i] = 1.0
    return spikes


def run_network(gt, delay: bool):
    x = gt['x'].numpy()
    batch, T, in_dim = x.shape
    fc1_w, fc1_b = gt['fc1_w'].numpy(), gt['fc1_b'].numpy()
    fc2_w, fc2_b = gt['fc2_w'].numpy(), gt['fc2_b'].numpy()
    fc3_w, fc3_b = gt['fc3_w'].numpy(), gt['fc3_b'].numpy()
    hid = fc1_w.shape[0]
    out = fc3_w.shape[0]

    inj1 = (np.einsum('btd,hd->bth', x, fc1_w * WEIGHT_SCALE) + fc1_b * WEIGHT_SCALE)
    inj1 = inj1.reshape(batch, T, hid).transpose(1, 0, 2).reshape(T, batch * hid)
    spk1 = simulate_layer(inj1, delay).reshape(T, batch, hid)

    inj2 = (np.einsum('bth,gh->btg', spk1.transpose(1, 0, 2), fc2_w * WEIGHT_SCALE)
            + fc2_b * WEIGHT_SCALE)
    inj2 = inj2.reshape(batch, T, hid).transpose(1, 0, 2).reshape(T, batch * hid)
    spk2 = simulate_layer(inj2, delay).reshape(T, batch, hid)

    inj3 = (np.einsum('btg,og->bto', spk2.transpose(1, 0, 2), fc3_w * WEIGHT_SCALE)
            + fc3_b * WEIGHT_SCALE)
    inj3 = inj3.reshape(batch, T, out).transpose(1, 0, 2).reshape(T, batch * out)
    spk3 = simulate_layer(inj3, delay).reshape(T, batch, out)

    rates = [spk1.mean(), spk2.mean(), spk3.mean()]
    return rates


if __name__ == '__main__':
    gt = load_ground_truth()

    print("=== Calibration: Brian2 delay=True vs manual_reimplementation delay=True ===")
    r_manual, _ = manual_run(gt, delay=True)
    r_brian_delay = run_network(gt, delay=True)
    print(f"manual   delay=True: {[f'{v:.4f}' for v in r_manual]}")
    print(f"brian2   delay=True: {[f'{v:.4f}' for v in r_brian_delay]}")
    calib_err = sum((a - b) ** 2 for a, b in zip(r_manual, r_brian_delay)) ** 0.5
    print(f"calibration error (should be ~0 if Brian2 model matches the formula): {calib_err:.4f}")
    print()

    print("=== Independent check: Brian2 delay=False vs hxtorch measured ===")
    r_brian_nodelay = run_network(gt, delay=False)
    print(f"brian2   delay=False: {[f'{v:.4f}' for v in r_brian_nodelay]}")
    print(f"hxtorch measured:     {[f'{v:.4f}' for v in HXTORCH_MEASURED]}")
    print()

    print("=== Final comparison ===")
    print(f"brian2 delay=True:  {[f'{v:.4f}' for v in r_brian_delay]}")
    print(f"brian2 delay=False: {[f'{v:.4f}' for v in r_brian_nodelay]}")
    print(f"hxtorch measured:   {[f'{v:.4f}' for v in HXTORCH_MEASURED]}")
    err_delay = sum((a - b) ** 2 for a, b in zip(r_brian_delay, HXTORCH_MEASURED)) ** 0.5
    err_nodelay = sum((a - b) ** 2 for a, b in zip(r_brian_nodelay, HXTORCH_MEASURED)) ** 0.5
    print(f"err vs hxtorch measured: brian2 delay=True -> {err_delay:.4f}   "
          f"brian2 delay=False -> {err_nodelay:.4f}")

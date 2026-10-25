# §107: real-hardware run of the ACTUAL trained SpikingEncoderV2 (700->128->
# 128->64), properly PARTITIONED this time. Step 2 of the partitioning fix -
# run run_real_hardware_v2_smoketest.py first and confirm it passes on real
# hardware before using this script (it validates the harness + the
# release_hardware try/finally fix on a trivial network first).
#
# Why partitioned: §106's audit (RESEARCH_NOTES.md §106 /
# project_ebrains_mock_real_run_2026-10-20 memory) confirmed BSS-2 gives each
# neuron circuit a fan-in of 128 signed synapse rows, 512 circuits per chip.
# fc1 alone (700 inputs -> 128 neurons) needs 6 circuits/neuron (we use 8,
# same "round up for convenience" reasoning as the project's own
# ts_13-network_partitioning.rst tutorial for its 784-input case) - that's
# 128 * 8 = 1024 circuits, already 2x the whole chip. fc2/fc3 (128+1 bias
# fan-in each) need 2 circuits/neuron: 128*2 + 64*2 = 384 circuits, fits in
# ONE chip together. So: fc1 is split into 2 independent partitions of 64
# hidden neurons each (each gets the full 700-wide input, per
# ts_13's "every sublayer receives all inputs from the presynaptic layer"
# rule for dense feedforward nets), run as 2 separate hardware passes,
# concatenated; then fc2+fc3 run together as a 3rd pass fed the concatenated
# fc1 output as a plain spike tensor (not a live cross-Experiment graph
# handle - each hardware pass is its own Experiment).
#
# Usage: pull this file in the Collab notebook and
# `exec(open('collab_scripts/run_real_hardware_v3_partitioned.py').read())`,
# don't retype it into a cell.

import json
import os
import sys

import torch
import hxtorch
import hxtorch.snn as hxsnn
import pygrenade_vx as grenade

# §109e: all BSS-2 connection + prepared-calibration logic lives in
# collab_scripts/bss2_calib.py, which documents why none of the obvious
# routes (exp.calibration, the demos' helpers, calib_helper's nightly paths)
# work on the deployed hxtorch. Scripts here are exec()'d from the repo root,
# so __file__ is undefined - resolve the import by cwd.
sys.path.insert(0, os.path.join(os.getcwd(), "collab_scripts"))
from bss2_calib import setup_hardware_client, attach_calibration  # noqa: E402



# SAFETY: same convention as v1/v2 - MOCK=True by default. Run mock first
# (should reproduce v1's mock cosine ~0.998, proving partitioning is
# mathematically lossless - same weights, just split across physical
# circuits/passes), THEN flip to False for the real, quota-consuming run.
MOCK = True

with open("trained_encoder_groundtruth.json") as f:
    GT = json.load(f)

IN_DIM, HID, OUT, T, BATCH = GT["in_dim"], GT["hidden_dim"], GT["out_dim"], GT["T"], GT["batch"]
WEIGHT_SCALE = 500.0
TAU = 4.48e-6  # alpha=beta=0.8 @ dt=1e-6, see RESEARCH_NOTES.md §49/§36g

assert HID % 2 == 0, "fc1 partitioning below assumes HID splits evenly in 2"
HID_PART = HID // 2  # 64 - fits in one chip at neuron_structure size=8 (512/8=64)

fc1_w_full = torch.tensor(GT["fc1_weight"])  # (HID, IN_DIM)
fc1_b_full = torch.tensor(GT["fc1_bias"])    # (HID,)
fc2_w = torch.tensor(GT["fc2_weight"])       # (HID, HID)
fc2_b = torch.tensor(GT["fc2_bias"])         # (HID,)
fc3_w = torch.tensor(GT["fc3_weight"])       # (OUT, HID)
fc3_b = torch.tensor(GT["fc3_bias"])         # (OUT,)
x = torch.tensor(GT["x"]).transpose(0, 1)    # GT: (batch,T,in_dim) -> (T,batch,in_dim)
expected = torch.tensor(GT["expected_repr_out"])


def run_fc1_partition(part_idx: int, mock: bool):
    """One hardware pass: 700 -> 64 (half of the 128 hidden neurons)."""
    lo, hi = part_idx * HID_PART, (part_idx + 1) * HID_PART
    setup_hardware_client()
    exp = hxsnn.Experiment(dt=1e-6, mock=mock)
    try:
        if not mock:
            # Attach the prepared calibration BEFORE init_hardware() and
            # before any module is built: the download opens its own
            # connection to read the chip identifier, and modules bind to
            # the execution instance at construction time.
            attach_calibration(exp)
            hxtorch.init_hardware()
        exp.inter_batch_entry_wait = int(125 * 8 * (TAU * 1e6))

        # 700-wide fan-in needs ceil(700/128)=6 circuits/neuron; use 8 for
        # convenience (same margin ts_13 uses for its 784-wide case).
        morph = hxtorch.core.morphology.SingleCompartmentNeuron(
            size=8, expand_horizontally=False)

        syn = hxsnn.Synapse(IN_DIM, HID_PART, exp)
        syn.weight.data = fc1_w_full[lo:hi] * WEIGHT_SCALE
        bias_syn = hxsnn.Synapse(1, HID_PART, exp)
        bias_syn.weight.data = fc1_b_full[lo:hi].unsqueeze(1) * WEIGHT_SCALE
        lif = hxsnn.LIF(HID_PART, exp, tau_mem=TAU, tau_syn=TAU, refractory_time=0,
                         enable_cadc_recording=True, enable_spike_recording=True,
                         neuron_structure=morph)

        ones = torch.ones(T, BATCH, 1)
        g = syn(hxsnn.LIFObservables(spikes=x))
        gb = bias_syn(hxsnn.LIFObservables(spikes=ones))
        z = lif(g, gb)

        hxsnn.run(exp, T)
        return z.spikes.detach()  # (T, batch, HID_PART)
    finally:
        if not mock:
            hxtorch.release_hardware()


def run_fc2_fc3(hidden_spikes: torch.Tensor, mock: bool):
    """One hardware pass: 128 -> 128 -> 64 (fc2+fc3 combined, 384 circuits fits one chip)."""
    setup_hardware_client()
    exp = hxsnn.Experiment(dt=1e-6, mock=mock)
    try:
        if not mock:
            # Attach the prepared calibration BEFORE init_hardware() and
            # before any module is built: the download opens its own
            # connection to read the chip identifier, and modules bind to
            # the execution instance at construction time.
            attach_calibration(exp)
            hxtorch.init_hardware()
        exp.inter_batch_entry_wait = int(125 * 8 * (TAU * 1e6))

        # fan-in 128(+1 bias)=129 > 128 -> 2 circuits/neuron for both layers.
        morph = hxtorch.core.morphology.SingleCompartmentNeuron(
            size=2, expand_horizontally=False)

        def make_layer(w, b, in_f, out_f):
            syn = hxsnn.Synapse(in_f, out_f, exp)
            syn.weight.data = w * WEIGHT_SCALE
            bias_syn = hxsnn.Synapse(1, out_f, exp)
            bias_syn.weight.data = b.unsqueeze(1) * WEIGHT_SCALE
            lif = hxsnn.LIF(out_f, exp, tau_mem=TAU, tau_syn=TAU, refractory_time=0,
                             enable_cadc_recording=True, enable_spike_recording=True,
                             neuron_structure=morph)
            return syn, bias_syn, lif

        syn2, bsyn2, lif2 = make_layer(fc2_w, fc2_b, HID, HID)
        syn3, bsyn3, lif3 = make_layer(fc3_w, fc3_b, HID, OUT)

        ones = torch.ones(T, BATCH, 1)
        g2 = syn2(hxsnn.LIFObservables(spikes=hidden_spikes))
        gb2 = bsyn2(hxsnn.LIFObservables(spikes=ones))
        z2 = lif2(g2, gb2)

        g3 = syn3(z2)
        gb3 = bsyn3(hxsnn.LIFObservables(spikes=ones))
        z3 = lif3(g3, gb3)

        hxsnn.run(exp, T)
        return z3.spikes.detach()  # (T, batch, OUT)
    finally:
        if not mock:
            hxtorch.release_hardware()


def run_full_pipeline(mock: bool):
    spikes_part0 = run_fc1_partition(0, mock=mock)
    spikes_part1 = run_fc1_partition(1, mock=mock)
    hidden_spikes = torch.cat([spikes_part0, spikes_part1], dim=2)  # (T, batch, HID)
    out_spikes = run_fc2_fc3(hidden_spikes, mock=mock)
    return out_spikes.sum(0) / T  # (batch, OUT)


tag = "DRYRUN(mock)" if MOCK else "REALHW"
repr_out = run_full_pipeline(mock=MOCK)
print(f"{tag} repr_out:", repr_out.tolist())
for b in range(BATCH):
    cos = torch.nn.functional.cosine_similarity(repr_out[b], expected[b], dim=0).item()
    print(f"{tag} batch {b}: cosine={cos:.4f}  norm={repr_out[b].norm().item():.4f} "
          f"(expected norm={expected[b].norm().item():.4f})")

if MOCK:
    print("\n*** This was a MOCK dry run of the PARTITIONED pipeline (zero "
          "hardware quota used). Cosine should land near v1's mock result "
          "(~0.998) - that would confirm partitioning didn't change the "
          "math, just how it's physically distributed. If it printed "
          "cleanly with cosine this high, set MOCK=False and re-run for "
          "the real hardware measurement (3 sequential hardware passes - "
          "expect this to take longer than v1's single-pass attempt). ***")

# §104: first REAL BrainScaleS-2 hardware run (mock=False) of SpikingEncoderV2
# (700->128->128->64), using the exact ws=500 + refractory_time=0 + bias-via-
# extra-synapse config validated against mock in §53-64 (cosine 0.9993-1.0000
# vs local repr_out, §64). Pushed via git per project convention - pull this
# file in the Collab notebook and `exec(open('collab_scripts/run_real_hardware_v1.py').read())`,
# don't retype it into a cell.
#
# This is the first time anything in this project touches real silicon
# instead of mock/simulation - see RESEARCH_NOTES.md §103 (guest account
# confirmed allowed on real hardware, extended to 2026-11-14) and the Opus
# audit in this session (mock/tool-estimate self-reports are not independent
# proof; real external hardware is).

import json
import torch
import hxtorch
import hxtorch.snn as hxsnn
from _static.common.helpers import setup_hardware_client, get_nightly_calibration

# SAFETY: this script has never been run (hxtorch only exists inside the
# EBRAINS platform - can't test it locally at all, not even for basic API/
# syntax mistakes). DO NOT flip this to False on the first attempt - run
# with MOCK=True first (same Collab kernel, zero hardware quota consumed,
# exact same code path minus init_hardware/mock=False) to catch any bug in
# this script itself, THEN flip to False for the real, quota-consuming run.
MOCK = True

setup_hardware_client()
if not MOCK:
    hxtorch.init_hardware()

with open("trained_encoder_groundtruth.json") as f:
    GT = json.load(f)

IN_DIM, HID, OUT, T, BATCH = GT["in_dim"], GT["hidden_dim"], GT["out_dim"], GT["T"], GT["batch"]
WEIGHT_SCALE = 500.0
TAU = 4.48e-6  # alpha=beta=0.8 @ dt=1e-6, see RESEARCH_NOTES.md §49/§36g

exp = hxsnn.Experiment(dt=1e-6, mock=MOCK)
if not MOCK:
    exp.calibration = get_nightly_calibration()
# §28a lesson: batch>1 needs enough settle time between samples on one chip,
# or state bleeds across the batch dimension. ~8x tau_mem, FPGA cycles=125/us
# (same rule of thumb as ts_13's inter_batch_entry_wait).
exp.inter_batch_entry_wait = int(125 * 8 * (TAU * 1e6))

def make_layer(w_key, b_key, in_f, out_f):
    # Scale applied by direct weight.data multiply, NOT a Synapse(weight_scale=...)
    # constructor kwarg - matching the exact mechanism validated in §53-64
    # (36i_smoke_test.ipynb cell 2). Passing both would double-scale.
    syn = hxsnn.Synapse(in_f, out_f, exp)
    syn.weight.data = torch.tensor(GT[w_key]) * WEIGHT_SCALE
    bias_syn = hxsnn.Synapse(1, out_f, exp)
    bias_syn.weight.data = torch.tensor(GT[b_key]).unsqueeze(1) * WEIGHT_SCALE
    lif = hxsnn.LIF(out_f, exp, tau_mem=TAU, tau_syn=TAU, refractory_time=0,
                     enable_cadc_recording=True, enable_spike_recording=True)
    return syn, bias_syn, lif

syn1, bsyn1, lif1 = make_layer("fc1_weight", "fc1_bias", IN_DIM, HID)
syn2, bsyn2, lif2 = make_layer("fc2_weight", "fc2_bias", HID, HID)
syn3, bsyn3, lif3 = make_layer("fc3_weight", "fc3_bias", HID, OUT)

x = torch.tensor(GT["x"]).transpose(0, 1)  # GT: (batch,T,in_dim) -> hxsnn wants (T,batch,in_dim)
ones = torch.ones(T, BATCH, 1)

g1 = syn1(hxsnn.LIFObservables(spikes=x))
gb1 = bsyn1(hxsnn.LIFObservables(spikes=ones))
z1 = lif1(g1, gb1)

g2 = syn2(z1)
gb2 = bsyn2(hxsnn.LIFObservables(spikes=ones))
z2 = lif2(g2, gb2)

g3 = syn3(z2)
gb3 = bsyn3(hxsnn.LIFObservables(spikes=ones))
z3 = lif3(g3, gb3)

hxsnn.run(exp, T)

spk1, spk2, spk3 = z1.spikes, z2.spikes, z3.spikes
repr_out = spk3.sum(0) / T  # (batch, OUT)

tag = "DRYRUN(mock)" if MOCK else "REALHW"
print(f"{tag} repr_out:", repr_out.tolist())
print(f"{tag} layer_spike_rates:", [
    (spk1.sum() / spk1.numel()).item(),
    (spk2.sum() / spk2.numel()).item(),
    (spk3.sum() / spk3.numel()).item(),
])

expected = torch.tensor(GT["expected_repr_out"])
for b in range(BATCH):
    cos = torch.nn.functional.cosine_similarity(repr_out[b], expected[b], dim=0).item()
    print(f"{tag} batch {b}: cosine={cos:.4f}  norm={repr_out[b].norm().item():.4f} "
          f"(expected norm={expected[b].norm().item():.4f})")
if MOCK:
    print("\n*** This was a MOCK dry run (zero hardware quota used). "
          "If everything above printed cleanly with no error, set MOCK=False "
          "and re-run for the real hardware measurement. ***")

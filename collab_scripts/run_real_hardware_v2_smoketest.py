# §106: real-hardware SMOKE TEST, step 1 of the partitioning fix.
#
# §105's run_real_hardware_v1.py failed on real silicon with
# "RuntimeError: No router of portfolio was successful." - confirmed via a
# rigorous audit (RESEARCH_NOTES.md §106 / project_ebrains_mock_real_run_2026-10-20
# memory) to be a genuine capacity bug, NOT resource contention: the 700-wide
# fc1 layer was fed into single-circuit LIF neurons (fan-in cap 128 signed
# rows per circuit on BSS-2), and the whole 3-layer network needed 1152
# neuron circuits vs 512 available on one chip. v1 also never called
# hxtorch.release_hardware() - no try/finally - which is the likely real
# explanation for the hour of stuck OnDemandUpload retry warnings.
#
# Before attempting the full partitioned real encoder (v3), this script
# validates the two fixes in isolation on a trivially small, well-under-
# capacity synthetic network (no partitioning needed at all - every layer's
# fan-in stays under 128): does real hardware actually run end-to-end, and
# does wrapping in try/finally + release_hardware() actually prevent the
# stuck-connection symptom. Validation is mock-vs-hardware cosine similarity
# on the SAME small network (no external ground-truth file needed, since
# this network isn't the real trained encoder - it's throwaway).
#
# Usage: pull this file in the Collab notebook and
# `exec(open('collab_scripts/run_real_hardware_v2_smoketest.py').read())`,
# don't retype it into a cell.

import torch
import hxtorch
import hxtorch.snn as hxsnn
from _static.common.helpers import setup_hardware_client, get_nightly_calibration

# SAFETY: same convention as v1 - MOCK=True by default, flip only after a
# clean mock pass. Run with MOCK=True first (zero hardware quota), confirm
# it prints cleanly, THEN flip to False.
MOCK = True

torch.manual_seed(42)

IN_DIM, HID, OUT, T, BATCH = 100, 60, 30, 12, 2
WEIGHT_SCALE = 500.0
TAU = 4.48e-6  # alpha=beta=0.8 @ dt=1e-6, see RESEARCH_NOTES.md §49/§36g

# Fixed random weights/input - same seed every run, so mock and hardware
# passes (run in separate cells/separate Experiments) are directly comparable.
fc1_w = torch.randn(HID, IN_DIM) * 0.05
fc1_b = torch.randn(HID) * 0.05
fc2_w = torch.randn(OUT, HID) * 0.05
fc2_b = torch.randn(OUT) * 0.05
x = (torch.rand(T, BATCH, IN_DIM) < 0.15).float()  # ~15% spike density


def run_once(mock: bool):
    setup_hardware_client()
    exp = hxsnn.Experiment(dt=1e-6, mock=mock)
    try:
        if not mock:
            hxtorch.init_hardware()
            exp.calibration = get_nightly_calibration()
        exp.inter_batch_entry_wait = int(125 * 8 * (TAU * 1e6))

        def make_layer(w, b, in_f, out_f):
            # Fan-in here is well under 128 for both layers (100 and 60),
            # so default single-circuit neurons are fine - no neuron_structure
            # needed, no partitioning needed. This is the whole point: a
            # network this small should just work.
            syn = hxsnn.Synapse(in_f, out_f, exp)
            syn.weight.data = w * WEIGHT_SCALE
            bias_syn = hxsnn.Synapse(1, out_f, exp)
            bias_syn.weight.data = b.unsqueeze(1) * WEIGHT_SCALE
            lif = hxsnn.LIF(out_f, exp, tau_mem=TAU, tau_syn=TAU, refractory_time=0,
                             enable_cadc_recording=True, enable_spike_recording=True)
            return syn, bias_syn, lif

        syn1, bsyn1, lif1 = make_layer(fc1_w, fc1_b, IN_DIM, HID)
        syn2, bsyn2, lif2 = make_layer(fc2_w, fc2_b, HID, OUT)

        ones = torch.ones(T, BATCH, 1)
        g1 = syn1(hxsnn.LIFObservables(spikes=x))
        gb1 = bsyn1(hxsnn.LIFObservables(spikes=ones))
        z1 = lif1(g1, gb1)

        g2 = syn2(z1)
        gb2 = bsyn2(hxsnn.LIFObservables(spikes=ones))
        z2 = lif2(g2, gb2)

        hxsnn.run(exp, T)
        return (z2.spikes.sum(0) / T).detach()  # (batch, OUT)
    finally:
        # §106 fix: always release, even if routing/run raises.
        if not mock:
            hxtorch.release_hardware()


repr_out_mock = run_once(mock=True)
print("MOCK repr_out:", repr_out_mock.tolist())

if not MOCK:
    repr_out_hw = run_once(mock=False)
    print("REALHW repr_out:", repr_out_hw.tolist())
    for b in range(BATCH):
        cos = torch.nn.functional.cosine_similarity(
            repr_out_hw[b], repr_out_mock[b], dim=0).item()
        print(f"REALHW batch {b}: cosine vs mock={cos:.4f}  "
              f"hw_norm={repr_out_hw[b].norm().item():.4f} "
              f"mock_norm={repr_out_mock[b].norm().item():.4f}")
    print("\n*** Smoke test complete. If this ran cleanly end-to-end "
          "(connect, run, release) and cosine is reasonably high, the "
          "harness + release_hardware fix are validated - move on to "
          "run_real_hardware_v3_partitioned.py for the real encoder. ***")
else:
    print("\n*** This was a MOCK-only dry run (zero hardware quota used). "
          "If it printed cleanly with no error, set MOCK=False "
          "above and re-run for the real smoke test. ***")

# §108: real-hardware smoke test, take 2 - v2's "trivial" 100->60->30
# network (2 layers, bias synapses) STILL failed real-hardware routing with
# "No router of portfolio was successful." Root-caused via an offline (zero
# hardware quota) routing probe: the limiting resource is NOT total neuron
# circuits (the §106 theory) but rows per PADI-bus (64 rows/bus, 4 busses =
# 256 rows/hemisphere). v2 needed 101 signed sources (100 input + 1 bias)
# into 60 neurons + 61 signed sources into 30 neurons = 324 physical rows,
# ~1.3-2x over the real per-bus budget - that's the actual constraint
# (confirmed directly: GreedyRouter's own error message is "Source
# distribution not successful. This can happen either if internal and
# background sources require more than the available synapse drivers on a
# single PADI-bus or if no distribution of external sources over the
# PADI-busses is found while using at most the available synapse drivers.").
#
# This script: single layer, no bias synapse, IN_DIM=50 -> HID=20 (50 signed
# sources = 100 physical rows, well under the 256-row hemisphere budget,
# confirmed to route successfully OFFLINE in the same Collab session before
# writing this file). Also explicitly sets exp.hw_routing_func to
# GreedyRouter() instead of relying on the default PortfolioRouter wrapper,
# since PortfolioRouter failed even when GreedyRouter alone succeeds on the
# identical network (reason not fully understood - PortfolioRouter tries
# multiple strategies and apparently all failed where GreedyRouter alone did
# not; using GreedyRouter directly is also how the error message above,
# which actually pinpointed the real constraint, was obtained).
#
# Usage: pull this file in the Collab notebook and
# `exec(open('collab_scripts/run_real_hardware_v2b_minimal.py').read())`,
# don't retype it into a cell.

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



# SAFETY: same convention as v1/v2 - MOCK=True by default, flip only after a
# clean mock pass. Run with MOCK=True first (zero hardware quota), confirm
# it prints cleanly, THEN flip to False.
MOCK = True

torch.manual_seed(42)

IN_DIM, HID, T, BATCH = 50, 20, 12, 2
WEIGHT_SCALE = 500.0
TAU = 4.48e-6  # alpha=beta=0.8 @ dt=1e-6, see RESEARCH_NOTES.md §49/§36g

fc1_w = torch.randn(HID, IN_DIM) * 0.05
x = (torch.rand(T, BATCH, IN_DIM) < 0.15).float()  # ~15% spike density


def run_once(mock: bool):
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
            # §108 fix: use GreedyRouter directly, not the default
            # PortfolioRouter - see module docstring above.
            exp.hw_routing_func = grenade.network.routing.GreedyRouter()
        exp.inter_batch_entry_wait = int(125 * 8 * (TAU * 1e6))

        # Single layer, no bias - 50 signed sources = 100 physical rows,
        # confirmed offline (zero quota) to route successfully on this
        # chip before this real-hardware attempt.
        syn = hxsnn.Synapse(IN_DIM, HID, exp)
        syn.weight.data = fc1_w * WEIGHT_SCALE
        lif = hxsnn.LIF(HID, exp, tau_mem=TAU, tau_syn=TAU, refractory_time=0,
                         enable_cadc_recording=True, enable_spike_recording=True)

        g = syn(hxsnn.LIFObservables(spikes=x))
        z = lif(g)

        hxsnn.run(exp, T)
        return (z.spikes.sum(0) / T).detach()  # (batch, HID)
    finally:
        # §106 fix (carried over): always release, even if routing/run raises.
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
          "harness + GreedyRouter fix are validated - move on to properly "
          "partitioning the real encoder (v3) using GreedyRouter + a "
          "per-PADI-bus-aware partition scheme, not just the neuron-circuit "
          "math from §106/v3's current form, which is now known incomplete. ***")
else:
    print("\n*** This was a MOCK-only dry run (zero hardware quota used). "
          "If it printed cleanly with no error, set MOCK=False "
          "above and re-run for the real smoke test. ***")

# §109f: zero-hardware-quota preflight for the calibration fix.
#
# History of this file (all of it on real EBRAINS kernels, zero experiment
# quota): round 1 dumped the hxtorch calibration API and disproved both the
# original bug AND the first attempt at a fix; rounds 2-3 crashed the kernel
# and taught us why. Everything learned is written up in
# collab_scripts/bss2_calib.py's docstring and in RESEARCH_NOTES.md §109.
#
# What this now does: exercise the ACTUAL fix path end to end, short of
# spending hardware experiment quota. No init_hardware(), no hxsnn.run() on
# real hardware - only the brief ManagedConnection handshake needed to read
# the chip identifier, plus a mock forward pass.
#
# Usage in the Collab (pull, don't retype):
#   import os, sys
#   os.chdir(os.path.expanduser('~/jepa-neuromorphic'))
#   exec(open('collab_scripts/probe_calib_api.py').read())

import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.getcwd(), "collab_scripts"))

LOGPATH = "probe_out.txt"
_log = open(LOGPATH, "w")  # noqa: SIM115


def P(*args):
    """Print and fsync - rounds 2-3 died hard enough to lose buffered stdout."""
    line = " ".join(str(a) for a in args)
    print(line)
    sys.stdout.flush()
    _log.write(line + "\n")
    _log.flush()
    os.fsync(_log.fileno())


def check(label, fn):
    try:
        result = fn()
        P(f"PASS  {label}" + (f" -> {result}" if result is not None else ""))
        return result
    except Exception:  # noqa: BLE001
        P(f"FAIL  {label}")
        P(traceback.format_exc().strip())
        return None


P("=" * 72)
P("kernel:", os.environ.get("LAB_KERNEL_NAME"), "| cwd:", os.getcwd())

import torch  # noqa: E402
import hxtorch  # noqa: E402
import hxtorch.snn as hxsnn  # noqa: E402
from bss2_calib import (setup_hardware_client, chip_identifier,  # noqa: E402
                        fetch_nightly_calib, attach_calibration, CALIB_FILE,
                        assert_calibration_in_effect)

P("hxtorch:", hxtorch.__file__)

# 1. quiggeldy must be configured before ANY call that opens a connection.
check("setup_hardware_client", setup_hardware_client)

# 2. the identifier, read inside a ManagedConnection (reading it outside one
#    is what hard-crashed rounds 2-3).
ident = check("chip_identifier", chip_identifier)

# 3. the calix-native .pkl must actually exist for this chip under
#    ebrains-stable (ebrains-experimental returns 404).
path = check("fetch_nightly_calib",
             lambda: f"{fetch_nightly_calib()} "
                     f"({os.path.getsize(CALIB_FILE)} bytes)")

# 4. the attachment itself, on a mock Experiment: proves the property exists,
#    the ExecutionInstance keeps calib_path, and the guards do not misfire.
exp = hxsnn.Experiment(dt=1e-6, mock=True)
check("attach_calibration(mock experiment)", lambda: attach_calibration(exp))
check("default_execution_instance.calib_path",
      lambda: exp.default_execution_instance.calib_path)

# 5. the guard must REFUSE a late attach - if this passes silently, the guard
#    is useless and a real run could still be calibrated from scratch.
def _late_attach():
    exp2 = hxsnn.Experiment(dt=1e-6, mock=True)
    hxsnn.Synapse(4, 2, exp2)
    try:
        attach_calibration(exp2)
    except RuntimeError as exc:
        return f"correctly refused: {exc}"
    raise AssertionError("late attach was NOT refused - guard is broken")
check("guard refuses attach-after-modules", _late_attach)


def _verifier_catches_uncalibrated():
    e = hxsnn.Experiment(dt=1e-6, mock=True)
    syn = hxsnn.Synapse(4, 2, e)
    lif = hxsnn.LIF(2, e)
    lif(syn(hxsnn.LIFObservables(spikes=torch.zeros(3, 1, 4))))
    try:
        assert_calibration_in_effect(e)
    except RuntimeError as exc:
        return f"correctly refused: {str(exc)[:90]}"
    raise AssertionError("verifier did NOT catch a network with no calibration")
check("verifier catches an uncalibrated network", _verifier_catches_uncalibrated)

# 6. a mock forward pass through the same shape v2b uses, so the wiring is
#    known good before any quota is spent.
def _mock_pass():
    e = hxsnn.Experiment(dt=1e-6, mock=True)
    attach_calibration(e)
    syn = hxsnn.Synapse(50, 20, e)
    torch.manual_seed(42)
    syn.weight.data = torch.randn(20, 50) * 0.05 * 500.0
    lif = hxsnn.LIF(20, e, tau_mem=4.48e-6, tau_syn=4.48e-6, refractory_time=0,
                    enable_cadc_recording=True, enable_spike_recording=True)
    x = (torch.rand(12, 2, 50) < 0.15).float()
    z = lif(syn(hxsnn.LIFObservables(spikes=x)))
    assert_calibration_in_effect(e)
    hxsnn.run(e, 12)
    rates = (z.spikes.sum(0) / 12).detach()
    return f"mock rates shape={tuple(rates.shape)} sum={rates.sum().item():.4f}"
check("mock forward pass with calibration attached", _mock_pass)

P("=" * 72)
P("*** Preflight done. No init_hardware(), no real-hardware run: zero "
  f"experiment quota. Full output in {os.path.abspath(LOGPATH)} ***")
_log.close()

# §109: zero-hardware-quota probe of the hxtorch calibration API.
#
# WHY: the §108 real-hardware attempt died mid-run, and the traceback showed
# it was inside calix's *implicit* full on-chip calibration
# (ExecutionInstance.calibrate() -> calix.spiking.calibrate -> neuron /
# synin / membrane-time-constant binary searches). That takes tens of
# minutes and it is what exhausted the quiggeldy session until
# lib-rcf.OnDemandUpload gave up ("Encountered 10, aborting!").
#
# Root cause: v1/v2b/v3 all did
#     exp.calibration = get_nightly_calibration()
# but helpers.get_nightly_calibration() returns a *lola.Chip* (via
# pynn.helper.chip_from_file) - that is the pyNN `initial_config` object,
# NOT an hxtorch calibration. The official hxtorch demo
# (ts_12-hxtorch_snn_intro.rst) instead does
#     save_nightly_calibration('spiking_cocolist.pbin')
#     exp.calibration = calib_helper.fixture_calibration_from_file(
#         'spiking_cocolist.pbin')
# with the comment "To avoid time consuming implict calibration we load a
# prepared calibration".
#
# Before spending more hardware quota, confirm *mechanically* how this
# version of hxtorch decides to skip calibration, so the fix is verified
# rather than copied on faith. Everything here is pure introspection +
# a file download: no init_hardware(), no chip time.
#
# Usage in the Collab notebook (do not retype - pull and exec):
#   exec(open('collab_scripts/probe_calib_api.py').read())

import inspect
import os

import hxtorch
import hxtorch.snn as hxsnn

print("=" * 72)
print("hxtorch module file:", hxtorch.__file__)
print("hxtorch.snn module file:", hxsnn.__file__)
print("hxtorch version attr:", getattr(hxtorch, "__version__", "<none>"))

# --- 1. Does Experiment.calibration even exist as a real property? --------
# If it is NOT a property/attribute of the class, then `exp.calibration = x`
# silently creates a dead attribute and calibration is never skipped.
print("=" * 72)
Experiment = hxsnn.Experiment
print("'calibration' in Experiment class dict:", "calibration" in vars(Experiment))
for name, obj in vars(Experiment).items():
    if "calib" in name.lower():
        print(f"  Experiment.{name}: {type(obj).__name__}")

# --- 2. The actual decision logic that chose calibrate() over the cache ---
print("=" * 72)
print("--- Experiment._calibrate source ---")
try:
    print(inspect.getsource(Experiment._calibrate))
except Exception as exc:  # noqa: BLE001
    print("could not get source:", repr(exc))

# --- 3. ExecutionInstance: where calib actually lives in this version ----
print("=" * 72)
try:
    from hxtorch.spiking.execution_instance import ExecutionInstance
    print("ExecutionInstance.__init__ signature:")
    print("  ", inspect.signature(ExecutionInstance.__init__))
    print("calib-related names on ExecutionInstance:")
    for name in dir(ExecutionInstance):
        if "calib" in name.lower():
            print("  ", name)
    print("--- ExecutionInstance.calibrate source ---")
    print(inspect.getsource(ExecutionInstance.calibrate))
except Exception as exc:  # noqa: BLE001
    print("ExecutionInstance probe failed:", repr(exc))

# --- 4. calib_helper: the loader the official demo uses -------------------
print("=" * 72)
calib_helper = None
for modpath in ("hxtorch.core.utils", "hxtorch.spiking.utils", "hxtorch.utils"):
    try:
        mod = __import__(modpath, fromlist=["calib_helper"])
        calib_helper = getattr(mod, "calib_helper")
        print(f"calib_helper found at {modpath}.calib_helper ->", calib_helper.__file__)
        break
    except Exception as exc:  # noqa: BLE001
        print(f"  {modpath}.calib_helper: {type(exc).__name__}")
if calib_helper is not None:
    print("public callables in calib_helper:")
    for name in dir(calib_helper):
        if not name.startswith("_") and callable(getattr(calib_helper, name)):
            try:
                sig = inspect.signature(getattr(calib_helper, name))
            except (TypeError, ValueError):
                sig = "<no signature>"
            print(f"   {name}{sig}")

# --- 5/6. ROUND 2 (§109b): the questions round 1 opened -------------------
# Round 1 (EBRAINS-25.10 kernel) established:
#   * 'calibration' is NOT in Experiment's class dict -> `exp.calibration = x`
#     is a dead attribute. Both the old code AND the first attempt at a fix
#     were wrong.
#   * calibration lives on ExecutionInstance(calib_path=..., calib_cache_dir=...)
#   * ExecutionInstance.calibrate() runs the expensive calix path only when
#     `not self.calib_path` AND some module exposes calibration_from_params()
#     (which our LIF does, because we pass tau_mem/tau_syn/refractory_time).
#     With calib_path set, that branch is never taken.
#   * calib_helper lives at hxtorch.spiking.utils.calib_helper (NOT
#     hxtorch.core.utils as the demos assume) and has NO
#     fixture_calibration_from_file at all.
# So: what exactly does calib_path want, and is a local nightly file already
# on disk (making the openproject.bioai.eu download unnecessary)?
print("=" * 72)
print("LAB_KERNEL_NAME:", os.environ.get("LAB_KERNEL_NAME"))
print("LAB_IMAGE_NAME:", os.environ.get("LAB_IMAGE_NAME"))

try:
    from hxtorch.spiking.execution_instance import ExecutionInstance
    print("--- ExecutionInstance.load_calib source ---")
    print(inspect.getsource(ExecutionInstance.load_calib))
except Exception as exc:  # noqa: BLE001
    print("load_calib probe failed:", repr(exc))

if calib_helper is not None:
    for fname in ("nightly_calib_path", "nightly_calix_native_path"):
        try:
            val = getattr(calib_helper, fname)()
            print(f"{fname}() = {val}  exists={os.path.exists(str(val))}")
        except Exception as exc:  # noqa: BLE001
            print(f"{fname}() raised: {exc!r}")

# Signatures we need to actually wire calib_path through to the modules.
print("--- signatures ---")
for obj, label in ((hxsnn.Experiment.__init__, "Experiment.__init__"),
                   (hxsnn.Synapse.__init__, "Synapse.__init__"),
                   (hxsnn.LIF.__init__, "LIF.__init__")):
    try:
        print(f"{label}{inspect.signature(obj)}")
    except Exception as exc:  # noqa: BLE001
        print(f"{label}: {exc!r}")

# Confirm the trigger: does LIF really expose calibration_from_params?
print("LIF has calibration_from_params:", hasattr(hxsnn.LIF, "calibration_from_params"))
print("LIF has params_from_calibration:", hasattr(hxsnn.LIF, "params_from_calibration"))
print("=" * 72)

# --- 7. Download the nightly calib file and show what the loader returns --
# Download only (no chip access beyond the identifier handshake that
# save_nightly_calibration itself needs); no init_hardware, no experiment.
print("=" * 72)
# Guarded: the demos' helpers module runs check_kernel() at import time and
# raises on any kernel but EBRAINS-experimental, which must not abort the
# rest of this probe.
FN = "spiking_cocolist.pbin"
try:
    from _static.common.helpers import (save_nightly_calibration,
                                        setup_hardware_client)
    # MUST come first: save_nightly_calibration opens a ManagedConnection to
    # read the chip identifier, and without QUIGGELDY_ENABLED/IP/PORT that
    # tries to talk to a local FPGA and takes the whole kernel down with it
    # (observed twice: hard crash, no traceback, on an otherwise idle kernel).
    setup_hardware_client()
    print("[probe] quiggeldy:", os.environ.get("QUIGGELDY_IP"),
          os.environ.get("QUIGGELDY_PORT"),
          "enabled=", os.environ.get("QUIGGELDY_ENABLED"))
    save_nightly_calibration(FN)
    print(f"{FN}: exists={os.path.exists(FN)} size={os.path.getsize(FN)} bytes")
except Exception as exc:  # noqa: BLE001
    print("save_nightly_calibration failed:", repr(exc))

if calib_helper is not None and os.path.exists(FN):
    try:
        calib_obj = calib_helper.fixture_calibration_from_file(FN)
        print("fixture_calibration_from_file ->", type(calib_obj).__module__
              + "." + type(calib_obj).__name__)
    except Exception as exc:  # noqa: BLE001
        print("fixture_calibration_from_file failed:", repr(exc))

# For contrast: what the (wrongly used) pyNN helper returns.
try:
    from _static.common.helpers import get_nightly_calibration
    chip_obj = get_nightly_calibration(FN)
    print("get_nightly_calibration ->", type(chip_obj).__module__
          + "." + type(chip_obj).__name__, "  <-- pyNN initial_config, not an hxtorch calib")
except Exception as exc:  # noqa: BLE001
    print("get_nightly_calibration probe failed:", repr(exc))


print("=" * 72)
print("*** Probe done. ZERO hardware experiment quota used (no init_hardware,"
      " no hxsnn.run). Paste the whole output back. ***")

# §109d: zero-experiment-quota probe of the hxtorch calibration API.
#
# WHY (§109): the §108 real-hardware run died inside calix's *implicit* full
# on-chip calibration, not a network fault. v1/v2/v2b/v3 set
#   exp.calibration = get_nightly_calibration()
# but that helper returns a lola.Chip (pyNN's `initial_config`), and round 1
# of this probe then showed 'calibration' is not even in Experiment's class
# dict - so the assignment was a dead attribute either way and the chip was
# recalibrated from scratch on every run.
#
# ROUND 1 ANSWERS (both EBRAINS-25.10 and EBRAINS-experimental, identical):
#   * 'calibration' in vars(Experiment) -> False. Dead attribute, confirmed.
#   * calibration lives on
#     ExecutionInstance(calib_path=..., calib_cache_dir=..., input_loopback=)
#   * ExecutionInstance.calibrate() takes the expensive calix branch only when
#     `not self.calib_path` AND some module exposes calibration_from_params()
#     - which our LIF does, because we pass tau_mem/tau_syn/refractory_time.
#     Set calib_path and that branch is unreachable.
#   * calib_helper is hxtorch.spiking.utils.calib_helper (NOT
#     hxtorch.core.utils as the demos assume) and has no
#     fixture_calibration_from_file; it has calib_from_calix_native,
#     chip_from_file, chip_from_portable_binary, nightly_calib_path,
#     nightly_calix_native_path, target_from_calix_native.
#
# WHAT ROUND 1 ALSO TAUGHT ABOUT RUNNING THIS PROBE:
#   * Several of these calls touch hardware. nightly_calib_path() and
#     save_nightly_calibration() both end up opening a connection, and with
#     QUIGGELDY_ENABLED/IP/PORT unset that tries to reach a local FPGA and
#     takes the whole kernel down - a hard crash, no traceback. So
#     setup_hardware_client() must run FIRST, before anything else here.
#   * A hard crash also loses buffered stdout, which is why the reordered
#     round-2 run printed nothing at all. Everything therefore goes through
#     P(), which appends to probe_out.txt and flushes - read that file from
#     a terminal if the kernel dies again.
#
# Usage in the Collab (pull, don't retype):
#   import os, sys
#   os.chdir(os.path.expanduser('~/jepa-neuromorphic'))
#   sys.path.insert(0, os.path.abspath('ebrains_brainscales2_demos'))
#   exec(open('collab_scripts/probe_calib_api.py').read())

import inspect
import os
import sys
import traceback

LOGPATH = "probe_out.txt"
_log = open(LOGPATH, "w")  # noqa: SIM115  (kept open for the whole probe)


def P(*args):
    """Print and durably append, so a kernel crash cannot eat the output."""
    line = " ".join(str(a) for a in args)
    print(line)
    sys.stdout.flush()
    _log.write(line + "\n")
    _log.flush()
    os.fsync(_log.fileno())


def section(name):
    P("=" * 72)
    P("### " + name)


def attempt(label, fn):
    """Run fn, logging the result or the exception - never abort the probe."""
    try:
        return fn()
    except Exception:  # noqa: BLE001
        P(f"{label}: RAISED")
        P(traceback.format_exc().strip())
        return None


# --- 0. hardware client FIRST: everything below may open a connection -----
section("0. environment + quiggeldy setup (must precede any hardware touch)")
P("LAB_KERNEL_NAME:", os.environ.get("LAB_KERNEL_NAME"))
P("LAB_IMAGE_NAME:", os.environ.get("LAB_IMAGE_NAME"))
attempt("setup_hardware_client", lambda: __import__(
    "_static.common.helpers", fromlist=["setup_hardware_client"]
).setup_hardware_client())
P("QUIGGELDY_ENABLED:", os.environ.get("QUIGGELDY_ENABLED"))
P("QUIGGELDY_IP:", os.environ.get("QUIGGELDY_IP"))
P("QUIGGELDY_PORT:", os.environ.get("QUIGGELDY_PORT"))

import hxtorch  # noqa: E402
import hxtorch.snn as hxsnn  # noqa: E402

section("1. module locations")
P("hxtorch:", hxtorch.__file__)
P("hxtorch.snn:", hxsnn.__file__)

section("2. is Experiment.calibration a real property?")
P("'calibration' in vars(Experiment):", "calibration" in vars(hxsnn.Experiment))

section("3. ExecutionInstance: the real calibration owner")
from hxtorch.spiking.execution_instance import ExecutionInstance  # noqa: E402
P("__init__:", inspect.signature(ExecutionInstance.__init__))
P("--- load_calib source ---")
P(attempt("load_calib", lambda: inspect.getsource(ExecutionInstance.load_calib)))

section("4. calib_helper + where a prepared calibration already lives")
from hxtorch.spiking.utils import calib_helper  # noqa: E402
P("calib_helper:", calib_helper.__file__)
for fname in ("nightly_calib_path", "nightly_calix_native_path"):
    def _probe(fname=fname):
        val = getattr(calib_helper, fname)()
        P(f"{fname}() = {val}  exists={os.path.exists(str(val))}")
    attempt(fname, _probe)

section("5. signatures needed to wire an ExecutionInstance through")
for obj, label in ((hxsnn.Experiment.__init__, "Experiment.__init__"),
                   (hxsnn.Synapse.__init__, "Synapse.__init__"),
                   (hxsnn.LIF.__init__, "LIF.__init__")):
    attempt(label, lambda obj=obj, label=label: P(f"{label}{inspect.signature(obj)}"))

section("6. confirm the trigger for the expensive calix branch")
P("LIF.calibration_from_params:", hasattr(hxsnn.LIF, "calibration_from_params"))
P("LIF.params_from_calibration:", hasattr(hxsnn.LIF, "params_from_calibration"))

section("7. does constructing an ExecutionInstance with calib_path work?")
def _ctor():
    path = calib_helper.nightly_calix_native_path()
    inst = ExecutionInstance(calib_path=path)
    P("ExecutionInstance(calib_path=...) ok; calib_path =", inst.calib_path)
    P("instance attrs of interest:",
      {k: type(v).__name__ for k, v in vars(inst).items()
       if "calib" in k.lower() or "chip" in k.lower()})
attempt("ExecutionInstance(calib_path=...)", _ctor)

P("=" * 72)
P("*** Probe done. ZERO hardware experiment quota used (no init_hardware, "
  f"no hxsnn.run). Full output also saved to {os.path.abspath(LOGPATH)} ***")
_log.close()

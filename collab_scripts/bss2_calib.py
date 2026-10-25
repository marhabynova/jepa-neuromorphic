"""§109e: BSS-2 connection + prepared-calibration helpers for the real-hardware runs.

Everything in here exists because of what the §109 probe established on the
actual EBRAINS kernels (see RESEARCH_NOTES.md §109). The short version:

1. `exp.calibration = <anything>` is a DEAD ATTRIBUTE in the deployed hxtorch.
   `'calibration' in vars(hxsnn.Experiment)` is False on both EBRAINS-25.10
   and EBRAINS-experimental. v1/v2/v2b/v3 all set it, so every `mock=False`
   run silently fell through to calix's full on-chip calibration - tens of
   minutes, which is what exhausted the quiggeldy session in §108.

2. Calibration actually lives on
   `ExecutionInstance(calib_path=..., calib_cache_dir=..., input_loopback=)`.
   `ExecutionInstance.calibrate()` only takes the expensive calix branch when
   `not self.calib_path` AND some module exposes `calibration_from_params()`
   - which our LIF does, because we pass tau_mem/tau_syn/refractory_time.
   Setting calib_path makes that branch unreachable.

3. `load_calib()` dispatches on the file extension: a `.pkl` goes through
   `calib_from_calix_native` and sets BOTH `self.calib` and `self.chip`,
   while anything else (e.g. the `.pbin` coco dump the demos download) only
   sets `self.chip` and leaves `self.calib` None - which makes
   `calibrate()` log "no readable calibration present ... coco binary ...
   Skipped" and skip `params_from_calibration`. So we want the calix-native
   `.pkl`, NOT `spiking_cocolist.pbin`.

4. `calib_helper.nightly_calix_native_path()` is useless here on two counts:
   it points at `/wang/data/calibration/...`, a Heidelberg-internal path not
   mounted in the collaboratory, and it calls
   `_hxtorch_core.get_unique_identifier()` OUTSIDE a connection, which hard
   crashes the kernel (no traceback - this killed three probe runs). Point 7
   below removes the need for it entirely.

5. We do NOT import `_static.common.helpers`: it runs `check_kernel()` at
   import time and hard-fails on anything but EBRAINS-experimental, and it
   hardcodes the experimental quiggeldy setup list. So connection setup is
   inlined here and the download URL is built directly.

6. THE KERNEL FLAVOUR PICKS EVERYTHING, AND MIXING THEM FAILS. There are two
   parallel deployments, each with its own quiggeldy setup, its own chip and
   its own calibration folder. Measured, not assumed:

       kernel            setup list                          port    chip                    calib folder
       EBRAINS-25.10     quiggeldy_setups.csv                11717   hxcube7fpga3chip61_1    ebrains-stable
       EBRAINS-exp...    quiggeldy_setups_experimental.csv   11750   hxcube7fpga0chip57_1    ebrains-experimental

   The matching pairs serve the .pkl with HTTP 200; both cross combinations
   return 404. Pointing a stable kernel at the experimental quiggeldy (what
   copying the demos' helper verbatim does) fails earlier and more cryptically,
   with "QuiggeldyConnection Could not request munge status from remote site:
   Input data format error" out of get_unique_identifier().

7. The setup CSVs carry an `Identifier` column, so the chip name is known
   without opening a connection at all. That removes the whole class of
   crashes in point 4 - nothing here needs hardware before init_hardware().
"""

import hashlib
import os
import urllib.request

import pandas as pd

from hxtorch.spiking.execution_instance import ExecutionInstance

SETUPS_URL_TEMPLATE = "https://quiggeldy.bioai.eu/nmpi/{csv}"
CALIB_FILE = "spiking_calix-native.pkl"
CALIB_URL_TEMPLATE = ("https://openproject.bioai.eu/data_calibration/"
                      "hicann-dls-sr-hx/{identifier}/stable/ebrains-{flavour}/"
                      "{filename}")


def kernel_flavour():
    """'experimental' or 'stable' - see point 6 in the module docstring."""
    return ("experimental"
            if "experimental" in (os.environ.get("LAB_KERNEL_NAME") or "")
            else "stable")


def _setups():
    csv = ("quiggeldy_setups_experimental.csv" if kernel_flavour() == "experimental"
           else "quiggeldy_setups.csv")
    return pd.read_csv(SETUPS_URL_TEMPLATE.format(csv=csv), dtype=str)


def my_setup():
    """The quiggeldy setup assigned to this user, as a (host, port, chip) row.

    Assignment is a stable hash of the JupyterHub username, so the same user
    always lands on the same chip - the same rule the demos use.
    """
    setups = _setups()
    username = os.environ.get("JUPYTERHUB_USER")
    assert username is not None, "not running in the EBRAINS collaboratory"
    index = int(hashlib.sha256(username.encode()).hexdigest(), 16) % len(setups)
    return username, setups.iloc[index]


def setup_hardware_client():
    """Point this kernel at the quiggeldy setup matching its flavour."""
    username, setup = my_setup()
    os.environ["QUIGGELDY_ENABLED"] = "1"
    os.environ["QUIGGELDY_USER_NO_MUNGE"] = username
    os.environ["QUIGGELDY_IP"] = setup["Host"]
    os.environ["QUIGGELDY_PORT"] = setup["Port"]
    return (kernel_flavour(), setup["Host"], setup["Port"], setup["Identifier"])


def chip_identifier():
    """The assigned chip's name, read from the setup list.

    Deliberately NOT `_hxtorch_core.get_unique_identifier()`: calling that
    outside a connection hard crashes the kernel (point 4), and calling it
    inside one needs a working connection we do not have yet.
    """
    return my_setup()[1]["Identifier"]


def fetch_nightly_calib(filename=CALIB_FILE, force=False):
    """Download the nightly calix-native calibration for our chip.

    Returns the local absolute path. Call before `hxtorch.init_hardware()`:
    this opens its own ManagedConnection to read the chip identifier.
    """
    if os.path.exists(filename) and not force:
        return os.path.abspath(filename)
    url = CALIB_URL_TEMPLATE.format(identifier=chip_identifier(),
                                    flavour=kernel_flavour(),
                                    filename=filename)
    urllib.request.urlretrieve(url, filename)
    return os.path.abspath(filename)


def attach_calibration(exp, calib_path=None):
    """Give `exp` an ExecutionInstance carrying a prepared calibration.

    Must be called BEFORE any module (Synapse/LIF/...) is constructed against
    `exp`: modules bind to the default execution instance at construction
    time, so a later swap would not reach them.

    Fails loudly rather than silently falling back to a full on-chip
    calibration - that silent fallback is the entire §108/§109 bug.
    """
    if "default_execution_instance" not in vars(type(exp)):
        raise RuntimeError(
            "hxsnn.Experiment has no 'default_execution_instance' property in "
            "this hxtorch build; re-run collab_scripts/probe_calib_api.py and "
            "find where the ExecutionInstance is meant to be attached.")
    # The preflight proved `exp.modules.nodes` is the WRONG signal here: it
    # stays empty until a module is *called*, while the binding we care about
    # happens in the module's __init__, which reads
    # exp.default_execution_instance. That getter lazily creates an instance
    # and caches it in _default_execution_instance - so a non-None cache is
    # exactly "something already took the default", and a later swap would
    # not reach it.
    cached = getattr(exp, "_default_execution_instance", None)
    if cached is not None:
        raise RuntimeError(
            "attach_calibration() called after the default execution instance "
            f"was already handed out ({cached}) - whatever took it is bound to "
            "that instance and would still be calibrated from scratch. Attach "
            "before constructing any module.")

    calib_path = calib_path or fetch_nightly_calib()
    if not os.path.exists(calib_path):
        raise RuntimeError(f"calibration file missing: {calib_path}")
    if not str(calib_path).endswith(".pkl"):
        raise RuntimeError(
            f"{calib_path} is not a calix-native .pkl; load_calib() would set "
            "only `chip` and leave `calib` None, so params_from_calibration "
            "gets skipped (see point 3 in this module's docstring).")

    instance = ExecutionInstance(calib_path=calib_path)
    exp.default_execution_instance = instance
    if not instance.calib_path:
        raise RuntimeError("ExecutionInstance did not keep calib_path")
    print(f"[calib] {calib_path} ({os.path.getsize(calib_path)} B) attached to "
          f"{instance}; the implicit calix calibration branch is now "
          "unreachable")
    return instance


def assert_calibration_in_effect(exp):
    """Verify, right before a run, that every module really has the calib.

    The ordering guard above is preventive and depends on an hxtorch internal;
    this is the actual safety net, and it reads only the documented path a
    module takes to its execution instance. Call it after the network is
    built and before hxsnn.run(), so a silent fall-back to a full calix
    calibration costs a traceback instead of a hardware session.
    """
    modules = list(getattr(getattr(exp, "modules", None), "nodes", None) or [])
    instances = {id(m.execution_instance): m.execution_instance
                 for m in modules if getattr(m, "execution_instance", None)}
    if not instances:
        raise RuntimeError(
            "no module reports an execution instance - cannot confirm the "
            "calibration is in effect, refusing to spend hardware time")
    uncalibrated = [inst for inst in instances.values() if not inst.calib_path]
    if uncalibrated:
        raise RuntimeError(
            f"{len(uncalibrated)} of {len(instances)} execution instance(s) "
            "have no calib_path; those would trigger the full on-chip calix "
            "calibration (the §108 failure). Instances: "
            f"{[str(i) for i in uncalibrated]}")
    print(f"[calib] verified: {len(modules)} module(s) across "
          f"{len(instances)} execution instance(s), all with calib_path set")

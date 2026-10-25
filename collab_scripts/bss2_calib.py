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
   crashes the kernel (no traceback - this killed three probe runs). The
   identifier must be read inside an `hxcomm.ManagedConnection()`.

5. We do NOT import `_static.common.helpers`: it runs `check_kernel()` at
   import time and hard-fails on anything but EBRAINS-experimental, while
   the calibration files are only published under `ebrains-stable`
   (verified: ebrains-stable/spiking_calix-native.pkl -> HTTP 200,
   ebrains-experimental/... -> HTTP 404). So `setup_hardware_client` is
   inlined here, minus the kernel check, and the download URL is built
   directly.
"""

import hashlib
import os
import urllib.request

import pandas as pd
from dlens_vx_v3 import hxcomm

from hxtorch.spiking.execution_instance import ExecutionInstance

QUIGGELDY_SETUPS_URL = ("https://quiggeldy.bioai.eu/nmpi/"
                        "quiggeldy_setups_experimental.csv")
CALIB_FILE = "spiking_calix-native.pkl"
CALIB_URL_TEMPLATE = ("https://openproject.bioai.eu/data_calibration/"
                      "hicann-dls-sr-hx/{identifier}/stable/ebrains-stable/"
                      "{filename}")


def setup_hardware_client():
    """Point this kernel at its assigned quiggeldy setup.

    Inlined from the demos' `_static.common.helpers.setup_hardware_client`,
    minus the EBRAINS-experimental kernel check - see point 5 above. The
    setup is chosen by a stable hash of the JupyterHub username, so the same
    user always lands on the same chip.
    """
    setups = pd.read_csv(QUIGGELDY_SETUPS_URL, dtype=str)
    username = os.environ.get("JUPYTERHUB_USER")
    assert username is not None, "not running in the EBRAINS collaboratory"
    index = int(hashlib.sha256(username.encode()).hexdigest(), 16) % len(setups)
    setup = setups.iloc[index]
    os.environ["QUIGGELDY_ENABLED"] = "1"
    os.environ["QUIGGELDY_USER_NO_MUNGE"] = username
    os.environ["QUIGGELDY_IP"] = setup["Host"]
    os.environ["QUIGGELDY_PORT"] = setup["Port"]
    return setup["Host"], setup["Port"]


def chip_identifier():
    """Read the assigned chip's identifier - only valid inside a connection."""
    with hxcomm.ManagedConnection() as connection:
        return connection.get_unique_identifier()[0]


def fetch_nightly_calib(filename=CALIB_FILE, force=False):
    """Download the nightly calix-native calibration for our chip.

    Returns the local absolute path. Call before `hxtorch.init_hardware()`:
    this opens its own ManagedConnection to read the chip identifier.
    """
    if os.path.exists(filename) and not force:
        return os.path.abspath(filename)
    url = CALIB_URL_TEMPLATE.format(identifier=chip_identifier(),
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
    built = getattr(getattr(exp, "modules", None), "nodes", None)
    if built:
        raise RuntimeError(
            "attach_calibration() called after modules were already built - "
            "they are bound to the old execution instance and would still be "
            "calibrated from scratch.")

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

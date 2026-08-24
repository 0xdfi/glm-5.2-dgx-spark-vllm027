"""
R17.1 nvfp4_ds_mla op-loader shim.

Provides `concat_and_cache_nvfp4_mla` (and vLLM-fork extras
`concat_and_cache_mla_rope_fused`, if present) in the `_C_cache_ops`
torch library namespace by loading the donor .so extracted from the
production glm52-exp1-sm121a-368-canary image.

Mechanism: the donor .so registers its ops through the torch::stable /
AOTI registration shim (`aoti_torch_library_def`), which catches
duplicate-registration errors per-op at the C ABI boundary and logs
them as non-fatal `[E] ... Duplicate registration` messages rather than
raising/aborting. Verified empirically (see load_test_full.log): 111
duplicate-registration errors were logged and caught for ops that
already exist in the 0.27.1 R17 canary image's own `_C`/`_C_cache_ops`
libraries (their ORIGINAL implementations are left untouched -- the
duplicate `def()` call simply fails and is discarded), while the two
genuinely new ops (`concat_and_cache_nvfp4_mla`,
`concat_and_cache_mla_rope_fused`) register cleanly because their names
don't collide.

Import this module once, early, before any code calls
`torch.ops._C_cache_ops.concat_and_cache_nvfp4_mla`. It is safe to
import multiple times (idempotent) and safe if the donor .so is absent
(no-op, existing fail-closed RuntimeError in _custom_ops.py still
fires).
"""

import ctypes
import logging
import os

import torch

logger = logging.getLogger(__name__)

_DONOR_SO_ENV = "VLLM_NVFP4_DS_MLA_DONOR_SO"
_DEFAULT_DONOR_SO = "/opt/vllm-extra/nvfp4_ds_mla/_C_stable_libtorch.abi3.so"

_loaded = False


def ensure_nvfp4_ds_mla_op_loaded() -> bool:
    """Load the donor .so (once) so that
    torch.ops._C_cache_ops.concat_and_cache_nvfp4_mla becomes available.

    Returns True if the op is available after this call (whether it was
    already loaded, freshly loaded, or -- in future -- provided by some
    other mechanism), False if it remains unavailable.
    """
    global _loaded

    def _op_present() -> bool:
        return hasattr(torch.ops, "_C_cache_ops") and hasattr(
            torch.ops._C_cache_ops, "concat_and_cache_nvfp4_mla"
        )

    if _op_present():
        return True

    if _loaded:
        # We already tried and it didn't stick; don't retry endlessly.
        return _op_present()

    donor_path = os.environ.get(_DONOR_SO_ENV, _DEFAULT_DONOR_SO)
    if not os.path.exists(donor_path):
        logger.warning(
            "nvfp4_ds_mla donor .so not found at %s (set %s to override); "
            "concat_and_cache_nvfp4_mla will remain unavailable",
            donor_path,
            _DONOR_SO_ENV,
        )
        _loaded = True
        return False

    try:
        torch.ops.load_library(donor_path)
    except OSError as e:
        # dlopen failed outright (missing deps, wrong torch ABI, etc.) --
        # this IS fatal/informative, unlike the caught duplicate-def errors.
        logger.error(
            "Failed to dlopen nvfp4_ds_mla donor .so %s: %r", donor_path, e
        )
        _loaded = True
        return False

    _loaded = True
    ok = _op_present()
    if ok:
        logger.info(
            "nvfp4_ds_mla: loaded concat_and_cache_nvfp4_mla from donor .so %s",
            donor_path,
        )
    else:
        logger.error(
            "nvfp4_ds_mla: donor .so %s loaded but concat_and_cache_nvfp4_mla "
            "still not registered -- check for a duplicate-registration "
            "failure specific to this op (not just the expected ones for "
            "pre-existing ops) in the loader's stderr output",
            donor_path,
        )
    return ok


# Best-effort eager load at import time. Failure here is non-fatal: the
# existing fail-closed check in _custom_ops.py's concat_and_cache_nvfp4_mla()
# still raises a clear RuntimeError if the op didn't materialize.
ensure_nvfp4_ds_mla_op_loaded()

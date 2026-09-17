"""
profiler_perf.py — Desktop performance profile for Profiler
Author: Yanis Zirem (patch proposal)

⚠️ IMPORTANT — import this module FIRST, before numpy / pandas / scipy /
sklearn / tensorflow / numexpr / dask / pandarallel, in every entry point
(Profiler_Desktop_Gui.py at the very top, and any module that used to set
its own OMP_NUM_THREADS/MKL_NUM_THREADS). BLAS libraries read these env
vars once, at import time — setting them later is a silent no-op.

WHY THIS MODULE EXISTS
-----------------------
Several modules were adapted from the web/VM version by simply *removing
the server cap* on thread/worker counts ("sans plafond serveur, utilise
tous les cœurs"). On a dedicated VM with many cores that's harmless. On a
normal laptop it creates classic nested-parallelism oversubscription:

  - profiler_preprocessing.py set OMP/MKL/OPENBLAS threads = N_CPUS
  - profiler_features_importance.py *also* set them (independently)
  - profiler_training.py spawns N_CPUS joblib/sklearn worker PROCESSES
    (n_jobs=-1) for cross-validation, AROUND models that themselves also
    use n_jobs=-1 internally (RandomForest, ExtraTrees, LGBM, KNN, ...)
  - TensorFlow claimed N_CPUS threads for its own inter/intra-op pools
  - pandarallel and dask each spun up their own N_CPUS-worker pools

Stacked together, a machine with e.g. 8 logical cores can end up trying to
run 60-100+ competing threads/processes at once. The OS scheduler thrashes,
the Streamlit UI freezes, the laptop hits thermal throttling — and the
whole thing is often *slower* than a modest, well-budgeted setup, not
faster. This is the "VM-shaped" behaviour you're noticing.

THE FIX
-------
One shared budget, computed once, with headroom reserved for the OS/UI,
and a hard rule: never nest two layers of "use every core" parallelism.
"""

import os


def _physical_cores():
    try:
        import psutil
        c = psutil.cpu_count(logical=False)
        if c:
            return c
    except Exception:
        pass
    return os.cpu_count() or 2


LOGICAL_CPUS = os.cpu_count() or 2
PHYSICAL_CPUS = _physical_cores()

# Reserve headroom for the OS, the Streamlit server thread, and the UI.
# Small machines need *proportionally more* headroom, not less — a 4-core
# laptop running a browser + Streamlit + this process cannot hand 4/4 cores
# to a training run without the interface freezing.
if LOGICAL_CPUS <= 4:
    _RESERVED = 2
elif LOGICAL_CPUS <= 8:
    _RESERVED = 2
else:
    _RESERVED = max(2, LOGICAL_CPUS // 8)

# The ONE outer parallelism budget: how many worker PROCESSES joblib /
# sklearn / pandarallel / CV are allowed to spawn at once.
OUTER_JOBS = max(1, LOGICAL_CPUS - _RESERVED)
PANDARALLEL_WORKERS = OUTER_JOBS

# Nested-parallelism guard: once OUTER_JOBS worker processes are already
# spread across the machine, each worker's own BLAS calls must NOT also
# try to grab every core — otherwise you get OUTER_JOBS × LOGICAL_CPUS
# threads fighting over LOGICAL_CPUS cores. Force BLAS to 1 thread/worker
# whenever we're already parallel at the outer (process) level.
BLAS_THREADS = 1 if OUTER_JOBS > 1 else LOGICAL_CPUS
for _env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_env] = str(BLAS_THREADS)


def has_usable_gpu() -> bool:
    """True only if TensorFlow can see a real GPU. mixed_precision /
    memory_growth are GPU-only wins; forcing them on a CPU-only laptop is a
    no-op at best (no tensor cores to exploit) and pure overhead at worst."""
    try:
        import tensorflow as tf
        return len(tf.config.list_physical_devices("GPU")) > 0
    except Exception:
        return False


def configure_tensorflow():
    """Call this once, before building any Keras model, instead of each
    module independently poking TF's global thread pools."""
    try:
        import tensorflow as tf
        if has_usable_gpu():
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            for gpu in tf.config.list_physical_devices("GPU"):
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            # CPU-only desktop: leave room for sklearn/joblib running in
            # the same session instead of also claiming every core.
            tf.config.threading.set_intra_op_parallelism_threads(max(1, OUTER_JOBS // 2))
            tf.config.threading.set_inter_op_parallelism_threads(2)
    except Exception:
        pass

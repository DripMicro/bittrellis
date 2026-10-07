"""BitTrellis — find the precision map the hardware actually wants."""

import os

__version__ = "0.1.0"
BLAS_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


def build_jobs() -> int:
    """Worker processes for a build: BITTRELLIS_BUILD_JOBS, else half the CPUs (at most 16); 1 means serial."""
    env = os.environ.get("BITTRELLIS_BUILD_JOBS")
    return max(1, int(env)) if env else max(1, min(16, (os.cpu_count() or 2) // 2))


def blas_threads() -> int:
    """BLAS threads per build worker: its share of the CPUs. One pool the size of the machine in every worker
    (12 workers x 24 threads on 24 CPUs) made a GPTQ build crawl: thousands of small matrix products, each
    waking 24 threads on an oversubscribed machine."""
    return max(1, (os.cpu_count() or 2) // build_jobs())

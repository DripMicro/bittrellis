"""CPU-only steps run encoders, many at once in a build: give each process its share of BLAS threads before
NumPy starts its pool (see `blas_threads`). GPU steps (quality, speed, tasks) keep the runtime's defaults.
Imported by the CLI before anything that loads NumPy."""

import os
import sys

from . import BLAS_VARS, blas_threads

if {"build", "regenerate", "audit", "fingerprint"} & set(sys.argv[1:4]):
    for _v in BLAS_VARS:
        os.environ.setdefault(_v, str(blas_threads()))

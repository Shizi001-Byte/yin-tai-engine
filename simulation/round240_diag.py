#!/usr/bin/env python3
# 验证子Agent：N×seed 2×2 定位 round240 反射误差根因。
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from round240_verify_measure import pair_error  # noqa: E402

L = 20.0 * np.pi / 3.0
for n in (96, 128):
    for seed in (237, 240):
        r = pair_error(n, L, 0.002, 100, 100, seed=seed)
        print(
            "DIAG",
            "N=%d" % n,
            "seed=%d" % seed,
            "refl=%.3e" % r["reflection_rel_error"],
            "chirSum=%.2e" % r["chirality_sum"],
            "P48diff=%.2e" % r["P4.8_diff"],
            "lam2nd_diff=%.2e" % r["lam2nd_diff"],
            "lam_stderr=%.2e" % r["lam_stderr_max"],
            flush=True,
        )

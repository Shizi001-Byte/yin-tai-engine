#!/usr/bin/env python3
# 验证子Agent：round240 短测程(measure40)报 beta3=0.25 lambda=0.47 sustained_chaotic，
# 而相邻 0.20/0.30 都≈0。长测程核查这是否只是过渡区瞬态（#237 同构：
# 瞬态正率随测程衰减到0）。N=128, warmup=100, measure=200, seed=240。
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from round231_P1_0 import make_grid, run_case  # noqa: E402
from round234_P1_0 import common_initial_state  # noqa: E402
import argparse  # noqa: E402

L = 20.0 * np.pi / 3.0
N = 128
k, mask = make_grid(N, L)
n_ext = int(round(4.8 * L / (2.0 * np.pi)))
u0 = common_initial_state(N, mask, n_ext, 240)
ns = argparse.Namespace(warmup=100.0, measure=200.0, renorm=1.0, seed=240)

for b3 in (0.20, 0.25):
    r = run_case(f"b{b3:+.2f}", b3, 1.0, u0, N, L, 0.002, k, mask, n_ext, ns)
    print(
        "LONG",
        "b3=%.2f" % b3,
        "P4.8=%.4f" % r["q4p8_power_fraction"],
        "lam_full=%.4f" % r["lambda_full"],
        "lam_2nd=%.4f+-%.4f" % (r["lambda_second_half"], r["lambda_second_half_stderr"]),
        "lam_last20=%.4f" % r["lambda_last20"],
        r["classification"],
        flush=True,
    )

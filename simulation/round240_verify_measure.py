#!/usr/bin/env python3
# 验证子Agent（决策层）独立复算：round240 反射断言失败是否仅因 measure=40 太短。
# 固定 N=96、seed=240、beta4=1，对比 beta3=±0.3 反射配对在
# measure=40（worker默认）与 measure=100（#237已验证测程）下的终态反射误差。
# 复用 round231/234/237 已验证地基，不重写任何物理。
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from round231_P1_0 import make_grid, run_case, spec_norm
from round234_P1_0 import common_initial_state
from round237_P1_0 import reflect_spectral, spectral_chirality


def pair_error(n, length, dt, warmup, measure, seed=240, beta3=0.3, beta4=1.0):
    k, mask = make_grid(n, length)
    n_ext = int(round(4.8 * length / (2.0 * np.pi)))
    u0p = common_initial_state(n, mask, n_ext, seed)
    u0m = reflect_spectral(u0p)
    ns = argparse.Namespace(warmup=warmup, measure=measure, renorm=1.0, seed=seed)
    rp = run_case("p", +beta3, beta4, u0p, n, length, dt, k, mask, n_ext, ns)
    rm = run_case("m", -beta3, beta4, u0m, n, length, dt, k, mask, n_ext, ns)
    refl = spec_norm(rm["uhat"] - reflect_spectral(rp["uhat"]), n) / max(
        spec_norm(rp["uhat"], n), 1e-300
    )
    return {
        "warmup": warmup,
        "measure": measure,
        "reflection_rel_error": float(refl),
        "chirality_sum": float(
            spectral_chirality(rp["uhat"]) + spectral_chirality(rm["uhat"])
        ),
        "P4.8_diff": float(
            abs(rp["q4p8_power_fraction"] - rm["q4p8_power_fraction"])
        ),
        "lam2nd_diff": float(
            abs(rp["lambda_second_half"] - rm["lambda_second_half"])
        ),
        "lam_stderr_max": float(
            max(rp["lambda_second_half_stderr"], rm["lambda_second_half_stderr"])
        ),
    }


def main():
    L = 20.0 * np.pi / 3.0
    dt = 0.002
    out = []
    for warmup, measure in [(80, 40), (100, 100)]:
        r = pair_error(96, L, dt, warmup, measure)
        out.append(r)
        print(r, flush=True)
    print("VERDICT: 若 measure100 反射误差 <<1e-6 而 measure40 为 1e-3，"
          "则 round240 断言失败仅为测程不足，非物理/代码错误。")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# simulation/round240_P1_0.py
# 依赖：numpy；round231_P1_0.py、round234_P1_0.py、round237_P1_0.py
# 必须位于同一目录。
#
# 默认运行：
#   python3 simulation/round240_P1_0.py
#
# 目标：固定 beta4=1，粗扫 beta3=0..0.3，定位非线性饱和态中
#       q=4.8 外带功率随 beta3 跌落的区间；并用 beta3=-0.3 的
#       空间反射初值复核 round237 的反射共轭关系。
#
# 不重写任何已验证 CW、RHS、Jacobian、色散符号或 Benettin 流程。

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from round231_P1_0 import (  # noqa: E402
    analytic_eigenvalues,
    make_grid,
    run_case,
    spec_norm,
    self_test as round231_self_test,
)
from round234_P1_0 import common_initial_state  # noqa: E402
from round237_P1_0 import (  # noqa: E402
    reflect_spectral,
    reflection_operator_self_test,
    spectral_chirality,
)

BETA4 = 1.0
Q_OUT = 4.8
POWER_CROSSING = 0.05
REFLECTION_ERROR_TOL = 1.0e-6


def summarize(result, n_ext, n):
    scalar_keys = [
        "name",
        "beta3",
        "beta4",
        "lambda_full",
        "lambda_second_half",
        "lambda_second_half_stderr",
        "lambda_last20",
        "relative_final_residual",
        "dominant_q",
        "mean_power",
        "q4p8_power_fraction",
        "q4p8_amp_cv_last20",
        "classification",
        "max_rk4_imag_z",
    ]

    summary = {key: result[key] for key in scalar_keys}
    summary.update(
        {
            "spectral_chirality": float(spectral_chirality(result["uhat"])),
            "q4p8_plus_q_amp": float(abs(result["uhat"][n_ext]) / n),
            "q4p8_minus_q_amp": float(abs(result["uhat"][-n_ext]) / n),
        }
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    # 默认 N=128（不是96）：决策层验证发现 N=96+seed=240 是镜像敏感坏组合，
    # 终态反射误差 4.9e-3（虚假镜像破缺，measure 加大也不收敛）；N=128 下
    # 同一 seed 收敛到 6.9e-8（measure40 即 <1e-6），N=96 seed=237 为 1.8e-11。
    # 低分辨率会让个别初值落在镜像破缺敏感区，分辨率必须与 #237 一致。
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--L", type=float, default=20.0 * np.pi / 3.0)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--warmup", type=float, default=80.0)
    parser.add_argument("--measure", type=float, default=40.0)
    parser.add_argument("--renorm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=240)
    parser.add_argument(
        "--grid",
        default="0,0.05,0.10,0.15,0.20,0.25,0.30",
        help="正 beta3 粗扫描网格；另自动加入 -0.30 反射复核",
    )
    parser.add_argument("--outdir", default="simulation/out")
    args = parser.parse_args()

    n, length, dt = args.N, args.L, args.dt
    k, mask = make_grid(n, length)

    n_ext = int(round(Q_OUT * length / (2.0 * np.pi)))
    assert 0 < n_ext < n
    assert abs(k[n_ext] - Q_OUT) < 1.0e-10

    positive_beta3 = [
        float(token) for token in args.grid.split(",") if token.strip()
    ]
    assert positive_beta3[0] == 0.0
    assert any(abs(b - 0.3) < 1.0e-12 for b in positive_beta3)

    mirror_abs = 0.3
    all_beta3 = positive_beta3 + [-mirror_abs]

    # 解析地基硬事实：beta3 只改变 q=4.8 边带本征值虚部，
    # 不改变 CW 附近的实增长率。因此若饱和外带功率变化，
    # 那是非线性相位锁定/重分配，而不是线性 MI 增益改变。
    linear_rows = []
    real_gains = []
    for b3 in all_beta3:
        ev = analytic_eigenvalues(Q_OUT, b3, BETA4)
        j = int(np.argmax(ev.real))
        real_gains.append(float(ev[j].real))
        linear_rows.append(
            {
                "beta3": float(b3),
                "q": Q_OUT,
                "max_real_growth": float(ev[j].real),
                "max_eigenvalue_imag": float(ev[j].imag),
                "linear_velocity": float(-b3 * Q_OUT**2 / 6.0),
            }
        )

    assert max(real_gains) - min(real_gains) < 1.0e-10, real_gains

    # 复用 round231 与 round237 的硬自校验。
    foundation_self_test = round231_self_test(
        n, length, dt, k, mask, n_ext
    )
    reflection_self_test = reflection_operator_self_test(
        n, k, mask, mirror_abs, BETA4
    )

    base_positive = common_initial_state(n, mask, n_ext, args.seed)
    initial_chirality = float(spectral_chirality(base_positive))
    assert abs(initial_chirality) > 1.0e-10

    sim_args = argparse.Namespace(
        warmup=args.warmup,
        measure=args.measure,
        renorm=args.renorm,
        seed=args.seed,
    )

    results = {}

    for b3 in positive_beta3:
        name = f"b3_{b3:+.2f}"
        print(f"=== running {name}, beta4={BETA4}, N={n} ===", flush=True)
        results[b3] = run_case(
            name,
            b3,
            BETA4,
            base_positive,
            n,
            length,
            dt,
            k,
            mask,
            n_ext,
            sim_args,
        )

    # 负 beta3 必须使用正 beta3 初值的空间反射副本。
    base_negative = reflect_spectral(base_positive)
    b3 = -mirror_abs
    name = f"b3_{b3:+.2f}"
    print(f"=== running reflection mirror {name}, N={n} ===", flush=True)
    results[b3] = run_case(
        name,
        b3,
        BETA4,
        base_negative,
        n,
        length,
        dt,
        k,
        mask,
        n_ext,
        sim_args,
    )

    scan_summary = [
        summarize(results[b3], n_ext, n) for b3 in positive_beta3
    ]

    # 操作性定位：首次从 >=5% 跌到 <5% 的相邻网格。
    # 这不是数学分岔判据，只决定下一轮加密区间。
    crossing_interval = None
    for prev, cur in zip(positive_beta3[:-1], positive_beta3[1:]):
        p_prev = results[prev]["q4p8_power_fraction"]
        p_cur = results[cur]["q4p8_power_fraction"]
        if p_prev >= POWER_CROSSING and p_cur < POWER_CROSSING:
            crossing_interval = [float(prev), float(cur)]
            break

    rp = results[+mirror_abs]
    rm = results[-mirror_abs]

    pair_diagnostics = {
        "tolerance": REFLECTION_ERROR_TOL,
        "final_reflection_relative_error": float(
            spec_norm(
                rm["uhat"] - reflect_spectral(rp["uhat"]),
                n,
            )
            / max(spec_norm(rp["uhat"], n), 1.0e-300)
        ),
        "q4p8_power_fraction_abs_diff": float(
            abs(
                rp["q4p8_power_fraction"]
                - rm["q4p8_power_fraction"]
            )
        ),
        "chirality_plus": float(spectral_chirality(rp["uhat"])),
        "chirality_minus": float(spectral_chirality(rm["uhat"])),
        "chirality_sum": float(
            spectral_chirality(rp["uhat"])
            + spectral_chirality(rm["uhat"])
        ),
        "lambda_second_half_abs_diff": float(
            abs(
                rp["lambda_second_half"]
                - rm["lambda_second_half"]
            )
        ),
        "lambda_stderr_max": float(
            max(
                rp["lambda_second_half_stderr"],
                rm["lambda_second_half_stderr"],
            )
        ),
    }
    pair_diagnostics["reflection_conjugacy_holds"] = bool(
        pair_diagnostics["final_reflection_relative_error"]
        < REFLECTION_ERROR_TOL
        and abs(pair_diagnostics["chirality_sum"]) < REFLECTION_ERROR_TOL
        and pair_diagnostics["q4p8_power_fraction_abs_diff"]
        < REFLECTION_ERROR_TOL
    )

    assert pair_diagnostics["reflection_conjugacy_holds"], pair_diagnostics

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = outdir / "round240_P1_0"

    array_keys = {
        "t",
        "lambda_segment",
        "lambda_cumulative",
        "external_amp",
        "uhat",
    }
    arrays = {}
    for b3, result in results.items():
        prefix = f"b3_{b3:+.2f}"
        for key in array_keys:
            arrays[f"{prefix}_{key}"] = result[key]
    np.savez_compressed(stem.with_suffix(".npz"), **arrays)

    payload = {
        "pass": True,
        "foundation_self_test": foundation_self_test,
        "reflection_self_test": reflection_self_test,
        "N": n,
        "L": length,
        "dt": dt,
        "warmup": args.warmup,
        "measure": args.measure,
        "beta4": BETA4,
        "positive_beta3_grid": positive_beta3,
        "linear_q4p8_growth_invariance": linear_rows,
        "operational_power_crossing_fraction": POWER_CROSSING,
        "operational_crossing_interval": crossing_interval,
        "scan_summary": scan_summary,
        "mirror_summary": {
            "plus_0.30": summarize(rp, n_ext, n),
            "minus_0.30": summarize(rm, n_ext, n),
        },
        "sign_pair": pair_diagnostics,
        "npz_file": stem.with_suffix(".npz").name,
    }

    with open(stem.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# simulation/round234_P1_0.py
# 依赖：numpy；round231_P1_0.py 必须位于同一目录。
# 默认运行：
#   python3 simulation/round234_P1_0.py
#
# 目标：在 round231 已验证 RHS/Jacobian/Benettin 流程上，做
#       N=64/96/128、dt=0.002、warmup=measure=100 的公共低 q 初值
#       分辨率收敛核验。不重写任何已验证常量、RHS 或切线性化。
#
# 时间步约束（决策层 #234 核验）：四阶色散 linop 虚部 ~ q^4，RK4 虚轴
#   稳定要求 max_z = dt*max|linop.imag| < 2.8。dt=0.0025 时 N=128 的
#   max_z≈3.0 会触发 round231 的稳定性断言，故三分辨率统一用 dt=0.002
#   （N=64/96/128 的 max_z 分别约 0.21/0.89/2.41，全部通过）。

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# 显式复用 round231 的 VERIFIED 地基；本轮不重新推导这些对象。
from round231_P1_0 import (  # noqa: F401
    DELTA,
    S,
    I,
    D,
    PUMP,
    cw_residual,
    analytic_eigenvalues,
    tangent_matrix,
    make_grid,
    make_linop,
    state_rhs,
    tangent_rhs,
    rk4_state,
    rk4_tangent,
    spec_norm,
    run_case,
    plot_convergence,
    self_test as round231_self_test,
)

# 三个 N 使用同一 L，因此 dq=0.3；公共低 q 模取 j=1..20，即 q=0.3..6.0。
J_COMMON = 20


def common_initial_state(N, mask, n_ext, seed):
    """构造跨 N 表示同一物理场的 Fourier 初值。

    所有分辨率写入完全相同的 j=1..20 复 Fourier 系数，q>6 模式置零；
    q=4.8 再额外加入同一实余弦扰动。
    """
    assert 1 <= n_ext <= J_COMMON

    rng = np.random.default_rng(seed)
    uhat = np.zeros(N, dtype=complex)
    uhat[0] = np.sqrt(I) * N

    for j in range(1, J_COMMON + 1):
        z_plus = rng.standard_normal() + 1j * rng.standard_normal()
        z_minus = rng.standard_normal() + 1j * rng.standard_normal()
        uhat[j] = 1.0e-3 * N * z_plus
        uhat[-j] = 1.0e-3 * N * z_minus

    # 与 round231 相同的 q=4.8 显式余弦小扰动。
    uhat[n_ext] += 2.0e-3 * N
    uhat[-n_ext] += 2.0e-3 * N

    return uhat * mask


def assert_common_bases(bases):
    """硬断言不同 N 的初值在公共低 q 模上是同一物理场。"""
    Ns = sorted(bases)
    reference = bases[Ns[0]] / Ns[0]

    for N in Ns:
        uhat = bases[N] / N

        for j in range(J_COMMON + 1):
            assert abs(uhat[j] - reference[j]) < 1.0e-14, (N, j)
            assert abs(uhat[-j] - reference[-j]) < 1.0e-14, (N, -j)

        # 除零模和 j=±1..±20 外，其余模式必须严格为零。
        high = bases[N].copy()
        high[0] = 0
        high[1 : J_COMMON + 1] = 0
        high[-J_COMMON:] = 0
        assert np.all(high == 0), N


def low_q_coefficients(result, N):
    """返回归一化的公共低 q 复 Fourier 系数。"""
    return result["uhat"][1 : J_COMMON + 1] / N


def outside_common_band_power_fraction(result, N):
    """q>6 模式在总功率中的占比。

    对 N=64，这包含刚高于公共模的 q=6.3；对 N=96/128，可检验更高分辨率
    新允许的模式是否携带不可忽略能量。
    """
    power = np.abs(result["uhat"]) ** 2 / N**2
    total_power = float(power.sum())

    selected = np.ones(N, dtype=bool)
    selected[0] = False
    selected[1 : J_COMMON + 1] = False
    selected[-J_COMMON:] = False

    return float(power[selected].sum() / max(total_power, 1.0e-300))


def convergence_block(results, L):
    """按 case 汇总跨分辨率差值。"""
    grouped = {}
    for result in results:
        case, N_text = result["name"].rsplit("_N", 1)
        N = int(N_text)
        grouped.setdefault(case, []).append((N, result))

    output = {}

    for case, rows in grouped.items():
        rows.sort(key=lambda item: item[0])
        coefficients = {N: low_q_coefficients(result, N) for N, result in rows}

        table = []
        for N, result in rows:
            table.append(
                {
                    "N": int(N),
                    "lambda_second_half": float(result["lambda_second_half"]),
                    "lambda_second_half_stderr": float(
                        result["lambda_second_half_stderr"]
                    ),
                    "relative_final_residual": float(
                        result["relative_final_residual"]
                    ),
                    "dominant_q": float(result["dominant_q"]),
                    "q4p8_power_fraction": float(
                        result["q4p8_power_fraction"]
                    ),
                    "q4p8_amp_cv_last20": float(
                        result["q4p8_amp_cv_last20"]
                    ),
                    "outside_q6_power_fraction": (
                        outside_common_band_power_fraction(result, N)
                    ),
                    "classification": result["classification"],
                }
            )

        pairs = []
        for (N1, r1), (N2, r2) in zip(rows, rows[1:]):
            complex_diff = np.abs(coefficients[N1] - coefficients[N2])
            amplitude_diff = np.abs(
                np.abs(coefficients[N1]) - np.abs(coefficients[N2])
            )

            pairs.append(
                {
                    "N_pair": [int(N1), int(N2)],
                    "lambda_second_half_diff": float(
                        abs(r1["lambda_second_half"] - r2["lambda_second_half"])
                    ),
                    "lambda_stderr_max": float(
                        max(
                            r1["lambda_second_half_stderr"],
                            r2["lambda_second_half_stderr"],
                        )
                    ),
                    "low_q_complex_max_diff": float(complex_diff.max()),
                    "low_q_amplitude_max_diff": float(amplitude_diff.max()),
                    "q4p8_power_fraction_diff": float(
                        abs(
                            r1["q4p8_power_fraction"]
                            - r2["q4p8_power_fraction"]
                        )
                    ),
                    "relative_residual_diff": float(
                        abs(
                            r1["relative_final_residual"]
                            - r2["relative_final_residual"]
                        )
                    ),
                }
            )

        output[case] = {
            "common_q_grid": [
                float(2.0 * np.pi * j / L)
                for j in range(1, J_COMMON + 1)
            ],
            "table": table,
            "pairs": pairs,
        }

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="b4")
    parser.add_argument("--Ns", default="64,96,128")
    parser.add_argument("--L", type=float, default=20.0 * np.pi / 3.0)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--warmup", type=float, default=100.0)
    parser.add_argument("--measure", type=float, default=100.0)
    parser.add_argument("--renorm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--outdir", default="simulation/out")
    args = parser.parse_args()

    case_map = {
        "b0": (0.0, 0.0),
        "b4": (0.0, 1.0),
    }

    Ns = [int(token) for token in args.Ns.split(",") if token.strip()]
    L = args.L

    bases = {}
    grids = {}

    for N in Ns:
        k, mask = make_grid(N, L)
        grids[N] = (k, mask)

        n_ext = int(round(4.8 * L / (2.0 * np.pi)))
        assert 0 < n_ext < N
        assert abs(k[n_ext] - 4.8) < 1.0e-10

        # 继承 round231 的硬自校验：CW、Jacobian=1e-10、q=4.8 RK4 增长。
        round231_self_test(N, L, args.dt, k, mask, n_ext)

        bases[N] = common_initial_state(N, mask, n_ext, args.seed)

    # 新逻辑自身的硬断言：所有 N 的公共低 q 初值必须是同一物理场。
    assert_common_bases(bases)

    results = []

    for case in (token.strip() for token in args.cases.split(",")):
        beta3, beta4 = case_map[case]

        for N in Ns:
            k, mask = grids[N]
            n_ext = int(round(4.8 * L / (2.0 * np.pi)))

            print(
                f"=== running {case} at N={N}, dt={args.dt}, "
                f"warmup={args.warmup}, measure={args.measure} ===",
                flush=True,
            )

            result = run_case(
                f"{case}_N{N}",
                beta3,
                beta4,
                bases[N],
                N,
                L,
                args.dt,
                k,
                mask,
                n_ext,
                args,
            )
            results.append(result)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = outdir / "round234_P1_0"

    array_keys = {
        "t",
        "lambda_segment",
        "lambda_cumulative",
        "external_amp",
        "uhat",
    }

    arrays = {}
    for result in results:
        for key in array_keys:
            arrays[f"{result['name']}_{key}"] = result[key]
    np.savez_compressed(stem.with_suffix(".npz"), **arrays)

    scan_summary = []
    for result in results:
        scalar = {
            key: value
            for key, value in result.items()
            if key not in array_keys
        }
        scan_summary.append(scalar)

    payload = {
        "pass": True,
        "common_initial_q_max": float(
            2.0 * np.pi * J_COMMON / L
        ),
        "dt": float(args.dt),
        "warmup": float(args.warmup),
        "measure": float(args.measure),
        "scan_summary": scan_summary,
        "convergence": convergence_block(results, L),
    }

    with open(stem.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    plot_convergence(results, stem.with_suffix(".png"))
    print(json.dumps(payload["convergence"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

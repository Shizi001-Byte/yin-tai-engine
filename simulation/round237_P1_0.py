#!/usr/bin/env python3
# simulation/round237_P1_0.py
# 依赖：numpy；round231_P1_0.py、round234_P1_0.py 必须位于同一目录。
#
# 默认运行：
#   python3 simulation/round237_P1_0.py
#
# 目标：在 round231/round234 已验证地基上检验
#       (beta3=+0.3, u0) 与 (beta3=-0.3, R u0) 的反射共轭关系，
#       并测量 beta3 非线性终态是否成为反向行波。
#       不重写、不改写任何已验证 RHS、切线性化或 Benettin 流程。

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from round231_P1_0 import (  # noqa: F401
    make_grid,
    make_linop,
    state_rhs,
    tangent_rhs,
    spec_norm,
    rk4_state,
    run_case,
    self_test as round231_self_test,
)
from round234_P1_0 import common_initial_state, J_COMMON  # noqa: F401

BETA3_ABS = 0.3
REFLECTION_ERROR_TOL = 1.0e-6


def reflect_spectral(uhat):
    """空间反射 R: u(x)->u(-x)，Fourier 系数满足 (Ru)_q = u_{-q}。"""
    n = uhat.shape[0]
    idx = (-np.arange(n)) % n
    return uhat[idx]


def reflection_operator_self_test(n, k, mask, beta3, beta4):
    """机器精度校验 beta3 反号 + 空间反射的算子恒等式。"""
    rng = np.random.default_rng(237031)

    linop_plus = make_linop(k, +beta3, beta4)
    linop_minus = make_linop(k, -beta3, beta4)

    # 只在 dealiasing 活跃模式上比较：Nyquist(j=N/2) 自配对 q≡-q，奇阶
    # β3 项反射后要求 q³ 变号却映到自身，恒等式在该点天然不成立（残差
    # 2|β3||q|³/6，N=128 约 707.8）；该模式已被 mask 屏蔽、从不参与动力学，
    # 故必须排除，否则全数组比较会误报。决策层 #237 实测活跃模式误差 1.1e-13。
    active = np.flatnonzero(mask)
    err_linop = float(
        np.max(np.abs((linop_minus - reflect_spectral(linop_plus))[active]))
    )

    u0 = np.zeros(n, dtype=complex)
    v0 = np.zeros(n, dtype=complex)
    u0[active] = n * (
        rng.standard_normal(active.size)
        + 1j * rng.standard_normal(active.size)
    )
    v0[active] = n * (
        rng.standard_normal(active.size)
        + 1j * rng.standard_normal(active.size)
    )

    ru0 = reflect_spectral(u0)
    rv0 = reflect_spectral(v0)

    rhs_plus = state_rhs(u0, linop_plus, mask)
    rhs_minus = state_rhs(ru0, linop_minus, mask)
    err_state = spec_norm(
        reflect_spectral(rhs_plus) - rhs_minus, n
    )

    tan_plus = tangent_rhs(u0, v0, linop_plus, mask)
    tan_minus = tangent_rhs(ru0, rv0, linop_minus, mask)
    err_tangent = spec_norm(
        reflect_spectral(tan_plus) - tan_minus, n
    )

    assert err_linop < 1.0e-12, err_linop
    assert err_state < 1.0e-10, err_state
    assert err_tangent < 1.0e-10, err_tangent

    return {
        "linop_reflection_max_error": err_linop,
        "state_rhs_reflection_rms_error": float(err_state),
        "tangent_rhs_reflection_rms_error": float(err_tangent),
    }


def spectral_chirality(uhat):
    """正 q 与负 q 功率的归一化不对称度；空间反射使其反号。"""
    n = uhat.shape[0]
    j = np.arange(1, n // 2)
    p_plus = float(np.sum(np.abs(uhat[j]) ** 2))
    p_minus = float(np.sum(np.abs(uhat[-j]) ** 2))
    denom = p_plus + p_minus
    return float((p_plus - p_minus) / denom) if denom > 0.0 else 0.0


def estimate_drift(
    uhat,
    linop,
    mask,
    k,
    dt,
    sample_time=0.5,
    total_time=1.0,
    relative_power_floor=1.0e-8,
):
    """由 û_q(t)=û_q(0) exp(-i q v t) 估计刚性漂移速度。

    若所有显著模给出近似相同的 v，则终态是刚性行波；
    若估计值高度分散，则不是简单行波。
    """
    n = uhat.shape[0]
    n_total = int(round(total_time / dt))
    n_sample = int(round(sample_time / dt))
    assert n_sample > 0
    assert n_total % n_sample == 0

    total_spectral_power = float(np.sum(np.abs(uhat) ** 2))
    power_floor = relative_power_floor * total_spectral_power

    u = uhat.copy()
    records = []
    current_time = 0.0

    for _ in range(n_total // n_sample):
        for _ in range(n_sample):
            u = rk4_state(u, dt, linop, mask)

        current_time += sample_time

        for j in range(1, n // 2):
            if not (mask[j] and mask[-j]):
                continue

            a0 = uhat[j]
            a1 = u[j]
            if abs(a0) ** 2 < power_floor or abs(a1) ** 2 < power_floor:
                continue

            phase = float(np.angle(a1 * np.conj(a0)))
            velocity = -phase / (float(k[j]) * current_time)
            weight = float(abs(a0) * abs(a1))
            records.append((velocity, weight))

    if not records:
        return {
            "drift_velocity_weighted_mean": 0.0,
            "drift_velocity_weighted_std": 0.0,
            "drift_velocity_median": 0.0,
            "n_drift_modes": 0,
        }

    data = np.asarray(records, dtype=float)
    velocities = data[:, 0]
    weights = data[:, 1]
    weights /= weights.sum()

    mean = float(np.sum(velocities * weights))
    variance = float(
        np.sum(weights * (velocities - mean) ** 2)
    )
    std = float(np.sqrt(variance))

    return {
        "drift_velocity_weighted_mean": mean,
        "drift_velocity_weighted_std": std,
        "drift_velocity_median": float(np.median(velocities)),
        "n_drift_modes": int(len(records)),
    }


def summarize_result(result, n_ext, drift):
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

    uhat = result["uhat"]
    n = uhat.shape[0]

    summary = {key: result[key] for key in scalar_keys}
    summary.update(drift)
    summary.update(
        {
            "spectral_chirality": spectral_chirality(uhat),
            "q4p8_plus_q_amp": float(abs(uhat[n_ext]) / n),
            "q4p8_minus_q_amp": float(abs(uhat[-n_ext]) / n),
            "linear_sideband_velocity_at_dominant_q": float(
                -result["beta3"] * result["dominant_q"] ** 2 / 6.0
            ),
        }
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--L", type=float, default=20.0 * np.pi / 3.0)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--warmup", type=float, default=100.0)
    parser.add_argument("--measure", type=float, default=100.0)
    parser.add_argument("--renorm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=237)
    parser.add_argument("--beta4", type=float, default=1.0)
    parser.add_argument("--drift-total", type=float, default=1.0)
    parser.add_argument("--drift-sample", type=float, default=0.5)
    parser.add_argument("--outdir", default="simulation/out")
    args = parser.parse_args()

    n, length, dt = args.N, args.L, args.dt
    k, mask = make_grid(n, length)

    n_ext = int(round(4.8 * length / (2.0 * np.pi)))
    assert 0 < n_ext < n
    assert abs(k[n_ext] - 4.8) < 1.0e-10

    # round231 的硬自校验：CW、2x2 Jacobian、q=4.8 RK4 增长。
    foundation_self_test = round231_self_test(
        n, length, dt, k, mask, n_ext
    )

    # 本轮新增的反射共轭硬自校验。
    reflection_self_test = reflection_operator_self_test(
        n, k, mask, BETA3_ABS, args.beta4
    )

    # 非对称低 q 初值；相反 beta3 使用其空间反射副本。
    base_plus = common_initial_state(n, mask, n_ext, args.seed)
    base_minus = reflect_spectral(base_plus)

    initial_chirality_plus = spectral_chirality(base_plus)
    initial_chirality_minus = spectral_chirality(base_minus)
    assert abs(initial_chirality_plus) > 1.0e-10, (
        "当前种子未产生足够显著的非对称初值"
    )
    assert abs(initial_chirality_plus + initial_chirality_minus) < 1.0e-14

    sim_args = argparse.Namespace(
        warmup=args.warmup,
        measure=args.measure,
        renorm=args.renorm,
        seed=args.seed,
    )

    print(
        f"=== beta3=+{BETA3_ABS}, beta4={args.beta4}, N={n} ===",
        flush=True,
    )
    result_plus = run_case(
        f"b3p_b4_{args.beta4:g}",
        +BETA3_ABS,
        args.beta4,
        base_plus,
        n,
        length,
        dt,
        k,
        mask,
        n_ext,
        sim_args,
    )

    print(
        f"=== beta3=-{BETA3_ABS}, beta4={args.beta4}, N={n} ===",
        flush=True,
    )
    result_minus = run_case(
        f"b3m_b4_{args.beta4:g}",
        -BETA3_ABS,
        args.beta4,
        base_minus,
        n,
        length,
        dt,
        k,
        mask,
        n_ext,
        sim_args,
    )

    drift_plus = estimate_drift(
        result_plus["uhat"],
        make_linop(k, +BETA3_ABS, args.beta4),
        mask,
        k,
        dt,
        sample_time=args.drift_sample,
        total_time=args.drift_total,
    )
    drift_minus = estimate_drift(
        result_minus["uhat"],
        make_linop(k, -BETA3_ABS, args.beta4),
        mask,
        k,
        dt,
        sample_time=args.drift_sample,
        total_time=args.drift_total,
    )

    summary_plus = summarize_result(result_plus, n_ext, drift_plus)
    summary_minus = summarize_result(result_minus, n_ext, drift_minus)

    final_reflection_error = float(
        spec_norm(
            result_minus["uhat"]
            - reflect_spectral(result_plus["uhat"]),
            n,
        )
        / max(spec_norm(result_plus["uhat"], n), 1.0e-300)
    )

    pair_diagnostics = {
        "final_reflection_relative_error": final_reflection_error,
        "reflection_conjugacy_holds": (
            final_reflection_error < REFLECTION_ERROR_TOL
        ),
        "lambda_second_half_abs_diff": float(
            abs(
                result_plus["lambda_second_half"]
                - result_minus["lambda_second_half"]
            )
        ),
        "lambda_stderr_max": float(
            max(
                result_plus["lambda_second_half_stderr"],
                result_minus["lambda_second_half_stderr"],
            )
        ),
        "drift_velocity_sum": float(
            drift_plus["drift_velocity_weighted_mean"]
            + drift_minus["drift_velocity_weighted_mean"]
        ),
        "chirality_sum": float(
            summary_plus["spectral_chirality"]
            + summary_minus["spectral_chirality"]
        ),
        "counter_propagating_pair": bool(
            drift_plus["drift_velocity_weighted_mean"]
            * drift_minus["drift_velocity_weighted_mean"]
            < 0.0
        ),
        "rigid_drift_pair": bool(
            drift_plus["drift_velocity_weighted_std"] < 0.05
            and drift_minus["drift_velocity_weighted_std"] < 0.05
            and abs(drift_plus["drift_velocity_weighted_mean"])
            > 3.0 * drift_plus["drift_velocity_weighted_std"]
            and abs(drift_minus["drift_velocity_weighted_mean"])
            > 3.0 * drift_minus["drift_velocity_weighted_std"]
        ),
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = outdir / "round237_P1_0"

    array_keys = {
        "t",
        "lambda_segment",
        "lambda_cumulative",
        "external_amp",
        "uhat",
    }
    arrays = {}
    for result in (result_plus, result_minus):
        for key in array_keys:
            arrays[f"{result['name']}_{key}"] = result[key]
    np.savez_compressed(stem.with_suffix(".npz"), **arrays)

    payload = {
        "pass": True,
        "foundation_self_test": foundation_self_test,
        "reflection_operator_self_test": reflection_self_test,
        "N": n,
        "L": length,
        "dt": dt,
        "warmup": args.warmup,
        "measure": args.measure,
        "common_q_max": float(2.0 * np.pi * J_COMMON / length),
        "initial_chirality_plus": initial_chirality_plus,
        "initial_chirality_minus": initial_chirality_minus,
        "runs": [summary_plus, summary_minus],
        "sign_pair": pair_diagnostics,
        "npz_file": stem.with_suffix(".npz").name,
    }

    with open(stem.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

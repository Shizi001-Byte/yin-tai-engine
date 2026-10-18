#!/usr/bin/env python3
# simulation/round243_P1_transition.py
# 依赖：numpy；round231/234/237/240 脚本必须位于同一目录。
#
# 默认完整运行：
#   python3 simulation/round243_P1_transition.py
#
# 目标：
#   固定 beta4=1，在 beta3=0.15..0.20 上以 0.01 步长加密；
#   warmup>=100、measure>=200；
#   N=128/dt=0.002 与 N=160/dt=0.001 双分辨率；
#   seed=240,241 多种子。
#
# 新增诊断只有两项：
#   1) q=4.8 外带功率与 lambda 的跨 N/seed 鲁棒性；
#   2) 从 measure 段时间序列拟合指数弛豫时间 tau，检验临界慢化。
#
# 不重写任何已验证物理核。

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
    self_test as round231_self_test,
)
from round234_P1_0 import (  # noqa: E402
    J_COMMON,
    assert_common_bases,
    common_initial_state,
)
from round237_P1_0 import reflection_operator_self_test  # noqa: E402
from round240_P1_0 import BETA4, Q_OUT, summarize  # noqa: E402


def parse_resolutions(text):
    configs = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            n_text, dt_text = token.split(":", 1)
            n = int(n_text)
            dt = float(dt_text)
        else:
            n = int(token)
            dt = 0.002 if n <= 128 else 0.001
        configs.append((n, dt))
    return configs


def parse_seeds(text):
    return [int(token) for token in text.split(",") if token.strip()]


def exponential_relaxation_fit(t, y):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    ptp = float(np.ptp(y))
    scale = float(np.max(np.abs(y)))
    if ptp < 1.0e-8 * max(scale, 1.0e-300):
        return {
            "tau": None,
            "asymptote": float(np.mean(y)),
            "amplitude": 0.0,
            "r2": 1.0,
            "reason": "flat_in_measurement_window",
        }

    best = None
    tau_grid = np.exp(
        np.linspace(
            np.log(2.0),
            np.log(max(10.0, 1.5 * t[-1])),
            80,
        )
    )

    for tau in tau_grid:
        design = np.column_stack(
            [np.ones_like(t), np.exp(-t / tau)]
        )
        coeff, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ coeff
        sse = float(residual @ residual)
        if best is None or sse < best[0]:
            best = (sse, float(tau), coeff)

    sse, tau, coeff = best
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - sse / sst if sst > 0.0 else 1.0

    return {
        "tau": float(tau),
        "asymptote": float(coeff[0]),
        "amplitude": float(coeff[1]),
        "r2": float(r2),
        "reason": "single_exponential_empirical_fit",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta3-grid", default="0.15,0.16,0.17,0.18,0.19,0.20")
    parser.add_argument("--resolutions", default="128:0.002,160:0.001")
    parser.add_argument("--seeds", default="240,241")
    parser.add_argument("--L", type=float, default=20.0 * np.pi / 3.0)
    parser.add_argument("--warmup", type=float, default=100.0)
    parser.add_argument("--measure", type=float, default=200.0)
    parser.add_argument("--renorm", type=float, default=1.0)
    parser.add_argument("--outdir", default="simulation/out")
    args = parser.parse_args()

    beta3_grid = [float(token) for token in args.beta3_grid.split(",") if token.strip()]
    resolutions = parse_resolutions(args.resolutions)
    seeds = parse_seeds(args.seeds)

    assert args.warmup >= 100.0
    assert args.measure >= 200.0
    assert abs(args.renorm - 1.0) < 1.0e-12
    assert len(beta3_grid) >= 5
    assert beta3_grid[0] <= 0.15 + 1.0e-12
    assert beta3_grid[-1] >= 0.20 - 1.0e-12
    assert max(np.diff(beta3_grid)) <= 0.0101
    assert {128, 160}.issubset({n for n, _ in resolutions})
    assert len(seeds) >= 2

    length = args.L
    grid_data = {}
    n_ext_by_N = {}

    for n, dt in resolutions:
        k, mask = make_grid(n, length)
        n_ext = int(round(Q_OUT * length / (2.0 * np.pi)))
        assert 0 < n_ext < n
        assert abs(k[n_ext] - Q_OUT) < 1.0e-10
        round231_self_test(n, length, dt, k, mask, n_ext)
        grid_data[n] = (k, mask, dt)
        n_ext_by_N[n] = n_ext

    reflection_checks = {}
    for n, (k, mask, _dt) in grid_data.items():
        reflection_checks[n] = reflection_operator_self_test(n, k, mask, 0.30, BETA4)

    linear_rows = []
    real_gains = []
    for b3 in beta3_grid:
        ev = analytic_eigenvalues(Q_OUT, b3, BETA4)
        j = int(np.argmax(ev.real))
        real_gains.append(float(ev[j].real))
        linear_rows.append({
            "beta3": float(b3),
            "q": Q_OUT,
            "max_real_growth": float(ev[j].real),
            "max_eigenvalue_imag": float(ev[j].imag),
            "linear_velocity": float(-b3 * Q_OUT**2 / 6.0),
        })
    assert max(real_gains) - min(real_gains) < 1.0e-10

    bases = {}
    for seed in seeds:
        bases_by_n = {}
        for n, _dt in resolutions:
            _k, mask, _ = grid_data[n]
            bases_by_n[n] = common_initial_state(n, mask, n_ext_by_N[n], seed)
            bases[(n, seed)] = bases_by_n[n]
        assert_common_bases(bases_by_n)

    sim_args = argparse.Namespace(warmup=args.warmup, measure=args.measure, renorm=args.renorm, seed=None)
    scan = []

    for b3 in beta3_grid:
        for seed in seeds:
            for n, dt in resolutions:
                k, mask, _ = grid_data[n]
                n_ext = n_ext_by_N[n]
                name = f"b3_{b3:.2f}_N{n}_s{seed}"
                print(f"=== {name}: warmup={args.warmup}, measure={args.measure}, dt={dt} ===", flush=True)
                run_args = argparse.Namespace(**vars(sim_args))
                run_args.seed = seed
                result = run_case(name, b3, BETA4, bases[(n, seed)], n, length, dt, k, mask, n_ext, run_args)
                summary = summarize(result, n_ext, n)
                summary.update({
                    "N": int(n),
                    "seed": int(seed),
                    "dt": float(dt),
                    "tau_external_amp": exponential_relaxation_fit(result["t"], result["external_amp"]),
                    "tau_lambda_segment": exponential_relaxation_fit(result["t"], result["lambda_segment"]),
                })
                assert np.isfinite(summary["lambda_second_half"])
                assert np.isfinite(summary["lambda_last20"])
                assert 0.0 <= summary["q4p8_power_fraction"] <= 1.0
                scan.append(summary)
                result["_store_name"] = name
                scan[-1]["_result"] = result

    lookup = {(row["beta3"], row["seed"], row["N"]): row for row in scan}
    robustness = []
    warnings = []

    for b3 in beta3_grid:
        item = {"beta3": float(b3), "resolution_pairs": [], "seed_spread": []}
        for seed in seeds:
            rows = [lookup[(b3, seed, n)] for n, _ in resolutions]
            for left, right in zip(rows[:-1], rows[1:]):
                diff = {
                    "seed": int(seed),
                    "N_pair": [int(left["N"]), int(right["N"])],
                    "lambda_second_half_diff": float(abs(left["lambda_second_half"] - right["lambda_second_half"])),
                    "lambda_last20_diff": float(abs(left["lambda_last20"] - right["lambda_last20"])),
                    "q4p8_power_fraction_diff": float(abs(left["q4p8_power_fraction"] - right["q4p8_power_fraction"])),
                    "residual_diff": float(abs(left["relative_final_residual"] - right["relative_final_residual"])),
                }
                item["resolution_pairs"].append(diff)
                if diff["lambda_second_half_diff"] > 0.03:
                    warnings.append(f"beta3={b3}, seed={seed}: lambda N diff={diff['lambda_second_half_diff']:.3f}")
                if diff["q4p8_power_fraction_diff"] > 0.03:
                    warnings.append(f"beta3={b3}, seed={seed}: q4.8 N diff={diff['q4p8_power_fraction_diff']:.3f}")
        for n, _dt in resolutions:
            rows = [lookup[(b3, seed, n)] for seed in seeds]
            lambdas = [row["lambda_second_half"] for row in rows]
            lambda_last20 = [row["lambda_last20"] for row in rows]
            powers = [row["q4p8_power_fraction"] for row in rows]
            spread = {
                "N": int(n),
                "lambda_second_half_range": [float(min(lambdas)), float(max(lambdas))],
                "lambda_last20_range": [float(min(lambda_last20)), float(max(lambda_last20))],
                "q4p8_power_fraction_range": [float(min(powers)), float(max(powers))],
                "classifications": sorted({row["classification"] for row in rows}),
            }
            item["seed_spread"].append(spread)
            if max(powers) - min(powers) > 0.05:
                warnings.append(f"beta3={b3}, N={n}: seed q4.8 spread={max(powers)-min(powers):.3f}")
        robustness.append(item)

    transition_curves = []
    for n, _dt in resolutions:
        for seed in seeds:
            rows = [lookup[(b3, seed, n)] for b3 in beta3_grid]
            powers = np.array([row["q4p8_power_fraction"] for row in rows])
            drops = np.diff(powers)
            sharp_idx = int(np.argmin(drops))
            crossing_interval = None
            for j, drop in enumerate(drops):
                if powers[j] >= 0.05 and powers[j + 1] < 0.05:
                    crossing_interval = [float(beta3_grid[j]), float(beta3_grid[j + 1])]
                    break
            transition_curves.append({
                "N": int(n),
                "seed": int(seed),
                "beta3": [float(b) for b in beta3_grid],
                "q4p8_power_fraction": [float(x) for x in powers],
                "lambda_last20": [float(row["lambda_last20"]) for row in rows],
                "lambda_second_half": [float(row["lambda_second_half"]) for row in rows],
                "tau_external_amp": [row["tau_external_amp"]["tau"] for row in rows],
                "sharpest_adjacent_drop": {
                    "interval": [float(beta3_grid[sharp_idx]), float(beta3_grid[sharp_idx + 1])],
                    "drop": float(drops[sharp_idx]),
                },
                "operational_5pct_crossing_interval": crossing_interval,
                "classifications": [row["classification"] for row in rows],
            })

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = outdir / "round243_P1_transition"
    array_keys = {"t", "lambda_segment", "lambda_cumulative", "external_amp", "uhat"}
    arrays = {}
    for row in scan:
        result = row.pop("_result")
        prefix = result["_store_name"]
        for key in array_keys:
            arrays[f"{prefix}_{key}"] = result[key]
    np.savez_compressed(stem.with_suffix(".npz"), **arrays)

    payload = {
        "pass": True,
        "reflection_operator_self_test": reflection_checks,
        "beta4": BETA4,
        "q_outer": Q_OUT,
        "L": float(length),
        "warmup": float(args.warmup),
        "measure": float(args.measure),
        "renorm": float(args.renorm),
        "resolutions": [{"N": int(n), "dt": float(dt)} for n, dt in resolutions],
        "seeds": seeds,
        "common_q_max": float(2.0 * np.pi * J_COMMON / length),
        "linear_q4p8_growth_invariance": linear_rows,
        "scan_summary": scan,
        "robustness": robustness,
        "transition_curves": transition_curves,
        "operational_warning_thresholds": {
            "resolution_lambda_diff": 0.03,
            "resolution_q4p8_diff": 0.03,
            "seed_q4p8_spread": 0.05,
        },
        "warnings": warnings,
        "npz_file": stem.with_suffix(".npz").name,
    }
    with open(stem.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(json.dumps({
        "pass": True,
        "n_runs": len(scan),
        "warnings": warnings,
        "transition_curves": transition_curves,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

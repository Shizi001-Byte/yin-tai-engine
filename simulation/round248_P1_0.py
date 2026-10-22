#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# round248_down_extend.py  —— #248 闭合迟滞下臂（工程前置）
# 目标：从 β3=0.22 的 common 平衡态出发，绝热降到 β3=0.10（步长 0.005），
#       seed 240/242 各自，找 L/M 态的失稳点（下临界点）。
# 复用已验证物理核 round246_basin.evolve / classify；只追加：
#   1) DOWN_B3 扩展到 0.10；2) 每步同时落盘 uhat(npz) 以支持断点续跑。
# 用法：python3 round248_down_extend.py --smoke     # 8 单位，秒级，验管道
#       setsid nohup python3 round248_down_extend.py > round248.log 2>&1 &
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from round231_P1_0 import make_grid                    # noqa: E402
from round234_P1_0 import common_initial_state         # noqa: E402
from round240_P1_0 import Q_OUT                         # noqa: E402
from round246_basin import evolve, L, N128, DT128       # noqa: E402

OUTP = HERE / "out" / "round248_down_extend.json"
NPZP = HERE / "out" / "round248_down_extend.npz"

# 0.22 降到 0.10，步长 0.005 -> 25 个点
DOWN_B3_FULL = [round(0.22 - 0.005 * i, 3) for i in range(25)]
SEEDS = [240, 242]


def load_state():
    d = json.load(open(OUTP)) if OUTP.exists() else {"rows": []}
    npz = dict(np.load(NPZP)) if NPZP.exists() else {}
    return d, npz


def save_state(d, npz):
    OUTP.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTP.with_suffix(".json.tmp")
    json.dump(d, open(tmp, "w"), indent=2, ensure_ascii=False)
    tmp.replace(OUTP)
    np.savez_compressed(NPZP, **npz)


def emit(rows, row, seed):
    row = dict(row)
    row["seed"] = int(seed)
    for i, x in enumerate(rows):
        if x["seed"] == seed and abs(float(x["beta3"]) - float(row["beta3"])) < 1e-9:
            rows[i] = row
            return
    rows.append(row)


def run_seed(d, npz, seed, grid, step_measure):
    k, mask = make_grid(N128, L)
    n_ext = int(round(Q_OUT * L / (2 * np.pi)))
    rows = d.setdefault("rows", [])

    # 断点续跑：沿 grid 顺序找最后一个已落盘 uhat 的点
    uhat = None
    start_i = 0
    for i, b3 in enumerate(grid):
        key = f"s{seed}_b{b3:g}"
        if key in npz:
            uhat = npz[key]
            start_i = i + 1
        else:
            break

    if start_i == 0:
        base = common_initial_state(N128, mask, n_ext, seed)
        row, uhat = evolve(grid[0], N128, DT128, seed, base, 100.0, 200.0)
        emit(rows, row, seed)
        npz[f"s{seed}_b{grid[0]:g}"] = uhat
        save_state(d, npz)
        start_i = 1
        print(f"[s{seed}] start b3={grid[0]:5.3f} ext={row['ext_tail_mean']:.3f} "
              f"{row['attractor']}", flush=True)

    for b3 in grid[start_i:]:
        row, uhat = evolve(b3, N128, DT128, seed, uhat, 0.0, float(step_measure))
        emit(rows, row, seed)
        npz[f"s{seed}_b{b3:g}"] = uhat
        save_state(d, npz)
        print(f"[s{seed}] DOWN b3={b3:5.3f} ext={row['ext_tail_mean']:.3f} "
              f"{row['attractor']:4s} q48={row['q48_final']*100:5.1f}% "
              f"lam={row['lambda_last20']:+.3f}", flush=True)


def summarize(d):
    """独立重算：找每 seed 的失稳点（LOW->HIGH 或 LOW->MID 跳变）。"""
    rows = d.get("rows", [])
    by = {}
    for r in rows:
        by.setdefault(int(r["seed"]), []).append(r)
    out = {}
    for seed, sub in by.items():
        sub.sort(key=lambda r: -float(r["beta3"]))  # 降序：0.22 -> 0.10
        jumps = []
        for prev, cur in zip(sub, sub[1:]):
            de = float(cur["ext_tail_mean"]) - float(prev["ext_tail_mean"])
            if abs(de) > 0.05:
                jumps.append({
                    "bracket": [float(cur["beta3"]), float(prev["beta3"])],
                    "from": prev["attractor"], "to": cur["attractor"],
                    "delta_ext": de,
                })
        out[seed] = {
            "range": [min(float(r["beta3"]) for r in sub),
                      max(float(r["beta3"]) for r in sub)],
            "attractor_path": [(float(r["beta3"]), r["attractor"],
                                float(r["ext_tail_mean"])) for r in sub],
            "jumps": jumps,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", default="240,242")
    ap.add_argument("--step-measure", type=float, default=250.0)
    args = ap.parse_args()

    grid = [0.22, 0.20, 0.18] if args.smoke else DOWN_B3_FULL
    step_measure = 8.0 if args.smoke else args.step_measure
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    d, npz = load_state()
    for seed in seeds:
        run_seed(d, npz, seed, grid, step_measure)

    save_state(d, npz)
    print("\n" + "=" * 64)
    print("【裁决输入】各 seed 降支 attractor 路径与跳变")
    print("=" * 64)
    print(json.dumps(summarize(d), indent=2, ensure_ascii=False))
    print("\n落盘 ->", OUTP, NPZP)


if __name__ == "__main__":
    main()

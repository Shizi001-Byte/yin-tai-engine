#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# round279_branch_track.py —— R276-1 第二步：同一支的 β3 连续扫描（定义 branch）
#
# 背景（#276/#278）：β3=0.18 下至少 3 个 HIGH 定常态共存，谱支撑近乎正交
#   （A: s240/s241 ext=0.48487；B: s237 ext=0.47384；C: s242 ext=0.42800）。
#   "多个分支"和"离散锁模族"在单点测量上不可区分，必须做同支连续扫描。
#
# 判据（主Agent设计，脚本只执行）：
#   - 支的连续性：相邻 β3（步长 0.005）之间 ext 变化 < 0.030 且分类仍为 HIGH → 视为同一支的连续延伸；
#     一步跳变 ≥ 0.050 → 该支在此 β3 处消失/合并/跳支。
#   - 支的合并：从不同锚点出发的两条支在某个 β3 收敛到同一 ext（差 < 0.005）→ 同一吸引子（合并）。
#   - 未定：λ > 1e-3 或 seg_ext 末两段仍在漂移 → 该点记 UNSETTLED，不参与连续性判断。
#   半长：measure=400（临界区附近不足以判稳，出现 UNSETTLED 就加长到 800 再说，不硬裁）。
#
# 每个锚点的终态各自作为起点，每一步用上一步终态续跑（warmup=0），初值直接读 npz。
#
# 用法：python3 round279_branch_track.py [--smoke]
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from round246_basin import evolve, N128, DT128   # noqa: E402

SRC = HERE / "out" / "round276_multiseed_coexistence.npz"
OUTP = HERE / "out" / "round279_branch_track.json"
NPZP = HERE / "out" / "round279_branch_track.npz"
B0 = 0.18
STEPS = [0.005, 0.010, 0.015]          # 上下各三步 → β3 ∈ {0.165 … 0.195}
ANCHORS = [
    ("A", "s240_b0.18_common_m800"),
    ("B", "s237_b0.18_common_m800"),
    ("C", "s242_b0.18_common_m800"),
]
MEASURE = 400.0
SMOKE_MEASURE = 8.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--measure", type=float, default=MEASURE)
    args = ap.parse_args()
    measure = SMOKE_MEASURE if args.smoke else args.measure

    src = dict(np.load(SRC))
    d = json.load(open(OUTP)) if OUTP.exists() else {"rows": [], "measure": measure}
    nz = dict(np.load(NPZP)) if NPZP.exists() else {}

    def save():
        tmp = OUTP.with_suffix(".json.tmp")
        json.dump(d, open(tmp, "w"), indent=2, ensure_ascii=False)
        tmp.replace(OUTP)
        np.savez_compressed(NPZP, **nz)

    print(f"#279 支追踪 β3 基准={B0} 步长±{STEPS} measure={measure:g}")

    for tag, key in ANCHORS:
        if key not in src:
            print(f"  [ERROR] 缺锚点 {key}"); continue
        # 下行
        for sgn, direction in ((-1.0, "down"), (1.0, "up")):
            b3 = B0
            u = src[key]
            for st in STEPS:
                b3 = round(b3 + sgn * st, 6)
                row_key = f"br{tag}_{direction}_{b3:.4f}"
                if row_key in nz:
                    print(f"  [skip] {row_key}"); u = nz[row_key]; continue
                row, uf = evolve(b3, N128, DT128, {"A": 240, "B": 237, "C": 242}[tag], u, 0.0, measure)
                row.update({"branch": tag, "direction": direction, "beta3": b3,
                            "row_key": row_key, "measure": measure})
                d["rows"].append(row); nz[row_key] = uf; save()
                print(f"  {row_key}: ext={row['ext_tail_mean']:.4f} {row['attractor']:4s} "
                      f"q48={row['q48_final']*100:5.1f}% lam={row['lambda_last20']:+.2e} "
                      f"seg={['%.3f' % x for x in row['seg_ext'][-3:]]}", flush=True)
                u = uf

    print("\n【判据汇总】")
    for tag, _ in ANCHORS:
        rows = [r for r in d["rows"] if r["branch"] == tag]
        for direction in ("down", "up"):
            seq = sorted([r for r in rows if r["direction"] == direction],
                         key=lambda r: r["beta3"], reverse=(direction == "down"))
            if not seq:
                continue
            chain = [f"{r['beta3']:.3f}:{r['ext_tail_mean']:.4f}" for r in seq]
            print(f"  支{tag} {direction}: " + " -> ".join(chain))
            for a, b in zip(seq, seq[1:]):
                jump = abs(b["ext_tail_mean"] - a["ext_tail_mean"])
                flag = "JUMP" if jump >= 0.050 else "cont"
                unsettled = "UNSETTLED" if max(abs(a["lambda_last20"]), abs(b["lambda_last20"])) > 1e-3 else ""
                print(f"     {a['beta3']:.3f}->{b['beta3']:.3f}: Δext={jump:.4f} {flag} {unsettled}")

    if args.smoke:
        print("### SMOKE OK ###")


if __name__ == "__main__":
    main()

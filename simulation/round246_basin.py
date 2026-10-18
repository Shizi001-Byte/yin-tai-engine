#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =====================================================================
# 决策层【验证子Agent·类型b 确定性长测程】#246 深度轮
# 背景（#243 已 VERIFIED，见 notes/glle-beta3-0.20-multistability.md）：
#   beta3=0.20 高功率锁模态(ext=0.440/q48~13%) 与 低功率抑制态(ext~0.147/q48~2%)
#   共存、初值依赖、跨 N128/160 稳定；beta3=0.21 非单调回中高态(ext0.343~0.423)，
#   疑 0.19~0.21 吸引子盆地快速重组，可能含第三中功率吸引子(ext~0.343)。
# 本脚本两个互斥实验，回答"稳定域在哪 / 是不是一阶共存":
#   实验A 盆地扫描(basin)：每个 beta3 用独立 common 初值 × 4 seed，measure800，
#         统计各 beta3 下高/中/低吸引子的盆地占比（前向标记、独立初值）。
#   实验B 绝热滞后(hysteresis)：沿用上一 beta3 终态 uhat 续跑(warmup=0)，
#         正向 0.18→0.22、反向 0.22→0.18，画 ext(beta3) 迟滞回线。
#         路径依赖(正向在某点跳、反向在另一点跳)是一阶共存/多稳态的金标准。
# 判据（主Agent设计，不靠单点）：取末 4 段(每段200单位) ext 均值分档
#         HIGH>0.38 / MID 0.28~0.38 / LOW<0.28；四段全程打印，最终人工看 seg_ext。
# 只复用已验证物理核 run_case/make_grid/common_initial_state；不 import、不读取
# 任何计算 worker 的结论，验证代码与 worker 代码路径完全独立。
# 用法：
#   python3 round246_basin.py --smoke      # 冒烟：measure=8 验管道(续跑/分类/落盘)
#   python3 round246_basin.py              # 全矩阵（约35-40min，建议后台）
# =====================================================================
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from round231_P1_0 import make_grid, run_case           # noqa: E402
from round234_P1_0 import common_initial_state          # noqa: E402
from round240_P1_0 import BETA4, Q_OUT                  # noqa: E402

L = 20.0 * np.pi / 3.0
N128, DT128 = 128, 0.002
SEEDS = [240, 241, 237, 242]
BASIN_B3 = [0.19, 0.20, 0.205, 0.21, 0.215, 0.22]
HYST_B3 = [0.18, 0.185, 0.19, 0.195, 0.20, 0.205, 0.21, 0.215, 0.22]


def classify(ext_last_mean):
    if ext_last_mean > 0.38:
        return "HIGH"
    if ext_last_mean >= 0.28:
        return "MID"
    return "LOW"


def evolve(b3, N, dt, seed, base, warmup, measure, renorm=1.0, seg=200.0):
    """跑一个工况，返回剔除 uhat 的可序列化结果 + 内部保留终态供绝热续跑。"""
    k, mask = make_grid(N, L)
    n_ext = int(round(Q_OUT * L / (2 * np.pi)))
    args = argparse.Namespace(warmup=warmup, measure=measure, renorm=renorm, seed=seed)
    r = run_case(f"b3_{b3}_N{N}_s{seed}", b3, BETA4, base, N, L, dt, k, mask, n_ext, args)
    ext = r["external_amp"]
    nseg = max(1, int(measure // seg))
    segs = []
    for i in range(nseg):
        a = int(i * seg)
        b = int(min((i + 1) * seg, measure))
        segs.append(float(ext[a:b].mean()))
    tail = segs[-4:] if len(segs) >= 4 else segs
    tail_mean = float(np.mean(tail))
    row = {
        "beta3": b3, "N": N, "seed": seed, "measure": measure,
        "q48_final": float(r["q4p8_power_fraction"]),
        "lambda_last20": float(r["lambda_last20"]),
        "classification": r["classification"],
        "seg_ext": segs,
        "ext_tail_mean": tail_mean,
        "attractor": classify(tail_mean),
    }
    return row, r["uhat"]


def basin_scan(N, dt, b3s, seeds, measure):
    k, mask = make_grid(N, L)
    n_ext = int(round(Q_OUT * L / (2 * np.pi)))
    rows = []
    for b3 in b3s:
        for sd in seeds:
            base = common_initial_state(N, mask, n_ext, sd)
            print(f"[A] basin b3={b3} N={N} s={sd} measure={measure}", flush=True)
            row, _ = evolve(b3, N, dt, sd, base, 100.0, float(measure))
            print(f"      tail={['%.3f' % x for x in row['seg_ext'][-4:]]} "
                  f"-> {row['attractor']:4s} q48={row['q48_final']*100:5.1f}% "
                  f"lam={row['lambda_last20']:+.3f}", flush=True)
            rows.append(row)
    # 盆地占比汇总
    print("\n" + "=" * 64)
    print("【A 盆地占比】beta3 -> HIGH/MID/LOW (各 seed 的吸引子)")
    print("=" * 64)
    for b3 in b3s:
        sub = [x for x in rows if x["beta3"] == b3]
        cnt = {"HIGH": 0, "MID": 0, "LOW": 0}
        for x in sub:
            cnt[x["attractor"]] += 1
        tag = " ".join(f"s{x['seed']}:{x['attractor']}" for x in sub)
        print(f"  b3={b3:5.3f}  H={cnt['HIGH']} M={cnt['MID']} L={cnt['LOW']}   {tag}")
    return rows


def hysteresis(N, dt, seed, b3s, step_measure):
    """正向从最低 b3 平衡态出发逐点续跑；反向从最高 b3 平衡态出发逐点续跑。"""
    k, mask = make_grid(N, L)
    n_ext = int(round(Q_OUT * L / (2 * np.pi)))

    # 正向（b3 升序）：首点 common 初值 warmup100 measure200 平衡，其后沿用上点终态
    base = common_initial_state(N, mask, n_ext, seed)
    up = []
    for i, b3 in enumerate(b3s):
        if i == 0:
            row, uhat = evolve(b3, N, dt, seed, base, 100.0, 200.0)
        else:
            row, uhat = evolve(b3, N, dt, seed, uhat, 0.0, float(step_measure))
        up.append(row)
        print(f"[B] UP   b3={b3:5.3f} ext_tail={row['ext_tail_mean']:.3f} "
              f"-> {row['attractor']}", flush=True)

    # 反向（b3 降序）：在最高 b3 用 common 初值平衡，其后逐点降
    base = common_initial_state(N, mask, n_ext, seed)
    row0, uhat = evolve(b3s[-1], N, dt, seed, base, 100.0, 200.0)
    down = []
    for b3 in reversed(b3s):
        if b3 == b3s[-1]:
            row = row0
        else:
            row, uhat = evolve(b3, N, dt, seed, uhat, 0.0, float(step_measure))
        down.append(row)
        print(f"[B] DOWN b3={b3:5.3f} ext_tail={row['ext_tail_mean']:.3f} "
              f"-> {row['attractor']}", flush=True)
    down.sort(key=lambda x: x["beta3"])

    print("\n" + "=" * 64)
    print(f"【B 迟滞回线 seed={seed}】beta3 : UP分支 ext / DOWN分支 ext")
    print("=" * 64)
    dmap = {x["beta3"]: x for x in down}
    for u in up:
        d = dmap[u["beta3"]]
        gap = abs(u["ext_tail_mean"] - d["ext_tail_mean"])
        flag = "  <== 路径分叉(共存)" if gap > 0.05 else ""
        print(f"  b3={u['beta3']:5.3f} : UP {u['ext_tail_mean']:.3f}({u['attractor']:4s}) / "
              f"DOWN {d['ext_tail_mean']:.3f}({d['attractor']:4s})  |gap|={gap:.3f}{flag}")
    return up, down


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="短测程验管道")
    ap.add_argument("--basin-only", action="store_true")
    ap.add_argument("--hyst-only", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        print("### SMOKE：measure=8 验管道（含绝热续跑）###", flush=True)
        b_rows = basin_scan(N128, DT128, [0.20, 0.21], [240], 8)
        up, down = hysteresis(N128, DT128, 240, [0.18, 0.20, 0.22], 8)
        out = {"smoke": True, "basin": b_rows, "hyst_up": up, "hyst_down": down}
        outp = HERE / "out" / "round246_smoke.json"
        json.dump(out, open(outp, "w"), indent=2, ensure_ascii=False)
        print("### SMOKE OK ->", outp, flush=True)
        return

    out = {}
    if not args.hyst_only:
        out["basin_N128"] = basin_scan(N128, DT128, BASIN_B3, SEEDS, 800)
    if not args.basin_only:
        # 主迟滞：敏感 seed240（0.20走低/0.21走高）；每绝热步 measure=200
        up240, down240 = hysteresis(N128, DT128, 240, HYST_B3, 200)
        out["hyst_s240"] = {"up": up240, "down": down240}
        # 第二 seed 复核路径依赖（242 在 0.20 走高态）
        up242, down242 = hysteresis(N128, DT128, 242, HYST_B3, 200)
        out["hyst_s242"] = {"up": up242, "down": down242}

    outp = HERE / "out" / "round246_basin_hyst.json"
    json.dump(out, open(outp, "w"), indent=2, ensure_ascii=False)
    print("\n全矩阵落盘 ->", outp, flush=True)


if __name__ == "__main__":
    main()

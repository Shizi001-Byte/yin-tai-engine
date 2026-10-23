#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# round276_multiseed_coexistence.py —— #276 深度轮：共存区多seed独立性验证
#
# 动机（#275 OPEN_ITEMS 反复标记的缺口）：
#   从 #256 到 #273，共存区测绘（β3≈0.16125-0.1875）全部使用单 seed s242。
#   v1.2 铁律"单初值/单分辨率结论不可信"，同一逻辑要求单 seed 结论也不可信：
#   需要回答"共存区是多稳态的普适结构，还是 s242 这一条随机场的偶然产物"。
#
# 设计（主Agent判据，不复用任何 worker 结论）：
#   对 β3 ∈ {0.170, 0.175, 0.180}（共存区内部，远离上下边界），
#   取 4 条独立随机通用初值 seed ∈ {240, 241, 237, 242}（与 #246 盆地扫描同套），
#   每条 measure=800，分类 HIGH>0.38 / MID 0.28-0.38 / LOW<0.28。
#   判据：同一 β3 下若 4 条独立初值落入 ≥2 个不同吸引子 → 该 β3 的多稳态
#         与初值选取强相关，而非单一 seed 偶然 → 共存区在 IC 层面稳健。
#   若 4 条全部落入同一吸引子，则不能仅由此断定"无共存"（盆地可能高度不对称），
#   需补 LOW 族初值对照 —— 故同时保留 s242 LOW 族锚点对照（round257_boundary.npz）。
#
# 用法：python3 round276_multiseed_coexistence.py            # 全矩阵
#       python3 round276_multiseed_coexistence.py --smoke    # measure=8 验管道
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from round246_basin import evolve, L, N128, DT128, classify   # noqa: E402
from round231_P1_0 import make_grid                            # noqa: E402
from round234_P1_0 import common_initial_state                 # noqa: E402
from round240_P1_0 import Q_OUT                                # noqa: E402

OUTP = HERE / "out" / "round276_multiseed_coexistence.json"
NPZP = HERE / "out" / "round276_multiseed_coexistence.npz"
CTRL_NPZ = HERE / "out" / "round257_boundary.npz"

B3S = [0.170, 0.175, 0.180]
SEEDS = [240, 241, 237, 242]
MEASURE = 800.0
CTRL = (0.175, "s242_b0.175_low", "low_family")


def load():
    d = json.load(open(OUTP)) if OUTP.exists() else {"rows": [], "ctrl": []}
    nz = dict(np.load(NPZP)) if NPZP.exists() else {}
    return d, nz


def save(d, nz):
    OUTP.parent.mkdir(parents=True, exist_ok=True)
    json.dump(d, open(OUTP.with_suffix(".json.tmp"), "w"), indent=2, ensure_ascii=False)
    OUTP.with_suffix(".json.tmp").replace(OUTP)
    np.savez_compressed(NPZP, **nz)


def run_one(d, nz, seed, b3, uhat, tag, measure):
    key = f"s{seed}_b{b3:g}_{tag}_m{measure:g}"
    if key in nz:
        print(f"  [skip] {key}", flush=True)
        return
    row, uf = evolve(b3, N128, DT128, seed, uhat, 100.0, measure)
    row.update({"seed": seed, "beta3": b3, "init": tag, "measure": measure})
    d["rows"].append(row)
    nz[key] = uf
    save(d, nz)
    print(f"  {key}: ext={row['ext_tail_mean']:.4f} {row['attractor']:4s} "
          f"q48={row['q48_final']*100:5.1f}% lam={row['lambda_last20']:+.3f}", flush=True)


def run_ctrl(d, nz, b3, init_key, tag, measure):
    key = f"s242_b{b3:g}_{tag}_m{measure:g}"
    if key in nz:
        print(f"  [skip] {key}", flush=True)
        return
    src = dict(np.load(CTRL_NPZ))
    if init_key not in src:
        print(f"  [WARN] 控制组初值缺失 {init_key}，跳过", flush=True)
        return
    row, uf = evolve(b3, N128, DT128, 242, src[init_key], 0.0, measure)
    row.update({"seed": 242, "beta3": b3, "init": tag, "measure": measure})
    d["ctrl"].append(row)
    nz[key] = uf
    save(d, nz)
    print(f"  [CTRL] {key}: ext={row['ext_tail_mean']:.4f} {row['attractor']:4s} "
          f"q48={row['q48_final']*100:5.1f}% lam={row['lambda_last20']:+.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    measure = 8.0 if args.smoke else MEASURE

    k, mask = make_grid(N128, L)
    n_ext = int(round(Q_OUT * L / (2 * np.pi)))

    d, nz = load()
    print(f"#276 共存区多seed独立性验证  beta3={B3S} seeds={SEEDS} measure={measure:g}")

    for b3 in B3S:
        for sd in SEEDS:
            base = common_initial_state(N128, mask, n_ext, sd)
            run_one(d, nz, sd, b3, base, "common", measure)

    b3c, init_key, tagc = CTRL
    run_ctrl(d, nz, b3c, init_key, tagc, measure)

    save(d, nz)

    print("\n" + "=" * 68)
    print("【裁决表】beta3 × seed → 吸引子（generic common 初值）")
    print("=" * 68)
    multi = 0
    for b3 in B3S:
        sub = [x for x in d["rows"] if abs(x["beta3"] - b3) < 1e-9 and x["init"] == "common"]
        attrs = {x["seed"]: x["attractor"] for x in sub}
        kind = set(attrs.values())
        flag = "  <== 多吸引子(IC依赖)" if len(kind) >= 2 else ""
        if len(kind) >= 2:
            multi += 1
        print(f"  b3={b3:5.3f}  " + "  ".join(f"s{sd}:{a:4s}" for sd, a in sorted(attrs.items()))
              + f"   kinds={len(kind)}{flag}")
    print()
    for r in d["ctrl"]:
        print(f"  [CTRL] b3={r['beta3']:g} {r['init']}: →{r['attractor']} "
              f"ext={r['ext_tail_mean']:.4f}")
    print(f"\n  多吸引子β3点数 = {multi}/{len(B3S)}")
    print("  解释：≥1 个β3点出现多种吸引子 ⇒ 该点的多稳态与初值强相关，"
          "共存区在 IC 层面稳健，非 s242 单seed偶然。")

    if args.smoke:
        print("\n### SMOKE OK ###")


if __name__ == "__main__":
    main()

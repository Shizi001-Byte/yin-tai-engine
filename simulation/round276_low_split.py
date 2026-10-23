#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# round276_low_split.py —— #276 异常质疑：LOW 是否分裂成两个不同低位
#
# 触发（#276 主矩阵 round276_multiseed_coexistence.json）：
#   β3=0.175 处，
#     seed=241 generic 初值 → LOW  ext=0.0841 q48=0.5%  λ=+0.000
#     seed=242 LOW 族初值   → LOW  ext=0.1949 q48=3.2%  λ=+0.010
#   两者同为 LOW 分类（<0.28）且 λ 均已沉降，但 ext 相差 0.11（约 2.3 倍）。
#   若两者都是真吸引子，则 LOW 档内部存在分裂（L1≈0.084 / L2≈0.195），
#   现行三档分类（HIGH/MID/LOW）会把两个不同吸引子混为一档 —— 分类口径缺陷。
#
# 本脚本只做一件事：把上述两个终态在**原 β3=0.175 继续演化 measure=400**（warmup=0），
#   若各自仍停在原水平（λ≈0、ext 不变）→ 两个独立低位吸引子成立（需再报主Agent裁决）；
#   若其中一支漂移 → 短测程假象，撤回分裂判断。
# 初值直接取自 round276_multiseed_coexistence.npz 的终态（独立复算路径，不重跑生成逻辑）。
#
# 用法：python3 round276_low_split.py [--smoke]
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from round246_basin import evolve, N128, DT128   # noqa: E402

SRC = HERE / "out" / "round276_multiseed_coexistence.npz"
OUTP = HERE / "out" / "round276_low_split.json"
NPZP = HERE / "out" / "round276_low_split.npz"
B3 = 0.175
CASES = [
    (241, "s241_b0.175_common_m800", "L1_s241_generic"),
    (242, "s242_b0.175_low_family_m800", "L2_s242_lowfamily"),
]
EXT_MEASURE = 400.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    measure = 8.0 if args.smoke else EXT_MEASURE

    src = dict(np.load(SRC))
    d = json.load(open(OUTP)) if OUTP.exists() else {"rows": []}
    nz = dict(np.load(NPZP)) if NPZP.exists() else {}

    def save():
        json.dump(d, open(OUTP.with_suffix(".json.tmp"), "w"), indent=2, ensure_ascii=False)
        OUTP.with_suffix(".json.tmp").replace(OUTP)
        np.savez_compressed(NPZP, **nz)

    print(f"#276 异常质疑：LOW 分裂验证 β3={B3} 续跑 measure={measure:g}")
    for sd, key, tag in CASES:
        if key not in src:
            print(f"  [ERROR] 缺初值 {key}"); continue
        u0 = src[key]
        if tag in nz:
            print(f"  [skip] {tag}"); continue
        row, uf = evolve(B3, N128, DT128, sd, u0, 0.0, measure)
        row.update({"seed": sd, "beta3": B3, "tag": tag, "measure_ext": measure})
        d["rows"].append(row); nz[tag] = uf; save()
        print(f"  {tag}: ext={row['ext_tail_mean']:.4f} {row['attractor']:4s} "
              f"q48={row['q48_final']*100:5.1f}% lam={row['lambda_last20']:+.4f} "
              f"seg={['%.3f' % x for x in row['seg_ext']]}", flush=True)

    print("\n【判断】")
    for r in d["rows"]:
        print(f"  {r['tag']}: ext={r['ext_tail_mean']:.4f} λ={r['lambda_last20']:+.4f}")
    if len(d["rows"]) == 2:
        gap = abs(d["rows"][0]["ext_tail_mean"] - d["rows"][1]["ext_tail_mean"])
        print(f"  ext 间距={gap:.4f}  "
              f"{'>> 两个不同低位吸引子（LOW 档需细分）' if gap > 0.05 else '>> 未分裂（短测程假象或同一吸引子）'}")

    if args.smoke:
        print("### SMOKE OK ###")


if __name__ == "__main__":
    main()

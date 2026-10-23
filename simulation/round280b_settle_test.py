#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# round280b_settle_test.py —— 判 B 支 @0.190/0.195 到底"settle 到新态"还是"还在慢漂"
#
# 背景（#280 细化）：B 支细步推进到 0.1900 时 ext 0.4338 且 seg=[0.409,0.458] 仍在爬升，
#   0.1925/0.1950 落到 0.4607/0.4596（与 0.18 锚点谱余弦掉到 0.905），
#   但 λ 仅 1e-3 量级 —— 无法区分"已停在新定常态"与"极慢暂态还没走完"。
#
# 判据（主Agent设计，脚本只执行）：
#   从该点终态 warmup=0 续跑，β3 固定不变，measure=1200（分 6 段 × 200）：
#     - 若 ext 末三段漂移 < 0.005 → SETTLED（真定常态，记录其 ext）
#     - 若末三段仍单调漂移 > 0.010 → DRIFTING（暂态未走完，需更长测程）
#     - 若末段跌出 HIGH → 实为级联失稳
#   λ 只作辅助提示，不作判据（实测 ext 六段恒定而 λ=-2.5e-3，见台账 #234）。
#
# 用法：python3 round280b_settle_test.py [--smoke]
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from round246_basin import evolve, N128, DT128   # noqa: E402

SRC = HERE / "out" / "round280_deathpoint_fine.npz"
OUTP = HERE / "out" / "round280b_settle_test.json"
NPZP = HERE / "out" / "round280b_settle_test.npz"
LONG = 1200.0
SMOKE = 8.0
CASES = [("B", 237, "dpB_0.1900", 0.1900), ("B", 237, "dpB_0.1950", 0.1950)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--measure", type=float, default=LONG)
    args = ap.parse_args()
    measure = SMOKE if args.smoke else args.measure
    src = dict(np.load(SRC))
    d = json.load(open(OUTP)) if OUTP.exists() else {"rows": [], "measure": measure}
    nz = dict(np.load(NPZP)) if NPZP.exists() else {}

    def save():
        tmp = OUTP.with_suffix(".json.tmp")
        json.dump(d, open(tmp, "w"), indent=2, ensure_ascii=False)
        tmp.replace(OUTP)
        np.savez_compressed(NPZP, **nz)

    for tag, seed, key, b3 in CASES:
        if key not in src:
            print(f"[ERROR] 缺起点 {key}")
            continue
        rk = f"settle_{tag}_{b3:.4f}"
        if rk in nz:
            print(f"[skip] {rk}")
            continue
        row, uf = evolve(b3, N128, DT128, seed, src[key], 0.0, measure)
        row.update({"branch": tag, "beta3": b3, "row_key": rk, "measure": measure})
        d["rows"].append(row)
        nz[rk] = uf
        save()
        seg = row["seg_ext"]
        tail = seg[-3:]
        drift = max(tail) - min(tail)
        print(f"{rk}: ext={row['ext_tail_mean']:.4f} {row['attractor']} q48={row['q48_final']*100:.1f}% "
              f"lam={row['lambda_last20']:+.2e} seg={['%.3f' % x for x in seg]} "
              f"tail_drift={drift:.4f}", flush=True)

    if args.smoke:
        print("### SMOKE OK ###")


if __name__ == "__main__":
    main()

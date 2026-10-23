#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# round281b_n160_time.py —— #281 N160 跨分辨率复核：步数按物理时间归一
#
# 坑：run_case 的 warmup/measure 单位是步，不是时间。
#   N128 dt=0.002 的 800 步 = 1.6 时间单位；N160 dt=0.001 跑 800 步只有 0.8。
#   跨 N 比较必须先按 dt 归一，否则会把"没跑够"读成分辨率差异。
#
#   --case A：N160, dt0.001, s240 @β3=0.18，warmup=1600(≡1.6)+measure=600(0.6)
#   --case B：N160, dt0.001, s237 @β3=0.18 先 1600+400 得锚点，再 0.19 续跑 2000 步(≡2.0)
#
# 实测（#281）：B 锚点 0.4738 与 N128 完全一致；但 B@0.19 在 N160 崩到 LOW 0.1757（N128 为 B′ 0.4616）；
#   A 锚点 N160 0.3821（主导模 n=16,1）vs N128 0.4849（n=16,4），谱余弦 0.737 —— 跨 N 冲突，待深度研究轮。
#
# 用法：python3 round281b_n160_time.py --case A|B [--smoke]
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from round231_P1_0 import make_grid
from round234_P1_0 import common_initial_state
from round240_P1_0 import Q_OUT
from round246_basin import evolve

N, DT = 160, 0.001
L = 20.0 * np.pi / 3.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=["A", "B"])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    smoke = a.smoke
    global OUTP
    OUTP = HERE / "out" / f"round281b_n160_time_{a.case}.json"
    _, mask = make_grid(N, L)
    n_ext = int(round(Q_OUT * L / (2 * np.pi)))
    d = json.load(open(OUTP)) if OUTP.exists() else {"rows": [], "N": N, "dt": DT,
                                                     "note": "measure 单位=步；时间=步*dt"}
    nz = dict(np.load(OUTP.with_suffix(".npz"))) if OUTP.with_suffix(".npz").exists() else {}

    def save():
        tmp = OUTP.with_suffix(".json.tmp")
        json.dump(d, open(tmp, "w"), indent=2, ensure_ascii=False)
        tmp.replace(OUTP)
        np.savez_compressed(OUTP.with_suffix(".npz"), **nz)

    def run(rk, b3, seed, base, warmup, measure):
        if rk in nz:
            print(f"[skip] {rk}")
            return nz[rk]
        row, uf = evolve(b3, N, DT, seed, base, warmup, measure)
        row.update({"row_key": rk, "beta3": b3, "warmup": warmup, "measure": measure,
                    "time_total": (warmup + measure) * DT})
        d["rows"].append(row); nz[rk] = uf; save()
        print(f"{rk}: ext={row['ext_tail_mean']:.4f} {row['attractor']:4s} q48={row['q48_final']*100:.1f}% "
              f"lam={row['lambda_last20']:+.2e} seg={['%.3f' % x for x in row['seg_ext']]}", flush=True)
        return uf

    if a.case == "A":
        s = 8 if smoke else 1600
        run("n160t_A_anchor_b0.180", 0.18, 240, common_initial_state(N, mask, n_ext, 240), s, 8 if smoke else 600)
    else:
        s = 8 if smoke else 1600
        u = run("n160t_B_anchor_b0.180", 0.18, 237, common_initial_state(N, mask, n_ext, 237), s, 8 if smoke else 400)
        run("n160t_B_b0.190", 0.19, 237, u, 0.0, 8 if smoke else 2000)
    if smoke:
        print("### SMOKE OK ###")


if __name__ == "__main__":
    main()

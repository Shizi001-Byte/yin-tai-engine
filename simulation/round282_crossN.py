#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# round282_crossN.py —— #282 R281-RES：跨分辨率一致性，参数化单点锚定
# 物理时间归一：warmup/measure 给的是步，时间 = 步 * dt。
# 标准测程：warmup 1.6 时间 + measure 0.6 时间（N128 800/300 步，N160 1600/600，N192 2000/750）。
# 注意：N192/dt0.0008 会触发 RK4 虚轴稳定性断言（dt*max|omega|=4.13），必须降 dt 到 0.0004（步数相应翻倍）。
# 用法：python3 round282_crossN.py --N 160 --dt 0.001 --seed 241 --warmup 1600 --measure 600 --tag n160_s241_b018
import argparse, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from round231_P1_0 import make_grid
from round234_P1_0 import common_initial_state
from round240_P1_0 import Q_OUT
from round246_basin import evolve

L = 20.0 * np.pi / 3.0
ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, required=True)
ap.add_argument("--dt", type=float, required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--b3", type=float, default=0.18)
ap.add_argument("--warmup", type=float, required=True)
ap.add_argument("--measure", type=float, required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
w, m = (8, 8) if a.smoke else (a.warmup, a.measure)
OUTP = HERE / "out" / f"round282_{a.tag}.json"
NPP = OUTP.with_suffix(".npz")
_, mask = make_grid(a.N, L)
n_ext = int(round(Q_OUT * L / (2 * np.pi)))
base = common_initial_state(a.N, mask, n_ext, a.seed)
row, uf = evolve(a.b3, a.N, a.dt, a.seed, base, w, m)
row.update({"row_key": a.tag, "N": a.N, "dt": a.dt, "seed": a.seed, "beta3": a.b3,
            "warmup": w, "measure": m, "time_total": (w + m) * a.dt})
tmp = OUTP.with_suffix(".json.tmp")
json.dump({"rows": [row]}, open(tmp, "w"), indent=2, ensure_ascii=False); tmp.replace(OUTP)
np.savez_compressed(NPP, **{a.tag: uf})
segs = [round(float(x), 3) for x in row["seg_ext"]]
print(f"{a.tag}: ext={row['ext_tail_mean']:.4f} {row['attractor']} q48={row['q48_final']*100:.1f}% "
      f"lam={row['lambda_last20']:+.2e} seg={segs} t={(w+m)*a.dt:.2f}", flush=True)
if a.smoke:
    print("### SMOKE OK ###")

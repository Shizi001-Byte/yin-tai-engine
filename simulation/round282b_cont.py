#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# round282b_cont.py —— #282 续跑：从已有终态出发，固定 β3 跑指定步数（物理时间=步*dt）
# 用途：同一个态往下推 β3，看是存活还是崩——判“死亡点”必须从该吸引子自己的态出发。
# 用法：python3 round282b_cont.py --src out/xxx.npz --key k --N 160 --dt 0.001 --b3 0.19 --steps 2000 --seed 241 --tag x
import argparse, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from round246_basin import evolve
ap = argparse.ArgumentParser()
for k, t in [("src", str), ("key", str), ("tag", str)]:
    ap.add_argument("--" + k, required=True, type=t)
ap.add_argument("--N", type=int, required=True); ap.add_argument("--dt", type=float, required=True)
ap.add_argument("--b3", type=float, required=True); ap.add_argument("--steps", type=float, required=True)
ap.add_argument("--seed", type=int, required=True); ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
st = 8 if a.smoke else a.steps
src = dict(np.load(HERE / a.src if not Path(a.src).is_absolute() else a.src))
row, uf = evolve(a.b3, a.N, a.dt, a.seed, src[a.key], 0.0, st)
row.update({"row_key": a.tag, "beta3": a.b3, "measure": st, "from": a.key, "time_total": st * a.dt})
OUTP = HERE / "out" / f"round282_{a.tag}.json"
tmp = OUTP.with_suffix(".json.tmp")
json.dump({"rows": [row]}, open(tmp, "w"), indent=2, ensure_ascii=False); tmp.replace(OUTP)
np.savez_compressed(OUTP.with_suffix(".npz"), **{a.tag: uf})
segs = [round(float(x), 3) for x in row["seg_ext"]]
print(f"{a.tag}: ext={row['ext_tail_mean']:.4f} {row['attractor']} lam={row['lambda_last20']:+.2e} seg={segs} t={st*a.dt:.2f}", flush=True)
if a.smoke: print("### SMOKE OK ###")

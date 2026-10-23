#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# round283_scan.py —— 参数续跑扫描（链式）：从给定态出发，按 β3 列表逐点续推，每点 warmup=0、measure=steps 步
# 用途：忠实沿同一条吸引子分支走临界区，避免"跳 β3"造成的路径性坦塌。
# 用法：python3 round283_scan.py --src out/x.npz --key k --N 160 --dt 0.001 --steps 800 --b3s 0.195,0.1975 --tag pfx
import argparse, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from round246_basin import evolve
ap = argparse.ArgumentParser()
for k in ("src", "key", "tag", "b3s"):
    ap.add_argument("--" + k, required=True)
ap.add_argument("--N", type=int, required=True); ap.add_argument("--dt", type=float, required=True)
ap.add_argument("--steps", type=float, required=True); ap.add_argument("--seed", type=int, default=241)
a = ap.parse_args()
src = dict(np.load(a.src if Path(a.src).is_absolute() else str(HERE / a.src)))
st = src[a.key]; rows = []; states = {}
for s in a.b3s.split(","):
    b3 = float(s)
    row, st = evolve(b3, a.N, a.dt, a.seed, st, 0.0, a.steps)
    states[f"{a.tag}_b{b3:.4f}"] = st
    row.update({"row_key": f"{a.tag}_b{b3:.4f}", "beta3": b3, "measure": a.steps, "time_total": a.steps * a.dt})
    rows.append(row)
    print(f"  b3={b3:.4f}: ext={row['ext_tail_mean']:.4f} {row['attractor']} "
          f"seg={[round(float(x),3) for x in row['seg_ext']]}", flush=True)
json.dump({"rows": rows}, open(HERE / "out" / f"round283_{a.tag}.json", "w"), indent=2, ensure_ascii=False)
np.savez_compressed(HERE / "out" / f"round283_{a.tag}.npz", **states)
print("SCAN DONE", flush=True)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# round259_mid_lifetime.py —— #259 MID亚稳态寿命分布测量
# 背景：#256发现β3=0.19/0.20的MID态为亚稳态(measure=800下失稳)，
#       但具体寿命分布(是否seed依赖、是否β3依赖)未定量。
# 本脚本：从#256 npz加载s242_b0.18 MID终态作初值，
#       在β3=0.19/0.20跑measure=400/600/800/1000，看MID态失稳时机。
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from round246_basin import evolve, N128, DT128

OUTP = HERE / "out" / "round259_mid_lifetime.json"
NPZP = HERE / "out" / "round259_mid_lifetime.npz"
SRC_NPZ = HERE / "out" / "round256_multistab.npz"

TEST_B3 = [0.19, 0.20]
MEASURES = [400.0, 600.0, 800.0, 1000.0]

def load():
    d = json.load(open(OUTP)) if OUTP.exists() else {"rows": []}
    nz = dict(np.load(NPZP)) if NPZP.exists() else {}
    return d, nz

def save(d, nz):
    OUTP.parent.mkdir(parents=True, exist_ok=True)
    json.dump(d, open(OUTP.with_suffix(".json.tmp"), "w"), indent=2, ensure_ascii=False)
    OUTP.with_suffix(".json.tmp").replace(OUTP)
    np.savez_compressed(NPZP, **nz)

def run(d, nz, seed, b3, uhat, measure):
    key = f"s{seed}_b{b3:g}_m{measure:g}"
    if key in nz:
        print(f"  [skip] {key}", flush=True); return
    row, uf = evolve(b3, N128, DT128, seed, uhat, 0.0, measure)
    row.update({"seed": seed, "beta3": b3, "measure": measure, "init_type": "mid"})
    d["rows"].append(row); nz[key] = uf; save(d, nz)
    print(f"  {key}: ext={row['ext_tail_mean']:.4f} {row['attractor']:4s} "
          f"q48={row['q48_final']*100:.1f}% lam={row['lambda_last20']:+.3f}", flush=True)

def main():
    d, nz = load()
    src = dict(np.load(SRC_NPZ))
    mid = src.get("s242_b0.18_mid")  # MID态初值(#256 β3=0.18终态)
    if mid is None:
        print(f"[ERROR] 无法加载MID初值"); sys.exit(1)
    print(f"#259 MID亚稳态寿命分布测量 初值=s242_b0.18_mid")
    for b3 in TEST_B3:
        print(f"\n--- β3={b3} ---")
        for m in MEASURES:
            run(d, nz, 242, b3, mid, m)
    save(d, nz)
    print("\n【裁决】MID态寿命分布")
    for b3 in TEST_B3:
        print(f"  β3={b3}:")
        for m in MEASURES:
            r = [x for x in d["rows"] if abs(x["beta3"]-b3)<1e-9 and abs(x["measure"]-m)<1e-9]
            if r:
                print(f"    measure={m:4.0f}: {r[0]['attractor']:4s} ext={r[0]['ext_tail_mean']:.3f} lam={r[0]['lambda_last20']:+.3f}")

if __name__ == "__main__":
    main()

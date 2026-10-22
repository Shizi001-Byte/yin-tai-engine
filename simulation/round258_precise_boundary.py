#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# round258_precise_boundary.py —— #258 共存区精确边界点测绘
# 下边界精确点: β3=0.1625 (从#257 npz加载s242_b0.17 LOW/MID终态)
# 上边界精确点: β3=0.1875 (从#257 npz加载s242_b0.185 LOW/MID终态)
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from round246_basin import evolve, N128, DT128

OUTP = HERE / "out" / "round258_precise_boundary.json"
NPZP = HERE / "out" / "round258_precise_boundary.npz"
SRC_NPZ = HERE / "out" / "round257_boundary.npz"
MEASURE = 800.0

# (beta3, low_init_key, mid_init_key)
TESTS = [
    (0.1625, "s242_b0.17_low", "s242_b0.17_mid"),   # 下边界精确点
    (0.1875, "s242_b0.185_low", "s242_b0.185_mid"), # 上边界精确点
]

def load():
    d = json.load(open(OUTP)) if OUTP.exists() else {"rows": []}
    nz = dict(np.load(NPZP)) if NPZP.exists() else {}
    return d, nz

def save(d, nz):
    OUTP.parent.mkdir(parents=True, exist_ok=True)
    json.dump(d, open(OUTP.with_suffix(".json.tmp"), "w"), indent=2, ensure_ascii=False)
    OUTP.with_suffix(".json.tmp").replace(OUTP)
    np.savez_compressed(NPZP, **nz)

def run(d, nz, seed, b3, uhat, itype):
    key = f"s{seed}_b{b3:g}_{itype}"
    if key in nz:
        print(f"  [skip] {key}", flush=True); return
    row, uf = evolve(b3, N128, DT128, seed, uhat, 0.0, MEASURE)
    row.update({"seed": seed, "beta3": b3, "init_type": itype, "measure": MEASURE})
    d["rows"].append(row); nz[key] = uf; save(d, nz)
    print(f"  {key}: ext={row['ext_tail_mean']:.4f} {row['attractor']:4s} "
          f"q48={row['q48_final']*100:.1f}% lam={row['lambda_last20']:+.3f}", flush=True)

def main():
    d, nz = load()
    src = dict(np.load(SRC_NPZ))
    print(f"#258 共存区精确边界点测绘 measure={MEASURE}")
    for b3, low_key, mid_key in TESTS:
        low = src.get(low_key); mid = src.get(mid_key)
        if low is None or mid is None:
            print(f"[ERROR] 无法加载初值 {low_key}/{mid_key}"); continue
        print(f"\n--- β3={b3} (LOW初值={low_key}, MID初值={mid_key}) ---")
        run(d, nz, 242, b3, low, "low")
        run(d, nz, 242, b3, mid, "mid")
    save(d, nz)
    print("\n【裁决】")
    for b3, _, _ in TESTS:
        lr = [r for r in d["rows"] if r["init_type"]=="low" and abs(r["beta3"]-b3)<1e-9]
        mr = [r for r in d["rows"] if r["init_type"]=="mid" and abs(r["beta3"]-b3)<1e-9]
        if lr and mr:
            lf, mf = lr[0]["attractor"], mr[0]["attractor"]
            print(f"  β3={b3}: LOW→{lf}, MID→{mf}, {'真共存' if lf=='LOW' and mf=='MID' else '非共存'}")

if __name__ == "__main__":
    main()

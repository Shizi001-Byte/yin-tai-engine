#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# round257_lower_boundary.py —— #257 下边界验证(β3=0.16/0.165)
# 从#257 npz加载s242_b0.17 LOW/MID终态作初值，在β3=0.16/0.165跑measure=800。
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from round246_basin import evolve, N128, DT128

OUTP = HERE / "out" / "round257_lower_boundary.json"
NPZP = HERE / "out" / "round257_lower_boundary.npz"
SRC_NPZ = HERE / "out" / "round257_boundary.npz"
TEST_B3 = [0.16, 0.165]
MEASURE = 800.0

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
    low = src.get("s242_b0.17_low"); mid = src.get("s242_b0.17_mid")
    if low is None or mid is None:
        print(f"[ERROR] 无法加载初值"); sys.exit(1)
    print(f"#257 下边界验证 measure={MEASURE} 测试点={TEST_B3}")
    print("--- LOW初值(s242_b0.17_low) ---")
    for b3 in TEST_B3: run(d, nz, 242, b3, low, "low")
    print("--- MID初值(s242_b0.17_mid) ---")
    for b3 in TEST_B3: run(d, nz, 242, b3, mid, "mid")
    save(d, nz)
    print("\n【裁决】")
    for b3 in TEST_B3:
        lr = [r for r in d["rows"] if r["init_type"]=="low" and abs(r["beta3"]-b3)<1e-9]
        mr = [r for r in d["rows"] if r["init_type"]=="mid" and abs(r["beta3"]-b3)<1e-9]
        if lr and mr:
            lf, mf = lr[0]["attractor"], mr[0]["attractor"]
            print(f"  β3={b3}: LOW→{lf}, MID→{mf}, {'真共存' if lf=='LOW' and mf=='MID' else '非共存'}")

if __name__ == "__main__":
    main()

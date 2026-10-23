#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# round273_b01875_mid_stability.py —— #273 验证共存区上边界点β3=0.1875 MID态measure=1000
# #258用measure=800测到MID稳定，#270发现β3=0.18875 MID→HIGH(measure=1000)
# 需确认β3=0.1875在measure=1000时是否仍保持MID（若变HIGH则上边界更窄）
# 初值从round257_boundary.npz加载s242_b0.185_mid
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from round246_basin import evolve, N128, DT128

OUTP = HERE / "out" / "round273_b01875_mid_stability.json"
NPZP = HERE / "out" / "round273_b01875_mid_stability.npz"
SRC_NPZ = HERE / "out" / "round257_boundary.npz"

TESTS = [
    (0.1875, "s242_b0.185_mid", "mid", 1000.0),
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

def run(d, nz, seed, b3, uhat, itype, measure):
    key = f"s{seed}_b{b3:g}_{itype}_m{measure:g}"
    if key in nz:
        print(f"  [skip] {key}", flush=True); return
    row, uf = evolve(b3, N128, DT128, seed, uhat, 0.0, measure)
    row.update({"seed": seed, "beta3": b3, "init_type": itype, "measure": measure})
    d["rows"].append(row); nz[key] = uf; save(d, nz)
    print(f"  {key}: ext={row['ext_tail_mean']:.4f} {row['attractor']:4s} "
          f"q48={row['q48_final']*100:.1f}% lam={row['lambda_last20']:+.3f}", flush=True)

def main():
    d, nz = load()
    src = dict(np.load(SRC_NPZ))
    print(f"#273 验证β3=0.1875 MID态 measure=1000")
    for b3, init_key, itype, measure in TESTS:
        uhat = src.get(init_key)
        if uhat is None:
            print(f"[ERROR] 无法加载初值 {init_key}"); continue
        print(f"\n--- β3={b3} (初值={init_key}, type={itype}) ---")
        run(d, nz, 242, b3, uhat, itype, measure)
    save(d, nz)
    print("\n【裁决】")
    for b3, _, _, _ in TESTS:
        rows = [r for r in d["rows"] if abs(r["beta3"]-b3)<1e-9]
        for r in rows:
            print(f"  β3={b3} {r['init_type']}: →{r['attractor']} ext={r['ext_tail_mean']:.4f} "
                  f"q48={r['q48_final']*100:.1f}% lam={r['lambda_last20']:+.3f}")

if __name__ == "__main__":
    main()

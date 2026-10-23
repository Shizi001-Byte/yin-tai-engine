#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# round269_b01625_low_stability.py —— #269 重新生成fuse故障丢失的β3=0.1625 LOW稳定性
# 同时测β3=0.16125精确下边界点（验证下边界在0.16-0.1625之间）
# 初值从round257_lower_boundary.npz加载s242_b0.165_low
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from round246_basin import evolve, N128, DT128

OUTP = HERE / "out" / "round269_b01625_low_stability.json"
NPZP = HERE / "out" / "round269_b01625_low_stability.npz"
SRC_NPZ = HERE / "out" / "round257_lower_boundary.npz"
MEASURE = 1000.0

# (beta3, init_key, init_type, measure)
TESTS = [
    (0.1625, "s242_b0.165_low", "low", 1000.0),   # 重新生成fuse故障丢失的数据
    (0.16125, "s242_b0.165_low", "low", 1000.0),  # 精确下边界点
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
    print(f"#269 β3=0.1625/0.16125 LOW稳定性 measure=1000")
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

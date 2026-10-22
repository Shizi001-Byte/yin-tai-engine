#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# round260_b020_bistability.py —— #260 β3=0.20 HIGH/LOW双稳态验证
# 背景：#256多seed common初值在β3=0.20全部→LOW；
#       #259 MID初值(s242_b0.18_mid)在β3=0.20→HIGH。
#       推测β3=0.20存在HIGH/LOW双稳态，终态依赖初值。
# 本脚本：用#256的HIGH初值(s240_b0.18_common, ext=0.485)在β3=0.20跑measure=800，
#       验证HIGH态是否稳定。同时用LOW初值(s242_b0.18_low)对照。
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from round246_basin import evolve, N128, DT128

OUTP = HERE / "out" / "round260_b020_bistability.json"
NPZP = HERE / "out" / "round260_b020_bistability.npz"
SRC_NPZ = HERE / "out" / "round256_multistab.npz"
B3 = 0.20
MEASURE = 800.0

# (初值key, 初值类型, 预期态)
TESTS = [
    ("s240_b0.18_common", "high", "HIGH"),  # #256 seed240 β3=0.18→HIGH(0.485)
    ("s242_b0.18_low", "low", "LOW"),       # #256 LOW态对照
    ("s242_b0.18_mid", "mid", "?"),         # #259 MID态→HIGH(已验证)
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
    key = f"s{seed}_b{b3:g}_{itype}"
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
    print(f"#260 β3={B3} HIGH/LOW双稳态验证 measure={MEASURE}")
    for init_key, itype, expected in TESTS:
        uhat = src.get(init_key)
        if uhat is None:
            print(f"[ERROR] 无法加载初值 {init_key}"); continue
        seed = int(init_key.split("_")[0][1:])
        print(f"\n--- 初值={init_key} (类型={itype}, 预期={expected}) ---")
        run(d, nz, seed, B3, uhat, itype, MEASURE)
    save(d, nz)
    print("\n【裁决】β3=0.20双稳态")
    for init_key, itype, expected in TESTS:
        r = [x for x in d["rows"] if x["init_type"]==itype]
        if r:
            print(f"  {itype:4s}初值→{r[0]['attractor']:4s} ext={r[0]['ext_tail_mean']:.3f} (预期{expected})")

if __name__ == "__main__":
    main()

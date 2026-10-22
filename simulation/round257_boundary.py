#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# round257_boundary.py —— #257 精确测绘s242 L/M共存区边界
# 背景：#256(measure=800)确认β3=0.18为L/M真共存点；
#       β3=0.19/0.20的MID态为亚稳态(最终演化到HIGH/LOW)。
#       共存区比#246测绘的(0.15-0.205)更窄，需精确测绘上下边界。
# 本脚本：从#256 npz加载s242的LOW/MID初值(β3=0.18终态)，
#       在β3=0.17/0.175/0.185跑measure=800，看初值依赖的终态。
#       判定：LOW初值→LOW且MID初值→MID = 真共存点；
#             MID初值→其他 = MID态不稳定(共存区上边界以上)。
# 复用已验证物理核 round246_basin.evolve / classify。
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from round246_basin import evolve, classify, L, N128, DT128  # noqa: E402

OUTP = HERE / "out" / "round257_boundary.json"
NPZP = HERE / "out" / "round257_boundary.npz"
SRC_NPZ = HERE / "out" / "round256_multistab.npz"

# 精确测绘边界测试点
TEST_B3 = [0.17, 0.175, 0.185]
LONG_MEASURE = 800.0


def load_state():
    d = json.load(open(OUTP)) if OUTP.exists() else {"rows": []}
    npz = dict(np.load(NPZP)) if NPZP.exists() else {}
    return d, npz


def save_state(d, npz):
    OUTP.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTP.with_suffix(".json.tmp")
    json.dump(d, open(tmp, "w"), indent=2, ensure_ascii=False)
    tmp.replace(OUTP)
    np.savez_compressed(NPZP, **npz)


def emit(rows, row):
    key = (row["seed"], row["beta3"], row["init_type"])
    for i, x in enumerate(rows):
        if (x["seed"], x["beta3"], x["init_type"]) == key:
            rows[i] = row
            return
    rows.append(row)


def run_from_uhat(d, npz, seed, b3, uhat, init_type, measure):
    key = f"s{seed}_b{b3:g}_{init_type}"
    if key in npz:
        print(f"  [skip] {key} 已存在", flush=True)
        return
    row, uhat_final = evolve(b3, N128, DT128, seed, uhat, 0.0, measure)
    row["seed"] = seed
    row["beta3"] = b3
    row["init_type"] = init_type
    row["measure"] = measure
    emit(d["rows"], row)
    npz[key] = uhat_final
    save_state(d, npz)
    print(f"  {key}: ext={row['ext_tail_mean']:.4f} {row['attractor']:4s} "
          f"q48={row['q48_final']*100:.1f}% lam={row['lambda_last20']:+.3f}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    measure = 8.0 if args.smoke else LONG_MEASURE
    d, npz = load_state()
    d.setdefault("rows", [])

    # 从#256 npz加载s242在β3=0.18的LOW/MID终态
    src = dict(np.load(SRC_NPZ))
    s242_low = src.get("s242_b0.18_low")   # LOW态初值(#256终态)
    s242_mid = src.get("s242_b0.18_mid")   # MID态初值(#256终态)

    if s242_low is None or s242_mid is None:
        print(f"[ERROR] 无法从{SRC_NPZ}加载s242初值", flush=True)
        sys.exit(1)

    print("=" * 64)
    print(f"#257 s242 L/M共存区边界精确测绘  measure={measure}")
    print(f"测试点: {TEST_B3}")
    print("=" * 64)

    # 实验1：LOW初值 → 各β3
    print("\n--- 实验1: LOW初值(s242_b0.18_low) → 边界测试点 ---")
    for b3 in TEST_B3:
        run_from_uhat(d, npz, 242, b3, s242_low, "low", measure)

    # 实验2：MID初值 → 各β3
    print("\n--- 实验2: MID初值(s242_b0.18_mid) → 边界测试点 ---")
    for b3 in TEST_B3:
        run_from_uhat(d, npz, 242, b3, s242_mid, "mid", measure)

    save_state(d, npz)

    # 裁决
    print("\n" + "=" * 64)
    print("【裁决】共存区边界判定")
    print("=" * 64)
    rows = d["rows"]
    for b3 in TEST_B3:
        low_r = [r for r in rows if r["init_type"] == "low" and abs(r["beta3"] - b3) < 1e-9]
        mid_r = [r for r in rows if r["init_type"] == "mid" and abs(r["beta3"] - b3) < 1e-9]
        if low_r and mid_r:
            lf, mf = low_r[0]["attractor"], mid_r[0]["attractor"]
            coexist = (lf == "LOW" and mf == "MID")
            print(f"  β3={b3:.3f}: LOW初值→{lf}, MID初值→{mf}, "
                  f"{'真共存' if coexist else '非共存(MID不稳定)'}")
    print(f"\n落盘 -> {OUTP} {NPZP}")


if __name__ == "__main__":
    main()

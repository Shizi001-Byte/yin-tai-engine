#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# round256_multistab_longcheck.py —— #256 验证s242 L/M多稳态共存区
# 背景：#246(measure300)中s242在β3=0.205跳MID，0.200-0.150全MID；
#       #248(measure250)中s242在β3=0.160跳MID，0.200-0.165全LOW。
#       两源在β3∈[0.165,0.205]冲突，疑L/M多稳态共存。
# 本脚本：从#248 npz加载s242的LOW/MID初值，在共存区β3=0.18/0.19/0.20
#       跑measure=800长测程，看LOW初值是否跳MID、MID初值是否跳LOW。
#       同时多seed(240/244/246/248)从common初值跑同样β3，统计盆地。
# 复用已验证物理核 round246_basin.evolve / classify。
# 用法：python3 round256_multistab_longcheck.py --smoke  # measure=8验管道
#       setsid nohup python3 round256_multistab_longcheck.py > round256.log 2>&1 &
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from round231_P1_0 import make_grid                    # noqa: E402
from round234_P1_0 import common_initial_state         # noqa: E402
from round240_P1_0 import Q_OUT                         # noqa: E402
from round246_basin import evolve, classify, L, N128, DT128  # noqa: E402

OUTP = HERE / "out" / "round256_multistab.json"
NPZP = HERE / "out" / "round256_multistab.npz"
SRC_NPZ = HERE / "out" / "round248_down_extend.npz"

# 共存区测试点
TEST_B3 = [0.18, 0.19, 0.20]
# 多seed盆地统计
BASIN_SEEDS = [240, 244, 246, 248]
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
    """从给定uhat初值跑长测程，warmup=0（绝热续跑）。"""
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


def run_from_common(d, npz, seed, b3, measure):
    """从common初值跑长测程（盆地统计）。"""
    k, mask = make_grid(N128, L)
    n_ext = int(round(Q_OUT * L / (2 * np.pi)))
    key = f"s{seed}_b{b3:g}_common"
    if key in npz:
        print(f"  [skip] {key} 已存在", flush=True)
        return
    base = common_initial_state(N128, mask, n_ext, seed)
    row, uhat_final = evolve(b3, N128, DT128, seed, base, 100.0, measure)
    row["seed"] = seed
    row["beta3"] = b3
    row["init_type"] = "common"
    row["measure"] = measure
    emit(d["rows"], row)
    npz[key] = uhat_final
    save_state(d, npz)
    print(f"  {key}: ext={row['ext_tail_mean']:.4f} {row['attractor']:4s} "
          f"q48={row['q48_final']*100:.1f}% lam={row['lambda_last20']:+.3f}",
          flush=True)


def summarize(d):
    """汇总：按init_type分组，看各β3下LOW/MID初值的终态。"""
    rows = d.get("rows", [])
    by_init = {}
    for r in rows:
        by_init.setdefault(r["init_type"], []).append(r)
    out = {}
    for it, sub in by_init.items():
        sub.sort(key=lambda r: (r["seed"], r["beta3"]))
        out[it] = [
            {"seed": r["seed"], "beta3": r["beta3"],
             "ext": round(r["ext_tail_mean"], 4),
             "attractor": r["attractor"],
             "lam": round(r["lambda_last20"], 4)}
            for r in sub
        ]
    # 共存判定：同一β3下LOW初值→LOW且MID初值→MID = 真多稳态
    coexistence = []
    for b3 in TEST_B3:
        low_rows = [r for r in rows if r["init_type"] == "low" and abs(r["beta3"] - b3) < 1e-9]
        mid_rows = [r for r in rows if r["init_type"] == "mid" and abs(r["beta3"] - b3) < 1e-9]
        if low_rows and mid_rows:
            low_final = low_rows[0]["attractor"]
            mid_final = mid_rows[0]["attractor"]
            coexistence.append({
                "beta3": b3,
                "low_init_final": low_final,
                "mid_init_final": mid_final,
                "true_multistability": (low_final == "LOW" and mid_final == "MID"),
            })
    out["coexistence_check"] = coexistence
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    measure = 8.0 if args.smoke else LONG_MEASURE
    d, npz = load_state()
    d.setdefault("rows", [])

    # 加载#248的s242 uhat
    src = dict(np.load(SRC_NPZ))
    s242_low = src.get("s242_b0.18")   # LOW态初值
    s242_mid = src.get("s242_b0.16")   # MID态初值

    if s242_low is None or s242_mid is None:
        print(f"[ERROR] 无法从{SRC_NPZ}加载s242初值", flush=True)
        sys.exit(1)

    print("=" * 64)
    print(f"#256 s242 L/M多稳态长测程验证  measure={measure}")
    print("=" * 64)

    # 实验1：LOW初值 → 共存区各β3
    print("\n--- 实验1: LOW初值(s242_b0.18) → 共存区 ---")
    for b3 in TEST_B3:
        run_from_uhat(d, npz, 242, b3, s242_low, "low", measure)

    # 实验2：MID初值 → 共存区各β3
    print("\n--- 实验2: MID初值(s242_b0.16) → 共存区 ---")
    for b3 in TEST_B3:
        run_from_uhat(d, npz, 242, b3, s242_mid, "mid", measure)

    # 实验3：多seed common初值盆地统计
    print("\n--- 实验3: 多seed common初值盆地统计 ---")
    for seed in BASIN_SEEDS:
        for b3 in TEST_B3:
            run_from_common(d, npz, seed, b3, measure)

    save_state(d, npz)
    print("\n" + "=" * 64)
    print("【裁决】共存区验证结果")
    print("=" * 64)
    print(json.dumps(summarize(d), indent=2, ensure_ascii=False))
    print(f"\n落盘 -> {OUTP} {NPZP}")


if __name__ == "__main__":
    main()

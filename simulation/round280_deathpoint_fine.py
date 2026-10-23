#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# round280_deathpoint_fine.py —— R276-1 第三步：支 A/B 上死亡点细化（步长 0.0025）
#
# 背景（#279）：A 支在 (0.195,0.210] 塌 MID、B 支在 (0.185,0.195] 先 MID 再 LOW，
#   但网格只有 0.015，死亡点只被夹在一个区间里，不知道具体在哪、也不知道
#   究竟是"一步跳死"还是"经过亚稳 MID 再死"。
#
# 判据（主Agent设计，脚本只执行）：
#   - 从 #279 的最后一个 HIGH 终态出发，β3 以 +0.0025 逐级推进（warmup=0，续跑）。
#   - 分支终止：分类不再是 HIGH（落 MID/LOW）→ 记死亡点在该步与上一步之间，并停止该支继续推进。
#   - 若某步仍 HIGH 但 λ>1e-3 → 标 MARGINAL（正在临界慢化），继续推进但结论里单独标注。
#   - measure=400（同 #279 口径，保证可比）；seg=200 → 2 段，段间 ext 差 >0.02 本身即预警。
#
# 实测结果（#280）：A 真死点 (0.2000,0.2025]、死前 λ 4.2e-3→1.9e-2→4.8e-1 三级慢化；
#   B 未死，在 0.1875→0.1900 之间跳到邻近稳定态 B′（ext 0.4616，见 round280b 续跑 1200 恒定）。
#
# 用法：python3 round280_deathpoint_fine.py [--smoke]
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from round246_basin import evolve, N128, DT128   # noqa: E402

SRC = HERE / "out" / "round279_branch_track.npz"
OUTP = HERE / "out" / "round280_deathpoint_fine.json"
NPZP = HERE / "out" / "round280_deathpoint_fine.npz"
MEASURE = 400.0
SMOKE_MEASURE = 8.0

PLAN = [
    # (tag, seed, 起点key, 逐级 β3 序列)
    ("A", 240, "brA_up_0.1950", [0.1975, 0.2000, 0.2025, 0.2050, 0.2075]),
    ("B", 237, "brB_up_0.1850", [0.1875, 0.1900, 0.1925, 0.1950]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--measure", type=float, default=MEASURE)
    args = ap.parse_args()
    measure = SMOKE_MEASURE if args.smoke else args.measure

    src = dict(np.load(SRC))
    d = json.load(open(OUTP)) if OUTP.exists() else {"rows": [], "measure": measure}
    nz = dict(np.load(NPZP)) if NPZP.exists() else {}

    def save():
        tmp = OUTP.with_suffix(".json.tmp")
        json.dump(d, open(tmp, "w"), indent=2, ensure_ascii=False)
        tmp.replace(OUTP)
        np.savez_compressed(NPZP, **nz)

    print(f"#280 死亡点细化 measure={measure:g}")

    for tag, seed, start_key, betas in PLAN:
        if start_key not in src:
            print(f"  [ERROR] 缺起点 {start_key}")
            continue
        u = src[start_key]
        print(f"  支{tag}: 起点 {start_key}")
        for b3 in betas:
            row_key = f"dp{tag}_{b3:.4f}"
            if row_key in nz:
                print(f"    [skip] {row_key}")
                u = nz[row_key]
                if d.get("stop", {}).get(tag) is not None:
                    break
                continue
            row, uf = evolve(b3, N128, DT128, seed, u, 0.0, measure)
            row.update({"branch": tag, "beta3": b3, "row_key": row_key, "measure": measure})
            d["rows"].append(row)
            nz[row_key] = uf
            save()
            lam = row["lambda_last20"]
            print(f"    β3={b3:.4f}: ext={row['ext_tail_mean']:.4f} {row['attractor']:4s} "
                  f"q48={row['q48_final']*100:5.1f}% lam={lam:+.2e} "
                  f"seg={['%.3f' % x for x in row['seg_ext']]}", flush=True)
            u = uf
            if row["attractor"] != "HIGH":
                d.setdefault("stop", {})[tag] = b3
                save()
                print(f"    -> 支{tag} 在 ({betas[betas.index(b3)-1] if betas.index(b3) else betas[0]}, {b3}] 内死亡")
                break
            elif lam > 1e-3:
                print("    -> MARGINAL（临界慢化，尚未跌出 HIGH）")

    print("\n【汇总】")
    for tag in ("A", "B"):
        rows = sorted([r for r in d["rows"] if r["branch"] == tag], key=lambda r: r["beta3"])
        if rows:
            print(f"  支{tag}: " + " -> ".join(f"{r['beta3']:.4f}:{r['ext_tail_mean']:.4f}{'/' + r['attractor']}"
                                                for r in rows))
    if args.smoke:
        print("### SMOKE OK ###")


if __name__ == "__main__":
    main()

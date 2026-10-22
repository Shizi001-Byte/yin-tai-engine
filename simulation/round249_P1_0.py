#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# simulation/round249_read_down.py
# #249 P1 只读审计：闭合 β3 迟滞降支下臂（0.22 -> 0.10）。
# 只复用已验证地基：round246_basin.classify（H/M/L 阈值）与 Q_OUT/L/N128；
# 不重写 evolve / RHS / 色散，不跑新积分。
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from round246_basin import classify, L, N128  # noqa: E402  已验证阈值/网格
from round240_P1_0 import Q_OUT               # noqa: E402  已验证 4.8

N_EXT = int(round(Q_OUT * L / (2.0 * np.pi)))  # =16
JUMP_TOL = 0.05          # #246 沿用的跳变判据
EXPECT_LO = 0.10         # 本轮降支下界
STEP = 0.005
EXPECT_B3 = [round(0.22 - STEP * i, 4) for i in range(int(round((0.22 - EXPECT_LO) / STEP)) + 1)]

SOURCES = [
    ("round248", HERE / "out" / "round248_down_extend.json", 0),
    ("round246_close", HERE / "out" / "round246_close.json", 1),
]


def _rows_from_json(path, name):
    if not path.exists():
        return []
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[warn] {name}: json 解析失败 {exc}", flush=True)
        return []
    rows = []
    if isinstance(d, dict):
        for key in ("down_extended", "down", "rows", "down_arm"):
            if isinstance(d.get(key), list):
                rows = d[key]
                break
    elif isinstance(d, list):
        rows = d
    out = []
    for r in rows:
        if not isinstance(r, dict) or "seed" not in r or "beta3" not in r:
            continue
        if not isinstance(r.get("seg_ext"), list) or not r["seg_ext"]:
            continue
        rr = dict(r)
        rr["_source"] = name
        out.append(rr)
    return out


def _rows_from_npz(name, prio):
    """降级：从每步落盘的 uhat 重建终态 ext（单段，无法恢复段均值）。"""
    out = []
    for p in sorted((HERE / "out").glob("round248*/*.npz")) + sorted(
        (HERE / "out").glob("round248*.npz")
    ):
        try:
            d = np.load(p)
        except Exception:
            continue
        if "uhat" not in d.files or "beta3" not in d.files or "seed" not in d.files:
            continue
        u = d["uhat"]
        n = u.shape[0]
        if not (0 < N_EXT < n // 2):
            continue
        ext = float(np.hypot(abs(u[N_EXT]), abs(u[-N_EXT])) / n)
        out.append(
            {
                "beta3": float(np.ravel(d["beta3"])[0]),
                "seed": int(np.ravel(d["seed"])[0]),
                "seg_ext": [ext],
                "measure": float(np.ravel(d["measure"])[0]) if "measure" in d.files else float("nan"),
                "_source": f"{name}:npz:{p.name}",
            }
        )
    return out


def audit_row(r):
    seg = [float(x) for x in r["seg_ext"]]
    tail = seg[-4:] if len(seg) >= 4 else seg
    tm = float(np.mean(tail))
    drift = float(max(tail) - min(tail)) if len(tail) > 1 else 0.0
    return {
        "beta3": round(float(r["beta3"]), 4),
        "seed": int(r["seed"]),
        "ext_tail_mean": tm,
        "attractor": classify(tm),
        "tail_drift": drift,
        "n_seg": len(seg),
        "q48": float(r.get("q48_final", float("nan"))),
        "lam": float(r.get("lambda_last20", float("nan"))),
        "measure": float(r.get("measure", float("nan"))),
        "source": r["_source"],
    }


def jumps_of(rows_desc):
    """rows_desc 按 beta3 降序；返回相邻 |Δext|>JUMP_TOL 的跳变。"""
    out = []
    for prev, cur in zip(rows_desc, rows_desc[1:]):
        d = cur["ext_tail_mean"] - prev["ext_tail_mean"]
        if abs(d) > JUMP_TOL:
            out.append(
                {
                    "bracket_beta3": [cur["beta3"], prev["beta3"]],
                    "ext_from": prev["ext_tail_mean"],
                    "ext_to": cur["ext_tail_mean"],
                    "delta_ext": d,
                    "attractor_from": prev["attractor"],
                    "attractor_to": cur["attractor"],
                }
            )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "out" / "round249_down_verdict.json"))
    args = ap.parse_args()

    # ---- 多源合并，优先级小者胜，冲突显式报出 ----
    merged, meta = {}, {}
    conflicts = []
    for name, path, prio in SOURCES:
        for r in _rows_from_json(path, name):
            a = audit_row(r)
            key = (a["seed"], a["beta3"])
            if key in merged and merged[key]["_prio"] <= prio:
                continue
            if key in merged:
                old = merged[key]
                if abs(old["ext_tail_mean"] - a["ext_tail_mean"]) > JUMP_TOL:
                    conflicts.append(
                        {
                            "seed": a["seed"], "beta3": a["beta3"],
                            "src_a": old["source"], "ext_a": old["ext_tail_mean"],
                            "src_b": a["source"], "ext_b": a["ext_tail_mean"],
                        }
                    )
            a["_prio"] = prio
            merged[key] = a
    if not merged:
        for r in _rows_from_npz("round248", 0):
            a = audit_row(r)
            a["_prio"] = 0
            merged[(a["seed"], a["beta3"])] = a

    rows = sorted(merged.values(), key=lambda x: (x["seed"], -x["beta3"]))

    seeds = sorted({r["seed"] for r in rows})
    report = {"pass_reader": True, "n_ext": N_EXT, "jump_tol": JUMP_TOL,
              "expected_grid": EXPECT_B3, "seeds": seeds,
              "conflicts": conflicts, "per_seed": []}

    present = {(r["seed"], r["beta3"]) for r in rows}
    for sd in seeds:
        sub = sorted([r for r in rows if r["seed"] == sd], key=lambda x: -x["beta3"])
        if not sub:
            continue
        miss = [b for b in EXPECT_B3 if (sd, b) not in present]
        jmps = jumps_of(sub)
        lowest = sub[-1]
        # 最低 β3 处仍 LOW => 下临界点在扫描窗外（未闭合）；否则闭合
        closed = lowest["attractor"] != "LOW" or bool(jmps)
        report["per_seed"].append(
            {
                "seed": sd,
                "n_points": len(sub),
                "beta3_range": [lowest["beta3"], sub[0]["beta3"]],
                "missing_beta3": miss,
                "table_desc": [
                    {
                        "beta3": r["beta3"],
                        "ext": round(r["ext_tail_mean"], 4),
                        "attractor": r["attractor"],
                        "q48": None if np.isnan(r["q48"]) else round(r["q48"], 4),
                        "lam_last20": None if np.isnan(r["lam"]) else round(r["lam"], 4),
                        "measure": None if np.isnan(r["measure"]) else r["measure"],
                        "source": r["source"],
                    }
                    for r in sub
                ],
                "jumps": jmps,
                "lowest_beta3_attractor": {
                    "beta3": lowest["beta3"], "ext": lowest["ext_tail_mean"],
                    "attractor": lowest["attractor"],
                },
                "down_arm_closed_in_window": bool(closed),
            }
        )

    # ---- 逐假说裁决 ----
    verdicts = {}
    for item in report["per_seed"]:
        sd = item["seed"]
        low = item["lowest_beta3_attractor"]
        if low["attractor"] == "LOW" and not item["jumps"]:
            v = "H1: 降支到 %.3f 仍 LOW，L 稳定域下界<=%.3f（窗内未闭合）" % (low["beta3"], low["beta3"])
        elif item["jumps"]:
            br = item["jumps"][-1]["bracket_beta3"]
            v = "H2/H3: 最低跳变落在 beta3∈[%.3f,%.3f]，%s->%s" % (
                br[0], br[1], item["jumps"][-1]["attractor_from"], item["jumps"][-1]["attractor_to"])
        else:
            v = "其它: 最低点 %s 且无跳变（检查是否数据不全）" % low["attractor"]
        verdicts[str(sd)] = v
    report["verdict_by_seed"] = verdicts
    report["down_arm_fully_closed"] = all(
        i["down_arm_closed_in_window"] for i in report["per_seed"]
    ) and not any(i["missing_beta3"] for i in report["per_seed"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 72)
    print(f"#249 降支审计  N_EXT={N_EXT}  jump_tol={JUMP_TOL}  seeds={seeds}")
    print("=" * 72)
    for item in report["per_seed"]:
        print(f"\nseed {item['seed']}  points={item['n_points']}  "
              f"range=[{item['beta3_range'][0]:.3f},{item['beta3_range'][1]:.3f}]")
        for row in item["table_desc"]:
            print(f"  b3={row['beta3']:5.3f} ext={row['ext']:.4f} "
                  f"{row['attractor']:4s} q48={row['q48']} lam={row['lam_last20']}")
        print("  jumps:", item["jumps"] or "无")
        print("  裁决:", report["verdict_by_seed"][str(item["seed"])])
        if item["missing_beta3"]:
            print("  缺测点:", item["missing_beta3"])
    print("\nCONFLICTS:", report["conflicts"] or "无")
    print("down_arm_fully_closed =", report["down_arm_fully_closed"])
    print("落盘 ->", out)


if __name__ == "__main__":
    main()

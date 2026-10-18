#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# simulation/round246_read_verdict.py
#
# #246 深度轮只读裁决：读取已完成的 round246_basin_hyst.json 与日志，
# 不重新积分、不修改物理参数。独立重算盆地占比、迟滞 gap、跳变点和
# 末段沉降标志，避免依据单个 ext 数值即兴下结论。
#
# 用法：
#   python3 simulation/round246_read_verdict.py
#   python3 simulation/round246_read_verdict.py \
#       --json simulation/out/round246_basin_hyst.json \
#       --log simulation/round246_basin.log

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

EXPECT_BASIN_B3 = [0.19, 0.20, 0.205, 0.21, 0.215, 0.22]
EXPECT_HYST_B3 = [0.18, 0.185, 0.19, 0.195, 0.20, 0.205, 0.21, 0.215, 0.22]
EXPECT_SEEDS = [240, 241, 237, 242]
EXPECT_HYST_SEEDS = [240, 242]

HIGH_THRESHOLD = 0.38
MID_THRESHOLD = 0.28
GAP_THRESHOLD = 0.05
JUMP_THRESHOLD = 0.05
SEGMENT_DRIFT_LIMIT = 0.03


def classify(ext_tail_mean: float) -> str:
    if ext_tail_mean > HIGH_THRESHOLD:
        return "HIGH"
    if ext_tail_mean >= MID_THRESHOLD:
        return "MID"
    return "LOW"


def audit_row(row: dict) -> dict:
    """从 seg_ext 独立重算尾段均值/吸引子，不信任已存储的 attractor 字段。"""
    seg = list(row.get("seg_ext", []))
    tail = seg[-4:]
    if not tail:
        return {
            "ok": False,
            "tail_mean": None,
            "attractor": None,
            "tail_drift": None,
            "settled": False,
            "label_mismatch": True,
        }

    tail_mean = sum(tail) / len(tail)
    attractor = classify(tail_mean)
    tail_drift = max(tail) - min(tail)

    # basin measure=800 应有 4 段；hyst measure=200 只有 1 段，不据此判沉降。
    settled = None if len(tail) < 4 else tail_drift <= SEGMENT_DRIFT_LIMIT

    label_mismatch = row.get("attractor") != attractor
    ext_mismatch = abs(row.get("ext_tail_mean", tail_mean) - tail_mean) > 1e-9

    return {
        "ok": True,
        "tail_mean": float(tail_mean),
        "attractor": attractor,
        "tail_drift": float(tail_drift),
        "settled": settled,
        "label_mismatch": bool(label_mismatch or ext_mismatch),
    }


def basin_verdict(rows: list[dict]) -> dict:
    warnings = []

    existing = {
        (round(float(r["beta3"]), 6), int(r["seed"]))
        for r in rows
    }
    missing = sorted(
        [b3, s]
        for b3 in EXPECT_BASIN_B3
        for s in EXPECT_SEEDS
        if (round(b3, 6), s) not in existing
    )
    if missing:
        warnings.append(f"basin 缺少 {len(missing)} 个工况: {missing}")

    table = []
    total_counts = {"HIGH": 0, "MID": 0, "LOW": 0}
    sustained_chaotic = []

    for b3 in EXPECT_BASIN_B3:
        sub = [r for r in rows if abs(float(r["beta3"]) - b3) < 1e-12]
        counts = {"HIGH": 0, "MID": 0, "LOW": 0}
        seed_map = {}
        q48_by_attractor = {"HIGH": [], "MID": [], "LOW": []}
        unsettled_seeds = []

        for row in sub:
            aud = audit_row(row)
            seed = int(row["seed"])
            attr = aud["attractor"]
            seed_map[str(seed)] = {
                "attractor": attr,
                "ext_tail_mean": aud["tail_mean"],
                "tail_drift": aud["tail_drift"],
                "settled": aud["settled"],
                "seg_ext": row.get("seg_ext", []),
                "q48_final": row.get("q48_final"),
                "lambda_last20": row.get("lambda_last20"),
            }

            if attr is not None:
                counts[attr] += 1
                total_counts[attr] += 1
                q48_by_attractor[attr].append(float(row["q48_final"]))

            if aud["label_mismatch"]:
                warnings.append(
                    f"basin b3={b3} seed={seed} 存储标签/尾段重算不一致"
                )
            if aud["settled"] is False:
                unsettled_seeds.append(seed)
                warnings.append(
                    f"basin b3={b3} seed={seed} 末4段漂移="
                    f"{aud['tail_drift']:.4f} > {SEGMENT_DRIFT_LIMIT}"
                )
            if float(row.get("measure", 0.0)) != 800.0:
                warnings.append(
                    f"basin b3={b3} seed={seed} measure="
                    f"{row.get('measure')} != 800"
                )
            if row.get("classification") == "sustained_chaotic":
                sustained_chaotic.append([b3, seed])

        n = len(sub)
        table.append(
            {
                "beta3": b3,
                "n": n,
                "counts": counts,
                "fractions": {
                    k: (counts[k] / n if n else None) for k in counts
                },
                "unsettled_seeds": unsettled_seeds,
                "seeds": seed_map,
                "q48_mean_by_attractor": {
                    k: (sum(v) / len(v) if v else None)
                    for k, v in q48_by_attractor.items()
                },
            }
        )

    return {
        "expected_rows": len(EXPECT_BASIN_B3) * len(EXPECT_SEEDS),
        "observed_rows": len(rows),
        "missing": missing,
        "total_counts": total_counts,
        "table": table,
        "sustained_chaotic_cases": sustained_chaotic,
        "warnings": warnings,
    }


def branch_jumps(rows_ascending: list[dict], direction: str) -> list[dict]:
    """direction='up' 按 beta3 升序；'down' 按绝热实验中的降序。"""
    sequence = (
        rows_ascending
        if direction == "up"
        else list(reversed(rows_ascending))
    )
    jumps = []

    for prev, cur in zip(sequence, sequence[1:]):
        prev_ext = float(prev["ext_tail_mean"])
        cur_ext = float(cur["ext_tail_mean"])
        delta = cur_ext - prev_ext

        if abs(delta) > JUMP_THRESHOLD:
            jumps.append(
                {
                    "direction": direction,
                    "from_beta3": float(prev["beta3"]),
                    "to_beta3": float(cur["beta3"]),
                    "bracket": sorted(
                        [float(prev["beta3"]), float(cur["beta3"])]
                    ),
                    "from_ext": prev_ext,
                    "to_ext": cur_ext,
                    "delta_ext_along_traversal": float(delta),
                    "from_attractor": prev.get("attractor"),
                    "to_attractor": cur.get("attractor"),
                }
            )

    return jumps


def contiguous_runs(values: list[float]) -> list[list[float]]:
    values = sorted(values)
    runs = []
    for value in values:
        if not runs or abs(value - runs[-1][-1]) > 0.011:
            runs.append([value])
        else:
            runs[-1].append(value)
    return runs


def hysteresis_one_seed(key: str, payload: dict) -> dict:
    warnings = []

    up = sorted(payload["up"], key=lambda r: float(r["beta3"]))
    down = sorted(payload["down"], key=lambda r: float(r["beta3"]))
    down_map = {round(float(r["beta3"]), 6): r for r in down}

    aligned = []
    coexistence_b3 = []

    for urow in up:
        b3 = float(urow["beta3"])
        drow = down_map.get(round(b3, 6))
        if drow is None:
            warnings.append(f"{key}: DOWN 缺 b3={b3}")
            continue

        u_aud = audit_row(urow)
        d_aud = audit_row(drow)
        u_ext = u_aud["tail_mean"]
        d_ext = d_aud["tail_mean"]
        gap = abs(u_ext - d_ext)

        item = {
            "beta3": b3,
            "up_ext": u_ext,
            "down_ext": d_ext,
            "up_attractor": u_aud["attractor"],
            "down_attractor": d_aud["attractor"],
            "gap": float(gap),
            "path_split": bool(gap > GAP_THRESHOLD),
        }
        aligned.append(item)
        if item["path_split"]:
            coexistence_b3.append(b3)

    for branch_name, rows in [("up", up), ("down", down)]:
        if len(rows) != len(EXPECT_HYST_B3):
            warnings.append(
                f"{key}.{branch_name} 有 {len(rows)} 行，"
                f"预期 {len(EXPECT_HYST_B3)}"
            )
        for row in rows:
            if float(row.get("measure", 0.0)) != 200.0:
                warnings.append(
                    f"{key}.{branch_name} b3={row['beta3']} "
                    f"measure={row.get('measure')} != 200"
                )

    match = re.search(r"s(\d+)$", key)
    seed_label = match.group(1) if match else key

    return {
        "seed": seed_label,
        "aligned": aligned,
        "coexistence_b3": coexistence_b3,
        "coexistence_runs": contiguous_runs(coexistence_b3),
        "up_jumps": branch_jumps(up, "up"),
        "down_jumps": branch_jumps(down, "down"),
        "warnings": warnings,
    }


def find_log(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    candidates = [
        HERE / "round246_basin.log",
        HERE.parent / "round246_basin.log",
        Path.cwd() / "round246_basin.log",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def audit_log(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path), "exists": False}

    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    error_lines = [
        line
        for line in lines
        if "Traceback" in line or "Error" in line or "RuntimeError" in line
    ]
    return {
        "path": str(path),
        "path": str(path),
        "n_lines": len(lines),
        "has_traceback": "Traceback" in text,
        "error_lines_tail": error_lines[-10:],
        "has_completion_marker": "全矩阵落盘" in text,
        "last_20_lines": lines[-20:],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        type=Path,
        default=HERE / "out" / "round246_basin_hyst.json",
    )
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=HERE / "out" / "round246_verdict.json",
    )
    args = parser.parse_args()

    if not args.json.exists():
        raise FileNotFoundError(
            f"找不到 {args.json}；#246 深度轮不得因此重跑全矩阵，"
            "应先检查后台任务或输出路径。"
        )

    data = json.loads(args.json.read_text(encoding="utf-8"))

    basin = basin_verdict(data.get("basin_N128", []))

    hyst_payloads = {
        key: value
        for key, value in data.items()
        if key.startswith("hyst_")
        and isinstance(value, dict)
        and "up" in value
        and "down" in value
    }
    hysteresis = [
        hysteresis_one_seed(key, payload)
        for key, payload in sorted(hyst_payloads.items())
    ]

    robust_sets = [set(h["coexistence_b3"]) for h in hysteresis]
    robust_b3 = sorted(set.intersection(*robust_sets)) if robust_sets else []

    all_warnings = list(basin["warnings"])
    for h in hysteresis:
        all_warnings.extend(h["warnings"])

    path_dependence_supported = (
        len(hysteresis) >= 2
        and len(robust_b3) > 0
        and all(h["up_jumps"] for h in hysteresis)
        and all(h["down_jumps"] for h in hysteresis)
    )

    basin_all_settled = all(
        seed_info["settled"] is not False
        for b3_row in basin["table"]
        for seed_info in b3_row["seeds"].values()
    )

    log_info = audit_log(find_log(args.log))

    verdict = {
        "pass_reader": True,
        "input_json": str(args.json),
        "thresholds": {
            "HIGH_min": HIGH_THRESHOLD,
            "MID_range": [MID_THRESHOLD, HIGH_THRESHOLD],
            "hysteresis_gap": GAP_THRESHOLD,
            "jump": JUMP_THRESHOLD,
            "basin_tail_drift_limit": SEGMENT_DRIFT_LIMIT,
        },
        "basin": basin,
        "hysteresis": hysteresis,
        "robust_coexistence_b3": robust_b3,
        "robust_coexistence_runs": contiguous_runs(robust_b3),
        "criteria": {
            "n_hysteresis_seeds": len(hysteresis),
            "all_expected_basin_rows_present": not basin["missing"],
            "all_basin_tail4_settled": basin_all_settled,
            "up_jump_in_every_seed": all(
                h["up_jumps"] for h in hysteresis
            ),
            "down_jump_in_every_seed": all(
                h["down_jumps"] for h in hysteresis
            ),
            "shared_gap_points_in_all_seeds": len(robust_b3) > 0,
            "no_sustained_chaotic_classification":
                not basin["sustained_chaotic_cases"],
            "first_order_path_dependence_supported_on_grid":
                path_dependence_supported,
        },
        "log": log_info,
        "warnings": all_warnings,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 72)
    print("实验 A：盆地占比")
    print("=" * 72)
    for row in basin["table"]:
        fractions = row["fractions"]
        print(
            f"b3={row['beta3']:5.3f}  "
            f"HIGH={row['counts']['HIGH']} "
            f"MID={row['counts']['MID']} "
            f"LOW={row['counts']['LOW']}  "
            f"({fractions['HIGH']:.0%}/{fractions['MID']:.0%}/"
            f"{fractions['LOW']:.0%})  "
            f"unsettled={row['unsettled_seeds']}"
        )

    print("\n" + "=" * 72)
    print("实验 B：绝热迟滞")
    print("=" * 72)
    for h in hysteresis:
        print(f"\nseed {h['seed']}")
        for item in h["aligned"]:
            flag = " <== SPLIT" if item["path_split"] else ""
            print(
                f"  b3={item['beta3']:5.3f}  "
                f"UP={item['up_ext']:.3f}({item['up_attractor']:4s})  "
                f"DOWN={item['down_ext']:.3f}({item['down_attractor']:4s})  "
                f"gap={item['gap']:.3f}{flag}"
            )
        print("  UP jumps:")
        for jump in h["up_jumps"]:
            print(
                f"    {jump['from_beta3']:.3f}->{jump['to_beta3']:.3f}: "
                f"{jump['from_ext']:.3f}->{jump['to_ext']:.3f}"
            )
        print("  DOWN jumps:")
        for jump in h["down_jumps"]:
            print(
                f"    {jump['from_beta3']:.3f}->{jump['to_beta3']:.3f}: "
                f"{jump['from_ext']:.3f}->{jump['to_ext']:.3f}"
            )

    print("\n" + "=" * 72)
    print("机器裁决")
    print("=" * 72)
    print(json.dumps(verdict["criteria"], indent=2, ensure_ascii=False))
    print(f"\n明细已写入: {args.out}")


if __name__ == "__main__":
    main()

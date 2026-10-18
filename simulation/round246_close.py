#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# #246 收尾实验（决策层类型b）：闭合多稳态迟滞回线 + 处理未沉降case
# 已确立(见 verdict)：三吸引子 H(ext~0.43-0.48)/M(ext~0.34)/L(ext~0.09-0.15)，
#   双seed共享gap在0.18-0.21；但①s240 DOWN低态保持到0.18未跳回(下临界点<0.18未测到)、
#   s242 DOWN落到中态保持到0.18(中态低β3失稳点未测)；②b3=0.20s241、0.205s242 measure800
#   末段未平台(临界慢化)。
# 本脚本：
#   实验C 延伸DOWN：从0.22 common平衡后绝热降到0.15(步长0.005,每步measure300)，
#         s240/s242 各自，找低态/中态在低β3跳回高态的下临界点；
#   实验D 延长未沉降case：(0.20,s241)与(0.205,s242)独立common初值 measure1600，看最终平台。
# 增量落盘、可断点续跑、setsid后台。
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
from round231_P1_0 import make_grid                       # noqa: E402
from round234_P1_0 import common_initial_state           # noqa: E402
from round240_P1_0 import Q_OUT                          # noqa: E402
from round246_basin import evolve, L, N128, DT128        # noqa: E402

OUTP = HERE / "out" / "round246_close.json"
# 从0.22降到0.15，步长0.005（含0.18以下延伸点）
DOWN_B3 = [round(0.22 - 0.005 * i, 3) for i in range(15)]  # 0.22..0.15
LONG_CASES = [(0.20, 241), (0.205, 242)]


def load():
    return json.load(open(OUTP)) if OUTP.exists() else {"down_extended": [], "long": []}


def save(d):
    tmp = OUTP.with_suffix(".json.tmp")
    json.dump(d, open(tmp, "w"), indent=2, ensure_ascii=False)
    tmp.replace(OUTP)


def upsert(lst, row, keys):
    for i, x in enumerate(lst):
        if all(abs(x[k] - row[k]) < 1e-9 if isinstance(row[k], float)
               else x[k] == row[k] for k in keys):
            lst[i] = row
            return
    lst.append(row)


def down_extended(d, seed, step_measure):
    k, mask = make_grid(N128, L)
    n_ext = int(round(Q_OUT * L / (2 * np.pi)))
    rows = d.setdefault("down_extended", [])
    # 0.22端点common平衡
    base = common_initial_state(N128, mask, n_ext, seed)
    row, uhat = evolve(DOWN_B3[0], N128, DT128, seed, base, 100.0, float(step_measure))
    row["seed"] = seed
    emit(rows, row, ("beta3", "seed"))
    print(f"[C s{seed}] DOWN b3={DOWN_B3[0]:5.3f} ext={row['ext_tail_mean']:.3f} {row['attractor']}", flush=True)
    save(d)
    for b3 in DOWN_B3[1:]:
        key = (round(b3, 4), seed)
        if any((round(x["beta3"], 4) == key[0] and x["seed"] == seed) for x in rows):
            # 续跑需要uhat，不能跳；为简单仍顺序重放（只在首次运行，这里不跳过）
            pass
        row, uhat = evolve(b3, N128, DT128, seed, uhat, 0.0, float(step_measure))
        row["seed"] = seed
        emit(rows, row, ("beta3", "seed"))
        print(f"[C s{seed}] DOWN b3={b3:5.3f} ext={row['ext_tail_mean']:.3f} {row['attractor']}", flush=True)
        save(d)


def emit(lst, row, keys):
    for i, x in enumerate(lst):
        if all(abs(x[k] - row[k]) < 1e-9 if isinstance(row[k], float) else x[k] == row[k]
               for k in keys):
            lst[i] = row
            return
    lst.append(row)


def long_settle(d, measure):
    k, mask = make_grid(N128, L)
    n_ext = int(round(Q_OUT * L / (2 * np.pi)))
    rows = d.setdefault("long", [])
    for b3, seed in LONG_CASES:
        if any(abs(x["beta3"] - b3) < 1e-9 and x["seed"] == seed and x["measure"] == measure
               for x in rows):
            continue
        base = common_initial_state(N128, mask, n_ext, seed)
        print(f"[D] long b3={b3} s={seed} measure={measure}", flush=True)
        row, _ = evolve(b3, N128, DT128, seed, base, 100.0, float(measure))
        emit(rows, row, ("beta3", "seed", "measure"))
        print(f"    seg(8段)= {['%.3f' % x for x in row['seg_ext']]}", flush=True)
        save(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--down-only", action="store_true")
    ap.add_argument("--long-only", action="store_true")
    args = ap.parse_args()
    step = 8 if args.smoke else 300
    mlong = 16 if args.smoke else 1600
    d = load()
    if not args.long_only:
        down_extended(d, 240, step)
        down_extended(d, 242, step)
    if not args.down_only:
        long_settle(d, mlong)
    save(d)
    print("close done ->", OUTP, flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# #246 恢复/补跑脚本（决策层类型b，复用 round246_basin 已验证管道）
# 背景：round246_basin.py 全矩阵在 20:31 被沙箱回收（停于盆地 b3=0.215 s237），
#       盆地A 18 个 case 结果在日志、滞后实验B未跑、json 未生成。
# 本脚本：①补跑盆地 b3=0.215/0.22 × 4seed（measure800）；②完整滞后 s240+s242。
# 关键：每完成一个 case / 一个绝热步立即增量落盘，再被回收也不丢；可断点续跑。
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
from round246_basin import evolve, L, N128, DT128, HYST_B3  # noqa: E402

OUTP = HERE / "out" / "round246_resume.json"
RESUME_B3 = [0.215, 0.22]
SEEDS = [240, 241, 237, 242]


def load():
    if OUTP.exists():
        return json.load(open(OUTP))
    return {"basin_resume": [], "hyst_resume": {}}


def save(d):
    tmp = OUTP.with_suffix(".json.tmp")
    json.dump(d, open(tmp, "w"), indent=2, ensure_ascii=False)
    tmp.replace(OUTP)


def basin_resume(d, measure):
    k, mask = make_grid(N128, L)
    n_ext = int(round(Q_OUT * L / (2 * np.pi)))
    rows = d.setdefault("basin_resume", [])
    done = {(round(r["beta3"], 4), r["seed"]) for r in rows}
    for b3 in RESUME_B3:
        for sd in SEEDS:
            if (round(b3, 4), sd) in done:
                continue
            base = common_initial_state(N128, mask, n_ext, sd)
            print(f"[A-resume] b3={b3} s={sd} measure={measure}", flush=True)
            row, _ = evolve(b3, N128, DT128, sd, base, 100.0, float(measure))
            print(f"           tail={['%.3f' % x for x in row['seg_ext'][-4:]]} "
                  f"-> {row['attractor']} q48={row['q48_final']*100:.1f}%", flush=True)
            rows.append(row)
            save(d)


def hyst_resume(d, seed, step_measure):
    k, mask = make_grid(N128, L)
    n_ext = int(round(Q_OUT * L / (2 * np.pi)))
    key = f"s{seed}"
    h = d.setdefault("hyst_resume", {}).setdefault(key, {"up": [], "down": []})
    done_up = {round(x["beta3"], 4) for x in h["up"]}
    done_down = {round(x["beta3"], 4) for x in h["down"]}

    # UP：需要从最低点连续绝热（续跑依赖前点 uhat，不能跳过中间点，故按链重放）
    base = common_initial_state(N128, mask, n_ext, seed)
    uhat = None
    for i, b3 in enumerate(HYST_B3):
        if round(b3, 4) in done_up and uhat is not None:
            # 已有点：为了续跑链仍需重算一次拿 uhat（但简单起见从头绝热重放，省分支）
            pass
        if i == 0:
            row, uhat = evolve(b3, N128, DT128, seed, base, 100.0, 200.0)
        else:
            row, uhat = evolve(b3, N128, DT128, seed, uhat, 0.0, float(step_measure))
        upsert(h["up"], row)
        print(f"[B-resume {key}] UP   b3={b3:5.3f} ext={row['ext_tail_mean']:.3f} "
              f"{row['attractor']}", flush=True)
        save(d)

    # DOWN：在最高 b3 平衡后逐点降
    base = common_initial_state(N128, mask, n_ext, seed)
    row0, uhat = evolve(HYST_B3[-1], N128, DT128, seed, base, 100.0, 200.0)
    upsert(h["down"], row0)
    save(d)
    for b3 in reversed(HYST_B3):
        if b3 == HYST_B3[-1]:
            continue
        row, uhat = evolve(b3, N128, DT128, seed, uhat, 0.0, float(step_measure))
        upsert(h["down"], row)
        print(f"[B-resume {key}] DOWN b3={b3:5.3f} ext={row['ext_tail_mean']:.3f} "
              f"{row['attractor']}", flush=True)
        save(d)


def upsert(lst, row):
    key = round(row["beta3"], 4)
    for i, x in enumerate(lst):
        if round(x["beta3"], 4) == key:
            lst[i] = row
            return
    lst.append(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyst-only", action="store_true")
    ap.add_argument("--basin-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    m = 8 if args.smoke else 800
    step = 8 if args.smoke else 200
    d = load()
    if not args.hyst_only:
        basin_resume(d, m)
    if not args.basin_only:
        hyst_resume(d, 240, step)
        hyst_resume(d, 242, step)
    save(d)
    print("resume done ->", OUTP, flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""round296_sweep.py —— 绝热扫描 + 终态落盘（可续跑）

补上历轮反复踩的坑：绝热降/升支每次都要从 0.22 重来，因为上一轮的终态 uhat 没存。
本脚本每走一步就把 (seed, beta3) 的终态 uhat 存进 npz，下次可用 --resume 从端点续跑。

用法：
  python3 round296_sweep.py --b3s 0.22:0.02:0.01 --seeds 240,242 --tag down02
  python3 round296_sweep.py --b3s 0.10:0.02:0.01 --seeds 240,242 --tag down02 \
      --resume out/round296_down26.npz --resume-from 0.10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from round231_P1_0 import make_grid            # noqa: E402
from round234_P1_0 import common_initial_state  # noqa: E402
from round240_P1_0 import Q_OUT                 # noqa: E402
from round246_basin import evolve, L, N128, DT128  # noqa: E402


def parse_range(s: str) -> list[float]:
    a, b, step = [float(x) for x in s.split(":")]
    step = abs(step)
    n = int(round(abs(b - a) / step)) + 1
    sign = 1.0 if b >= a else -1.0
    return [round(a + sign * step * i, 4) for i in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b3s", required=True, help="start:stop:step，如 0.22:0.02:0.01")
    ap.add_argument("--seeds", default="240,242")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--measure", type=float, default=300.0)
    ap.add_argument("--seg", type=float, default=50.0)
    ap.add_argument("--warmup", type=float, default=100.0)
    ap.add_argument("--smoke", action="store_true", help="measure=60/seg=10 只验管道")
    ap.add_argument("--resume", default=None, help="已有 npz，含终态 uhat")
    ap.add_argument("--resume-from", default=None, help="续跑起点 beta3")
    args = ap.parse_args()

    b3s = parse_range(args.b3s)
    seeds = [int(s) for s in args.seeds.split(",")]
    measure = 60.0 if args.smoke else args.measure
    seg = 10.0 if args.smoke else args.seg
    warm0 = 20.0 if args.smoke else args.warmup

    jsonp = HERE / "out" / f"round296_{args.tag}.json"
    npzp = HERE / "out" / f"round296_{args.tag}.npz"

    rows: list[dict] = []
    states: dict[str, np.ndarray] = {}
    if args.resume:
        rp = Path(args.resume)
        rp = rp if rp.is_absolute() else HERE / rp
        old = dict(np.load(rp))
        states.update({k: v for k, v in old.items() if not k.startswith("meta_")})
    if jsonp.exists():
        rows = json.loads(jsonp.read_text(encoding="utf-8")).get("rows", [])

    k, mask = make_grid(N128, L)
    n_ext = int(round(Q_OUT * L / (2 * np.pi)))

    for seed in seeds:
        base = common_initial_state(N128, mask, n_ext, seed)
        uhat = None
        start_idx = 0
        if args.resume:
            cand = [b for b in b3s if f"s{seed}_b{b:.4f}" in states]
            if args.resume_from:
                cand = [b for b in b3s if abs(b - float(args.resume_from)) < 1e-9]
            if not cand:
                print(f"[s{seed}] resume 无可用端点，改从 common_initial 冷启", flush=True)
            else:
                b0 = cand[0]
                uhat = states[f"s{seed}_b{b0:.4f}"]
                start_idx = b3s.index(b0)
                print(f"[s{seed}] 从 b3={b0:.3f} 的存盘终态续跑", flush=True)

        for i, b3 in enumerate(b3s):
            if i < start_idx:
                continue
            if uhat is None:
                row, uhat = evolve(b3, N128, DT128, seed, base, warm0, measure, seg=seg)
            else:
                row, uhat = evolve(b3, N128, DT128, seed, uhat, 0.0, measure, seg=seg)
            row["beta3"] = float(b3)
            row["seed"] = int(seed)
            row["source"] = f"round296_{args.tag}"
            rows = [x for x in rows if not (abs(float(x["beta3"]) - b3) < 1e-9 and int(x["seed"]) == seed)]
            rows.append(row)
            states[f"s{seed}_b{b3:.4f}"] = uhat.copy()
            np.savez_compressed(npzp, **states)
            jsonp.write_text(json.dumps({"rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[s{seed}] b3={b3:5.3f} ext_tail={row['ext_tail_mean']:.3f} "
                  f"{row['attractor']:4s} segs={['%.3f' % x for x in row['seg_ext']]}", flush=True)

    print("done ->", jsonp, npzp, flush=True)


if __name__ == "__main__":
    main()

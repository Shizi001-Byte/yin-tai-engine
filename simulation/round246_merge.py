#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# #246 合并：把日志恢复的 partial(18case) 与 resume(补全0.215/0.22+滞后)
# 合并成 round246_basin.py 标准输出格式 round246_basin_hyst.json，供 worker 裁决脚本 read_verdict。
# 去重：同一 (beta3,seed) 以 resume 直接输出为准（覆盖 partial）。
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
part = json.load(open(OUT / "round246_basin_partial.json"))
res = json.load(open(OUT / "round246_resume.json"))

by = {}
for r in part.get("basin_partial", []):
    r = {k: v for k, v in r.items() if k != "source"}
    by[(round(float(r["beta3"]), 4), int(r["seed"]))] = r
for r in res.get("basin_resume", []):
    by[(round(float(r["beta3"]), 4), int(r["seed"]))] = r

basin = sorted(by.values(), key=lambda r: (float(r["beta3"]), int(r["seed"])))
out = {"basin_N128": basin}
for key in ("s240", "s242"):
    h = res["hyst_resume"][key]
    out["hyst_" + key] = {
        "up": sorted(h["up"], key=lambda r: float(r["beta3"])),
        "down": sorted(h["down"], key=lambda r: float(r["beta3"])),
    }

json.dump(out, open(OUT / "round246_basin_hyst.json", "w"), indent=2, ensure_ascii=False)
print("basin rows:", len(basin))
from collections import Counter
for b3 in sorted(set(r["beta3"] for r in basin)):
    c = Counter(r["attractor"] for r in basin if r["beta3"] == b3)
    print(f"  b3={b3:5.3f}", dict(c))

#!/usr/bin/env python3
# 决策层【验证子Agent·超长测程】脚本：针对 #243 独立验证发现的异常点 beta3=0.20
# （measure=200 下跨 seed 功率 spread 10.57pp、N128s241 仍高功率 12.6%、N160s241 λ=+0.231/tau=2151）
# 目标：区分两种互斥解释
#   (A) 真双稳态/一阶分岔：同参数不同 seed 永久落在高/低两个不同吸引子，长测程后仍不收敛到同一态；
#   (B) 亚稳态超长暂态/临界慢化：高功率态只是寿命 >200 的亚稳态，长测程后全部跌落至低功率态。
# 做法：b3=0.20，4 seed（240/241/237/242）× N128/160，warmup=100、measure=800，
#   分段（每200单位）报告 q4.8 外带振幅/功率与 lambda，看高功率态何时（或是否）跌落；
#   附 b3=0.21 作“越临界后”对照（measure=600）。
# 只复用已验证物理核（round231 run_case / round234 公共初值 / round240 summarize），不重写。
import argparse, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
from round231_P1_0 import make_grid, run_case
from round234_P1_0 import common_initial_state
from round240_P1_0 import BETA4, Q_OUT, summarize


def run_long(b3, N, dt, seed, L, warmup, measure, seg=200.0):
    k, mask = make_grid(N, L)
    n_ext = int(round(Q_OUT * L / (2 * np.pi)))
    base = common_initial_state(N, mask, n_ext, seed)
    args = argparse.Namespace(warmup=warmup, measure=measure, renorm=1.0, seed=seed)
    r = run_case(f"b3_{b3}_N{N}_s{seed}", b3, BETA4, base, N, L, dt, k, mask, n_ext, args)
    s = summarize(r, n_ext, N)
    # 分段：external_amp 均值（外带强弱）、lambda 分段均值
    nseg = int(measure // seg)
    ext = r["external_amp"]; lam = r["lambda_segment"]
    seg_ext, seg_lam = [], []
    for i in range(nseg):
        a, b = int(i * seg), int((i + 1) * seg)
        seg_ext.append(float(ext[a:b].mean()))
        seg_lam.append(float(lam[a:b].mean()))
    return {
        "beta3": b3, "N": N, "seed": seed, "measure": measure,
        "q48_final": s["q4p8_power_fraction"],
        "lambda_last20": s["lambda_last20"],
        "classification": s["classification"],
        "seg_ext_amp": seg_ext,   # 外带振幅随时间：高功率态是否跌落
        "seg_lambda": seg_lam,    # 分段 lambda
    }


def main():
    L = 20.0 * np.pi / 3.0
    seeds = [240, 241, 237, 242]
    res = {(128, 0.002): [], (160, 0.001): []}
    rows = []
    # 主：b3=0.20 measure=800
    for (N, dt) in res:
        for sd in seeds:
            print(f"=== b3=0.20 N={N} s={sd} measure=800 ===", flush=True)
            row = run_long(0.20, N, dt, sd, L, 100.0, 800.0)
            rows.append(row)
            print(f"   q48_final={row['q48_final']*100:5.1f}% lam20={row['lambda_last20']:+.4f} "
                  f"seg_ext={['%.3f'%x for x in row['seg_ext_amp']]}", flush=True)
    # 对照：b3=0.21 measure=600
    for (N, dt) in res:
        for sd in [240, 241]:
            print(f"=== b3=0.21 N={N} s={sd} measure=600 ===", flush=True)
            row = run_long(0.21, N, dt, sd, L, 100.0, 600.0)
            rows.append(row)
            print(f"   q48_final={row['q48_final']*100:5.1f}% lam20={row['lambda_last20']:+.4f} "
                  f"seg_ext={['%.3f'%x for x in row['seg_ext_amp']]}", flush=True)

    out = HERE / "out" / "round243_longcheck.json"
    json.dump(rows, open(out, "w"), indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("【裁决】b3=0.20 高功率态(N128s241 measure200=12.6%)在 measure800 的命运：")
    print("=" * 70)
    for row in rows:
        if row["beta3"] == 0.20:
            first, last = row["seg_ext_amp"][0], row["seg_ext_amp"][-1]
            verdict = "跌落(亚稳态/暂态)" if last < 0.5 * first else "维持(疑似双稳态)"
            print(f"  N={row['N']} s{row['seed']}: ext首段={first:.4f} 末段={last:.4f} "
                  f"q48终={row['q48_final']*100:5.1f}% -> {verdict}")


if __name__ == "__main__":
    main()

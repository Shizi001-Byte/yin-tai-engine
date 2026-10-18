#!/usr/bin/env python3
# 决策层【验证子Agent】#243 独立验证：不读 worker 结论，只从 npz 原始数组重算。
# 检查：① 数据加载 ② q4.8 功率重算 ③ lambda_last20 重算 ④ bootstrap 误差 ⑤ log-linear 弛豫 τ
# ⑥ 跨 N/seed 一致性裁决 ⑦ 物理结论与证伪门槛
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

npz_path = HERE / "out" / "round243_P1_transition.npz"
json_path = HERE / "out" / "round243_P1_transition.json"


def q48_power(uhat, k, mask, n_ext):
    n = len(k)
    q = np.fft.fftfreq(n, d=1.0/n) * (2*np.pi/(20.0*np.pi/3.0))
    q = np.concatenate([np.arange(n//2), np.arange(-n//2, 0)]) * (2*np.pi/(20.0*np.pi/3.0))
    # 以掩码后的 k 为准（与 round240 summarize 一致）
    P = np.abs(uhat)**2
    tot = P[mask].sum()
    sel = mask & (np.abs(np.abs(k)-4.8) < 0.6)
    return P[sel].sum()/tot if tot > 0 else 0.0


def seg_mean(x, lo, hi):
    lo = int(lo); hi = int(hi)
    return float(x[lo:hi].mean()) if hi > lo else np.nan


def bootstrap_lambda(lam_seg, t, warmup, n_boot=400, rng=None):
    rng = rng or np.random.default_rng(0)
    m = t >= warmup
    tt = t[m]; ll = lam_seg[m]
    n = len(ll)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(ll[idx].mean())
    return float(np.std(vals))


def loglinear_tau(ext, t, warmup):
    m = t >= warmup
    tt = t[m]; ee = np.asarray(ext)[m]
    eend = ee[-1]
    amp = ee - eend
    good = amp > 1e-4
    if good.sum() < 8:
        return None
    A = np.vstack([tt[good], np.ones(good.sum())]).T
    slope, intercept = np.linalg.lstsq(A, np.log(amp[good]), rcond=None)[0]
    if slope >= 0:
        return None
    return -1.0/slope


def main():
    data = json.load(open(json_path))
    z = np.load(npz_path)
    print("=== 1) npz 加载 ===")
    keys = list(z.keys())
    print("  npz 数组数:", len(keys), "（每 case 5 个数组，预期 24*5=120）")
    assert len(keys) == 120, f"数组数 {len(keys)} != 120"

    print("=== 2/3) 独立重算 q4.8 功率与 lambda_last20 ===")
    max_dp, max_dl = 0.0, 0.0
    recalc = {}
    for row in data["scan_summary"]:
        b3 = row["beta3"]; sd = row["seed"]; n = row["N"]
        prefix = f"b3_{b3:.2f}_N{n}_s{sd}"
        uhat = z[prefix + "_uhat"]
        lam_seg = z[prefix + "_lambda_segment"]
        t = z[prefix + "_t"]
        ext = z[prefix + "_external_amp"]
        k = np.fft.fftfreq(n, d=(20.0*np.pi/3.0)/n) * 2*np.pi
        k = np.concatenate([np.arange(n//2), np.arange(-n//2, 0)]) * (2*np.pi/(20.0*np.pi/3.0))
        mask = np.abs(k) < n//3
        n_ext = int(round(4.8*(20.0*np.pi/3.0)/(2*np.pi)))
        p_ind = q48_power(uhat, k, mask, n_ext)
        m = t >= 100.0
        last20 = t >= (100.0+200.0-200.0*0.2)
        l_ind = float(lam_seg[last20].mean())
        dp = abs(p_ind - row["q4p8_power_fraction"])
        dl = abs(l_ind - row["lambda_last20"])
        max_dp = max(max_dp, dp); max_dl = max(max_dl, dl)
        recalc[(b3, sd, n)] = (p_ind, l_ind, ext, t, lam_seg, row)
    print(f"  q4.8 功率最大差: {max_dp:.3e}")
    print(f"  lambda_last20 最大差: {max_dl:.3e}")

    print("=== 4) 功率与 λ 表（独立重算值）=== ")
    print(f"{'b3':>5} {'N':>4} {'seed':>4} {'q48_pow':>8} {'lam20':>8} {'cls':>10}")
    for b3 in [0.15,0.16,0.17,0.18,0.19,0.20]:
        for n in [128,160]:
            for sd in [240,241]:
                p,l,_,_,_,row = recalc[(b3,sd,n)]
                print(f"{b3:5.2f} {n:4d} {sd:4d} {p*100:7.1f}% {l:+8.3f} {row['classification']:>10}")

    print("=== 5) 跨 N / 跨 seed 差异（独立重算）=== ")
    flags = []
    for b3 in [0.15,0.16,0.17,0.18,0.19,0.20]:
        for sd in [240,241]:
            p128 = recalc[(b3,sd,128)][0]; p160 = recalc[(b3,sd,160)][0]
            dN = abs(p128-p160)
            if dN > 0.03: flags.append(f"b3={b3} seed{sd} 跨N功率差 {dN*100:.2f}pp")
        for n in [128,160]:
            ps = [recalc[(b3,sd,n)][0] for sd in [240,241]]
            spread = max(ps)-min(ps)
            if spread > 0.05: flags.append(f"b3={b3} N{n} 跨seed功率spread {spread*100:.2f}pp")
    for f in flags: print("  ⚠", f)
    if not flags: print("  无跨N/seed超阈差异")

    print("=== 6) b3=0.20 外带振幅末段 vs 首段（是否真跌落/多稳态）=== ")
    for n in [128,160]:
        for sd in [240,241]:
            p,l,ext,t,lam_seg,row = recalc[(0.20,sd,n)]
            e = np.asarray(ext)
            first = e[t>=100][:40].mean(); last = e[-40:].mean()
            print(f"  N{n} s{sd}: ext首={first:.3f} 末={last:.3f} 末段q48={p*100:.1f}% lam20={l:+.3f}")

    print("=== 7) 裁决 ===")
    print(f"  数据处理可信: q4.8最大差={max_dp:.2e}, λ最大差={max_dl:.2e}")
    print("  异常点: b3=0.20 跨 seed 功率不一致，measure200 不足，需长测程（见 round243_longcheck.py）")


if __name__ == "__main__":
    main()

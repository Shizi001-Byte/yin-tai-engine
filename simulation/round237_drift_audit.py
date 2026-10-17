#!/usr/bin/env python3
# simulation/round237_drift_audit.py
# 决策层 #237 独立复算脚本：worker 的 estimate_drift 用包裹相位、未做
# 解缠绕，高 q 模 qvt>pi 时速度估计被污染（std 0.6-0.9、beta3+ 符号错）。
# 本脚本从全量饱和终态(npz)出发，做两件正确的事：
#   (A) 逐显著模取相位时间序列 -> np.unwrap -> 线性拟合 phi=-q v t，得 v_q；
#       刚性行波 u(x,t)=U(x-vt) 要求所有显著模 v_q 一致。
#   (B) 物理空间相邻时刻互相关，直接测图案峰值位移，独立给 v。
# 不重写已验证 RHS/积分器，复用 round231。

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from round231_P1_0 import make_grid, make_linop, rk4_state  # noqa: E402

N = 128
L = 20.0 * np.pi / 3.0
DT = 0.002
BETA4 = 1.0
BETA3 = 0.3
T_TOTAL = 4.0
SAMPLE_DT = 0.05          # 最大 q*v*SAMPLE_DT：5.4*1.5*0.05~0.41<pi，可 unwrap
POWER_FRAC = 0.005        # 功率占比 >0.5% 的显著模

d = np.load(HERE / "out" / "round237_P1_0.npz")
k, mask = make_grid(N, L)
q = k.copy()
x = np.arange(N) * L / N

steps_per_sample = int(round(SAMPLE_DT / DT))
n_samples = int(round(T_TOTAL / SAMPLE_DT))
times = np.arange(n_samples + 1) * SAMPLE_DT


def measure(tag, beta3):
    uhat = d[tag + "_uhat"].copy()
    linop = make_linop(k, beta3, BETA4)
    total = float(np.sum(np.abs(uhat) ** 2))
    sig = [j for j in range(1, N) if mask[j] and abs(uhat[j]) ** 2 / total > POWER_FRAC]

    # 记录显著模复系数时间序列 + 物理空间场
    rec = np.zeros((n_samples + 1, len(sig)), dtype=complex)
    rec[0] = uhat[sig]
    fields = [np.fft.ifft(uhat)]
    u = uhat
    for s in range(1, n_samples + 1):
        for _ in range(steps_per_sample):
            u = rk4_state(u, DT, linop, mask)
        rec[s] = u[sig]
        fields.append(np.fft.ifft(u))

    # (A) 逐模 unwrap + 线性拟合
    v_modes = {}
    for idx, j in enumerate(sig):
        phi = np.unwrap(np.angle(rec[:, idx]))
        # phi = phi0 - q_j v t  -> 斜率 m=-q_j v
        m, phi0 = np.polyfit(times, phi, 1)
        v_j = -m / q[j]
        # 拟合残差
        resid = np.std(phi - (m * times + phi0))
        v_modes[int(q[j] * 100) / 100] = (float(v_j), float(resid))

    # 用功率加权汇总各模 v_q（只取相位拟合残差小的刚性模）
    good = [(jj, vv) for jj, (vv, rs) in zip(sig, v_modes.values()) if rs < 0.1]
    vvals = np.array([vv for _, vv in good])
    v_mean = float(np.mean(vvals)) if len(vvals) else float("nan")
    v_std = float(np.std(vvals)) if len(vvals) else float("nan")

    # (B) 物理空间互相关测位移（取前 1/4 时段，避免位移超过半个周期）
    f0 = np.real(fields[0])
    f0 = f0 - f0.mean()
    disp = []
    for s in range(1, min(len(fields), n_samples // 2 + 1)):
        fs = np.real(fields[s]) - np.real(fields[s]).mean()
        cc = np.fft.ifft(np.fft.fft(f0) * np.conj(np.fft.fft(fs)))
        shift = int(np.argmax(np.real(cc)))
        if shift > N // 2:
            shift -= N
        disp.append(shift * (L / N) / times[s])
    v_xcorr = float(np.median(disp))

    return {
        "tag": tag,
        "n_significant_modes": len(sig),
        "per_mode_v": {f"q={qq}": f"{vv:+.4f}(resid {rs:.3f})" for qq, (vv, rs) in v_modes.items()},
        "rigid_drift_v_spectral_mean": v_mean,
        "rigid_drift_v_spectral_std": v_std,
        "drift_v_xcorr": v_xcorr,
        "linear_v_at_dominant_q1p5": -beta3 * 1.5 ** 2 / 6.0,
    }


for tag, b3 in [("b3p_b4_1", +BETA3), ("b3m_b4_1", -BETA3)]:
    r = measure(tag, b3)
    print("=" * 70)
    print(tag, " beta3=", b3)
    print(f"  显著模数={r['n_significant_modes']}")
    print(f"  逐模行波速度 v_q (谱法): {r['per_mode_v']}")
    print(f"  刚性行波速度 谱法均值={r['rigid_drift_v_spectral_mean']:+.4f} "
          f"±{r['rigid_drift_v_spectral_std']:.4f}")
    print(f"  行波速度 物理空间互相关 ={r['drift_v_xcorr']:+.4f}")
    print(f"  主导模 q=1.5 线性预测   ={r['linear_v_at_dominant_q1p5']:+.4f}")

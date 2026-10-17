#!/usr/bin/env python3
# simulation/round231_P1_0.py
# 依赖：numpy；绘图可选 matplotlib。
# 默认：python3 simulation/round231_P1_0.py
#
# 目标：在 round213 已验证 RHS/Jacobian 上做长测程 Benettin 最大 Lyapunov，
#       区分 β4=1 外带 q=4.8 是稳定锁模，还是持续时空混沌。

import argparse
import json
import os
import numpy as np

# ------------------------- round213 已验证常量/函数，原样复用 -------------------------
DELTA = 3.0
S = 2.0
I = 2.0
D = 1.0  # beta2/2

# 使 CW 解 A0=sqrt(I) 为实数的泵浦复相量；|PUMP| == S。
PUMP = np.sqrt(I) * (1.0 + 1j * (DELTA - I))


def cw_residual(intensity):
    """均匀稳态关系 S^2 = I[1+(Delta-I)^2]。"""
    return intensity * (1.0 + (DELTA - intensity) ** 2) - S * S


def tangent_matrix(q, beta3, beta4):
    """q>0 边带对 (u_q, u^*_{-q}) 的线性化矩阵。"""
    q = float(q)
    gamma3 = beta3 * q**3 / 6.0
    x4 = 2.0 * I - DELTA - D * q**2 + beta4 * q**4 / 24.0

    return np.array(
        [
            [-1.0 + 1j * (x4 + gamma3), 1j * I],
            [-1j * I, -1.0 - 1j * (x4 - gamma3)],
        ],
        dtype=complex,
    )


def analytic_eigenvalues(q, beta3, beta4):
    """round213 解析公式：lambda = -1 + i gamma3 ± sqrt(I^2-X4^2)。"""
    q = float(q)
    gamma3 = beta3 * q**3 / 6.0
    x4 = 2.0 * I - DELTA - D * q**2 + beta4 * q**4 / 24.0
    root = np.sqrt(I * I - x4 * x4 + 0j)
    return np.array(
        [
            -1.0 + 1j * gamma3 + root,
            -1.0 + 1j * gamma3 - root,
        ],
        dtype=complex,
    )


# ------------------------- 伪谱 RHS / RK4 / 切向量 -------------------------
def make_grid(N, L):
    n_idx = np.fft.fftfreq(N) * N
    k = 2.0 * np.pi * n_idx / L
    nyquist = np.pi * N / L
    dealias = np.abs(k) < (2.0 / 3.0) * nyquist
    return k, dealias


def make_linop(k, beta3, beta4):
    # 与 tangent_matrix 严格对应：
    # -1-iΔ - iD k² + iβ3 k³/6 + iβ4 k⁴/24
    return (
        -1.0
        - 1j * DELTA
        - 1j * D * k**2
        + 1j * beta3 * k**3 / 6.0
        + 1j * beta4 * k**4 / 24.0
    )


def state_rhs(uhat, linop, mask):
    u = np.fft.ifft(uhat)
    pump_kerr = PUMP + 1j * np.abs(u) ** 2 * u
    out = np.fft.fft(pump_kerr) + linop * uhat
    return out * mask


def tangent_rhs(uhat, vhat, linop, mask):
    u = np.fft.ifft(uhat)
    v = np.fft.ifft(vhat)
    # D[i|u|^2 u] = i(2|u|^2 v + u^2 v*)
    lin = 1j * (2.0 * np.abs(u) ** 2 * v + (u**2) * np.conj(v))
    out = np.fft.fft(lin) + linop * vhat
    return out * mask


def spec_norm(xhat, N):
    """FFT 未归一化时的物理域 RMS 范数。"""
    return np.linalg.norm(xhat) / N


def rk4_state(uhat, dt, linop, mask):
    a1 = state_rhs(uhat, linop, mask)
    a2 = state_rhs(uhat + 0.5 * dt * a1, linop, mask)
    a3 = state_rhs(uhat + 0.5 * dt * a2, linop, mask)
    a4 = state_rhs(uhat + dt * a3, linop, mask)
    return uhat + dt * (a1 + 2.0 * a2 + 2.0 * a3 + a4) / 6.0


def rk4_tangent(uhat, vhat, dt, linop, mask):
    ku1 = state_rhs(uhat, linop, mask)
    kv1 = tangent_rhs(uhat, vhat, linop, mask)

    u2 = uhat + 0.5 * dt * ku1
    v2 = vhat + 0.5 * dt * kv1
    ku2 = state_rhs(u2, linop, mask)
    kv2 = tangent_rhs(u2, v2, linop, mask)

    u3 = uhat + 0.5 * dt * ku2
    v3 = vhat + 0.5 * dt * kv2
    ku3 = state_rhs(u3, linop, mask)
    kv3 = tangent_rhs(u3, v3, linop, mask)

    u4 = uhat + dt * ku3
    v4 = vhat + dt * kv3
    ku4 = state_rhs(u4, linop, mask)
    kv4 = tangent_rhs(u4, v4, linop, mask)

    unext = uhat + dt * (ku1 + 2.0 * ku2 + 2.0 * ku3 + ku4) / 6.0
    vnext = vhat + dt * (kv1 + 2.0 * kv2 + 2.0 * kv3 + kv4) / 6.0
    return unext, vnext


# ------------------------- 新代码自洽性 assert -------------------------
def self_test(N, L, dt, k, mask, n_ext):
    assert abs(cw_residual(I)) < 1e-12
    assert abs(abs(PUMP) - S) < 1e-12

    # round213 解析事实：q=4.8 在 β4=1 时增长率先约 0.99846，β4=0 时实部为 -1。
    ana_ext = analytic_eigenvalues(4.8, 0.0, 1.0)
    assert abs(ana_ext.real.max() - 0.99846) < 2e-3
    ana0 = analytic_eigenvalues(4.8, 0.0, 0.0)
    assert abs(ana0.real.max() + 1.0) < 1e-10

    ucw = np.zeros(N, dtype=complex)
    ucw[0] = np.sqrt(I) * N

    # 多个色散参数下，数值切向 RHS 必须等于 round213 的 2x2 Jacobian。
    for beta3, beta4 in ((0.0, 0.0), (0.3, 1.0)):
        linop = make_linop(k, beta3, beta4)

        residual = state_rhs(ucw, linop, mask)
        assert spec_norm(residual, N) < 1e-11

        nq = 5
        q = k[nq]
        assert mask[nq]

        M_num = np.zeros((2, 2), dtype=complex)
        for col, (a, b) in enumerate(((1.0, 0.0), (0.0, 1.0))):
            vhat = np.zeros(N, dtype=complex)
            vhat[nq] = a
            vhat[-nq] = np.conj(b)
            dr = tangent_rhs(ucw, vhat, linop, mask)
            M_num[0, col] = dr[nq]
            M_num[1, col] = np.conj(dr[-nq])

        M_ana = tangent_matrix(q, beta3, beta4)
        assert np.allclose(M_num, M_ana, atol=1e-10, rtol=0.0), (M_num, M_ana)

    # q=4.8 不稳定本征向量的 1 时间单位 RK4 增长必须等于解析 exp(Re lambda*t)。
    if abs(k[n_ext] - 4.8) < 1e-10 and mask[n_ext]:
        linop = make_linop(k, 0.0, 1.0)
        M = tangent_matrix(4.8, 0.0, 1.0)
        vals, vecs = np.linalg.eig(M)
        j = int(np.argmax(vals.real))

        vhat = np.zeros(N, dtype=complex)
        a, b = vecs[:, j]
        vhat[n_ext] = a
        vhat[-n_ext] = np.conj(b)

        norm0 = spec_norm(vhat, N)
        nsteps = int(round(1.0 / dt))
        for _ in range(nsteps):
            ucw, vhat = rk4_tangent(ucw, vhat, dt, linop, mask)

        growth = spec_norm(vhat, N) / norm0
        expected = float(np.exp(vals[j].real * 1.0))
        assert abs(growth - expected) < 2e-5 * expected, (growth, expected)
    else:
        print("[self_test] 当前网格不含精确 q=4.8，跳过该模 RK4 增长检查")


# ------------------------- 初值与单工况 -------------------------
def initial_state(N, mask, n_ext, seed):
    rng = np.random.default_rng(seed)
    uhat = np.zeros(N, dtype=complex)
    uhat[0] = np.sqrt(I) * N

    idx = np.flatnonzero(mask)
    idx = idx[idx != 0]
    uhat[idx] = 1e-3 * N * (
        rng.standard_normal(idx.size) + 1j * rng.standard_normal(idx.size)
    )

    # 明确注入 q=4.8 余弦小扰动；所有工况共用同一初值。
    uhat[n_ext] += 2e-3 * N
    uhat[-n_ext] += 2e-3 * N
    return uhat * mask


def run_case(name, beta3, beta4, base, N, L, dt, k, mask, n_ext, args):
    linop = make_linop(k, beta3, beta4)
    max_z = float(dt * np.max(np.abs(linop[mask].imag)))
    assert max_z < 2.8, f"RK4 虚轴稳定性可能不足: dt*max|omega|={max_z}"

    uhat = base.copy()

    warmup_steps = int(round(args.warmup / dt))
    for _ in range(warmup_steps):
        uhat = rk4_state(uhat, dt, linop, mask)

    # 进入吸引子后重置切向量，避免 warmup 瞬态污染饱和 λ1。
    rng = np.random.default_rng(args.seed + 1009)
    idx = np.flatnonzero(mask)
    idx = idx[idx != 0]
    vhat = np.zeros(N, dtype=complex)
    vhat[idx] = rng.standard_normal(idx.size) + 1j * rng.standard_normal(idx.size)
    vhat *= mask
    vhat /= spec_norm(vhat, N)

    seg_steps = int(round(args.renorm / dt))
    n_seg = int(round(args.measure / args.renorm))
    assert abs(seg_steps * dt - args.renorm) < 1e-12
    assert abs(n_seg * args.renorm - args.measure) < 1e-12

    times, seg_lambda, cum_lambda, ext_amp = [], [], [], []
    log_total = 0.0

    for iseg in range(n_seg):
        for _ in range(seg_steps):
            uhat, vhat = rk4_tangent(uhat, vhat, dt, linop, mask)

        g = spec_norm(vhat, N)
        vhat /= g
        log_total += float(np.log(g))

        t = (iseg + 1) * args.renorm
        times.append(t)
        seg_lambda.append(np.log(g) / args.renorm)
        cum_lambda.append(log_total / t)
        ext_amp.append(
            np.hypot(np.abs(uhat[n_ext]), np.abs(uhat[-n_ext])) / N
        )

    times = np.asarray(times)
    seg_lambda = np.asarray(seg_lambda)
    cum_lambda = np.asarray(cum_lambda)
    ext_amp = np.asarray(ext_amp)

    # 终态诊断
    rhs_norm = spec_norm(state_rhs(uhat, linop, mask), N)
    field_rms = spec_norm(uhat, N)
    rel_residual = float(rhs_norm / max(field_rms, 1e-30))

    pk = np.abs(uhat) ** 2 / N**2
    total_power = float(np.sum(pk))

    dom_selector = np.where(np.arange(N) == 0, 0.0, pk)
    dom_idx = int(np.argmax(dom_selector))

    ext_power_fraction = float(
        (pk[n_ext] + pk[-n_ext]) / max(total_power, 1e-30)
    )
    ext_last = ext_amp[-20:]
    ext_cv = float(
        np.std(ext_last, ddof=1) / max(np.abs(np.mean(ext_last)), 1e-30)
    )

    half = n_seg // 2
    second = seg_lambda[half:]
    lambda_second = float(np.mean(second))
    stderr = float(np.std(second, ddof=1) / np.sqrt(len(second)))
    positive = lambda_second > 0.0 and lambda_second - 2.0 * stderr > 0.0
    negative = lambda_second + 2.0 * stderr < 0.0

    if rel_residual < 1e-4 and not positive:
        classification = "stationary_locked"
    elif rel_residual < 1e-4 and positive:
        classification = "inconsistent_stationary_with_positive_lambda"
    elif positive:
        classification = "sustained_chaotic"
    elif negative:
        classification = "contracting_nonchaotic"
    else:
        classification = "marginal_or_undetermined"

    return {
        "name": name,
        "beta3": float(beta3),
        "beta4": float(beta4),
        "lambda_full": float(cum_lambda[-1]),
        "lambda_second_half": lambda_second,
        "lambda_second_half_stderr": stderr,
        "lambda_last20": float(np.mean(seg_lambda[-20:])),
        "relative_final_residual": rel_residual,
        "dominant_q": float(abs(k[dom_idx])),
        "mean_power": total_power,
        "q4p8_power_fraction": ext_power_fraction,
        "q4p8_amp_cv_last20": ext_cv,
        "classification": classification,
        "max_rk4_imag_z": max_z,
        "t": times,
        "lambda_segment": seg_lambda,
        "lambda_cumulative": cum_lambda,
        "external_amp": ext_amp,
        "uhat": uhat,
    }


def plot_convergence(results, png_path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot skipped] {exc}")
        return

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    for r in results:
        axes[0].plot(
            r["t"], r["lambda_segment"], alpha=0.25,
            label=f"{r['name']} segment"
        )
        window = 10
        if len(r["t"]) >= window:
            run = np.convolve(
                r["lambda_segment"],
                np.ones(window) / window,
                mode="valid",
            )
            axes[0].plot(
                r["t"][window - 1:], run, lw=2.0,
                label=f"{r['name']} 10-window mean"
            )
        axes[1].plot(
            r["t"], r["lambda_cumulative"], lw=2.0, label=r["name"]
        )

    axes[0].axhline(0.0, color="k", lw=0.8)
    axes[1].axhline(0.0, color="k", lw=0.8)
    axes[0].set_ylabel(r"segment $\lambda_1$")
    axes[1].set_ylabel(r"cumulative $\lambda_1$")
    axes[1].set_xlabel("measurement time")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="b0,b4")
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--L", type=float, default=20.0 * np.pi / 3.0)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--warmup", type=float, default=50.0)
    parser.add_argument("--measure", type=float, default=100.0)
    parser.add_argument("--renorm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--outdir", default="simulation/out")
    args = parser.parse_args()

    case_map = {
        "b0": (0.0, 0.0),
        "b4": (0.0, 1.0),
        "b3p": (0.3, 0.0),
        "b3m": (-0.3, 0.0),
    }

    N, L, dt = args.N, args.L, args.dt
    k, mask = make_grid(N, L)

    n_ext = int(round(4.8 * L / (2.0 * np.pi)))
    assert 0 < n_ext < N

    self_test(N, L, dt, k, mask, n_ext)
    base = initial_state(N, mask, n_ext, args.seed)

    results = []
    for name in (x.strip() for x in args.cases.split(",")):
        beta3, beta4 = case_map[name]
        print(f"=== running {name}: beta3={beta3}, beta4={beta4} ===", flush=True)
        results.append(
            run_case(
                name, beta3, beta4, base, N, L, dt, k, mask, n_ext, args
            )
        )

    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.join(args.outdir, "round231_P1_0")

    arrays = {}
    array_keys = {
        "t", "lambda_segment", "lambda_cumulative", "external_amp", "uhat"
    }
    for r in results:
        for key in array_keys:
            arrays[f"{r['name']}_{key}"] = r[key]
    np.savez_compressed(stem + ".npz", **arrays)

    summary = []
    for r in results:
        summary.append({key: value for key, value in r.items() if key not in array_keys})

    with open(stem + ".json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    plot_convergence(results, stem + ".png")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

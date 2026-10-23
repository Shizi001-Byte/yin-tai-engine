#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# round283_newton.py —— 定常行波（腔孤子）解的直接求解与延拓
# 定常方程：F(u)=mask*(FFT(PUMP+i|u|^2 u)+linop(b3)·uhat)，行波ansatz u(x,t)=U(x-c t)  =>  F(U)+c*(i k U)=0
# 未知量：活跃模的实/虚部（含DC）+ 漂移速度 c；Gauss-Newton + 回溯线搜索；逐列缩放差分步长。
# 延拓：b3 逐步走，用上一点解作初值；Newton 失败或 sv_min 塌陷 ==> 解支终点（折叠）。
# 用法：python3 round283_newton.py --N 128 --src out/x.npz --key k --b3s 0.18,0.19 --tag t
import argparse, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from round231_P1_0 import make_grid, make_linop, state_rhs
from round246_basin import BETA4
L = 20.0 * np.pi / 3.0
Q_OUT = 4.8
ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, required=True)
ap.add_argument("--src", required=True); ap.add_argument("--key", required=True)
ap.add_argument("--b3s", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--maxit", type=int, default=40); ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
N = a.N
k, mask = make_grid(N, L)
n_ext = int(round(Q_OUT * L / (2 * np.pi)))
idx = np.flatnonzero(mask)
dof = 2 * idx.size
src = dict(np.load(a.src if Path(a.src).is_absolute() else str(HERE / a.src)))
u0 = src[a.key]

def pack(u): return np.concatenate([u[idx].real, u[idx].imag])
def unpack(x):
    u = np.zeros(N, dtype=complex); u[idx] = x[:idx.size] + 1j * x[idx.size:]; return u * mask
def resid(x, b3, c):
    u = unpack(x)
    return state_rhs(u, make_linop(k, b3, BETA4), mask) + c * (1j * k * u) * mask
def resid_v(x, b3, c):
    r = resid(x, b3, c); return np.concatenate([r[idx].real, r[idx].imag])
def ext_of(u): return float(np.hypot(abs(u[n_ext]), abs(u[-n_ext])) / N)

def newton(x, b3, maxit=40, c=0.0):
    x = np.concatenate([x, [c]])
    def ev(z): return resid_v(z[:-1], b3, z[-1])
    def jac(z, r):
        J = np.zeros((dof, dof + 1))
        for j in range(dof + 1):
            e = 1e-7 * max(1.0, abs(z[j]))
            zp = z.copy(); zp[j] += e
            J[:, j] = (ev(zp) - r) / e
        return J
    r = ev(x); n0 = np.linalg.norm(r); J = jac(x, r)
    for it in range(maxit):
        rn = np.linalg.norm(r)
        if rn < 1e-11: break
        dx = np.linalg.lstsq(J, -r, rcond=None)[0]
        s = 1.0
        while s > 1e-4:
            xnew = x + s * dx; rnew = ev(xnew)
            if np.linalg.norm(rnew) < rn: x, r = xnew, rnew; break
            s *= 0.5
        else:
            break
        J = jac(x, r)
    sv = np.linalg.svd(J, compute_uv=False)
    lam = np.linalg.eigvals(J[:, :-1]); imax = int(np.argmax(lam.real))
    return x[:-1], float(np.linalg.norm(r)), float(n0), float(sv[-1]), float(lam.real[imax]), float(lam.imag[imax]), float(x[-1])

rows, states = [], {}
x = pack(u0); c_prev = 0.0; c_init = None
b3s = a.b3s.split(",")
for s0 in b3s[:1]:
    best = (1e9, 0.0)
    for cc in np.arange(-1.0, 1.0, 0.02):
        rr = resid_v(x, float(s0), cc)
        if np.linalg.norm(rr) < best[0]: best = (np.linalg.norm(rr), cc)
    c_init = best[1]
    print(f"  c_init={c_init:+.3f}", flush=True)
for s in b3s:
    b3 = float(s)
    x, rn, r0, svmin, lre, lim, c = newton(x, b3, a.maxit if not a.smoke else 8, c_init if c_init is not None else c_prev)
    c_prev = c; c_init = None
    u = unpack(x); e = ext_of(u); conv = rn < 1e-8
    rows.append({"b3": b3, "c": c, "ext": e, "res_after": rn, "res_before": r0, "sv_min": svmin,
                 "jac_max_real": lre, "jac_max_imag": lim, "converged": bool(conv)})
    states[f"{a.tag}_b{b3:.4f}"] = u
    print(f"  b3={b3:.4f} ext={e:.4f} res={rn:.2e}(from {r0:.1e}) svmin={svmin:.2e} "
          f"Re(lam)={lre:+.2e} Im={lim:.2f} c={c:+.4f} {'OK' if conv else 'FAIL'}", flush=True)
    if not conv and len(rows) > 1:
        print("  >>> 解支终止（Newton 未收敛）", flush=True); break
json.dump({"rows": rows}, open(HERE / "out" / f"round283_{a.tag}.json", "w"), indent=2, ensure_ascii=False)
np.savez_compressed(HERE / "out" / f"round283_{a.tag}.npz", **states)
print("NEWTON DONE", flush=True)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# round287b_stability_evolve.py —— 稳定性独立仲裁：从精确定常行波解出发，加 1e-5 微扰，
# 走真 PDE（实验室系），只看 ext(t) 守不守得住。
# 用法：python3 round287b_stability_evolve.py --b3 0.20 --tag b020 [--T 200] [--pert 1e-5]
import sys, json, argparse
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from round231_P1_0 import make_grid, make_linop, state_rhs
from round246_basin import BETA4
ap = argparse.ArgumentParser(); ap.add_argument('--b3', type=float, required=True); ap.add_argument('--tag', required=True)
ap.add_argument('--T', type=float, default=200.0); ap.add_argument('--pert', type=float, default=1e-5)
a = ap.parse_args()
L = 20.0*np.pi/3.0; N = 128; dt = 0.002
k, mask = make_grid(N, L); n_ext = 16
u = dict(np.load(HERE/'out/round283_smokeN5.npz'))['smokeN5_b0.1800'].copy()
lin = make_linop(k, a.b3, BETA4)
rng = np.random.default_rng(287)
u = u + a.pert*(rng.standard_normal(N)+1j*rng.standard_normal(N))*mask
def ext(u): return float(np.hypot(abs(u[n_ext]), abs(u[-n_ext]))/N)
ts, es = [], []
for n in range(int(a.T/dt)):
    k1 = state_rhs(u, lin, mask); k2 = state_rhs(u+0.5*dt*k1, lin, mask)
    k3 = state_rhs(u+0.5*dt*k2, lin, mask); k4 = state_rhs(u+dt*k3, lin, mask)
    u = u + dt/6.0*(k1+2*k2+2*k3+k4)
    if n % 50 == 0: ts.append(n*dt); es.append(ext(u))
es = np.array(es); ts = np.array(ts)
json.dump({"b3": a.b3, "ts": ts.tolist(), "ext": es.tolist()}, open(HERE/f'out/round287b_{a.tag}.json','w'))
print(f"EVOLVE b3={a.b3} ext_start={es[0]:.6f} ext_end={es[-1]:.4f} ext_min={es.min():.4f} ext_max={es.max():.4f} last50_spread={es[-50:].max()-es[-50:].min():.4f}")

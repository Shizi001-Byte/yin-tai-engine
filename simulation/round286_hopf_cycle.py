#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# round286_hopf_cycle.py —— 尝试看 Hopf 之后是否有极限环（本版结论：不可用，见下）
# 失败原因：把“行波解”直接放进实验室系演化，ext 因平移不变而恒定；且共动系步进不稳定会发散。
# 保留作为踩坑记录；正确做法见 round287b_stability_evolve.py（实验室系 + 看 ext 守不守）。
import sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from round231_P1_0 import make_grid, make_linop, state_rhs
from round246_basin import BETA4
L = 20.0*np.pi/3.0; N = 128; dt = 0.002; T_TOTAL = 400.0
k, mask = make_grid(N, L); n_ext = 16
lin = make_linop(k, 0.2, BETA4)
u = dict(np.load(HERE/'out/round283_n128_A_up.npz'))['n128_A_up_b0.2000'].copy()
rng = np.random.default_rng(286)
u = u + 1e-4*(rng.standard_normal(N)+1j*rng.standard_normal(N))*mask
def ext(u): return float(np.hypot(abs(u[n_ext]), abs(u[-n_ext]))/N)
ts, es = [], []
for n in range(int(T_TOTAL/dt)):
    k1 = state_rhs(u, lin, mask); k2 = state_rhs(u+0.5*dt*k1, lin, mask)
    k3 = state_rhs(u+0.5*dt*k2, lin, mask); k4 = state_rhs(u+dt*k3, lin, mask)
    u = u + dt/6.0*(k1+2*k2+2*k3+k4)
    if n % 25 == 0: ts.append(n*dt); es.append(ext(u))
json.dump({"b3": 0.2, "ts": ts, "ext": es}, open(HERE/'out/round286_hopf_cycle.json','w'))
print("HOPF CYCLE (legacy, see caveat)")

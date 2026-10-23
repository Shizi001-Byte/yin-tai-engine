#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# round288_spectrum_audit.py —— 残差门控谱审计
# 规矩：任何“读回 npz 再算谱”必须先复算残差；c 的符号用残差最小的那个；残差>1e-8 的谱一律作废。
# 同时给出伪谱量 g(T)=ln||expm(J*T)v||/T，用来区分“渐近不稳定”与“非正规性带来的长时暂态衰减”。
import sys, json, argparse
from pathlib import Path
import numpy as np
from scipy.linalg import expm
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from round231_P1_0 import make_grid, make_linop, state_rhs
from round246_basin import BETA4
L = 20.0*np.pi/3.0; N = 128
k, mask = make_grid(N, L); idx = np.flatnonzero(mask); dof = 2*idx.size
def pk(u): return np.concatenate([u[idx].real, u[idx].imag])
def up(x):
    u = np.zeros(N, dtype=complex); u[idx] = x[:idx.size]+1j*x[idx.size:]; return u*mask
def audit(npz, jsonf, keyprefix):
    st = dict(np.load(HERE/npz)); rows = json.load(open(HERE/jsonf))['rows']; out = []
    for r in rows:
        b3 = r['b3']; key = f"{keyprefix}_b{b3:.4f}"
        if key not in st: continue
        x = pk(st[key])
        def rv(z, c):
            u = up(z); rr = state_rhs(u, make_linop(k, b3, BETA4), mask) + c*(1j*k*u)*mask
            return np.concatenate([rr[idx].real, rr[idx].imag])
        c = min([r['c'], -r['c']], key=lambda cc: np.linalg.norm(rv(x, cc)))
        rr = rv(x, c); res = float(np.linalg.norm(rr))
        if res > 1e-8:
            out.append({"b3": b3, "res": res, "verdict": "VOID"}); continue
        J = np.zeros((dof, dof))
        for j in range(dof):
            e = 1e-7*max(1.0, abs(x[j])); xp = x.copy(); xp[j] += e
            J[:, j] = (rv(xp, c)-rr)/e
        l = np.linalg.eigvals(J); o = np.argsort(-l.real)
        i = o[0] if abs(l.imag[o[0]]) > 1e-6 else o[1]
        v = np.random.default_rng(7).standard_normal(dof); v /= np.linalg.norm(v)
        g = float(np.log(np.linalg.norm(expm(J*120)@v))/120)
        out.append({"b3": b3, "c": float(c), "res": res, "re": float(l.real[i]), "im": float(abs(l.imag[i])), "g120": g})
    return out
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--npz', required=True); ap.add_argument('--json', required=True)
    ap.add_argument('--prefix', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); res = audit(a.npz, a.json, a.prefix)
    json.dump(res, open(HERE/f'out/{a.out}.json', 'w'), indent=2)
    print("b3      res      Re(lambda)   Im      g120")
    for r in res:
        if r.get('verdict') == 'VOID': print(f"{r['b3']:.4f} {r['res']:.1e} VOID"); continue
        print(f"{r['b3']:.4f} {r['res']:.1e} {r['re']:+.6f} {r['im']:.3f} {r['g120']:+.4f}")

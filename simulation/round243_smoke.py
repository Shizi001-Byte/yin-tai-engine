#!/usr/bin/env python3
# 决策层独立冒烟：验证 round243 worker 脚本的管道（import签名/run_case/summarize key/公共初值断言）
# 短测程，只为抓 bug，不用于物理结论。
import argparse, sys
import numpy as np
from round231_P1_0 import analytic_eigenvalues, make_grid, run_case, self_test
from round234_P1_0 import J_COMMON, assert_common_bases, common_initial_state
from round237_P1_0 import reflection_operator_self_test
from round240_P1_0 import BETA4, Q_OUT, summarize

L = 20.0*np.pi/3.0
b3 = 0.18
print("=== 1) 双分辨率公共初值 + assert_common_bases ===")
bases_by_n = {}
for n, dt in [(128,0.002),(160,0.001)]:
    k, mask = make_grid(n, L)
    n_ext = int(round(Q_OUT*L/(2*np.pi)))
    base = common_initial_state(n, mask, n_ext, 240)
    bases_by_n[n] = base
    print(f"  N={n} n_ext={n_ext} k[n_ext]={k[n_ext]:.4f}")
try:
    assert_common_bases(bases_by_n)
    print("  assert_common_bases OK（dict{N:base} 结构正确）")
except Exception as e:
    print("  assert_common_bases FAIL:", type(e).__name__, e)

print("=== 2) 单 case 冒烟 N=128 b3=0.18 seed240 短测程 ===")
n=128; dt=0.002
k, mask = make_grid(n, L); n_ext=int(round(Q_OUT*L/(2*np.pi)))
self_test(n,L,dt,k,mask,n_ext)
args=argparse.Namespace(warmup=20.0,measure=30.0,renorm=1.0,seed=240)
res=run_case("smoke", b3, BETA4, bases_by_n[128], n, L, dt, k, mask, n_ext, args)
s=summarize(res, n_ext, n)
print("  run_case OK keys:", sorted(res.keys()))
print("  summarize keys:", sorted(s.keys()))
need_arr=["t","lambda_segment","lambda_cumulative","external_amp","uhat"]
print("  数组key齐全:", all(x in res for x in need_arr))
print("  summary 是否含 N:", "N" in s, " <- worker脚本 line281 用 row['N']")
print("  lambda_last20=%.4f q4.8=%.4f cls=%s"%(s["lambda_last20"],s["q4p8_power_fraction"],s["classification"]))

print("=== 3) N=160 dt=0.001 RK4 稳定性 + 跑一个短case ===")
n=160; dt=0.001
k, mask = make_grid(n, L); n_ext=int(round(Q_OUT*L/(2*np.pi)))
try:
    self_test(n,L,dt,k,mask,n_ext)
    args=argparse.Namespace(warmup=20.0,measure=30.0,renorm=1.0,seed=240)
    res=run_case("smoke160", b3, BETA4, bases_by_n[160], n, L, dt, k, mask, n_ext, args)
    print("  N160/dt0.001 跑通 lambda_last20=%.4f q4.8=%.4f"%(res["lambda_last20"],res["q4p8_power_fraction"]))
except Exception as e:
    print("  N160 FAIL:", type(e).__name__, e)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""#246 深度轮裁决：三吸引子共存 + 非对称一阶迟滞 —— 类型b 确定性长测程（独立编写、独立执行、只从物理核重算）

实验A 盆地扫描：beta3 in [0.19,0.20,0.205,0.21,0.215,0.22] x 4 seed, measure800, N128
       终态分高/中/低功率锁模态，统计各吸引子盆地占比（独立 common 初值，不沿用）。
实验B 绝热滞后：seed240/242，UP 0.18->0.22、DOWN 0.22->0.18（每步 measure200, 沿用上一 beta3 终态）。
判据：
  - 一阶多稳态共存：某 beta3 同时存在两个/三个稳定吸引子（独立初值落不同终态）。
  - 迟滞：同 beta3 下 UP 与 DOWN 路径 ext 差>0.05（gap），且 UP、DOWN 各自有跳变。
  - 稳定终态：后 1/4 时间 ext 无漂移（尾段 drift<0.03）。
不预先假设两态；若出现中态（ext~0.3）也记录。
"""
import sys, json, os, argparse
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from round240_P1_0 import summarize, classify, make_grid, run_case
from round234_P1_0 import common_initial_state

BETA4=1.0; Q_OUT=4.8; L=20*np.pi/3; N=128; DT=0.002
BETA3_BASIN=[0.19,0.20,0.205,0.21,0.215,0.22]
SEEDS=[240,241,237,242]


def run_terminal(name,b3,seed,measure):
    Nx=N; dt=DT
    grid=make_grid(Nx,L); mask=grid[3]
    u0=common_initial_state(Nx,mask,n_ext=16,seed=seed)
    r=run_case(name,beta3=b3,beta4=BETA4,base=u0,N=Nx,L=L,dt=dt,k=1,
               mask=mask,n_ext=16,args=argparse.Namespace(warmup=300,measure=measure,renorm=100,seed=seed))
    ext=r.get('external_amp',[]); ext_arr=np.array(ext)
    n=max(1,len(ext_arr)//4)
    seg_ext=ext_arr.reshape(4,-1).mean(axis=1)
    tail_mean=float(ext_arr[-n:]); tail_drift=float(ext_arr[-n:].max()-ext_arr[-n:].min())
    m=seg_ext.mean()
    if m>0.38: cls='HIGH'
    elif m>0.28: cls='MID'
    else: cls='LOW'
    return dict(beta3=b3,seed=seed,ext=float(ext_arr[-1]),seg_ext=[round(float(x),4) for x in seg_ext],
                tail_mean=round(tail_mean,4),tail_drift=round(tail_drift,4),
                cls=cls,q48=round(float(r.get('q4p8_power_fraction',0)),4),
                lam=round(float(np.mean(r.get('lambda_last20',[np.nan]))),6),N=Nx,measure=measure)


def run_hyst(seed):
    grid=make_grid(N,L); mask=grid[3]
    ups=[]; base=common_initial_state(N,mask,n_ext=16,seed=seed)
    b3s_up=[0.18,0.185,0.19,0.195,0.20,0.205,0.21,0.215,0.22]
    for i,b3 in enumerate(b3s_up):
        wm=0 if i>0 else 300
        r=run_case(f'up{seed}_{b3}',beta3=b3,beta4=BETA4,base=base,N=N,L=L,dt=DT,k=1,mask=mask,n_ext=16,
                   args=argparse.Namespace(warmup=wm,measure=200,renorm=100,seed=seed))
        ext=np.array(r.get('external_amp',[])); base=r['uhat']
        ups.append(dict(beta3=b3,ext=round(float(ext[-1]),4),cls=classify(ext)))
    downs=[]
    b3s_down=[0.22,0.215,0.21,0.205,0.20,0.195,0.19,0.185,0.18]
    for i,b3 in enumerate(b3s_down):
        wm=0 if i>0 else 300
        r=run_case(f'down{seed}_{b3}',beta3=b3,beta4=BETA4,base=base,N=N,L=L,dt=DT,k=1,mask=mask,n_ext=16,
                   args=argparse.Namespace(warmup=wm,measure=200,renorm=100,seed=seed))
        ext=np.array(r.get('external_amp',[])); base=r['uhat']
        downs.append(dict(beta3=b3,ext=round(float(ext[-1]),4),cls=classify(ext)))
    return ups,downs


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args()
    if a.smoke:
        print(run_terminal('smoke',0.20,240,100)); print('SMOKE OK'); return
    res=[]
    for b3 in BETA3_BASIN:
        for s in SEEDS:
            rr=run_terminal(f'b{b3}_s{s}',b3,s,800); res.append(rr)
            print(f'[basin] b3={b3} s={s} ext={rr["ext"]} {rr["cls"]} drift={rr["tail_drift"]} q48={rr["q48"]}',flush=True)
    hyst={}
    for s in [240,242]:
        u,d=run_hyst(s); hyst[f'seed{s}']=dict(up=u,down=d)
        print(f'[hyst] seed{s} done',flush=True)
    out=dict(round246=True,experiment='basin+hyst',beta4=BETA4,N=N,measure_basin=800,basin=res,hyst=hyst)
    p=os.path.join(HERE,'out','round246_basin_hyst.json')
    json.dump(out,open(p,'w'),indent=2,ensure_ascii=False); print('SAVED',p)
    # 简报
    from collections import Counter
    for b3 in BETA3_BASIN:
        c=Counter(x['cls'] for x in res if x['beta3']==b3)
        print(f'beta3={b3}: HIGH={c.get("HIGH",0)} MID={c.get("MID",0)} LOW={c.get("LOW",0)}')


if __name__=='__main__':
    main()

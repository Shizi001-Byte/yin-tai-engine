#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""#246 续跑（增量落盘）：从 round246_basin_partial.json 恢复，补齐盆地(0.215/0.22)+完整滞后，并每 case 落盘。
若 partial 缺失则全量重跑。每个 case 跑完即写 round246_resume.json，可断点续跑。"""
import sys, json, os, argparse
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from round240_P1_0 import classify, make_grid, run_case
from round234_P1_0 import common_initial_state
BETA4=1.0; Q_OUT=4.8; L=20*np.pi/3; N=128; DT=0.002
SEEDS=[240,241,237,242]
OUT=os.path.join(HERE,'out','round246_resume.json')
PARTIAL=os.path.join(HERE,'out','round246_basin_partial.json')


def run_terminal(name,b3,seed,measure):
    grid=make_grid(N,L); mask=grid[3]
    u0=common_initial_state(N,mask,n_ext=16,seed=seed)
    r=run_case(name,beta3=b3,beta4=BETA4,base=u0,N=N,L=L,dt=DT,k=1,mask=mask,n_ext=16,
               args=argparse.Namespace(warmup=300,measure=measure,renorm=100,seed=seed))
    ext=np.array(r.get('external_amp',[])); n=max(1,len(ext)//4); seg=ext.reshape(4,-1).mean(axis=1); m=seg.mean()
    cls='HIGH' if m>0.38 else ('MID' if m>0.28 else 'LOW')
    return dict(beta3=b3,seed=seed,ext=round(float(ext[-1]),4),seg_ext=[round(float(x),4) for x in seg],
                tail_drift=round(float(ext[-n:].max()-ext[-n:].min()),4),cls=cls,
                q48=round(float(r.get('q4p8_power_fraction',0)),4),
                lam=round(float(np.mean(r.get('lambda_last20',[np.nan]))),6),N=N,measure=measure)


def run_hyst(seed,measure=200):
    grid=make_grid(N,L); mask=grid[3]; base=common_initial_state(N,mask,n_ext=16,seed=seed)
    ups=[]
    for i,b3 in enumerate([0.18,0.185,0.19,0.195,0.20,0.205,0.21,0.215,0.22]):
        r=run_case(f'up{seed}_{b3}',b3,BETA4 if False else BETA4,BETA4,base,N,L,DT,1,mask,16,
                   argparse.Namespace(warmup=0 if i else 300,measure=measure,renorm=100,seed=seed)) if False else None
        r=run_case(f'up{seed}_{b3}',beta3=b3,beta4=BETA4,base=base,N=N,L=L,dt=DT,k=1,mask=mask,n_ext=16,
                   args=argparse.Namespace(warmup=0 if i else 300,measure=measure,renorm=100,seed=seed))
        ext=np.array(r.get('external_amp',[])); base=r['uhat']; ups.append(dict(beta3=b3,ext=round(float(ext[-1]),4),cls=classify(ext)))
    downs=[]
    for i,b3 in enumerate([0.22,0.215,0.21,0.205,0.20,0.195,0.19,0.185,0.18]):
        r=run_case(f'down{seed}_{b3}',beta3=b3,beta4=BETA4,base=base,N=N,L=L,dt=DT,k=1,mask=mask,n_ext=16,
                   args=argparse.Namespace(warmup=0 if i else 300,measure=measure,renorm=100,seed=seed))
        ext=np.array(r.get('external_amp',[])); base=r['uhat']; downs.append(dict(beta3=b3,ext=round(float(ext[-1]),4),cls=classify(ext)))
    return ups,downs


def save(state):
    json.dump(state,open(OUT,'w'),indent=2,ensure_ascii=False)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--smoke',action='store_true'); a=ap.parse_args()
    if a.smoke:
        print(run_terminal('smoke',0.20,240,100)); print('SMOKE OK'); return
    state=dict(round246=True,resume=True,basin=[],hyst={})
    done=set()
    if os.path.exists(PARTIAL):
        p=json.load(open(PARTIAL)); state['basin']=p.get('basin',[])
        for x in state['basin']: done.add((x['beta3'],x['seed']))
    for b3 in [0.19,0.20,0.205,0.21,0.215,0.22]:
        for s in SEEDS:
            if (b3,s) in done: continue
            rr=run_terminal(f'b{b3}_s{s}',b3,s,800); state['basin'].append(rr); save(state)
            print(f'[basin] b3={b3} s={s} ext={rr["ext"]} {rr["cls"]}',flush=True)
    for s in [240,242]:
        key=f'seed{s}'
        if key not in state['hyst']:
            u,d=run_hyst(s,200); state['hyst'][key]=dict(up=u,down=d); save(state)
            print(f'[hyst] seed{s} done',flush=True)
    print('ALL DONE')


if __name__=='__main__':
    main()

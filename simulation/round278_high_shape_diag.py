"""#278 轻量轮诊断（R276-1 第一步，只读 npz，不重跑演化）
问题：同一 beta3 下 4 条独立初值收敛出的 HIGH 态 ext 互不相同（0.428~0.535），
      λ 却都 ≈0。它们到底是"同一个吸引子被采样成不同读数"，还是"不同的锁定态"？
判据（主Agent设计）：
  A. 若只是空间平移/相位差别 → ext（外带模幅度，平移不变）应相同；实测不同，故排除纯平移。
  B. 计算每条终态的实空间波形：峰数 n_peak、峰值 max|u|、主模序号 m_star、能量 q48 占比。
     若同一 beta3 下各态 n_peak 或 m_star 不同 → 至少两个不同的锁定结构，不是同一不动点。
     若 n_peak、m_star 全同而仅幅度不同 → 同一族的连续/离散缩放，需要进一步测谱。
本脚本只做确定性读取与统计，独立于 round276 的生成路径。
"""
import json
import numpy as np

NPZ = 'out/round276_multiseed_coexistence.npz'
JSON = 'out/round276_multiseed_coexistence.json'
Q_OUT = 4.8
L = 20 * np.pi / 3
N = 128


def analyze(uhat):
    uhat = np.asarray(uhat).ravel()
    n_ext = int(round(Q_OUT * L / (2 * np.pi)))
    ext = float(np.hypot(abs(uhat[n_ext]), abs(uhat[-n_ext])) / N)
    pk = np.abs(uhat) ** 2 / N ** 2
    total_power = float(pk.sum())
    # 实空间波形：uhat 按 [0..N/2-1, -N/2..-1] 排布
    k = np.fft.fftfreq(N, d=1.0 / N).astype(int)
    u = np.fft.ifft(uhat)
    amp = np.abs(u) * N
    # 峰数：局部极大且 > 60% 最大值
    thr = 0.6 * amp.max()
    peaks = [i for i in range(N)
             if amp[i] >= amp[(i - 1) % N] and amp[i] > amp[(i + 1) % N] and amp[i] >= thr]
    # 主模：去掉直流后的最强模
    pkc = pk.copy()
    pkc[0] = 0.0
    m_star = int(np.argmax(pkc))
    m_star = m_star if m_star <= N // 2 else m_star - N
    # 谱宽：pk > 1% 主峰的模式数
    n_modes = int(np.sum(pkc > 0.01 * pkc.max()))
    return {
        'ext': ext,
        'total_power': total_power,
        'max_amp': float(amp.max()),
        'n_peak': len(peaks),
        'm_star': m_star,
        'n_modes': n_modes,
        'q48_frac': float((pk[n_ext] + pk[-n_ext]) / max(total_power, 1e-30)),
        'phase_q48': float(np.angle(uhat[n_ext])),
    }


def main():
    d = np.load(NPZ)
    out = {}
    for key in d.keys():
        out[key] = analyze(d[key])
    with open(JSON, encoding='utf-8') as f:
        rows = json.load(f)

    ref = {(r['beta3'], r['seed']): r for r in rows['rows']}
    print(f"{'key':34s} {'ext':>7s} {'json_ext':>8s} {'pow':>7s} {'max':>6s} {'npk':>3s} {'m*':>4s} {'nmod':>4s} {'q48%':>6s}")
    for key, v in out.items():
        parts = key.split('_')
        seed = int(parts[0][1:])
        b3 = float(parts[1][1:])
        j = ref.get((b3, seed))
        je = f"{j['ext_tail_mean']:.4f}" if j else "-"
        print(f"{key:34s} {v['ext']:7.4f} {je:>8s} {v['total_power']:7.4f} "
              f"{v['max_amp']:6.3f} {v['n_peak']:3d} {v['m_star']:4d} "
              f"{v['n_modes']:4d} {100*v['q48_frac']:5.1f}%")

    print("\n=== 同 beta3 内 HIGH 态结构对比 ===")
    for b3 in (0.17, 0.175, 0.18):
        keys = [k for k in out if k.split('_')[1] == f'b{b3:g}'.replace('b0.', 'b0.')]
        keys = [k for k in out if k.startswith(('s240', 's241', 's237', 's242'))
                and k.split('_')[1] == 'b%g' % b3]
        hi = [(k, out[k]) for k in keys if out[k]['ext'] > 0.38]
        if not hi:
            continue
        print(f"beta3={b3}:")
        for k, v in hi:
            print(f"   {k:32s} ext={v['ext']:.4f} n_peak={v['n_peak']} "
                  f"m_star={v['m_star']} n_modes={v['n_modes']} q48={100*v['q48_frac']:.1f}%")
        exts = sorted(v['ext'] for _, v in hi)
        print(f"   -> ext 极差 {exts[-1]-exts[0]:.4f}; n_peak 集合 {sorted(set(v['n_peak'] for _, v in hi))}; "
              f"m_star 集合 {sorted(set(v['m_star'] for _, v in hi))}")

    with open('out/round278_high_shape_diag.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\nwrote out/round278_high_shape_diag.json")


if __name__ == '__main__':
    main()

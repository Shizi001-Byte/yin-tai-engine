#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# round288_old_fixedc_wrong.py —— 踩坑留档：固定 c 的牛顿（错）
# 错在哪：定常行波方程 F(U) - c*d_x U = 0 只有 c 取真值才有解；把 c 钉死后方程无解，
# 牛顿只能收敛到残差 0.4~8 的“最小二乘妥协点”，在那上面算谱得到 −0.0025±4.87i，与演化结论相反。
# 正确做法：c 作为未知量一起解（round283_newton.py），或读回已有解后用“残差最小符号”复算。
print("DO NOT USE: fixed-c Newton has no solution; keep c as unknown.")

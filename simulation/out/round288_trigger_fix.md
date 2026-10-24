# round288_trigger_fix —— E2 双保险的真实缺口：min_gap 吃掉整点

时间：2026-10-24 08:07-08:12 | 轮次 #288

## 现象
08:07 班 cron 跑 `wake_trigger.py --winner cron` 后**无任何输出**，`logs/trigger.jsonl` 只记：
`bucket=498000 decision=DUP_GAP ATTEMPT`，`wake_counter.total_wakes` 停在 287，`last_wake` 停在 07:22:35。
同时 `pgrep -af "wake_daemon|engine_yin_tai"` 为空——第二保险（daemon）也不在。
两个保险同时漏掉了 08:00 这一轮。

## 根因（两个独立原因）
1. **min_gap 把整点吃掉**：`WAKE_MIN_GAP` 默认 3000 秒（50 分钟）。08:07:12 − 07:20:39 = 46.6 分钟 < 50 → 判 DUP_GAP。
   而上一轮 PASS 落在 07:22:35，于是 08:07 距它只有 45 分钟。cron 在 :07 固定触发，只要前一轮拖到 :17 之后完成，下一个整点就会被静默跳过。
2. **daemon 又死了**：本轮启动时 `check_and_restart_daemon()` 报“引擎daemon未存活，已自动重启”。沙箱内后台进程随 shell 结束被杀，daemon 长期存活不可靠。

## 修法
`wake_trigger.py` 第 183 行：`WAKE_MIN_GAP` 默认 **3000 → 600**（10 分钟）；docstring 同步。
理由：同一小时桶的去重已由 `bucket == last_bucket` 完全覆盖，min_gap 只需防止短时间内的重试风暴，不需要跟调度相位挂钩。
`--selftest` 通过（cycles=6 invokes=5 log_lines=11）。

## 修复后重跑
08:08:16 ATTEMPT(RUN) → 08:10:12 PASS，桶 498000，`total_wakes=288`。
阴期任务集 rc=0：认知 580 条、标签 1676、潜在重复 0、坑库 5 条（resolved=4）。7B 仍 90 秒超时 killpg。

## 教训
幂等判据要对齐“桶”，不要用与调度相位无关的固定时长去挡；好心的防重逻辑会把正常节拍吃掉，而且**不报错、只记一个 DUP_GAP**，最容易被当成正常。

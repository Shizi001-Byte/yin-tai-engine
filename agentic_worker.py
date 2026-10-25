#!/usr/bin/env python3
"""
agentic_worker.py —— 自主执行 worker（前端轻、后端重）

把一件完整的事交给它：自己写代码 → 自己跑 → 看结果 → 迭代 → 交最终报告。
主 Agent 只拿最终报告扫一眼，不亲手推车。

用法：
  python3 agentic_worker.py --round 296 --worker 0 --task "任务描述" [--context "背景/路径"] [--max-iters 6]
输出：
  yin_tai_data/self_wake_logs/agentic_{round}_{worker}.md   最终报告
  simulation/agent_ws/w{worker}/                           隔离工作区（代码/产物）
  退出码：0=拿到 final；1=超时/失败
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
import ark_call  # noqa: E402

SIM_DIR = os.path.join(BASE_DIR, "simulation")
WS_ROOT = os.path.join(SIM_DIR, "agent_ws")
LOG_DIR = os.path.join(BASE_DIR, "yin_tai_data", "self_wake_logs")

EXEC_TIMEOUT = 120          # 每步代码最多跑 120 秒
OUT_CAP = 4000              # 喂回给模型的输出截断

SYSTEM_PROMPT = """你是寂雷引擎里的自主执行 worker。你要自己写代码、自己运行、看结果、修错、迭代，直到把任务完成。
你有手：你写的 python 代码会被真实执行，stdout/stderr 会在下一轮返回给你。

每轮严格按这个格式输出（不要多余的东西）：
思考：
（这一步要做什么、为什么、预期看到什么）
ACTION: run_code
（然后一个且仅一个 ```python 代码块；代码必须独立可运行、必须 print 关键结果；禁止无限循环）

或者任务完成时：
思考：
（简短）
ACTION: final
FINAL:
（最终报告：结论、关键数字、证据、仍不确定的地方；用中文）

纪律：
- 报错就读错误、修代码，同一个错误修两次还没好就换方法。
- 不要重复跑已经验证过的东西；不要硬猜数字，数字必须来自你跑出的结果。
- 物理核脚本路径、参数、已知坑都在背景里，直接 import 复用。
- 轮次有限，必须在最后一轮前给出 FINAL。"""


def parse_step(text):
    """从模型输出解析 action / code / final。"""
    m = re.search(r"ACTION:\s*(run_code|final)", text)
    action = m.group(1) if m else None
    code = None
    blocks = re.findall(r"```python\s*(.*?)```", text, re.S)
    if blocks:
        code = blocks[-1]
    final = None
    fm = re.search(r"FINAL:\s*(.*?)(?:$)", text, re.S)
    if fm:
        final = fm.group(1).strip()
    return action, code, final


def run_code(ws, step, code):
    path = os.path.join(ws, f"step_{step}.py")
    open(path, "w", encoding="utf-8").write(code)
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    try:
        p = subprocess.run(
            [sys.executable, path],
            cwd=ws,
            capture_output=True,
            text=True,
            timeout=EXEC_TIMEOUT,
            env=env,
        )
        out = p.stdout or ""
        err = p.stderr or ""
        if len(out) > OUT_CAP:
            out = out[:OUT_CAP] + f"\n...[截断，共{len(p.stdout)}字]"
        if len(err) > OUT_CAP:
            err = err[:OUT_CAP] + f"\n...[截断，共{len(p.stderr)}字]"
        return f"exit={p.returncode}\n--- stdout ---\n{out}\n--- stderr ---\n{err}"
    except subprocess.TimeoutExpired:
        return f"exit=TIMEOUT（超过{EXEC_TIMEOUT}秒被强杀，检查是否有死循环/测程过长）"
    except Exception as e:
        return f"exit=EXEC_ERROR: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--worker", type=int, required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--context", default="")
    ap.add_argument("--max-iters", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=12000)
    args = ap.parse_args()

    ws = os.path.join(WS_ROOT, f"w{args.worker}")
    os.makedirs(ws, exist_ok=True)

    print(f"[agentic w{args.worker}] 任务：{args.task[:80]}", flush=True)

    user_head = (
        f"背景：\n{args.context}\n\n"
        f"你的任务：\n{args.task}\n\n"
        f"工作目录（你代码的运行 cwd）：{ws}\n"
        f"物理核脚本目录：{SIM_DIR}（可 sys.path.insert(0, r'{SIM_DIR}') 后 import）\n"
        f"python：{sys.executable}\n"
    )
    messages_ctx = user_head
    final_report = None
    history = []

    for step in range(1, args.max_iters + 1):
        last_round = step == args.max_iters
        prompt = messages_ctx
        if last_round:
            prompt += "\n【这是最后一轮，必须输出 ACTION: final 和 FINAL 报告。】"
        prompt += "\n请按格式输出。"

        stream_file = os.path.join(LOG_DIR, f"agentic_{args.round}_w{args.worker}_step{step}.md")
        os.makedirs(LOG_DIR, exist_ok=True)
        t0 = time.time()
        resp = ark_call.call_api(
            prompt,
            system_prompt=SYSTEM_PROMPT,
            max_tokens=args.max_tokens,
            output_file=stream_file,
        )
        if resp is None and os.path.exists(stream_file):
            raw = open(stream_file, encoding="utf-8").read()
            if "[流式输出完整结束]" in raw:
                resp = raw.split("[流式输出完整结束]")[0]
        if not resp or not resp.strip():
            print(f"[agentic w{args.worker}] step{step} 空正文，重试", flush=True)
            continue

        action, code, final = parse_step(resp)
        print(f"[agentic w{args.worker}] step{step} action={action} ({time.time()-t0:.0f}s)", flush=True)

        if action == "final" or (final and last_round):
            final_report = final or "（模型未给 FINAL 正文）"
            break
        if action == "run_code" and code:
            result = run_code(ws, step, code)
            print(f"[agentic w{args.worker}] step{step} 执行完成：{result.splitlines()[0]}", flush=True)
            history.append(f"第{step}轮你写的代码（已执行，存在 step_{step}.py）：\n{code[:1200]}")
            messages_ctx = (
                user_head
                + "\n".join(history)
                + f"\n\n第{step}步代码执行结果：\n{result}\n\n"
                + "基于结果继续下一步（或在完成时给 ACTION: final）。"
            )
        else:
            messages_ctx += "\n【没解析出合法 ACTION/代码，请严格按格式重输出】"

    if not final_report:
        final_report = f"（{args.max_iters} 轮内未拿到 final，任务未完成，待主 Agent 处理）"

    report_path = os.path.join(LOG_DIR, f"agentic_{args.round}_w{args.worker}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Agentic worker {args.worker} · round {args.round} · {datetime.now():%Y-%m-%d %H:%M}\n\n")
        f.write(f"**任务**：{args.task}\n\n")
        f.write(final_report + "\n")
    print(f"[agentic w{args.worker}] 报告：{report_path}", flush=True)

    meta = {
        "round": args.round,
        "worker": args.worker,
        "task": args.task,
        "finished": final_report and not final_report.startswith("（"),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(ws, "result.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    sys.exit(0 if meta["finished"] else 1)


if __name__ == "__main__":
    main()

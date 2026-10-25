#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
火山方舟API通用调用脚本（寂雷引擎版）
用法：python3 ark_call.py "你的问题" [--max_tokens 8000] [--system "系统提示"]
密钥从 secrets/ark_config.json 读取，不硬编码。
默认：流式输出，max_tokens=8000
"""
import urllib.request
import json
import sys
import os
import time
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "secrets", "ark_config.json")


def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def call_api(prompt, system_prompt="你是一个资深助手。", max_tokens=8000, output_file=None):
    cfg = load_config()
    payload = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "stream": True,
    }
    req = urllib.request.Request(
        f"{cfg['base_url']}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg['api_key']}",
        },
    )
    full_content = ""
    out_fh = open(output_file, "w", encoding="utf-8") if output_file else None
    # evolving深度模型先输出reasoning_content(内部推理)，单独落盘，持续flush防止进程被判定无响应回收
    reasoning_file = output_file.replace(".md", "_reasoning.md") if output_file else None
    reason_fh = open(reasoning_file, "w", encoding="utf-8") if reasoning_file else None
    reason_chars = 0
    actual_model = ""
    last_valid = time.time()
    IDLE_LIMIT = 600  # 连续600秒无有效reasoning/content即判定挂起（evolving深度思考TTFT长）
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for line in resp:
                # 空闲看门狗：代理可能发keepalive/注释行维持连接但无真实数据
                if time.time() - last_valid > IDLE_LIMIT:
                    raise TimeoutError(f"连续{IDLE_LIMIT}秒无有效数据，判定连接挂起")
                line = line.decode("utf-8").strip()
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        actual_model = data.get("model", actual_model) or actual_model
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        # 内部推理过程：实时落盘（不刷屏），让进程持续有IO
                        reasoning = delta.get("reasoning_content", "")
                        if reasoning:
                            print(reasoning, file=sys.stderr, end="", flush=True)  # 牛式输出:实时打印推理
                            if reason_fh:
                                reason_fh.write(reasoning)
                                reason_fh.flush()
                            reason_chars += len(reasoning)
                            last_valid = time.time()
                        # 正式正文
                        content = delta.get("content", "")
                        if content:
                            full_content += content
                            print(content, end="", flush=True)
                            if out_fh:
                                out_fh.write(content)
                                out_fh.flush()
                            last_valid = time.time()
                    except json.JSONDecodeError:
                        pass
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "ignore")
        print(f"\n[API HTTP错误] {e.code} {body[:500]}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"\n[API错误] {e}", file=sys.stderr)
        if out_fh:
            out_fh.write(f"\n\n[中断于异常] {e}\n")
            out_fh.close()
        return None
    finally:
        if reason_fh:
            reason_fh.write(f"\n\n[推理结束，共{reason_chars}字]\n")
            reason_fh.close()
    print()
    if actual_model:
        print(f"[实际模型] {actual_model}", file=sys.stderr)
    if out_fh:
        out_fh.write(f"\n\n[实际模型: {actual_model}]\n[流式输出完整结束]\n")
        out_fh.close()
    return full_content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="火山方舟API通用调用")
    parser.add_argument("prompt", help="问题/提示词")
    parser.add_argument("--system", default="你是一个资深助手。", help="系统提示")
    parser.add_argument("--max_tokens", type=int, default=8000, help="最大输出token数")
    args = parser.parse_args()
    result = call_api(args.prompt, args.system, args.max_tokens)
    if result is None:
        sys.exit(1)
    if not result.strip():
        print("[失败] 正文为空（reasoning 吃光预算？），请提高 max_tokens", file=sys.stderr)
        sys.exit(2)
    sys.exit(0)

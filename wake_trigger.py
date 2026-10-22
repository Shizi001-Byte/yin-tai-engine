#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wake_trigger.py — E2 双保险的统一幂等入口（cron 与 daemon 都调它）

契约：
- 并发：flock 非阻塞互斥；抢不到锁 => LOCK_BUSY，退出码 0
- 重复：bucket = floor(now / WAKE_INTERVAL)，同桶第二次 => DUP_BUCKET；
        WAKE_MIN_GAP 秒间隔兜底
- 语义：at-least-once。self_wake.py 返回 0 才提交桶；失败留待同桶重试。
        轮次仲裁权在 logs/history.jsonl，本脚本不碰该文件。
- --force：跳过重复闸（手工补救），仍走锁与提交。
仅标准库，POSIX。
环境变量：WAKE_INTERVAL（默认3600，测试可设60）、WAKE_MIN_GAP（默认3000）
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    import fcntl
except ImportError:
    fcntl = None

BASE = Path(__file__).resolve().parent


def log(base, event, **kw):
    path = Path(base) / "logs" / "trigger.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {"ts": time.time(),
           "iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "event": event}
    rec.update(kw)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")


def decide(now, last_bucket, last_ts, interval, min_gap, force):
    """纯函数：给定现状返回 (bucket, decision)。"""
    bucket = int(now // interval)
    if force:
        return bucket, "FORCE_RUN"
    if last_bucket is not None and bucket == last_bucket:
        return bucket, "DUP_BUCKET"
    if last_ts is not None and 0 < (now - last_ts) < min_gap:
        return bucket, "DUP_GAP"
    return bucket, "RUN"


def read_bucket(base):
    p = Path(base) / "locks" / "last_bucket.json"
    if not p.exists():
        return None, None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d.get("bucket"), d.get("ts")
    except (ValueError, OSError):
        return None, None  # 损坏按"无桶"处理：触发优先，日志留痕


def write_bucket(base, bucket, now, winner):
    p = Path(base) / "locks" / "last_bucket.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(
        json.dumps({"bucket": bucket, "ts": now, "winner": winner},
                   sort_keys=True),
        encoding="utf-8")
    os.replace(tmp, p)


def real_invoke(base, child_args):
    cmd = [sys.executable, str(Path(base) / "self_wake.py")] + list(child_args)
    return subprocess.run(cmd, cwd=str(base)).returncode


def cycle(base, interval, min_gap, winner, force, child_args,
          invoke=None, now_func=time.time):
    if fcntl is None:
        raise SystemExit("wake_trigger requires POSIX fcntl")
    base = Path(base)
    (base / "locks").mkdir(parents=True, exist_ok=True)
    invoke = invoke or (lambda: real_invoke(base, child_args))

    lock_f = open(base / "locks" / "wake.lock", "a+")
    try:
        try:
            fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log(base, "LOCK_BUSY", winner=winner, force=force)
            return 0

        last_bucket, last_ts = read_bucket(base)
        now = now_func()
        bucket, decision = decide(now, last_bucket, last_ts,
                                  interval, min_gap, force)
        log(base, "ATTEMPT", winner=winner, bucket=bucket,
            decision=decision, force=force)
        if decision.startswith("DUP_"):
            return 0

        rc = invoke()
        if rc == 0:
            write_bucket(base, bucket, now, winner)
            log(base, "PASS", winner=winner, bucket=bucket, decision=decision)
            return 0
        log(base, "FAIL", winner=winner, bucket=bucket, rc=rc)
        return rc
    finally:
        try:
            fcntl.flock(lock_f, fcntl.LOCK_UN)
        except OSError:
            pass
        lock_f.close()


def _selftest():
    import tempfile
    base = Path(tempfile.mkdtemp(prefix="jilei_trigger_selftest_"))
    calls = []

    def invoke_ok():
        # 子进程窗口内，另一把非阻塞锁必须抢失败 => 证明锁确实持有
        lf2 = open(base / "locks" / "wake.lock", "a+")
        try:
            fcntl.flock(lf2, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked_again = True
        except BlockingIOError:
            locked_again = False
        finally:
            lf2.close()
        assert not locked_again, "lock must be held during child invocation"
        calls.append("ok")
        return 0

    def invoke_fail():
        calls.append("fail")
        return 7

    # interval=60；时间戳刻意构造：桶16×2、桶17×2(含force)、桶18失败后重试
    clocks = iter([1000, 1005, 1060, 1061, 1080, 1085])
    nowf = lambda: next(clocks)

    assert cycle(base, 60, 0, "cron",   False, [], invoke_ok,   nowf) == 0   # 桶16 PASS
    assert cycle(base, 60, 0, "daemon", False, [], invoke_ok,   nowf) == 0   # 桶16 DUP
    assert cycle(base, 60, 0, "cron",   False, [], invoke_ok,   nowf) == 0   # 桶17 PASS
    assert cycle(base, 60, 0, "manual", True,  [], invoke_ok,   nowf) == 0   # 桶17 FORCE
    assert cycle(base, 60, 0, "cron",   False, [], invoke_fail, nowf) == 7   # 桶18 FAIL不提交
    assert cycle(base, 60, 0, "daemon", False, [], invoke_ok,   nowf) == 0   # 桶18 重试PASS

    assert calls == ["ok", "ok", "ok", "fail", "ok"], calls
    events = [json.loads(x)["event"]
              for x in (base / "logs" / "trigger.jsonl").read_text().splitlines()]
    expect = ["ATTEMPT", "PASS",                      # 桶16
              "ATTEMPT",                              # 桶16 DUP
              "ATTEMPT", "PASS",                      # 桶17
              "ATTEMPT", "PASS",                      # 桶17 FORCE
              "ATTEMPT", "FAIL",                      # 桶18 失败
              "ATTEMPT", "PASS"]                      # 桶18 重试
    assert events == expect, events
    print("SELFTEST PASS: cycles=6 invokes=%d log_lines=%d"
          % (len(calls), len(events)))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--winner", choices=["cron", "daemon", "manual"],
                   default="manual")
    p.add_argument("--force", action="store_true")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("child_args", nargs="*",
                   help="透传给 self_wake.py 的参数（默认无）")
    a = p.parse_args(argv)
    if a.selftest:
        return _selftest()
    interval = int(os.environ.get("WAKE_INTERVAL", "3600"))
    min_gap = int(os.environ.get("WAKE_MIN_GAP", "3000"))
    return cycle(BASE, interval, min_gap, a.winner, a.force, a.child_args)


if __name__ == "__main__":
    sys.exit(main())

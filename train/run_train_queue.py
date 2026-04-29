import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# 项目根目录。所有训练命令默认都在这里执行，保证脚本里的相对路径
# 例如 ./processed_9ch_vibrant_label 和 ./log 仍然按原项目根目录解析。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMMAND_FILE = PROJECT_ROOT / "train" / "commands.txt"


def load_commands(path):
    """读取命令文件：每一行是一条训练命令，空行和 # 开头的行会被跳过。"""
    commands = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        command = raw_line.strip()
        if not command or command.startswith("#"):
            continue
        commands.append((line_no, command))
    return commands


def append_history(history_path, message):
    """把队列运行状态追加到日志文件，方便训练结束后回看每条命令的结果。"""
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def run_queue(command_file, cwd, history_path, dry_run=False):
    """按顺序执行命令；某条命令失败时只记录失败，不中断后续训练。"""
    commands = load_commands(command_file)
    if not commands:
        print(f"No commands found in {command_file}")
        return 0

    print(f"Loaded {len(commands)} command(s) from {command_file}")
    print(f"Working directory: {cwd}")
    print(f"History log: {history_path}")

    failures = 0
    for index, (line_no, command) in enumerate(commands, start=1):
        start_time = datetime.now()
        header = f"[{start_time:%Y-%m-%d %H:%M:%S}] START {index}/{len(commands)} line {line_no}: {command}"
        print("\n" + header, flush=True)
        append_history(history_path, header)

        if dry_run:
            continue

        started = time.monotonic()
        try:
            # shell=True 是为了支持命令文件中的重定向写法：
            # py train/train_swinhr.py > log/swinhr_v9.txt 2>&1
            process = subprocess.Popen(command, cwd=str(cwd), shell=True)
            return_code = process.wait()
        except KeyboardInterrupt:
            msg = "Interrupted by user. Stop queue."
            print(msg)
            append_history(history_path, msg)
            return 130
        except Exception as exc:
            failures += 1
            msg = f"FAILED TO START line {line_no}: {exc}"
            print(msg, flush=True)
            append_history(history_path, msg)
            continue

        elapsed = time.monotonic() - started
        end_time = datetime.now()
        status = "OK" if return_code == 0 else f"FAILED return_code={return_code}"
        if return_code != 0:
            failures += 1

        footer = f"[{end_time:%Y-%m-%d %H:%M:%S}] END {index}/{len(commands)} {status} elapsed={elapsed:.1f}s"
        print(footer, flush=True)
        append_history(history_path, footer)

    summary = f"Queue finished: total={len(commands)}, failures={failures}"
    print("\n" + summary)
    append_history(history_path, summary)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Run training commands one by one and continue after failures.")
    parser.add_argument("--commands", type=Path, default=DEFAULT_COMMAND_FILE,
                        help="Text file containing one shell command per line.")
    parser.add_argument("--cwd", type=Path, default=PROJECT_ROOT,
                        help="Working directory used to run every command.")
    parser.add_argument("--history", type=Path, default=PROJECT_ROOT / "log" / "train_queue_history.txt",
                        help="Queue status log path.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    args = parser.parse_args()

    command_file = args.commands.resolve()
    cwd = args.cwd.resolve()
    history_path = args.history.resolve()

    if not command_file.exists():
        print(f"Command file not found: {command_file}", file=sys.stderr)
        return 2
    if not cwd.exists():
        print(f"Working directory not found: {cwd}", file=sys.stderr)
        return 2

    return run_queue(command_file, cwd, history_path, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())

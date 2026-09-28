#!/usr/bin/env python3
"""fractionkeep：精确分数运算库与命令行。

值一律表示为整数对 (p, q)，q > 0 且 gcd(|p|, q) = 1，零唯一写作 0/1。
全程整数运算，不出现浮点。仅使用标准库。
"""

from __future__ import annotations

import functools
import html
import math
import os
import re
import signal
import sys
from collections import namedtuple

BUDGET = 64
MAX_POWER = 1000

VALUE_RE = re.compile(r"-?\d+(?:/-?\d+)?$")
INT_RE = re.compile(r"-?\d+$")
OPS = ("+", "-", "*", "/", "//", "%", "^")

Step = namedtuple("Step", "index op operand raw_p raw_q rd p q nd")
ChainResult = namedtuple(
    "ChainResult",
    "name start steps final peak raw_peak minimum maximum",
)


class InputError(Exception):
    """命令行用法、链文件或数值不合法。"""


# ---------- 规范化与位数 ----------

def normalize(p, q):
    """约到最简；要求 q > 0；零写作 (0, 1)。"""
    if q <= 0:
        raise InputError(f"分母必须为正整数: {p}/{q}")
    g = math.gcd(abs(p), q)
    p //= g
    q //= g
    if p == 0:
        return 0, 1
    return p, q


def digits(n):
    """|n| 的十进制位数，L(0) = 1。"""
    return len(str(abs(n)))


def pair_digits(p, q):
    return digits(p) + digits(q)


def fmt(p, q):
    return f"{p}/{q}"


def parse_value(token):
    """解析 `p` 或 `p/q` 形式的值并规范化；分母 <= 0 是输入错误。"""
    if not VALUE_RE.match(token):
        raise InputError(f"无法解析的分数: {token}")
    parts = token.split("/")
    p = int(parts[0])
    q = int(parts[1]) if len(parts) == 2 else 1
    if q <= 0:
        raise InputError(f"分母必须为正整数: {token}")
    return normalize(p, q)


def parse_exponent(token):
    if not INT_RE.match(token):
        raise InputError(f"指数必须是整数: {token}")
    k = int(token)
    if abs(k) > MAX_POWER:
        raise InputError(f"指数绝对值超过 {MAX_POWER}: {token}")
    return k


# ---------- 运算 ----------

def apply_op(op, p, q, operand):
    """对规范化值 (p, q) 执行一步运算。

    operand：'^' 时为整数 k，其余为规范化对 (s, t)。
    返回 (P, Q, np, nq)：未约分对（分母恒为正）与规范化结果。
    """
    if op == "^":
        k = operand
        if k >= 0:
            P, Q = p ** k, q ** k
        else:
            if p == 0:
                raise InputError("0 的负数次幂")
            P, Q = q ** (-k), p ** (-k)
        if Q < 0:
            P, Q = -P, -Q
        np_, nq = normalize(P, Q)
        return P, Q, np_, nq
    s, t = operand
    if op == "+":
        P, Q = p * t + s * q, q * t
    elif op == "-":
        P, Q = p * t - s * q, q * t
    elif op == "*":
        P, Q = p * s, q * t
    elif op == "/":
        if s == 0:
            raise InputError("除数为零")
        P, Q = p * t, q * s
    elif op == "//":
        if s == 0:
            raise InputError("除数为零")
        P, Q = p * t, q * s
        if Q < 0:
            P, Q = -P, -Q
        k0 = (p * t) // (q * s)
        return P, Q, k0, 1
    elif op == "%":
        if s == 0:
            raise InputError("除数为零")
        k0 = (p * t) // (q * s)
        P, Q = p * t - s * q * k0, q * t
        np_, nq = normalize(P, Q)
        return P, Q, np_, nq
    else:
        raise InputError(f"未知操作: {op}")
    if Q < 0:
        P, Q = -P, -Q
    np_, nq = normalize(P, Q)
    return P, Q, np_, nq


def compare(a, b):
    """三路比较：交叉相乘，q、t > 0，不用浮点。"""
    left = a[0] * b[1]
    right = b[0] * a[1]
    return (left > right) - (left < right)


# ---------- 运算链 ----------

def load_chain(path):
    """解析链文件，返回 (start, [(op, token), ...])。"""
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        raise InputError(f"无法读取链文件: {path}")
    start = None
    steps = []
    for lineno, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if start is None:
            if len(parts) != 2 or parts[0] != "start":
                raise InputError(f"{path}:{lineno}: 第一条有效行必须是 start <值>")
            start = parse_value(parts[1])
        else:
            if len(parts) != 2 or parts[0] not in OPS:
                raise InputError(f"{path}:{lineno}: 无法解析的步骤: {line}")
            steps.append((parts[0], parts[1]))
    if start is None:
        raise InputError(f"{path}: 缺少 start 行")
    if not steps:
        raise InputError(f"{path}: 链至少需要一步")
    return start, steps


def run_chain(name, start, raw_steps):
    """逐步执行运算链，每步先写未约分对再规范化，返回 ChainResult。"""
    p, q = start
    peak = pair_digits(p, q)
    raw_peak = 0
    minimum = maximum = (p, q)
    records = []
    for index, (op, token) in enumerate(raw_steps, 1):
        if op == "^":
            k = parse_exponent(token)
            P, Q, np_, nq = apply_op(op, p, q, k)
            operand = str(k)
        else:
            s, t = parse_value(token)
            P, Q, np_, nq = apply_op(op, p, q, (s, t))
            operand = fmt(s, t)
        rd = pair_digits(P, Q)
        nd = pair_digits(np_, nq)
        records.append(Step(index, op, operand, P, Q, rd, np_, nq, nd))
        if rd > raw_peak:
            raw_peak = rd
        if nd > peak:
            peak = nd
        if compare((np_, nq), minimum) < 0:
            minimum = (np_, nq)
        if compare((np_, nq), maximum) > 0:
            maximum = (np_, nq)
        p, q = np_, nq
    return ChainResult(name, start, records, (p, q), peak, raw_peak, minimum, maximum)


# ---------- 命令 ----------

def cmd_run(args):
    if len(args) != 1:
        raise InputError("用法: run <链文件>")
    path = args[0]
    start, raw_steps = load_chain(path)
    name = os.path.splitext(os.path.basename(path))[0]
    result = run_chain(name, start, raw_steps)
    lines = [
        f"chain {name} start={fmt(*start)} steps={len(result.steps)} "
        f"final={fmt(*result.final)} peak={result.peak} "
        f"raw_peak={result.raw_peak} budget={BUDGET}"
    ]
    for st in result.steps:
        lines.append(
            f"step {st.index} {st.op} {st.operand} raw={fmt(st.raw_p, st.raw_q)} "
            f"rd={st.rd} norm={fmt(st.p, st.q)} nd={st.nd}"
        )
    return lines


def cmd_cmp(args):
    if len(args) != 2:
        raise InputError("用法: cmp <值> <值>")
    a = parse_value(args[0])
    b = parse_value(args[1])
    return [f"cmp={compare(a, b)}"]


def cmd_sort(args):
    if not args:
        raise InputError("用法: sort <值> [<值> …]")
    values = {parse_value(token) for token in args}
    ordered = sorted(values, key=functools.cmp_to_key(compare))
    return [fmt(*v) for v in ordered]


# ---------- 期望脚本核对 ----------

def parse_expected(path):
    """解析期望脚本，返回 [(参数, 期望 stdout 行列表), ...]。"""
    try:
        with open(path, encoding="utf-8", newline="") as fh:
            text = fh.read()
    except OSError:
        raise InputError(f"无法读取期望脚本: {path}")
    commands = []
    current_args = None
    expected_lines = []
    for line in text.split("\n"):
        if line.startswith("#"):
            continue
        if line.startswith("? "):
            if current_args is not None:
                commands.append((current_args, expected_lines))
            current_args = line[2:]
            expected_lines = []
        elif current_args is not None:
            expected_lines.append(line)
    if current_args is not None:
        commands.append((current_args, expected_lines))
    for args_str, lines in commands:
        if lines and lines[-1] == "":
            lines.pop()
    return commands


def dispatch(args):
    """执行期望脚本里的一条命令，返回 stdout 行列表。"""
    if not args:
        raise InputError("期望脚本中的空命令")
    cmd, rest = args[0], args[1:]
    if cmd == "run":
        return cmd_run(rest)
    if cmd == "cmp":
        return cmd_cmp(rest)
    if cmd == "sort":
        return cmd_sort(rest)
    raise InputError(f"期望脚本中的未知命令: {cmd}")


# ---------- HTML 报告 ----------

HTML_HEAD = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>fractionkeep 精确分数运算报告</title>
<style>
body { font-family: "Segoe UI", "Microsoft YaHei", sans-serif; margin: 2em; color: #222; }
h1 { font-size: 1.5em; }
h2 { font-size: 1.15em; margin-top: 2em; border-bottom: 2px solid #456; padding-bottom: .2em; }
table { border-collapse: collapse; margin: .6em 0; }
th, td { border: 1px solid #bbb; padding: .2em .7em; text-align: left; }
th { background: #eef1f4; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.summary td:first-child { background: #f7f8fa; font-weight: bold; }
.ok { color: #1a7f37; font-weight: bold; }
.over { color: #b42318; font-weight: bold; }
.steps td { font-family: Consolas, "Courier New", monospace; font-size: .92em; }
</style>
</head>
<body>
<h1>fractionkeep 精确分数运算报告</h1>
<p>口径：每步结果约到最简、分母为正、零写作 0/1；位数预算 64（规范化结果的分子分母位数之和）。
本页全部数值由引擎计算后写入，页面内不重算。</p>
"""

HTML_TAIL = "</body>\n</html>\n"


def render_section(result):
    esc = html.escape
    within = result.peak <= BUDGET
    verdict = (
        f'<span class="ok">peak={result.peak} ≤ {BUDGET}，在预算内</span>'
        if within
        else f'<span class="over">peak={result.peak} &gt; {BUDGET}，超出预算</span>'
    )
    parts = [f"<h2>{esc(result.name)}</h2>\n"]
    parts.append('<table class="summary">\n')
    summary_rows = [
        ("步数", str(len(result.steps))),
        ("起始值", fmt(*result.start)),
        ("最终值", fmt(*result.final)),
        ("规范化位数峰值 vs 预算", verdict),
        ("约分前位数峰值 (raw_peak)", str(result.raw_peak)),
        ("最小值（含起始值）", fmt(*result.minimum)),
        ("最大值（含起始值）", fmt(*result.maximum)),
    ]
    for label, value in summary_rows:
        parts.append(f"<tr><td>{esc(label)}</td><td>{value}</td></tr>\n")
    parts.append("</table>\n")
    parts.append(
        '<table class="steps">\n<thead><tr>'
        "<th>步号</th><th>操作</th><th>未约分对</th>"
        '<th class="num">rd</th><th>规范化结果</th><th class="num">nd</th>'
        "</tr></thead>\n<tbody>\n"
    )
    for st in result.steps:
        parts.append(
            f"<tr><td>{st.index}</td>"
            f"<td>{esc(st.op)} {esc(st.operand)}</td>"
            f"<td>{esc(fmt(st.raw_p, st.raw_q))}</td>"
            f'<td class="num">{st.rd}</td>'
            f"<td>{esc(fmt(st.p, st.q))}</td>"
            f'<td class="num">{st.nd}</td></tr>\n'
        )
    parts.append("</tbody>\n</table>\n")
    return "".join(parts)


def write_report(root, output):
    chains_dir = os.path.join(root, "chains")
    try:
        names = sorted(n for n in os.listdir(chains_dir) if n.endswith(".txt"))
    except OSError:
        raise InputError(f"无法读取链目录: {chains_dir}")
    if not names:
        raise InputError(f"链目录为空: {chains_dir}")
    sections = []
    for fname in names:
        path = os.path.join(chains_dir, fname)
        start, raw_steps = load_chain(path)
        name = os.path.splitext(fname)[0]
        sections.append(render_section(run_chain(name, start, raw_steps)))
    page = HTML_HEAD + "".join(sections) + HTML_TAIL
    directory = os.path.dirname(output)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(output, "w", encoding="utf-8", newline="") as fh:
        fh.write(page)


def cmd_report(args, out=None):
    if out is None:
        out = sys.stdout
    if len(args) > 2:
        raise InputError("用法: report [<样例根目录>=samples] [<输出文件>=var/fractions.html]")
    root = args[0] if len(args) >= 1 else "samples"
    output = args[1] if len(args) >= 2 else os.path.join("var", "fractions.html")
    expected_dir = os.path.join(root, "expected")
    try:
        names = sorted(os.listdir(expected_dir))
    except OSError:
        raise InputError(f"无法读取期望目录: {expected_dir}")
    commands = []
    for name in names:
        if name.endswith(".txt"):
            commands.extend(parse_expected(os.path.join(expected_dir, name)))
    mismatches = []
    for args_str, expected_lines in commands:
        actual = "".join(line + "\n" for line in dispatch(args_str.split()))
        expected = "".join(line + "\n" for line in expected_lines)
        if actual != expected:
            mismatches.append(args_str)
    if mismatches:
        for args_str in mismatches:
            print(f"mismatch={args_str}", file=out)
        return 1
    print(f"ok={len(commands)}", file=out)
    write_report(root, output)
    return 0


# ---------- 入口 ----------

def main(argv):
    if not argv:
        print("error: 用法: fractionkeep.py <run|cmp|sort|report> …", file=sys.stderr)
        return 2
    cmd, args = argv[0], argv[1:]
    try:
        if cmd == "run":
            lines = cmd_run(args)
        elif cmd == "cmp":
            lines = cmd_cmp(args)
        elif cmd == "sort":
            lines = cmd_sort(args)
        elif cmd == "report":
            return cmd_report(args)
        else:
            raise InputError(f"未知命令: {cmd}")
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    sys.stdout.write("".join(line + "\n" for line in lines))
    return 0


if __name__ == "__main__":
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    sys.exit(main(sys.argv[1:]))

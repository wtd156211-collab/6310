#!/usr/bin/env python3
"""fractionkeep — 精确分数运算引擎与命令行工具。

值一律表示为 (p, q) 整数对：q > 0、gcd(|p|, q) = 1、零唯一写作 (0, 1)。
全程整数运算，不出现浮点；每步先写出未约分对 (P, Q)（Q 恒为正），
约到最简后才进入下一步。
"""

from __future__ import annotations

import functools
import html
import math
import sys
from pathlib import Path

BUDGET = 64
MAX_EXPONENT = 1000
OPERATORS = ("+", "-", "*", "/", "//", "%", "^")

USAGE = (
    "用法: python3 fractionkeep.py run <链文件> | cmp <值> <值> | "
    "sort <值> [<值> …] | report [<样例根目录>=samples] [<输出文件>=var/fractions.html]"
)


class InputError(Exception):
    """用法或输入错误（退出码 2）。"""


# ---------- 基础 ----------

def digits(n: int) -> int:
    """十进制位数 L(n) = |n| 的位数，L(0) = 1。"""
    return len(str(abs(n)))


def pair_digits(p: int, q: int) -> int:
    """一对数的位数 = L(p) + L(q)。"""
    return digits(p) + digits(q)


def reduce_pair(p: int, q: int) -> tuple[int, int]:
    """约到最简；要求 q > 0；零记作 (0, 1)。"""
    if p == 0:
        return (0, 1)
    g = math.gcd(p, q)
    return (p // g, q // g)


def compare(a: tuple[int, int], b: tuple[int, int]) -> int:
    """三路比较：交叉相乘 p*t 与 s*q（q、t > 0），不用浮点。"""
    left = a[0] * b[1]
    right = b[0] * a[1]
    return (left > right) - (left < right)


def fmt(fr: tuple[int, int]) -> str:
    return f"{fr[0]}/{fr[1]}"


# ---------- 解析 ----------

def _parse_int(text: str, whole: str) -> int:
    body = text[1:] if text.startswith("-") else text
    if not body or not body.isascii() or not body.isdigit():
        raise InputError(f"无法解析的整数: {whole}")
    return int(text)


def parse_value(text: str) -> tuple[int, int]:
    """解析 `p` 或 `p/q` 并规范化；分母 ≤ 0 是输入错误。"""
    if "/" in text:
        num, _, den = text.partition("/")
        p = _parse_int(num, text)
        q = _parse_int(den, text)
    else:
        p, q = _parse_int(text, text), 1
    if q <= 0:
        raise InputError(f"分母必须为正: {text}")
    return reduce_pair(p, q)


def parse_exponent(text: str) -> int:
    """`^` 的操作数：整数且 |k| ≤ 1000。"""
    k = _parse_int(text, text)
    if abs(k) > MAX_EXPONENT:
        raise InputError(f"指数超出范围 |k| <= {MAX_EXPONENT}: {text}")
    return k


def parse_chain(text: str) -> tuple[tuple[int, int], list]:
    """解析链文件，返回 (起始值, [(操作, 操作数), …])。"""
    start = None
    steps = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if start is None:
            if len(parts) != 2 or parts[0] != "start":
                raise InputError(f"第一条有效行必须是 start <值>: {line}")
            start = parse_value(parts[1])
            continue
        if len(parts) != 2 or parts[0] not in OPERATORS:
            raise InputError(f"无法解析的步骤: {line}")
        op = parts[0]
        operand = parse_exponent(parts[1]) if op == "^" else parse_value(parts[1])
        steps.append((op, operand))
    if start is None:
        raise InputError("链为空: 缺少 start 行")
    if not steps:
        raise InputError("链至少需要 1 步")
    return start, steps


def parse_expected(text: str) -> list:
    """解析期望脚本，返回 [(参数列表, 期望 stdout), …]。"""
    commands = []
    current_lines = None
    for line in text.split("\n"):
        if line.startswith("? "):
            current_lines = []
            commands.append((line[2:].split(), current_lines))
        elif line.startswith("#"):
            continue
        elif current_lines is not None:
            current_lines.append(line)
    result = []
    for args, lines in commands:
        while lines and lines[-1] == "":
            lines.pop()
        expected = "\n".join(lines) + "\n" if lines else ""
        result.append((args, expected))
    return result


# ---------- 运算 ----------

def raw_pair(x: tuple[int, int], op: str, operand) -> tuple[int, int]:
    """一步运算的未约分对 (P, Q)，Q 恒为正（负分母把符号挪到分子）。"""
    p, q = x
    if op == "+":
        s, t = operand
        return (p * t + s * q, q * t)
    if op == "-":
        s, t = operand
        return (p * t - s * q, q * t)
    if op == "*":
        s, t = operand
        return (p * s, q * t)
    if op == "/" or op == "//":
        s, t = operand
        if s == 0:
            raise InputError("除数为 0")
        P, Q = p * t, q * s
        return (P, Q) if Q > 0 else (-P, -Q)
    if op == "%":
        s, t = operand
        if s == 0:
            raise InputError("除数为 0")
        k0 = (p * t) // (q * s)
        return (p * t - s * q * k0, q * t)
    if op == "^":
        k = operand
        if k >= 0:
            return (p ** k, q ** k)
        if p == 0:
            raise InputError("0 的负数次幂")
        P, Q = q ** (-k), p ** (-k)
        return (P, Q) if Q > 0 else (-P, -Q)
    raise InputError(f"未知操作: {op}")


def apply_op(x: tuple[int, int], op: str, operand):
    """返回 (未约分对, 规范化结果)。"""
    P, Q = raw_pair(x, op, operand)
    if op == "//":
        return (P, Q), (P // Q, 1)
    return (P, Q), reduce_pair(P, Q)


class ChainResult:
    __slots__ = ("name", "start", "step_count", "final", "peak", "raw_peak",
                 "minimum", "maximum", "records")

    def __init__(self, name, start, step_count, final, peak, raw_peak,
                 minimum, maximum, records):
        self.name = name
        self.start = start
        self.step_count = step_count
        self.final = final
        self.peak = peak
        self.raw_peak = raw_peak
        self.minimum = minimum
        self.maximum = maximum
        self.records = records


def run_chain(name: str, start: tuple[int, int], steps: list) -> ChainResult:
    """逐步执行；每步记录 (步号, 操作, 操作数, (P, Q), rd, 规范化结果, nd)。"""
    records = []
    x = start
    peak = pair_digits(*start)
    raw_peak = 0
    minimum = maximum = start
    for index, (op, operand) in enumerate(steps, 1):
        (P, Q), norm = apply_op(x, op, operand)
        rd = pair_digits(P, Q)
        nd = pair_digits(*norm)
        records.append((index, op, operand, (P, Q), rd, norm, nd))
        x = norm
        if nd > peak:
            peak = nd
        if rd > raw_peak:
            raw_peak = rd
        if compare(x, minimum) < 0:
            minimum = x
        if compare(x, maximum) > 0:
            maximum = x
    return ChainResult(name, start, len(steps), x, peak, raw_peak,
                       minimum, maximum, records)


def operand_text(op: str, operand) -> str:
    return str(operand) if op == "^" else fmt(operand)


# ---------- 命令 ----------

def read_text(path: Path) -> str:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise InputError(f"无法读取文件: {path}") from exc
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InputError(f"文件不是合法 UTF-8: {path}") from exc
    if text.startswith("\ufeff"):
        raise InputError(f"文件含 BOM: {path}")
    return text


def run_chain_file(path_str: str) -> ChainResult:
    path = Path(path_str)
    start, steps = parse_chain(read_text(path))
    return run_chain(path.stem, start, steps)


def run_text(result: ChainResult) -> str:
    lines = [
        f"chain {result.name} start={fmt(result.start)} steps={result.step_count} "
        f"final={fmt(result.final)} peak={result.peak} raw_peak={result.raw_peak} "
        f"budget={BUDGET}"
    ]
    for (index, op, operand, raw, rd, norm, nd) in result.records:
        lines.append(
            f"step {index} {op} {operand_text(op, operand)} "
            f"raw={fmt(raw)} rd={rd} norm={fmt(norm)} nd={nd}"
        )
    return "\n".join(lines) + "\n"


def command_run(args: list) -> str:
    if len(args) != 1:
        raise InputError("用法: run <链文件>")
    return run_text(run_chain_file(args[0]))


def command_cmp(args: list) -> str:
    if len(args) != 2:
        raise InputError("用法: cmp <值> <值>")
    return f"cmp={compare(parse_value(args[0]), parse_value(args[1]))}\n"


def command_sort(args: list) -> str:
    if not args:
        raise InputError("用法: sort <值> [<值> …]")
    values = {parse_value(a) for a in args}
    ordered = sorted(values, key=functools.cmp_to_key(compare))
    return "".join(fmt(v) + "\n" for v in ordered)


def execute(args: list) -> tuple[int, str]:
    """执行一条命令，返回 (退出码, stdout 文本)。"""
    if not args:
        raise InputError(USAGE)
    cmd, rest = args[0], args[1:]
    if cmd == "run":
        return 0, command_run(rest)
    if cmd == "cmp":
        return 0, command_cmp(rest)
    if cmd == "sort":
        return 0, command_sort(rest)
    if cmd == "report":
        return command_report(rest)
    raise InputError(f"未知命令: {cmd}")


def command_report(args: list) -> tuple[int, str]:
    if len(args) > 2:
        raise InputError(
            "用法: report [<样例根目录>=samples] [<输出文件>=var/fractions.html]")
    root = Path(args[0]) if len(args) >= 1 else Path("samples")
    out_path = Path(args[1]) if len(args) >= 2 else Path("var/fractions.html")
    expected_dir = root / "expected"
    if not expected_dir.is_dir():
        raise InputError(f"找不到期望目录: {expected_dir}")
    total = 0
    mismatches = []
    for path in sorted(expected_dir.glob("*.txt")):
        for cmd_args, expected in parse_expected(read_text(path)):
            total += 1
            if cmd_args and cmd_args[0] == "report":
                actual = None
            else:
                try:
                    _, actual = execute(cmd_args)
                except InputError:
                    actual = None
            if actual != expected:
                mismatches.append(" ".join(cmd_args))
    if mismatches:
        return 1, "".join(f"mismatch={m}\n" for m in mismatches)
    chains_dir = root / "chains"
    if not chains_dir.is_dir():
        raise InputError(f"找不到链目录: {chains_dir}")
    results = [run_chain_file(str(p)) for p in sorted(chains_dir.glob("*.txt"))]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        fh.write(render_html(results))
    return 0, f"ok={total}\n"


# ---------- HTML 报告 ----------

def render_section(result: ChainResult) -> str:
    name = html.escape(result.name)
    within = result.peak <= BUDGET
    if within:
        verdict = f"peak={result.peak} ≤ 预算 {BUDGET}，在预算内（余量 {BUDGET - result.peak}）"
        verdict_cls = "ok"
    else:
        verdict = f"peak={result.peak} > 预算 {BUDGET}，超出预算 {result.peak - BUDGET}"
        verdict_cls = "bad"
    rows = [
        "<tr><th>步号</th><th>操作</th><th>操作数</th><th>未约分对 P/Q</th>"
        "<th>rd</th><th>规范化结果 p/q</th><th>nd</th></tr>"
    ]
    for (index, op, operand, raw, rd, norm, nd) in result.records:
        rows.append(
            f"<tr><td>{index}</td><td>{html.escape(op)}</td>"
            f"<td>{html.escape(operand_text(op, operand))}</td>"
            f"<td>{fmt(raw)}</td><td>{rd}</td>"
            f"<td>{fmt(norm)}</td><td>{nd}</td></tr>"
        )
    table = "\n".join(rows)
    return f"""<section>
<h2>{name}</h2>
<ul class="summary">
<li>步数：{result.step_count}</li>
<li>起始值：{fmt(result.start)}</li>
<li>最终值：{fmt(result.final)}</li>
<li>规范化位数峰值：<span class="{verdict_cls}">{verdict}</span></li>
<li>约分前位数峰值 raw_peak：{result.raw_peak}</li>
<li>最小值：{fmt(result.minimum)}　最大值：{fmt(result.maximum)}</li>
</ul>
<table>
{table}
</table>
</section>"""


def render_html(results: list) -> str:
    sections = "\n".join(render_section(r) for r in results)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>精确分数运算报告</title>
<style>
body {{ font-family: "Segoe UI", "Microsoft YaHei", sans-serif; margin: 2em; color: #222; }}
h1 {{ font-size: 1.5em; }}
h2 {{ font-size: 1.2em; border-bottom: 2px solid #456; padding-bottom: .2em; }}
.meta {{ color: #555; }}
.summary {{ list-style: none; padding: 0; columns: 2; }}
.summary li {{ margin: .2em 0; break-inside: avoid; }}
.ok {{ color: #137333; font-weight: 600; }}
.bad {{ color: #c5221f; font-weight: 600; }}
table {{ border-collapse: collapse; margin-top: .8em;
font-family: Consolas, "Courier New", monospace; font-size: .9em; }}
th, td {{ border: 1px solid #ccc; padding: .15em .6em; text-align: right; }}
th {{ background: #eef1f4; position: sticky; top: 0; }}
tr:nth-child(even) td {{ background: #f8f9fa; }}
</style>
</head>
<body>
<h1>精确分数运算报告</h1>
<p class="meta">口径：每步先写出未约分对再约到最简；分母恒为正；零唯一写作 0/1；
规范化位数预算 {BUDGET}。页面上的数字一律由引擎预先算好。</p>
{sections}
</body>
</html>
"""


# ---------- 入口 ----------

def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(newline="\n")
    try:
        code, out = execute(argv)
    except InputError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    sys.stdout.write(out)
    return code


if __name__ == "__main__":
    sys.exit(main())

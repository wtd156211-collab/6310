#!/usr/bin/env python3
"""fractionkeep 的 unittest 测试。"""

import contextlib
import io
import math
import tempfile
import unittest
from pathlib import Path

import fractionkeep as fk

REPO_ROOT = Path(__file__).resolve().parent
SAMPLES = REPO_ROOT / "samples"


def run_cli(argv):
    """在进程内执行 CLI，返回 (退出码, stdout, stderr)。"""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = fk.main(argv)
    return code, out.getvalue(), err.getvalue()


class DigitsTest(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(fk.digits(0), 1)
        self.assertEqual(fk.digits(7), 1)
        self.assertEqual(fk.digits(-7), 1)
        self.assertEqual(fk.digits(10), 2)
        self.assertEqual(fk.digits(-999), 3)
        self.assertEqual(fk.pair_digits(-12, 345), 5)


class NormalizeTest(unittest.TestCase):
    def test_reduce(self):
        self.assertEqual(fk.reduce_pair(2, 4), (1, 2))
        self.assertEqual(fk.reduce_pair(-6, 3), (-2, 1))
        self.assertEqual(fk.reduce_pair(7, 7), (1, 1))

    def test_zero_is_unique(self):
        self.assertEqual(fk.reduce_pair(0, 5), (0, 1))
        self.assertEqual(fk.reduce_pair(0, 1), (0, 1))

    def test_parse_normalizes(self):
        self.assertEqual(fk.parse_value("2/4"), (1, 2))
        self.assertEqual(fk.parse_value("-0/7"), (0, 1))
        self.assertEqual(fk.parse_value("0/9"), (0, 1))
        self.assertEqual(fk.parse_value("4/8"), (1, 2))
        self.assertEqual(fk.parse_value("-6/12"), (-1, 2))
        self.assertEqual(fk.parse_value("5"), (5, 1))
        self.assertEqual(fk.parse_value("-3"), (-3, 1))

    def test_parse_rejects_bad_input(self):
        for bad in ("3/0", "3/-4", "-3/-4", "1/2/3", "abc", "1/", "/2",
                    "--1", "+1", "1.5", "", "1 /2"):
            with self.assertRaises(fk.InputError, msg=bad):
                fk.parse_value(bad)


class ArithmeticTest(unittest.TestCase):
    def test_add_normalizes_each_step(self):
        raw, norm = fk.apply_op((1, 2), "+", (1, 4))
        self.assertEqual(raw, (6, 8))
        self.assertEqual(norm, (3, 4))

    def test_sub(self):
        raw, norm = fk.apply_op((1, 2), "-", (1, 3))
        self.assertEqual(raw, (1, 6))
        self.assertEqual(norm, (1, 6))

    def test_mul(self):
        raw, norm = fk.apply_op((2, 3), "*", (3, 4))
        self.assertEqual(raw, (6, 12))
        self.assertEqual(norm, (1, 2))

    def test_div(self):
        raw, norm = fk.apply_op((1, 2), "/", (3, 4))
        self.assertEqual(raw, (4, 6))
        self.assertEqual(norm, (2, 3))

    def test_div_negative_denominator_moves_sign(self):
        raw, norm = fk.apply_op((1, 2), "/", (-1, 3))
        self.assertEqual(raw, (-3, 2))
        self.assertEqual(norm, (-3, 2))

    def test_div_by_zero(self):
        with self.assertRaises(fk.InputError):
            fk.apply_op((1, 2), "/", (0, 1))
        with self.assertRaises(fk.InputError):
            fk.apply_op((1, 2), "//", (0, 1))
        with self.assertRaises(fk.InputError):
            fk.apply_op((1, 2), "%", (0, 1))

    def test_zero_result_is_unique(self):
        raw, norm = fk.apply_op((49, 16), "*", (0, 1))
        self.assertEqual(raw, (0, 16))
        self.assertEqual(norm, (0, 1))


class PowTest(unittest.TestCase):
    def test_positive_exponent(self):
        self.assertEqual(fk.apply_op((2, 3), "^", 3)[1], (8, 27))

    def test_zero_exponent(self):
        self.assertEqual(fk.apply_op((7, 5), "^", 0)[1], (1, 1))
        self.assertEqual(fk.apply_op((0, 1), "^", 0)[1], (1, 1))

    def test_negative_exponent(self):
        self.assertEqual(fk.apply_op((2, 3), "^", -2)[1], (9, 4))
        raw, norm = fk.apply_op((-4, 7), "^", -2)
        self.assertEqual((raw, norm), ((49, 16), (49, 16)))

    def test_negative_exponent_negative_base_flips_sign(self):
        raw, norm = fk.apply_op((-2, 3), "^", -3)
        self.assertEqual(raw, (-27, 8))
        self.assertEqual(norm, (-27, 8))

    def test_zero_negative_exponent_is_error(self):
        with self.assertRaises(fk.InputError):
            fk.apply_op((0, 1), "^", -1)

    def test_exponent_bounds(self):
        self.assertEqual(fk.parse_exponent("1000"), 1000)
        self.assertEqual(fk.parse_exponent("-1000"), -1000)
        for bad in ("1001", "-1001", "1/2", "x"):
            with self.assertRaises(fk.InputError, msg=bad):
                fk.parse_exponent(bad)


class FloorModTest(unittest.TestCase):
    def test_floor_division_rounds_toward_negative_infinity(self):
        self.assertEqual(fk.apply_op((-1, 2), "//", (1, 3)), ((-3, 2), (-2, 1)))
        self.assertEqual(fk.apply_op((7, 3), "//", (1, 2)), ((14, 3), (4, 1)))
        self.assertEqual(fk.apply_op((1, 1), "//", (1, 1)), ((1, 1), (1, 1)))

    def test_mod_sign_follows_divisor(self):
        self.assertEqual(fk.apply_op((-7, 3), "%", (1, 2))[1], (1, 6))
        self.assertEqual(fk.apply_op((-2, 1), "%", (-5, 7))[1], (-4, 7))
        self.assertEqual(fk.apply_op((7, 3), "%", (-1, 2))[1], (-1, 6))

    def test_mod_zero_when_divisible(self):
        self.assertEqual(fk.apply_op((3, 2), "%", (1, 2))[1], (0, 1))

    def test_mod_invariant(self):
        # r = x - m * floor(x / m)，且 0 <= r/|m| < 1
        cases = [((-7, 3), (1, 2)), ((7, 3), (-1, 2)), ((-2, 1), (-5, 7)),
                 ((5, 4), (3, 8)), ((-1, 6), (2, 9))]
        for x, m in cases:
            r = fk.apply_op(x, "%", m)[1]
            k = fk.apply_op(x, "//", m)[1]
            back = fk.apply_op(fk.apply_op(m, "*", k)[1], "+", r)[1]
            self.assertEqual(back, x)
            self.assertEqual(fk.compare((abs(r[0]), r[1]), (abs(m[0]), m[1])), -1)
            self.assertEqual(r[0] * m[0] >= 0 or r[0] == 0, True)


class CompareTest(unittest.TestCase):
    def test_cross_multiplication_with_negatives(self):
        self.assertEqual(fk.compare((-1, 3), (-1, 2)), 1)
        self.assertEqual(fk.compare((-1, 2), (-1, 3)), -1)
        self.assertEqual(fk.compare((-7, 3), (0, 1)), -1)
        self.assertEqual(fk.compare((0, 1), (1, 6)), -1)
        self.assertEqual(fk.compare((1, 2), (2, 4)), 0)

    def test_cmp_cli(self):
        self.assertEqual(run_cli(["cmp", "-1/7", "0/1"])[1], "cmp=-1\n")
        self.assertEqual(run_cli(["cmp", "-3/2", "-7/3"])[1], "cmp=1\n")
        self.assertEqual(run_cli(["cmp", "0/1", "0/9"])[1], "cmp=0\n")


class SortTest(unittest.TestCase):
    def test_dedupe_and_order(self):
        code, out, _ = run_cli(["sort", "2/4", "1/2", "-1/2", "0", "-0/7", "3"])
        self.assertEqual(code, 0)
        self.assertEqual(out, "-1/2\n0/1\n1/2\n3/1\n")

    def test_negatives_before_zero_before_positives(self):
        _, out, _ = run_cli(["sort", "1/6", "-7/3", "0/1", "-1/1", "49/16", "1/1", "-1/2"])
        self.assertEqual(out, "-7/3\n-1/1\n-1/2\n0/1\n1/6\n1/1\n49/16\n")

    def test_sort_requires_values(self):
        code, _, err = run_cli(["sort"])
        self.assertEqual(code, 2)
        self.assertTrue(err.startswith("error: "))


class ChainParseTest(unittest.TestCase):
    def test_comments_and_blank_lines(self):
        start, steps = fk.parse_chain("# 注释\n\nstart 2/4\n+ 1/2\n\n# 尾注\n^ 2\n")
        self.assertEqual(start, (1, 2))
        self.assertEqual(steps, [("+", (1, 2)), ("^", 2)])

    def test_errors(self):
        for bad, msg in [
            ("", "空链"),
            ("+ 1/3\n", "缺少 start"),
            ("start 1/2\n", "没有步骤"),
            ("start 1/2\n& 1/3\n", "未知操作"),
            ("start 1/2\n+ 1/3 extra\n", "多余记号"),
            ("start 1/2\n+\n", "缺操作数"),
            ("start 1/2\nstart 1/3\n", "重复 start"),
            ("start 1/2\n^ 1/2\n", "指数不是整数"),
        ]:
            with self.assertRaises(fk.InputError, msg=msg):
                fk.parse_chain(bad)


class ChainRunTest(unittest.TestCase):
    def test_small_chain_text(self):
        chain = SAMPLES / "chains" / "05_signs.txt"
        code, out, _ = run_cli(["run", str(chain)])
        self.assertEqual(code, 0)
        first = out.splitlines()[0]
        self.assertEqual(
            first,
            "chain 05_signs start=0/1 steps=13 final=0/1 peak=4 raw_peak=4 budget=64",
        )

    def test_every_step_normalized_and_budget(self):
        for chain_path in sorted((SAMPLES / "chains").glob("*.txt")):
            result = fk.run_chain_file(str(chain_path))
            with self.subTest(chain=result.name):
                for value in [result.start] + [r[5] for r in result.records]:
                    p, q = value
                    self.assertGreater(q, 0)
                    if p == 0:
                        self.assertEqual(q, 1)
                    else:
                        self.assertEqual(math.gcd(abs(p), q), 1)
                self.assertLessEqual(result.peak, fk.BUDGET)

    def test_coprime_is_closest_to_budget(self):
        peaks = {}
        for chain_path in sorted((SAMPLES / "chains").glob("*.txt")):
            result = fk.run_chain_file(str(chain_path))
            peaks[result.name] = result.peak
        self.assertEqual(peaks["03_coprime"], 56)
        self.assertEqual(max(peaks, key=peaks.get), "03_coprime")

    def test_deterministic_repeated_runs(self):
        chain = str(SAMPLES / "chains" / "06_scale.txt")
        first = fk.command_run([chain])
        second = fk.command_run([chain])
        self.assertEqual(first, second)


class ExpectedScriptsTest(unittest.TestCase):
    def test_all_expected_commands_match(self):
        total = 0
        for path in sorted((SAMPLES / "expected").glob("*.txt")):
            for args, expected in fk.parse_expected(fk.read_text(path)):
                total += 1
                with self.subTest(command=" ".join(args)):
                    _, actual = fk.execute(args)
                    self.assertEqual(actual, expected)
        self.assertEqual(total, 16)


class ReportTest(unittest.TestCase):
    def test_report_ok_and_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "r.html"
            code, out, _ = run_cli(["report", "samples", str(out_path)])
            self.assertEqual(code, 0)
            self.assertEqual(out, "ok=16\n")
            page = out_path.read_bytes().decode("utf-8")
            self.assertIn("01_addchain", page)
            self.assertIn("06_scale", page)
            self.assertIn("peak=56", page)
            self.assertNotIn("<script", page)
            self.assertTrue(page.endswith("\n"))

    def test_report_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "a.html"
            second = Path(tmp) / "b.html"
            run_cli(["report", "samples", str(first)])
            run_cli(["report", "samples", str(second)])
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_report_mismatch_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "expected").mkdir()
            (root / "expected" / "x.txt").write_text(
                "? cmp 1/2 1/3\ncmp=0\n", encoding="utf-8")
            code, out, _ = run_cli(
                ["report", str(root), str(root / "out.html")])
            self.assertEqual(code, 1)
            self.assertEqual(out, "mismatch=cmp 1/2 1/3\n")
            self.assertFalse((root / "out.html").exists())


class CliTest(unittest.TestCase):
    def test_usage_errors_exit_2(self):
        for argv in ([], ["run"], ["run", "a", "b"], ["cmp", "1/2"],
                     ["bogus"], ["report", "a", "b", "c"]):
            with self.subTest(argv=argv):
                code, _, err = run_cli(argv)
                self.assertEqual(code, 2)
                self.assertTrue(err.startswith("error: "))
                self.assertEqual(err.count("\n"), 1)

    def test_input_errors_exit_2(self):
        for argv in (["run", "no-such-file.txt"], ["cmp", "1/0", "1/2"],
                     ["sort", "1/2", "x"]):
            with self.subTest(argv=argv):
                code, _, err = run_cli(argv)
                self.assertEqual(code, 2)
                self.assertTrue(err.startswith("error: "))

    def test_success_exit_0(self):
        code, out, _ = run_cli(["run", str(SAMPLES / "chains" / "01_addchain.txt")])
        self.assertEqual(code, 0)
        self.assertTrue(out.startswith("chain 01_addchain "))


if __name__ == "__main__":
    unittest.main()

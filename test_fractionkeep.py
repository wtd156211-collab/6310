#!/usr/bin/env python3
"""fractionkeep 的 unittest 测试。"""

import io
import math
import os
import tempfile
import unittest

import fractionkeep as fk

ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(ROOT, "samples")


class NormalizeTest(unittest.TestCase):
    def test_reduce(self):
        self.assertEqual(fk.normalize(2, 4), (1, 2))
        self.assertEqual(fk.normalize(-6, 9), (-2, 3))
        self.assertEqual(fk.normalize(7, 1), (7, 1))

    def test_zero_unique(self):
        self.assertEqual(fk.normalize(0, 5), (0, 1))
        self.assertEqual(fk.normalize(0, 1), (0, 1))

    def test_denominator_must_be_positive(self):
        with self.assertRaises(fk.InputError):
            fk.normalize(1, 0)
        with self.assertRaises(fk.InputError):
            fk.normalize(1, -3)

    def test_parse_value(self):
        self.assertEqual(fk.parse_value("2/4"), (1, 2))
        self.assertEqual(fk.parse_value("-0/7"), (0, 1))
        self.assertEqual(fk.parse_value("5"), (5, 1))
        self.assertEqual(fk.parse_value("-3"), (-3, 1))
        for bad in ("1/0", "1/-2", "a/b", "1/2/3", "", "1.5", "1_0"):
            with self.assertRaises(fk.InputError, msg=bad):
                fk.parse_value(bad)

    def test_digits(self):
        self.assertEqual(fk.digits(0), 1)
        self.assertEqual(fk.digits(-7), 1)
        self.assertEqual(fk.digits(999), 3)
        self.assertEqual(fk.pair_digits(-12, 345), 5)


class ArithmeticTest(unittest.TestCase):
    def test_add_sub(self):
        self.assertEqual(fk.apply_op("+", 1, 2, (1, 3)), (5, 6, 5, 6))
        self.assertEqual(fk.apply_op("-", 1, 2, (1, 2)), (0, 4, 0, 1))

    def test_mul_div(self):
        self.assertEqual(fk.apply_op("*", 2, 3, (3, 4)), (6, 12, 1, 2))
        self.assertEqual(fk.apply_op("/", 2, 3, (3, 4)), (8, 9, 8, 9))
        # 除以负数：未约分对分母挪号为正
        P, Q, np_, nq = fk.apply_op("/", 1, 2, (-1, 3))
        self.assertEqual((P, Q, np_, nq), (-3, 2, -3, 2))
        self.assertGreater(Q, 0)
        with self.assertRaises(fk.InputError):
            fk.apply_op("/", 1, 2, (0, 1))

    def test_power(self):
        self.assertEqual(fk.apply_op("^", 2, 3, 3), (8, 27, 8, 27))
        self.assertEqual(fk.apply_op("^", 2, 3, -2), (9, 4, 9, 4))
        self.assertEqual(fk.apply_op("^", 0, 1, 0), (1, 1, 1, 1))  # 0^0 = 1
        self.assertEqual(fk.apply_op("^", 0, 1, 5), (0, 1, 0, 1))
        # 负底数负指数：分母挪号为正
        P, Q, np_, nq = fk.apply_op("^", -2, 1, -3)
        self.assertEqual((np_, nq), (-1, 8))
        self.assertGreater(Q, 0)
        with self.assertRaises(fk.InputError):
            fk.apply_op("^", 0, 1, -1)

    def test_floordiv_floors_toward_negative_infinity(self):
        self.assertEqual(fk.apply_op("//", -1, 2, (1, 3)), (-3, 2, -2, 1))
        self.assertEqual(fk.apply_op("//", 7, 3, (1, 2))[2:], (4, 1))
        with self.assertRaises(fk.InputError):
            fk.apply_op("//", 1, 2, (0, 1))

    def test_mod_sign_follows_divisor(self):
        # -7/3 % 1/2 = 1/6
        self.assertEqual(fk.apply_op("%", -7, 3, (1, 2))[2:], (1, 6))
        # -2 % -5/7 = -4/7
        self.assertEqual(fk.apply_op("%", -2, 1, (-5, 7))[2:], (-4, 7))
        # 整除时余数为 0/1
        self.assertEqual(fk.apply_op("%", 3, 2, (1, 2))[2:], (0, 1))
        with self.assertRaises(fk.InputError):
            fk.apply_op("%", 1, 2, (0, 1))

    def test_compare(self):
        self.assertEqual(fk.compare((-1, 3), (-1, 2)), 1)  # -1/3 > -1/2
        self.assertEqual(fk.compare((-1, 7), (0, 1)), -1)
        self.assertEqual(fk.compare((2, 4), (1, 2)), 0)
        self.assertEqual(fk.compare((0, 1), (0, 1)), 0)


class ChainTest(unittest.TestCase):
    def test_load_chain_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            def write(text):
                path = os.path.join(tmp, "c.txt")
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(text)
                return path

            with self.assertRaises(fk.InputError):
                fk.load_chain(write("+ 1/2\n"))
            with self.assertRaises(fk.InputError):
                fk.load_chain(write("start 1/2\n"))
            with self.assertRaises(fk.InputError):
                fk.load_chain(write("start 1/2\n~ 1/3\n"))
            start, steps = fk.load_chain(write("start 1/2\n^ 1001\n"))
            with self.assertRaises(fk.InputError):
                fk.run_chain("c", start, steps)
            with self.assertRaises(fk.InputError):
                fk.load_chain(os.path.join(tmp, "missing.txt"))

    def test_every_step_normalized(self):
        chains_dir = os.path.join(SAMPLES, "chains")
        for fname in sorted(os.listdir(chains_dir)):
            start, raw_steps = fk.load_chain(os.path.join(chains_dir, fname))
            result = fk.run_chain(fname[:-4], start, raw_steps)
            for st in result.steps:
                self.assertGreater(st.q, 0, st)
                self.assertGreater(st.raw_q, 0, st)
                if st.p == 0:
                    self.assertEqual(st.q, 1, st)
                else:
                    self.assertEqual(math.gcd(abs(st.p), st.q), 1, st)
            self.assertLessEqual(result.peak, fk.BUDGET, fname)

    def test_known_summaries(self):
        expected = {
            "01_addchain": ("1/2", 38, "1592457339642613/485721041551200", 31, 34),
            "02_muldiv": ("2/3", 4000, "2/3", 13, 24),
            "03_coprime": ("1/1", 7,
                           "1130065282276738967948504620/1001056056190169315273666729", 56, 56),
            "04_nearint": ("3/2", 7,
                           "999999999991999999999981/999999999991999999999967", 48, 50),
            "05_signs": ("0/1", 13, "0/1", 4, 4),
            "06_scale": ("0/1", 12000, "4000/999983", 14, 26),
        }
        for name, (start_s, steps, final_s, peak, raw_peak) in expected.items():
            lines = fk.cmd_run([os.path.join(SAMPLES, "chains", name + ".txt")])
            head = lines[0].split()
            self.assertEqual(head[0], "chain")
            self.assertEqual(head[1], name)
            self.assertEqual(head[2], f"start={start_s}")
            self.assertEqual(head[3], f"steps={steps}")
            self.assertEqual(head[4], f"final={final_s}")
            self.assertEqual(head[5], f"peak={peak}")
            self.assertEqual(head[6], f"raw_peak={raw_peak}")
            self.assertEqual(head[7], f"budget={fk.BUDGET}")
            self.assertEqual(len(lines), steps + 1)


class CliTest(unittest.TestCase):
    def test_cmp(self):
        self.assertEqual(fk.cmd_cmp(["-1/7", "0/1"]), ["cmp=-1"])
        self.assertEqual(fk.cmd_cmp(["-3/2", "-7/3"]), ["cmp=1"])
        self.assertEqual(fk.cmd_cmp(["0/1", "0/9"]), ["cmp=0"])

    def test_sort_dedup_ascending(self):
        lines = fk.cmd_sort(["0/1", "-7/3", "1/6", "-1/1", "-1/2", "49/16", "1/1"])
        self.assertEqual(lines, ["-7/3", "-1/1", "-1/2", "0/1", "1/6", "1/1", "49/16"])
        # 同值不同写法去重
        self.assertEqual(fk.cmd_sort(["2/4", "1/2", "0/5", "-0/3"]), ["0/1", "1/2"])

    def test_usage_errors(self):
        for args in (["run"], ["run", "a", "b"], ["cmp", "1/2"], ["sort"]):
            with self.assertRaises(fk.InputError, msg=args):
                fk.dispatch(args)
        with self.assertRaises(fk.InputError):
            fk.dispatch(["report"])

    def test_main_exit_codes(self):
        import contextlib
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(fk.main(["badcmd"]), 2)
        self.assertTrue(err.getvalue().startswith("error: "))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(fk.main(["cmp", "1/2", "2/4"]), 0)
        self.assertEqual(out.getvalue(), "cmp=0\n")


class ReportTest(unittest.TestCase):
    def setUp(self):
        self.old_cwd = os.getcwd()
        os.chdir(ROOT)

    def tearDown(self):
        os.chdir(self.old_cwd)

    def test_report_ok_16(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "fractions.html")
            buf = io.StringIO()
            self.assertEqual(fk.cmd_report(["samples", out], out=buf), 0)
            self.assertEqual(buf.getvalue(), "ok=16\n")
            with open(out, encoding="utf-8") as fh:
                page = fh.read()
            for name in ("01_addchain", "02_muldiv", "03_coprime",
                         "04_nearint", "05_signs", "06_scale"):
                self.assertIn(name, page)
            self.assertIn("peak=56", page)  # 03_coprime 离预算最近
            self.assertNotIn("<script", page)
            self.assertNotIn("http://", page)
            self.assertNotIn("https://", page)

    def test_report_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            pages = []
            for i in range(2):
                out = os.path.join(tmp, f"r{i}.html")
                fk.cmd_report(["samples", out], out=io.StringIO())
                with open(out, "rb") as fh:
                    pages.append(fh.read())
            self.assertEqual(pages[0], pages[1])

    def test_report_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "expected"))
            os.makedirs(os.path.join(tmp, "chains"))
            with open(os.path.join(tmp, "chains", "x.txt"), "w", encoding="utf-8") as fh:
                fh.write("start 1/2\n+ 1/3\n")
            with open(os.path.join(tmp, "expected", "x.txt"), "w", encoding="utf-8") as fh:
                fh.write("? run " + os.path.join(tmp, "chains", "x.txt") + "\nwrong\n")
            buf = io.StringIO()
            rc = fk.cmd_report([tmp, os.path.join(tmp, "out.html")], out=buf)
            self.assertEqual(rc, 1)
            self.assertTrue(buf.getvalue().startswith("mismatch="))
            self.assertFalse(os.path.exists(os.path.join(tmp, "out.html")))


if __name__ == "__main__":
    unittest.main()

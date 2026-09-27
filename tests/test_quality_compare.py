"""compare_tier1 statistics: exact McNemar, paired SE, group rule, verdicts."""
from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "quality"))
import compare_tier1 as C  # noqa: E402


def rows(group: str, task: str, outcomes: list[int], start: int = 0, error_ids=()) -> dict:
    out = {}
    for k, v in enumerate(outcomes):
        r = {"task": task, "group": group, "id": start + k, "correct": v}
        if start + k in error_ids:
            r["error"] = "HTTP 500"
        out[(task, str(start + k))] = r
    return out


class McNemarTests(unittest.TestCase):
    def test_exact_values(self):
        self.assertEqual(C.mcnemar_p(0, 0), 1.0)
        self.assertAlmostEqual(C.mcnemar_p(10, 0), 2 * 0.5 ** 10)
        self.assertAlmostEqual(C.mcnemar_p(0, 10), 2 * 0.5 ** 10)
        # b=8, c=2: 2 * P(X<=2 | n=10) = 2 * (1 + 10 + 45) / 1024
        self.assertAlmostEqual(C.mcnemar_p(8, 2), 2 * 56 / 1024)
        self.assertEqual(C.mcnemar_p(5, 5), 1.0)
        self.assertLess(C.mcnemar_p(400, 330), 0.02)  # large n stays finite

    def test_paired_se(self):
        st = C.paired_stats([(1, 0), (0, 0), (1, 1), (0, 1)])
        self.assertEqual((st["diff"], st["a_only"], st["b_only"]), (0.0, 1, 1))
        d = [-1, 0, 0, 1]
        se = math.sqrt(sum(x * x for x in d) / 3 / 4)
        self.assertAlmostEqual(st["se"], round(100 * se, 3))


class CompareTests(unittest.TestCase):
    def test_identical_pass(self):
        a = rows("GSM8K", "gsm8k", [1, 0] * 100)
        res = C.compare(a, a)
        self.assertEqual(res["verdict"], "PASS")
        self.assertEqual(res["pooled"]["diff"], 0.0)

    def test_group_fail_needs_both_p_and_drop(self):
        # 200 items: A right on 180; B loses 20 of them -> drop 10 points, b=20 c=0
        a = rows("GSM8K", "gsm8k", [1] * 180 + [0] * 20)
        b = rows("GSM8K", "gsm8k", [1] * 160 + [0] * 40)
        other = rows("MMLU-Pro", "mmlu_pro", [1] * 800)
        res = C.compare({**a, **other}, {**b, **other})
        g = res["groups"]["GSM8K"]
        self.assertTrue(g["fail"])
        self.assertAlmostEqual(g["acc_a"] - g["acc_b"], 10.0)
        self.assertGreaterEqual(res["pooled"]["diff"], -2.0)  # pooled passes, group still fails
        self.assertEqual(res["verdict"], "FAIL")

    def test_small_significant_drop_is_not_a_fail(self):
        # drop of 4 points (8/200) with b=8, c=0 -> p ~ 0.0078 < 0.01 but drop < 5
        a = rows("BFCL", "bfcl", [1] * 200)
        b = rows("BFCL", "bfcl", [1] * 192 + [0] * 8)
        g = C.compare(a, b)["groups"]["BFCL"]
        self.assertLess(g["p_mcnemar"], 0.01)
        self.assertFalse(g["fail"])

    def test_large_noisy_drop_is_not_a_fail(self):
        # drop of 6 points but many discordant pairs both ways -> p > 0.01
        a = rows("Vision", "chartqa", [1] * 50 + [0] * 50)
        b = rows("Vision", "chartqa", [0] * 28 + [1] * 22 + [1] * 22 + [0] * 28)
        g = C.compare(a, b)["groups"]["Vision"]
        self.assertAlmostEqual(g["acc_a"] - g["acc_b"], 6.0)
        self.assertGreater(g["p_mcnemar"], 0.01)
        self.assertFalse(g["fail"])

    def test_pooled_floor(self):
        a = rows("IFEval", "ifeval", [1] * 1000)
        b = rows("IFEval", "ifeval", [1] * 979 + [0] * 21)
        res = C.compare(a, b)
        self.assertAlmostEqual(res["pooled"]["diff"], -2.1)
        self.assertFalse(res["pooled"]["pass"])
        self.assertEqual(res["verdict"], "FAIL")

    def test_errors_and_invalid(self):
        a = rows("GSM8K", "gsm8k", [1] * 100)
        b = rows("GSM8K", "gsm8k", [1] * 100, error_ids={3})
        res = C.compare(a, b)
        self.assertEqual((res["paired"], res["error_pairs"], res["verdict"]), (99, 1, "PASS"))
        b = rows("GSM8K", "gsm8k", [1] * 100, error_ids=set(range(5)))
        self.assertEqual(C.compare(a, b)["verdict"], "INVALID")

    def test_load_last_row_wins_and_cli(self):
        with tempfile.TemporaryDirectory() as d:
            pa, pb = Path(d, "a.jsonl"), Path(d, "b.jsonl")
            pa.write_text("\n".join(json.dumps(r) for r in [
                {"meta": {}}, {"task": "gsm8k", "group": "GSM8K", "id": 1, "correct": 0, "error": "x"},
                {"task": "gsm8k", "group": "GSM8K", "id": 1, "correct": 1}]) + "\n")
            pb.write_text(json.dumps({"task": "gsm8k", "group": "GSM8K", "id": 1, "correct": 1}) + "\n")
            self.assertEqual(C.load(pa)[("gsm8k", "1")]["correct"], 1)
            self.assertEqual(C.main([str(pa), str(pb), "--json", str(Path(d, "c.json"))]), 0)
            self.assertEqual(json.loads(Path(d, "c.json").read_text())["verdict"], "PASS")


if __name__ == "__main__":
    unittest.main()

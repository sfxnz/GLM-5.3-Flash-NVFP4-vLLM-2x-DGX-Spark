"""Tier-0 math, judges and a record/compare round trip against a fake serve.

  python3 -m unittest discover -s tests -p 'test_quality*.py'
"""
from __future__ import annotations

import json
import math
import re
import sys
import tempfile
import unittest
from array import array
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "quality"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import tier0  # noqa: E402
from test_quality_fakes import FakeServe  # noqa: E402


# ----------------------------------------------------------------- tests

class KLTests(unittest.TestCase):
    def test_identical_is_zero(self):
        ids, lp = [1, 2, 3], [math.log(0.5), math.log(0.3), math.log(0.1)]
        self.assertAlmostEqual(tier0.kl_topk(ids, lp, ids, lp), 0.0, places=12)

    def test_hand_value(self):
        # p = (.5, .3 | rest .2), q = (.4, .4 | rest .2)
        p = [math.log(0.5), math.log(0.3)]
        q = [math.log(0.4), math.log(0.4)]
        want = 0.5 * math.log(0.5 / 0.4) + 0.3 * math.log(0.3 / 0.4)
        self.assertAlmostEqual(tier0.kl_topk([1, 2], p, [1, 2], q), want, places=9)

    def test_missing_token_gets_bounded_fill(self):
        p_ids, p_lp = [1, 2], [math.log(0.6), math.log(0.3)]
        q_ids, q_lp = [1, 3], [math.log(0.6), math.log(0.35)]
        kl = tier0.kl_topk(p_ids, p_lp, q_ids, q_lp)
        # token 2 gets min(q_min=.35, unreturned .05/2) = .025
        want = 0.6 * 0 + 0.3 * math.log(0.3 / 0.025) + 0.1 * math.log(0.1 / (1 - 0.6 - 0.025))
        self.assertAlmostEqual(kl, want, places=9)
        # an exact extra logprob for the missing token replaces the fill
        kl2 = tier0.kl_topk(p_ids, p_lp, q_ids, q_lp, {2: math.log(0.3)})
        self.assertLess(kl2, kl)

    @staticmethod
    def doc(lp, swap_first=False):
        k = tier0.TOPK
        ids = [100 + j for j in range(k)] * 2
        lps = [-1.0 - 0.2 * j for j in range(k)] * 2
        if swap_first:
            ids[0], ids[1] = ids[1], ids[0]
        return {"id": "d", "domain": "prose", "ids": [5, 6, 100, 100], "start": 2,
                "lp": array("f", lp), "tk_ids": array("i", ids), "tk_lp": array("f", lps)}

    def test_compare_captures(self):
        ref = self.doc([-1.0, -2.0])
        same = tier0.compare_captures([ref], [self.doc([-1.0, -2.0])])
        self.assertEqual((same["delta"], same["top1_agree"], same["kl"]), (0.0, 1.0, 0.0))
        res = tier0.compare_captures([ref], [self.doc([-1.5, -2.5], swap_first=True)])
        self.assertAlmostEqual(res["delta"], 0.5)
        self.assertAlmostEqual(res["top1_agree"], 0.5)
        self.assertGreater(res["kl"], 0)
        other = dict(self.doc([-1.0, -2.0]), ids=[5, 6, 7, 9])
        self.assertEqual(tier0.compare_captures([ref], [other])["mismatched_docs"], ["d"])

    def test_pack_roundtrip(self):
        cap = [self.doc([-1.25, -2.5])]
        back = tier0.unpack_capture(json.loads(json.dumps(tier0.pack_capture(cap))))
        self.assertEqual(back[0]["tk_ids"], cap[0]["tk_ids"])
        self.assertEqual(back[0]["lp"], cap[0]["lp"])
        self.assertEqual(tier0.compare_captures(cap, back)["kl"], 0.0)


class JudgeTests(unittest.TestCase):
    def test_hazard_and_divergence(self):
        self.assertIsNone(tier0.first_divergence([1, 2], [1, 2]))
        self.assertEqual(tier0.first_divergence([1, 2, 3], [1, 5, 3]), 1)
        h = tier0.hazard([([1, 2, 3, 4], [1, 2, 3, 4]), ([1, 2], [1, 9])])
        self.assertEqual((h["diverged"], h["hazard"]), (1, round(1 / 6, 6)))

    def test_count(self):
        good = ", ".join(str(i) for i in range(1, 201))
        self.assertTrue(tier0.count_ok(good)["pass"])
        bad = good.replace("117", "171")
        r = tier0.count_ok(bad)
        self.assertFalse(r["pass"])
        self.assertEqual(r["first_bad"], 116)

    def test_kwarg_cell(self):
        ok_off = {"content": "Paris", "reasoning": "", "finish_reason": "stop"}
        self.assertTrue(tier0.judge_kwarg_cell(ok_off, think=False)["pass"])
        leak = {"content": "Okay, the user asks for the capital. Paris", "reasoning": "", "finish_reason": "stop"}
        self.assertFalse(tier0.judge_kwarg_cell(leak, think=False)["pass"])
        # QUAL-2 shape: thinking requested, answer lands in reasoning, content empty
        lost = {"content": "", "reasoning": "The capital is Paris.", "finish_reason": "stop"}
        cell = tier0.judge_kwarg_cell(lost, think=True)
        self.assertFalse(cell["pass"])
        self.assertFalse(cell["checks"]["content"])
        on = {"content": "Paris", "reasoning": "France's capital is Paris.", "finish_reason": "stop"}
        self.assertTrue(tier0.judge_kwarg_cell(on, think=True)["pass"])
        self.assertFalse(tier0.judge_kwarg_cell(on, think=False)["pass"])
        # effort low: <think> opened, model closed it at once (live 2026-09-27); allowed only there
        closed = {"content": "Paris", "reasoning": "", "finish_reason": "stop"}
        self.assertFalse(tier0.judge_kwarg_cell(closed, think=True)["pass"])
        self.assertTrue(tier0.judge_kwarg_cell(closed, think=True, reasoning_optional=True)["pass"])
        self.assertTrue(tier0.judge_kwarg_cell(on, think=True, reasoning_optional=True)["pass"])
        # with no reasoning, content is still checked for chain-of-thought
        self.assertFalse(tier0.judge_kwarg_cell(leak, think=True, reasoning_optional=True)["pass"])

    def test_utf8(self):
        rows = "\n".join(f"| {n} | {n * n} | {n ** 3} | {tier0.chinese_numeral(n)} |" for n in range(1, 41))
        table = "| n | n² | n³ | 中文 |\n|---|---|---|---|\n" + rows
        self.assertTrue(tier0.judge_utf8({"content": table, "finish_reason": "stop"})["pass"])
        self.assertEqual(tier0.chinese_numeral(23), "二十三")
        self.assertEqual(tier0.chinese_numeral(10), "十")
        self.assertEqual(tier0.chinese_numeral(40), "四十")
        bad = table.replace("二十三", "二�三")
        r = tier0.judge_utf8({"content": bad, "finish_reason": "stop"})
        self.assertEqual((r["pass"], r["fffd"], r["chinese_numeral_errors"]), (False, 1, 1))
        wrong_sq = table.replace("| 7 | 49 |", "| 7 | 48 |")
        self.assertEqual(tier0.judge_utf8({"content": wrong_sq, "finish_reason": "stop"})["square_errors"], 1)

    def test_tool_item(self):
        item = {"id": "t", "expect": {"name": "f", "args": {"city": "Paris", "n": 2}}, "ignore": ["note"]}
        good = {"tool_calls": [{"name": "f", "arguments": '{"city": "paris", "n": 2.0, "note": "x"}'}]}
        r = tier0.judge_tool_item(item, good)
        self.assertTrue(r["json_valid"] and r["args_ok"])
        self.assertFalse(tier0.judge_tool_item(item, {"tool_calls": [{"name": "f", "arguments": '{"city": '}]})["json_valid"])
        self.assertFalse(tier0.judge_tool_item(item, {"tool_calls": []})["json_valid"])

    def test_sse_parse(self):
        out = {"content": "hello world", "reasoning": "think", "tool_calls": [{"name": "f", "arguments": '{"a": 1}'}]}
        got = common.parse_sse(FakeServe.sse(out).split(b"\n"))
        self.assertEqual((got["content"], got["reasoning"]), ("hello world", "think"))
        self.assertEqual(got["tool_calls"], [{"name": "f", "arguments": '{"a": 1}'}])
        self.assertEqual(got["finish_reason"], "stop")

    def test_needle_is_fixed(self):
        self.assertEqual(tier0.needle_code(1000), tier0.needle_code(1000))
        self.assertEqual(tier0.filler(7, 3), tier0.filler(7, 3))


class CriteriaTests(unittest.TestCase):
    REF = {"nll": {"rerun": {"abs_delta": 0.002, "top1_agree": 0.995}}, "greedy": {"aa": {"hazard": 0.0}}}

    def _nll(self, delta, top1=0.996, kl=5e-4):
        return {"nll": {"vs_ref": {"delta": delta, "top1_agree": top1, "kl": kl, "mismatched_docs": []}}}

    def test_nll_floor_and_sigma(self):
        rows = {r["name"]: r for r in tier0.criteria(self._nll(0.0049), self.REF, None, set())}
        self.assertTrue(rows["nll.delta"]["pass"])
        self.assertEqual(rows["nll.delta"]["limit"], 0.006)  # 3 * sigma beats the 0.005 floor
        rows = {r["name"]: r for r in tier0.criteria(self._nll(-0.0061), self.REF, None, set())}
        self.assertFalse(rows["nll.delta"]["pass"])

    def test_stage_thresholds(self):
        rows = {r["name"]: r for r in tier0.criteria(self._nll(0.0, top1=0.985, kl=2e-3), self.REF, "nvfp4", set())}
        self.assertTrue(rows["nll.top1_stage"]["pass"] and rows["nll.kl_stage"]["pass"])
        self.assertFalse(rows["nll.top1_rerun"]["pass"])
        rows = {r["name"]: r for r in tier0.criteria(self._nll(0.0, top1=0.985, kl=2e-3), self.REF, "fp8", set())}
        self.assertFalse(rows["nll.top1_stage"]["pass"] or rows["nll.kl_stage"]["pass"])

    def test_behavioural_and_skip(self):
        comps = {"tools": {"json_valid": 0.96}, "count": {"pass": True, "n_numbers": 200},
                 "greedy": {"golden": {"hazard": 0.02}}, "utf8": {"error": "boom"}}
        rows = tier0.criteria(comps, self.REF, None, {"greedy.hazard"})
        by = {r["name"]: r for r in rows}
        self.assertFalse(by["tools.json_valid"]["pass"])
        self.assertTrue(by["count"]["pass"])
        self.assertTrue(by["greedy.hazard"]["skipped"])
        self.assertFalse(by["utf8.error"]["pass"])
        line = tier0.verdict("compare", "r", rows)
        self.assertTrue(line.startswith("FAIL"))
        self.assertIn("tools.json_valid", line)
        self.assertNotIn("greedy.hazard", line)

    def test_kwargs_split(self):
        # today's template: only the two thinking:true cells fail (QUAL-2)
        cells = [{"cell": f"{n}.{s}", "pass": not n.startswith("thinking_true")}
                 for n, _, _ in tier0.KWARG_SHAPES for s in ("block", "stream")]
        rows = tier0.criteria({"kwargs": {"pass": False, "cells": cells}}, None, None, {"kwargs.thinking_alias"})
        by = {r["name"]: r for r in rows}
        self.assertEqual((by["kwargs.core"]["pass"], by["kwargs.core"]["value"], by["kwargs.core"]["limit"]),
                         (True, 10, 10))
        self.assertFalse(by["kwargs.thinking_alias"]["pass"])
        self.assertTrue(tier0.verdict("compare", "r", rows).startswith("PASS"))
        # chain-of-thought leaking with thinking off still fails the core gate
        cells[2]["pass"] = False  # enable_thinking_false.block
        rows = tier0.criteria({"kwargs": {"cells": cells}}, None, None, {"kwargs.thinking_alias"})
        self.assertIn("kwargs.core", tier0.verdict("compare", "r", rows))


def probe_chat(body):
    """Answers every Tier-0 probe correctly, the way a healthy serve would."""
    text = body["messages"][-1]["content"]
    if body.get("tools"):
        spec = json.loads((tier0.DATA / "tools50.json").read_text())
        item = next(i for i in spec["items"] if i["prompt"] == text)
        args = {**item["expect"]["args"], **{k: "x" for k in item.get("ignore", [])}}
        return {"content": "", "tool_calls": [{"name": item["expect"]["name"], "arguments": json.dumps(args)}]}
    if text == tier0.KWARG_PROMPT:
        kw = body.get("chat_template_kwargs") or {}
        think = kw.get("enable_thinking", kw.get("thinking", body.get("reasoning_effort", "none") != "none"))
        return {"content": "Paris", "reasoning": "France's capital." if think else ""}
    if text == tier0.UTF8_PROMPT:
        rows = "\n".join(f"| {n} | {n * n} | {n ** 3} | {tier0.chinese_numeral(n)} |" for n in range(1, 41))
        return {"content": "| n | n² | n³ | 中文 |\n|---|---|---|---|\n" + rows}
    m = re.search(r"The vault code assigned to (.+?) is (\S+)\.", text)
    if m:
        return {"content": m.group(2)}
    return count_chat(body)


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.s = FakeServe(probe_chat)
        self.c = common.Client(self.s.url)

    def tearDown(self):
        self.s.close()

    def test_kwargs_matrix(self):
        res = tier0.run_kwargs(self.c, default_thinking=False)
        self.assertEqual((res["passed"], res["total"]), (12, 12), [x for x in res["cells"] if not x["pass"]])
        sent = [b for p, b in self.s.requests if p == "/v1/chat/completions"]
        self.assertEqual(sum(1 for b in sent if b.get("stream")), 6)
        self.assertTrue(any("chat_template_kwargs" not in b and "reasoning_effort" not in b for b in sent))

    def test_kwargs_matrix_effort_low_closes_thinking(self):
        def chat(body):
            out = probe_chat(body)
            if body.get("reasoning_effort") == "low":
                out["reasoning"] = ""
            return out

        s = FakeServe(chat)
        try:
            res = tier0.run_kwargs(common.Client(s.url), default_thinking=False)
        finally:
            s.close()
        self.assertEqual((res["passed"], res["total"]), (12, 12), [x for x in res["cells"] if not x["pass"]])

    def test_utf8_tools_needle(self):
        self.assertTrue(tier0.run_utf8(self.c)["pass"])
        tools = tier0.run_tools(self.c)
        self.assertEqual((tools["n"], tools["json_valid"], tools["args_ok"]), (50, 1.0, 1.0))
        needle = tier0.run_needle(self.c, (3000,))
        self.assertTrue(needle["pass"], needle)
        self.assertEqual(needle["per_length"], {"3000": 3})


def count_chat(body):
    text = body["messages"][-1]["content"]
    if text.startswith("Count from 1 to 200"):
        return {"content": ", ".join(str(i) for i in range(1, 201))}
    return {"content": "reply", "token_ids": [len(text) % 7, 1, 2, 3]}


class RoundTripTests(unittest.TestCase):
    """record then compare through real HTTP, on a 3-doc corpus."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._saved = (tier0.EVALS_DIR, tier0.corpus)
        tier0.EVALS_DIR = Path(self.tmp.name)
        docs = [{"id": f"d{k}", "domain": "prose", "text": " ".join(f"w{k}{j}" for j in range(60))} for k in range(3)]
        tier0.corpus = lambda: (docs, "sha-test")

    def tearDown(self):
        tier0.EVALS_DIR, tier0.corpus = self._saved
        self.tmp.cleanup()

    def _run(self, serve, *argv):
        out = Path(self.tmp.name) / "t0.json"
        code = tier0.main([*argv, "--url", serve.url, "--only", "nll,greedy,count", "--out", str(out)])
        return code, json.loads(out.read_text())

    def test_record_then_compare(self):
        s = FakeServe(count_chat)
        try:
            code, rec = self._run(s, "record", "--name", "base")
            self.assertEqual(code, 0, rec["verdict"])
            ref = json.loads((Path(self.tmp.name) / "ref/base.json").read_text())
            self.assertEqual(ref["nll"]["rerun"]["abs_delta"], 0.0)
            doc0 = tier0.unpack_capture(ref["nll"]["capture"])[0]
            self.assertEqual(len(doc0["tk_ids"]), 20 * len(doc0["lp"]))
            self.assertEqual(len(doc0["lp"]), 60 - tier0.NLL_SKIP)
            self.assertEqual(doc0["start"], 2 + tier0.NLL_SKIP)
            nll_req = [b for p, b in s.requests if p == "/v1/completions"][0]
            self.assertEqual((nll_req["prompt_logprobs"], nll_req["max_tokens"]), (20, 1))
            code, cmp_ = self._run(s, "compare", "--ref", "base", "--stage", "fp8")
            self.assertEqual(code, 0, cmp_["verdict"])
            vs = cmp_["components"]["nll"]["vs_ref"]
            self.assertEqual((vs["delta"], vs["top1_agree"], vs["kl"]), (0.0, 1.0, 0.0))
        finally:
            s.close()
        worse = FakeServe(count_chat, nll_shift=0.01, swap_top=True)
        try:
            code, cmp_ = self._run(worse, "compare", "--ref", "base", "--stage", "fp8")
            self.assertEqual(code, 1)
            failed = {c["name"] for c in cmp_["criteria"] if not c["pass"]}
            self.assertTrue({"nll.delta", "nll.top1_stage", "nll.top1_rerun", "nll.kl_stage"} <= failed, failed)
            self.assertTrue(cmp_["verdict"].startswith("FAIL"))
        finally:
            worse.close()

    def test_missing_token_ids_fails_loudly(self):
        def no_ids(body):
            out = count_chat(body)
            out.pop("token_ids", None)
            return out

        s = FakeServe(no_ids)
        try:
            code, rec = self._run(s, "record", "--name", "noids")
        finally:
            s.close()
        self.assertEqual(code, 1)
        self.assertIn("greedy.error", {c["name"] for c in rec["criteria"] if not c["pass"]})
        self.assertFalse((Path(self.tmp.name) / "ref/noids.json").exists())  # no vacuous reference


if __name__ == "__main__":
    unittest.main()

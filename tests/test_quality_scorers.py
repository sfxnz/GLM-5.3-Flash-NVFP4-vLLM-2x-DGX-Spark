"""Tier-1 scorers and IFEval checks on hand-made examples."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "quality"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import ifeval  # noqa: E402
import scorers as S  # noqa: E402
import tier1  # noqa: E402
import vision  # noqa: E402
from test_quality_fakes import FakeServe  # noqa: E402


class GSM8KTests(unittest.TestCase):
    def test_gold_and_extract(self):
        self.assertEqual(S.gsm8k_gold("step\n#### 1,234"), "1234")
        self.assertTrue(S.gsm8k_correct("48 + 24 = 72\nAnswer: 72", "72"))
        self.assertTrue(S.gsm8k_correct("So the answer is $1,080.", "1080"))
        self.assertTrue(S.gsm8k_correct("Answer: **18.0**", "18"))
        self.assertFalse(S.gsm8k_correct("Answer: 17", "18"))
        self.assertFalse(S.gsm8k_correct("no number here", "18"))
        self.assertEqual(S.gsm8k_extract("It costs 5 then 7. Answer: 12\nDone"), "12")


class LetterTests(unittest.TestCase):
    def test_mmlu_pro(self):
        self.assertEqual(S.mmlu_pro_extract("Reasoning... The answer is (J)."), "J")
        self.assertEqual(S.mmlu_pro_extract("the answer is C"), "C")
        self.assertEqual(S.mmlu_pro_extract("Answer: (b)"), "B")
        self.assertEqual(S.mmlu_pro_extract("I pick\nD"), "D")
        self.assertIsNone(S.mmlu_pro_extract("I am not sure"))
        self.assertIsNone(S.mmlu_pro_extract("The answer is (J)", n_options=4))
        # a hedge word is not a letter; the lone final letter still counts
        self.assertEqual(S.mmlu_pro_extract("the answer is a bit unclear; I would pick\nC"), "C")
        self.assertIsNone(S.mmlu_pro_extract("the answer is a bit unclear"))
        self.assertEqual(S.mmlu_pro_extract("The answer is a bit unclear. The answer is (e)."), "E")
        self.assertEqual(S.mmlu_pro_extract("The answer is B."), "B")

    def test_mmmu(self):
        opts = ["Aurelia", "Matilda", "Hermione", "Juno"]
        self.assertEqual(S.mmmu_extract("C", opts), "C")
        self.assertEqual(S.mmmu_extract("(B) Matilda", opts), "B")
        self.assertEqual(S.mmmu_extract("C. Hermione", opts), "C")
        self.assertEqual(S.mmmu_extract("The answer is D", opts), "D")
        self.assertEqual(S.mmmu_extract("<|begin_of_box|>A<|end_of_box|>", opts), "A")
        self.assertEqual(S.mmmu_extract("it is hermione", opts), "C")
        self.assertEqual(S.mmmu_options("['a', 'b']"), ["a", "b"])


class VisionScorerTests(unittest.TestCase):
    def test_chartqa_relaxed(self):
        self.assertTrue(S.chartqa_correct("14", ["14"]))
        self.assertTrue(S.chartqa_correct("104.9", ["100"]))   # within 5%
        self.assertFalse(S.chartqa_correct("106", ["100"]))
        self.assertTrue(S.chartqa_correct("45%", ["45"]))
        self.assertTrue(S.chartqa_correct("Yes.", ["Yes"]))
        self.assertFalse(S.chartqa_correct("No", ["Yes"]))
        self.assertTrue(S.chartqa_correct("0", ["0"]))
        self.assertTrue(S.chartqa_correct("<|begin_of_box|>1,200<|end_of_box|>", ["1200"]))

    def test_ocrbench(self):
        self.assertTrue(S.ocrbench_correct("The text reads CENTRE.", ["CENTRE"], "IIIT5K"))
        self.assertFalse(S.ocrbench_correct("CENTER", ["CENTRE"], "IIIT5K"))
        self.assertTrue(S.ocrbench_correct("x ^ { 2 } + 1", ["x^{2}+1"], "HME100k"))


class BFCLTests(unittest.TestCase):
    FUNCS = [{"name": "triangle_properties.get", "description": "d", "parameters": {
        "type": "dict", "properties": {
            "side1": {"type": "integer"}, "scale": {"type": "float"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "opts": {"type": "dict", "properties": {"deep": {"type": "boolean"}}},
            "get_area": {"type": "boolean", "default": True, "optional": True}},
        "required": ["side1"]}}]
    TRUTH = [{"triangle_properties.get": {"side1": [5], "scale": [1.5, ""], "tags": [["a b", "c"], ""],
                                          "get_area": ["", True]}}]

    def test_tool_conversion(self):
        (tool,) = S.bfcl_tools(self.FUNCS)
        fn = tool["function"]
        self.assertEqual(fn["name"], "triangle_properties_get")
        props = fn["parameters"]["properties"]
        self.assertEqual(fn["parameters"]["type"], "object")
        self.assertEqual((props["scale"]["type"], props["opts"]["type"]), ("number", "object"))
        self.assertEqual(props["tags"]["items"], {"type": "string"})
        self.assertNotIn("optional", props["get_area"])

    def call(self, args):
        return [{"name": "triangle_properties_get", "arguments": json.dumps(args)}]

    def test_match(self):
        self.assertTrue(S.bfcl_correct(self.call({"side1": 5}), self.TRUTH)[0])
        self.assertTrue(S.bfcl_correct(self.call({"side1": 5, "scale": 1.5, "tags": ["A-B", "c"],
                                                  "get_area": True}), self.TRUTH)[0])
        self.assertTrue(S.bfcl_correct(self.call({"side1": 5.0}), self.TRUTH)[0])
        nested = [{"triangle_properties.get": {"side1": [5], "opts": [{"deep": [True]}]}}]
        self.assertTrue(S.bfcl_correct(self.call({"side1": 5, "opts": {"deep": True}}), nested)[0])
        self.assertFalse(S.bfcl_correct(self.call({"side1": 5, "opts": {"deep": False}}), nested)[0])

    def test_mismatch(self):
        self.assertFalse(S.bfcl_correct(self.call({"side1": 6}), self.TRUTH)[0])
        self.assertFalse(S.bfcl_correct(self.call({"scale": 1.5}), self.TRUTH)[0])          # required missing
        self.assertFalse(S.bfcl_correct(self.call({"side1": 5, "color": "red"}), self.TRUTH)[0])  # unknown arg
        self.assertFalse(S.bfcl_correct(self.call({"side1": 5, "get_area": False}), self.TRUTH)[0])
        self.assertFalse(S.bfcl_correct([], self.TRUTH)[0])
        self.assertFalse(S.bfcl_correct([{"name": "other", "arguments": "{}"}], self.TRUTH)[0])
        self.assertFalse(S.bfcl_correct([{"name": "triangle_properties_get", "arguments": "{bad"}], self.TRUTH)[0])


class IFEvalTests(unittest.TestCase):
    def c(self, iid, resp, **kw):
        return ifeval.check(iid, resp, kw)

    def test_checks(self):
        self.assertTrue(self.c("punctuation:no_comma", "no commas here"))
        self.assertFalse(self.c("punctuation:no_comma", "a, b"))
        self.assertTrue(self.c("length_constraints:number_words", "one two three", relation="less than", num_words=4))
        self.assertFalse(self.c("length_constraints:number_words", "one two three", relation="at least", num_words=4))
        self.assertTrue(self.c("length_constraints:number_sentences", "Hi there. Mr. Smith came. Bye!",
                               relation="less than", num_sentences=4))
        self.assertEqual(ifeval.count_sentences("Hi there. Mr. Smith came. Bye!"), 3)
        self.assertTrue(self.c("detectable_format:number_highlighted_sections", "*a* and **b**", num_highlights=2))
        self.assertTrue(self.c("detectable_format:title", "<<My Title>>\nbody"))
        self.assertFalse(self.c("detectable_format:title", "<< >>"))
        self.assertTrue(self.c("detectable_format:json_format", '```json\n{"a": 1}\n```'))
        self.assertTrue(self.c("detectable_format:number_bullet_lists", "* a\n* b\n- c", num_bullets=3))
        self.assertTrue(self.c("detectable_content:postscript", "text\nP.S. more", postscript_marker="P.S."))
        self.assertTrue(self.c("detectable_content:number_placeholders", "[name] at [address]", num_placeholders=2))
        self.assertTrue(self.c("length_constraints:number_paragraphs", "a\n***\nb", num_paragraphs=2))
        self.assertFalse(self.c("length_constraints:number_paragraphs", "a\n***\n***\nb", num_paragraphs=2))
        self.assertTrue(self.c("length_constraints:nth_paragraph_first_word", "Weekend plans.\n\nSecond.",
                               num_paragraphs=2, nth_paragraph=1, first_word="weekend"))
        self.assertTrue(self.c("detectable_format:multiple_sections", "SECTION 1 a SECTION 2 b",
                               section_spliter="SECTION", num_sections=2))
        self.assertTrue(self.c("combination:two_responses", "one\n******\ntwo"))
        self.assertFalse(self.c("combination:two_responses", "same\n******\nsame"))
        self.assertTrue(self.c("combination:repeat_prompt", "Write X. Answer", prompt_to_repeat="write x."))
        self.assertTrue(self.c("startend:end_checker", 'Bye. Any other questions?"', end_phrase="Any other questions?"))
        self.assertTrue(self.c("startend:quotation", '"quoted"'))
        self.assertTrue(self.c("change_case:english_capital", "ALL CAPS HERE"))
        self.assertFalse(self.c("change_case:english_capital", "ВСЕ ЗАГЛАВНЫЕ"))
        self.assertTrue(self.c("change_case:english_lowercase", "all lower"))
        self.assertTrue(self.c("change_case:capital_word_frequency", "NASA and ESA", capital_relation="at least",
                               capital_frequency=2))
        self.assertTrue(self.c("keywords:existence", "Correlated data", keywords=["correlated"]))
        self.assertTrue(self.c("keywords:frequency", "story Story", keyword="story", relation="at least", frequency=2))
        self.assertFalse(self.c("keywords:forbidden_words", "a rock", forbidden_words=["rock"]))
        self.assertTrue(self.c("keywords:forbidden_words", "rocky road", forbidden_words=["rock"]))
        self.assertTrue(self.c("keywords:letter_frequency", "###", letter="#", let_relation="at least", let_frequency=3))
        self.assertTrue(self.c("detectable_format:constrained_response", "My answer is yes."))

    def test_none_kwargs_and_score(self):
        item = {"instruction_id_list": ["punctuation:no_comma", "startend:quotation"],
                "kwargs": [{}, {"unused": None}]}
        self.assertEqual(ifeval.score(item, '"ok"'), {"correct": 1, "instructions": [True, True]})
        self.assertEqual(ifeval.score(item, '"a, b"')["correct"], 0)

    def test_every_supported_id_has_a_check(self):
        ids = json.loads((Path(tier1.__file__).parent / "data/tier1_ids.json").read_text())
        self.assertEqual(len(ids["ifeval"]), 150)
        self.assertEqual(len(set(ifeval.CHECKS)), 24)


class Tier1BuildTests(unittest.TestCase):
    def test_prompts(self):
        msgs, extra, mt = tier1.build("mmlu_pro", {"category": "law", "question": "Q?", "options": ["x", "y"]})
        self.assertIn("A. x\nB. y", msgs[0]["content"])
        self.assertIn('"The answer is (X)"', msgs[0]["content"])
        msgs, extra, mt = tier1.build("bfcl", {"messages": [{"role": "user", "content": "hi"}],
                                               "functions": BFCLTests.FUNCS})
        self.assertEqual(extra["tool_choice"], "auto")
        self.assertEqual(extra["tools"][0]["function"]["name"], "triangle_properties_get")

    def test_summarize_last_row_wins(self):
        rows = [{"meta": {}}, {"task": "gsm8k", "id": 1, "correct": 0, "error": "x"},
                {"task": "gsm8k", "id": 1, "correct": 1}, {"task": "gsm8k", "id": 2, "correct": 0}]
        s = tier1.summarize(rows)
        self.assertEqual((s["items"], s["errors"], s["tasks"]["gsm8k"]["acc"]), (2, 0, 0.5))


class Tier1RunTests(unittest.TestCase):
    """run -> JSONL -> resume, against a fake serve and a two-task cache."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._saved = tier1.DS_DIR
        tier1.DS_DIR = Path(self.tmp.name)
        (tier1.DS_DIR / "gsm8k").mkdir()
        (tier1.DS_DIR / "gsm8k/items.jsonl").write_text(
            "".join(json.dumps({"id": i, "question": f"q{i}", "answer": str(i)}) + "\n" for i in range(4)))
        (tier1.DS_DIR / "chartqa/images").mkdir(parents=True)
        (tier1.DS_DIR / "chartqa/images/0.png").write_bytes(vision.solid("red"))
        (tier1.DS_DIR / "chartqa/items.jsonl").write_text(json.dumps(
            {"id": 0, "query": "colour?", "labels": ["red"], "image": "chartqa/images/0.png"}) + "\n")

    def tearDown(self):
        tier1.DS_DIR = self._saved
        self.tmp.cleanup()

    def test_run_and_resume(self):
        calls = []

        def chat(body):
            calls.append(body)
            text = body["messages"][-1]["content"]
            if isinstance(text, list):
                return {"content": "Red"}
            n = int(text.split()[0][1:])
            if n == 3 and len(calls) < 6:
                return {"http_error": 500, "message": "boom"}
            return {"content": f"Answer: {n if n != 2 else 99}"}

        s = FakeServe(chat)
        out = Path(self.tmp.name) / "run.jsonl"
        argv = ["run", "--out", str(out), "--url", s.url, "--tasks", "gsm8k,chartqa"]
        try:
            self.assertEqual(tier1.main(argv), 1)  # one infrastructure error
            summ = tier1.summarize(common.jsonl(out))
            self.assertEqual((summ["items"], summ["errors"]), (5, 1))
            self.assertEqual(tier1.main(argv), 0)  # resume retries only the errored item
            summ = tier1.summarize(common.jsonl(out))
            self.assertEqual((summ["errors"], summ["tasks"]["gsm8k"]["correct"], summ["tasks"]["chartqa"]["acc"]),
                             (0, 3, 1.0))
            self.assertEqual(len(calls), 6)
            self.assertTrue(calls[0]["temperature"] == 0 and "seed" not in calls[0])
            self.assertNotIn("chat_template_kwargs", calls[0])
            with self.assertRaises(SystemExit):  # a different config refuses to append
                tier1.main(argv + ["--reasoning-effort", "low"])
            with self.assertRaises(SystemExit):  # another served model refuses to append
                tier1.main(argv + ["--model", "other-model"])
            saved_ids, tier1.IDS_FILE = tier1.IDS_FILE, Path(self.tmp.name) / "resampled_ids.json"
            tier1.IDS_FILE.write_text('{"seed": 1}\n')
            try:
                with self.assertRaises(SystemExit):  # a resampled id set refuses to append
                    tier1.main(argv)
            finally:
                tier1.IDS_FILE = saved_ids
        finally:
            s.close()


if __name__ == "__main__":
    unittest.main()

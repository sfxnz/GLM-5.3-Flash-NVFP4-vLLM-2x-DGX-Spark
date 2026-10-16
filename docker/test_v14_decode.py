"""CPU tests for patch_v14_decode.py.

    GLM53_V13_SRC=/path/to/v13src python3 -m unittest docker/test_v14_decode.py -v

GLM53_V13_SRC is the directory that holds the `vllm/` package tree copied out
of glm53-sm121-v13. Without it the source tests skip.
"""

import os
import py_compile
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import patch_v14_decode as P  # noqa: E402

SRC = os.environ.get("GLM53_V13_SRC", "")


class PlanEditsTest(unittest.TestCase):
    def test_refuses_missing_anchor(self):
        with self.assertRaises(SystemExit):
            P.plan_edits("nothing here", [("x", "anchor", "new")])

    def test_already_applied_is_noop(self):
        self.assertEqual(P.plan_edits("a new b", [("x", "old", "new")]), ("a new b", 0))


@unittest.skipUnless(SRC and (Path(SRC) / "vllm").is_dir(), "GLM53_V13_SRC not set")
class SourceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "vllm"
        for rel in P.FILE_EDITS:
            (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(Path(SRC) / "vllm" / rel, self.root / rel)
        (self.root / "__init__.py").touch()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_every_anchor_once(self):
        for rel, edits in P.FILE_EDITS.items():
            src = (self.root / rel).read_text()
            for what, old, _new in edits:
                self.assertEqual(src.count(old), 1, what)

    def test_apply_twice_and_compile(self):
        self.assertEqual(len(P.main(self.root)), 3)
        self.assertEqual(P.main(self.root), [])
        for rel in P.FILE_EDITS:
            py_compile.compile(str(self.root / rel), doraise=True)

    def test_switches_default_off(self):
        P.main(self.root)
        model = (self.root / "models/glm5next/nvidia/model.py").read_text()
        attn = (self.root / "models/glm5next/nvidia/attention.py").read_text()
        utils = (self.root / "v1/worker/gpu/attn_utils.py").read_text()
        self.assertIn('os.environ.get("GLM53_ROUTER_DEDUPE") == "1"', model)
        self.assertIn('os.environ.get("GLM53_HEADGATE_MM") == "1"', attn)
        self.assertIn('os.environ.get("GLM53_LATE_MLA_PLAN") == "1"', utils)
        # The off branch keeps v13's exact code.
        self.assertIn("router_logits, _ = self.gate(hidden_states)", model)
        self.assertIn("weights = torch.mm(hidden_states.float(), self._wp_fp32)", attn)

    def test_refuses_drift(self):
        path = self.root / "models/glm5next/nvidia/model.py"
        path.write_text(path.read_text().replace("router_logits, _ = self.gate(hidden_states)", "x = 1"))
        with self.assertRaises(SystemExit):
            P.main(self.root)


if __name__ == "__main__":
    unittest.main()

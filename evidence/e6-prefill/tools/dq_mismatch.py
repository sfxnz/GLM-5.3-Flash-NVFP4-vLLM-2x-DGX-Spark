#!/usr/bin/env python3
"""Classify the elements behind the two DequantTest failures (CPU interpreter, glm53-sm121-v11).

Reruns docker/test_v13_fp8.py's DequantTest cases with a check() that sorts every
bit mismatch into: sign-of-zero only (kernel +0, reference -0, or the reverse),
NaN, or a real value difference. Run from the repo root like evidence/e4-tools/cpu_tests.sh:

  docker run --rm --entrypoint python3 -v "$PWD":/r -w /r \
    -v ~/.cache/huggingface:/root/.cache/huggingface:ro \
    -e GLM53_V11_SRC=/usr/local/lib/python3.12/dist-packages glm53-sm121-v11 \
    evidence/e6-prefill/tools/dq_mismatch.py
"""
import sys
import unittest

sys.path.insert(0, ".")
import torch  # noqa: E402

from docker import test_v13_fp8 as t  # noqa: E402

totals = {}


def check(self, mode, parts, ref, n, k, gs=None):
    layer = self.layer(mode, parts, n, k, gs)
    out = torch.empty(n, k, dtype=torch.bfloat16)
    self.m.dequantize_marlin(layer, self.m.dequant_spec(layer, mode, gs), out)
    r = ref.to(torch.bfloat16)
    bad = out.view(torch.int16) != r.view(torch.int16)
    zero = bad & (out == 0) & (r == 0)
    nan = bad & (out.isnan() | r.isnan())
    real = bad & ~zero & ~nan
    kneg = int((zero & torch.signbit(out)).sum())
    rneg = int((zero & torch.signbit(r)).sum())
    key = f"{self._testMethodName} {mode} g{gs} {n}x{k}"
    totals[key] = dict(elements=n * k, bit_mismatch=int(bad.sum()), sign_of_zero=int(zero.sum()),
                       ref_minus0_kernel_plus0=rneg, kernel_minus0_ref_plus0=kneg,
                       nan=int(nan.sum()), value_diff=int(real.sum()))
    print(key, totals[key], flush=True)
    self.assertFalse(real.any() or nan.any(), f"{key}: {int(real.sum())} value / {int(nan.sum())} NaN")
    return layer


t.DequantTest.check = check
suite = unittest.TestSuite()
for name in ("test_int_matches_reference", "test_fp8_matches_reference",
             "test_nvfp4_matches_reference", "test_every_code_matches_reference"):
    suite.addTest(t.DequantTest(name))
res = unittest.TextTestRunner(verbosity=2).run(suite)
bits = sum(v["bit_mismatch"] for v in totals.values())
sz = sum(v["sign_of_zero"] for v in totals.values())
val = sum(v["value_diff"] + v["nan"] for v in totals.values())
print(f"SUMMARY bit mismatches {bits}: sign-of-zero {sz}, value/NaN {val}; "
      f"tests {'OK' if res.wasSuccessful() else 'FAILED'} (value-equal check)")
sys.exit(0 if res.wasSuccessful() else 1)

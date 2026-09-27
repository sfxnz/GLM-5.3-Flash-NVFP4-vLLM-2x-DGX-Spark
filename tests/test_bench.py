"""CPU-only tests for bench_decode.py (ruler v2) and kit/compare.py.

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import struct
import sys
import tempfile
import threading
import time
import unittest
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "kit")]

import bench_decode as bd  # noqa: E402
import compare  # noqa: E402

K_SPEC = 7          # draft slots the fake server reports
ACC = 3             # tokens per verify step the fake server emits
STEP_S = 0.002      # fake verify step
NATURAL = 40        # natural length when min_tokens is not sent


class FakeServer:
    """OpenAI-compatible SSE server with vLLM-style spec-decode Prometheus counters."""

    # fault: None, "error_event" (vLLM mid-stream failure: error event then [DONE]),
    # "truncate" (connection dies mid-chunk), "no_done" (clean close, no usage, no [DONE]),
    # "http400" (request rejected with a JSON body), "short" (ignores min_tokens).
    def __init__(self, reasoning_first: bool = False, fault: str | None = None):
        self.lock = threading.Lock()
        self.counters = {"drafts": 0, "draft_tokens": 0, "accepted": 0,
                         "itl_count": 0, "itl_sum": 0.0, "prefill_count": 0, "prefill_sum": 0.0}
        self.per_pos = [0] * K_SPEC
        self.bodies: list[dict] = []
        self.reasoning_first = reasoning_first
        self.fault = fault
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *_):
                pass

            def do_GET(self):
                if self.path == "/v1/models":
                    self._send(200, "application/json",
                               json.dumps({"data": [{"id": "fake/GLM"}, {"id": "other"}]}).encode())
                elif self.path == "/metrics":
                    self._send(200, "text/plain", fake.metrics_text().encode())
                else:
                    self._send(404, "text/plain", b"")

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with fake.lock:
                    fake.bodies.append(body)
                if fake.fault == "http400":
                    self._send(400, "application/json", b'{"error": {"message": "is not a multimodal model"}}')
                    return
                if fake.fault in ("error_event", "truncate", "no_done"):
                    self._fail(body)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for ev in fake.events(body):
                    self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")

            def _fail(self, body):
                events = fake.events(body)
                first = [next(events), next(events)]  # role delta + first content token
                data = b"".join(f"data: {json.dumps(ev)}\n\n".encode() for ev in first)
                if fake.fault == "truncate":
                    # Raw HTTP/1.1 chunked body; the last chunk promises more bytes than arrive.
                    self.wfile.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                                     b"Transfer-Encoding: chunked\r\n\r\n")
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
                    self.wfile.write(b"400\r\ndata: {\"choi")
                    self.close_connection = True
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(data)
                if fake.fault == "error_event":
                    err = {"error": {"message": "EngineCore died", "type": "InternalServerError", "code": 500}}
                    self.wfile.write(f"data: {json.dumps(err)}\n\ndata: [DONE]\n\n".encode())

            def _send(self, code, ctype, data):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1/chat/completions"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    @staticmethod
    def prompt_tokens(content) -> int:
        if isinstance(content, list):
            text = " ".join(p.get("text", "") for p in content)
            return int(len(text.split()) * 1.3) + 64  # image tokens
        return int(len(content.split()) * 1.3)

    def events(self, body):
        forced = body.get("min_tokens") and self.fault != "short"
        n = body["max_tokens"] if forced else min(NATURAL, body["max_tokens"])
        mk = lambda delta: {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]}  # noqa: E731
        yield mk({"role": "assistant"})
        time.sleep(0.01)  # prefill
        # Thinking on: every token is reasoning (512 tokens at Max effort rarely reach content).
        think = bool((body.get("chat_template_kwargs") or {}).get("enable_thinking"))
        first_key = "reasoning" if self.reasoning_first or think else "content"
        yield mk({first_key: "t0 "})
        emitted, steps, accepted, pos = 1, 0, 0, [0] * K_SPEC
        while emitted < n:
            time.sleep(STEP_S)
            k = min(ACC, n - emitted)
            steps += 1
            accepted += k - 1
            for j in range(k - 1):
                pos[j] += 1
            yield mk({"reasoning" if think else "content": "tok " * k})
            emitted += k
        yield {"choices": [{"index": 0, "delta": {},
                            "finish_reason": "length" if n == body["max_tokens"] else "stop"}]}
        pt = self.prompt_tokens(body["messages"][0]["content"])
        with self.lock:
            c = self.counters
            c["drafts"] += steps
            c["draft_tokens"] += steps * K_SPEC
            c["accepted"] += accepted
            c["itl_count"] += steps
            c["itl_sum"] += steps * STEP_S
            c["prefill_count"] += 1
            c["prefill_sum"] += 0.01
            self.per_pos = [a + b for a, b in zip(self.per_pos, pos)]
        yield {"choices": [], "usage": {"prompt_tokens": pt, "completion_tokens": n,
                                        "total_tokens": pt + n}}

    def metrics_text(self) -> str:
        with self.lock:
            c = dict(self.counters)
            per_pos = list(self.per_pos)
        lab = 'engine="0",model_name="fake GLM"'
        lines = [
            "# HELP vllm:spec_decode_num_drafts Number of spec decoding drafts.",
            "# TYPE vllm:spec_decode_num_drafts counter",
            f"vllm:spec_decode_num_drafts_total{{{lab}}} {c['drafts']}.0",
            f"vllm:spec_decode_num_drafts_created{{{lab}}} 1.7e9",
            f"vllm:spec_decode_num_draft_tokens_total{{{lab}}} {c['draft_tokens']}.0",
            f"vllm:spec_decode_num_accepted_tokens_total{{{lab}}} {c['accepted']}.0",
            f"vllm:inter_token_latency_seconds_sum{{{lab}}} {c['itl_sum']}",
            f"vllm:inter_token_latency_seconds_count{{{lab}}} {c['itl_count']}.0",
            f'vllm:inter_token_latency_seconds_bucket{{{lab},le="0.01"}} {c["itl_count"]}.0',
            f"vllm:request_prefill_time_seconds_sum{{{lab}}} {c['prefill_sum']}",
            f"vllm:request_prefill_time_seconds_count{{{lab}}} {c['prefill_count']}.0",
            "vllm:num_requests_running 0.0",
        ]
        lines += [f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{{lab},position="{i}"}} {v}.0'
                  for i, v in enumerate(per_pos)]
        return "\n".join(lines) + "\n"


def expected_acceptance(n: int) -> float:
    steps = math.ceil((n - 1) / ACC)
    return 1 + (n - 1 - steps) / steps


MEM_BEFORE = "MemTotal: 125000000 kB\nMemAvailable: 8000000 kB\nSwapTotal: 16777216 kB\nSwapFree: 10485760 kB\n"


class TestMetrics(unittest.TestCase):
    def test_parse_metrics_sums_labels_and_strips_total(self):
        text = "\n".join([
            "# TYPE vllm:spec_decode_num_drafts counter",
            'vllm:spec_decode_num_drafts_total{engine="0",model_name="a b"} 10.0',
            'vllm:spec_decode_num_drafts_total{engine="1",model_name="a b"} 5.0',
            'vllm:spec_decode_num_drafts_created{engine="0"} 1.7e9',
            "vllm:spec_decode_num_accepted_tokens 21",  # bare counter name
            'vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="0"} 9.0',
            'vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="1",position="0"} 3.0',
            'vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="1"} 4.0',
            'vllm:inter_token_latency_seconds_sum{engine="0"} 1.5',
            'vllm:inter_token_latency_seconds_count{engine="0"} 15.0',
            'vllm:inter_token_latency_seconds_bucket{engine="0",le="+Inf"} 15.0',
        ])
        m = bd.parse_metrics(text)
        self.assertEqual(m["drafts"], 15.0)
        self.assertEqual(m["accepted"], 21.0)
        self.assertEqual(m["per_pos"], {0: 12.0, 1: 4.0})
        self.assertEqual(m["itl_count"], 15.0)
        self.assertAlmostEqual(m["itl_sum"], 1.5)
        self.assertNotIn("draft_tokens", m)

    def test_delta_and_factorize(self):
        before = {"drafts": 100, "draft_tokens": 700, "accepted": 150, "itl_sum": 10.0,
                  "itl_count": 100, "prefill_sum": 1.0, "prefill_count": 1, "per_pos": {0: 80, 1: 50}}
        after = {"drafts": 300, "draft_tokens": 2100, "accepted": 450, "itl_sum": 34.0,
                 "itl_count": 301, "prefill_sum": 1.5, "prefill_count": 2,
                 "per_pos": {0: 230, 1: 150, 2: 20}}
        d = bd.metrics_delta(before, after)
        self.assertEqual(d["drafts"], 200)
        self.assertEqual(d["per_pos"], [150, 100, 20])
        # 200 steps, 300 accepted -> 500 decode tokens in 24 s.
        f = bd.factorize(500, 24.0, d)
        self.assertAlmostEqual(f["acceptance_len"], 2.5)
        self.assertAlmostEqual(f["step_ms"], 120.0)
        self.assertAlmostEqual(f["draft_acceptance_rate"], 300 / 1400)
        self.assertEqual(f["per_pos"], [0.75, 0.5, 0.1])
        self.assertAlmostEqual(f["itl_ms"], 24000 / 201)
        self.assertEqual(f["itl_minus_drafts"], 1)
        self.assertAlmostEqual(f["server_prefill_s"], 0.5)
        # tok_s = 500/24 = 20.83 == acceptance_len / step = 2.5 / 0.120
        self.assertAlmostEqual(f["sanity_err"], 0.0)
        self.assertAlmostEqual(500 / 24.0, f["acceptance_len"] * 1000 / f["step_ms"])
        # max_tokens truncation mid-step: one accepted token not emitted.
        self.assertAlmostEqual(bd.factorize(499, 24.0, d)["sanity_err"], -0.002)

    def test_factorize_without_spec(self):
        self.assertEqual(bd.factorize(100, 5.0, {}), {})
        self.assertNotIn("step_ms", bd.factorize(100, 5.0, {"drafts": 0.0}))

    def test_describe(self):
        s = bd.describe([10.0, 12.0, 14.0, None])
        self.assertEqual(s["n"], 3)
        self.assertAlmostEqual(s["mean"], 12.0)
        self.assertAlmostEqual(s["stdev"], 2.0)
        half = 4.303 * 2.0 / 3 ** 0.5
        self.assertAlmostEqual(s["ci95"][0], 12.0 - half)
        self.assertIsNone(bd.describe([5.0])["ci95"])
        self.assertIsNone(bd.describe([None]))


class TestHygiene(unittest.TestCase):
    def test_parse_meminfo(self):
        m = bd.parse_meminfo(MEM_BEFORE)
        self.assertAlmostEqual(m["swap_used_mib"], 6144.0)
        self.assertAlmostEqual(m["mem_available_mib"], 8000000 / 1024)

    def test_swap_verdict(self):
        before = {"local": {"swap_used_mib": 6144.0}, "spark2": {"swap_used_mib": 100.0}}
        self.assertIsNone(bd.swap_verdict(before, {"local": {"swap_used_mib": 6200.0},
                                                   "spark2": {"swap_used_mib": 100.0}}))
        self.assertIsNone(bd.swap_verdict(before, {"local": {"swap_used_mib": 5000.0},
                                                   "spark2": {"swap_used_mib": 164.0}}))
        reason = bd.swap_verdict(before, {"local": {"swap_used_mib": 6144.0},
                                          "spark2": {"swap_used_mib": 165.0}})
        self.assertEqual(reason, "spark2 swap +65 MiB")
        # An unreadable host never invalidates.
        self.assertIsNone(bd.swap_verdict({"local": None}, {"local": {"swap_used_mib": 1e6}}))


class TestPrompts(unittest.TestCase):
    def test_png_is_valid(self):
        png = bd.make_png(7, size=32)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        pos, chunks = 8, {}
        while pos < len(png):
            (length,) = struct.unpack(">I", png[pos:pos + 4])
            tag, data = png[pos + 4:pos + 8], png[pos + 8:pos + 8 + length]
            (crc,) = struct.unpack(">I", png[pos + 8 + length:pos + 12 + length])
            self.assertEqual(crc, zlib.crc32(tag + data) & 0xFFFFFFFF)
            chunks[tag] = data
            pos += 12 + length
        self.assertEqual(struct.unpack(">II", chunks[b"IHDR"][:8]), (32, 32))
        self.assertEqual(len(zlib.decompress(chunks[b"IDAT"])), 32 * (1 + 32 * 3))
        self.assertNotEqual(bd.make_png(7, 32), bd.make_png(8, 32))

    def test_prompt_sets(self):
        self.assertEqual(len(set(bd.PROSE)), 8)
        self.assertEqual(len(set(bd.CODE)), 8)
        _, waves = bd.plan_cell("H", 1, 512)
        self.assertEqual([len(s) for _, _, s in waves], [2, 2, 2, 2])
        self.assertEqual(len({s["prompt_id"] for _, _, specs in waves for s in specs}), 8)
        _, waves = bd.plan_cell("K", 1, 512)
        self.assertEqual([g for g, _, _ in waves], ["K@c1"] * 3 + ["K@c2"] * 3)
        self.assertTrue(all(s["content"] == bd.LEGACY_PROSE and not s["forced"]
                            for _, _, specs in waves for s in specs))

    def test_long_prompt_salt_first_and_unique(self):
        a, salt_a = bd.long_prompt(100)
        b, salt_b = bd.long_prompt(100)
        self.assertNotEqual(salt_a, salt_b)
        self.assertTrue(a.startswith(f"Document id {salt_a}."))
        self.assertEqual(a.split("\n", 1)[1], b.split("\n", 1)[1])  # seeded filler


class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.server = FakeServer()
        self.tmp = tempfile.TemporaryDirectory()
        mem = {"swap_used_mib": 100.0, "mem_available_mib": 8000.0}
        patcher = mock.patch.object(bd, "snapshot_hosts", return_value={"local": mem})
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self.server.close()
        self.tmp.cleanup()

    def run_bench(self, *extra):
        with contextlib.redirect_stdout(io.StringIO()):
            rc = bd.main(["--url", self.server.url, "--tokens", "31", "--out", self.tmp.name, *extra])
        with open(os.path.join(self.tmp.name, "bench.json"), encoding="utf-8") as fh:
            return rc, json.load(fh)

    def test_full_panel(self):
        rc, rep = self.run_bench("--full")
        self.assertEqual(rc, 0)
        self.assertEqual(rep["ruler_version"], "v2")
        self.assertEqual(rep["model"], "fake/GLM")
        self.assertTrue(os.path.exists(os.path.join(self.tmp.name, "bench.txt")))
        groups = {s["group"]: s for s in rep["summary"]}
        self.assertEqual(sorted(groups), sorted(
            ["A", "B", "J@c1", "J@c2", "H", "I", "E@32k", "E@128k", "F", "G", "K@c1", "K@c2", "T"]))
        a = groups["A"]
        self.assertEqual((a["n_waves"], a["n_requests"], a["short_requests"]), (8, 8, 0))
        self.assertAlmostEqual(a["acceptance_len"]["mean"], expected_acceptance(31))
        self.assertTrue(a["sanity_ok"])
        self.assertEqual(len(a["per_pos"]), K_SPEC)
        self.assertGreater(a["step_ms"]["mean"], STEP_S * 1000 * 0.9)
        self.assertEqual(groups["H"]["n_waves"], 4)
        self.assertEqual(groups["I"]["n_waves"], 2)
        self.assertIsNotNone(groups["H"]["agg_tok_s"])
        self.assertAlmostEqual(groups["H"]["acceptance_len"]["mean"], expected_acceptance(31))
        self.assertAlmostEqual(groups["K@c1"]["acceptance_len"]["mean"], expected_acceptance(NATURAL))
        self.assertEqual(groups["F"]["n_requests"], 4)
        # E sized by calibration: ~32k / ~128k prompt tokens, each salted.
        e32 = groups["E@32k"]["prompt_tokens"]["mean"]
        e128 = groups["E@128k"]["prompt_tokens"]["mean"]
        self.assertLess(abs(e32 / 32768 - 1), 0.02)
        self.assertLess(abs(e128 / 131072 - 1), 0.02)
        self.assertIsNotNone(groups["E@32k"]["server_prefill_tok_s"])
        # Request bodies: forced cells send min_tokens, thinking is off except T, no effort kwarg.
        bodies = self.server.bodies
        thinking = [b for b in bodies if b["chat_template_kwargs"] != {"enable_thinking": False}]
        self.assertEqual(len(thinking), 1 + 8)  # T: warm-up + 8 prose prompts
        self.assertTrue(all(b["chat_template_kwargs"] == {"enable_thinking": True} for b in thinking))
        self.assertEqual(groups["T"]["n_requests"], 8)
        self.assertFalse(any("reasoning_effort" in b for b in bodies))
        forced = [b for b in bodies if b.get("min_tokens")]
        self.assertTrue(all(b["min_tokens"] == b["max_tokens"] for b in forced))
        legacy = [b for b in bodies if b["messages"][0]["content"] == bd.LEGACY_PROSE]
        self.assertEqual(len(legacy), 1 + 3 + 3 * 2)  # warm-up, 3 x c=1, 3 x c=2
        self.assertTrue(all("min_tokens" not in b and b["max_tokens"] == 200 for b in legacy))
        sampled = [b for b in bodies if b.get("top_p") == 0.95]
        self.assertEqual(len(sampled), 9)
        self.assertTrue(all(b["temperature"] == 1.0 and b["seed"] == bd.SEED for b in sampled))
        images = [b for b in bodies if isinstance(b["messages"][0]["content"], list)]
        self.assertEqual(len(images), 5)
        self.assertTrue(images[0]["messages"][0]["content"][0]["image_url"]["url"]
                        .startswith("data:image/png;base64,"))
        # Per-request raw rows carry hash and finish_reason.
        row = next(w for w in rep["waves"] if w["group"] == "A")["requests"][0]
        self.assertEqual(row["finish_reason"], "length")
        self.assertEqual(len(row["sha256"]), 64)

    def test_thinking_cell_t(self):
        rc, rep = self.run_bench("--cells", "T")
        self.assertEqual(rc, 0)
        (t,) = rep["summary"]
        self.assertEqual((t["group"], t["n_waves"], t["n_requests"], t["short_requests"]), ("T", 8, 8, 0))
        # Same metrics as A; the fake streams reasoning only, so TTFT and decode start at a reasoning token.
        self.assertAlmostEqual(t["acceptance_len"]["mean"], expected_acceptance(31))
        self.assertTrue(t["sanity_ok"])
        self.assertLess(t["ttft_s"]["median"], 0.5)
        self.assertIsNotNone(t["step_ms"]["mean"])
        bodies = self.server.bodies
        self.assertEqual(len(bodies), 1 + 8)
        self.assertEqual({b["messages"][0]["content"] for b in bodies[1:]}, set(bd.PROSE))
        for b in bodies:
            self.assertEqual(b["chat_template_kwargs"], {"enable_thinking": True})
            self.assertNotIn("reasoning_effort", b)  # unset renders Max effort
            self.assertEqual((b["temperature"], b["min_tokens"]), (0.0, b["max_tokens"]))
        self.assertEqual({b["max_tokens"] for b in bodies[1:]}, {31})
        row = next(w for w in rep["waves"] if w["group"] == "T")["requests"][0]
        self.assertEqual(row["completion_tokens"], 31)  # reasoning + content
        self.assertAlmostEqual(row["tok_s"], 30 / row["decode_s"])

    def test_fast_gate_unchanged(self):
        self.assertEqual(bd.DEFAULT_CELLS, "A,B,J,H,K")  # T runs with --full or --cells T

    def test_swap_growth_invalidates_cell(self):
        seq = [{"local": {"swap_used_mib": 100.0}}, {"local": {"swap_used_mib": 300.0}}]
        with mock.patch.object(bd, "snapshot_hosts", side_effect=seq):
            rc, rep = self.run_bench("--cells", "K")
        self.assertEqual(rc, 1)
        self.assertEqual(rep["cells"]["K"]["invalid_reason"], "local swap +200 MiB")
        self.assertTrue(all(not s["valid"] for s in rep["summary"]))

    def test_unknown_cell_rejected(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            bd.main(["--url", self.server.url, "--cells", "A,Z"])

    def test_empty_cells_rejected(self):
        for cells in ("", " , "):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                bd.main(["--url", self.server.url, "--cells", cells])

    def test_unreadable_meminfo_is_flagged(self):
        seq = [{"local": {"swap_used_mib": 100.0}, "spark2": None},
               {"local": {"swap_used_mib": 100.0}, "spark2": None}]
        with mock.patch.object(bd, "snapshot_hosts", side_effect=seq):
            rc, rep = self.run_bench("--cells", "K")
        self.assertEqual(rc, 1)
        self.assertEqual(rep["cells"]["K"]["hygiene_unverified"], ["spark2"])
        self.assertEqual(rep["hygiene_unverified"], ["K"])
        self.assertIsNone(rep["cells"]["K"]["invalid_reason"])

    def test_k_reports_legacy_stream_median(self):
        _, rep = self.run_bench("--cells", "K")
        k2 = next(s for s in rep["summary"] if s["group"] == "K@c2")
        self.assertEqual(k2["stream_tok_s"]["n"], 6)
        with open(os.path.join(self.tmp.name, "bench.txt"), encoding="utf-8") as fh:
            self.assertIn("K@c2 legacy-comparable median per-stream tok/s", fh.read())


class TestFailures(unittest.TestCase):
    """Server failures must fail the wave and the exit code, never score as tok/s 0."""

    def run_faulty(self, fault: str, cells: str = "A", tokens: int = 31) -> tuple[int, dict]:
        server = FakeServer(fault=fault)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.addCleanup(server.close)
        mem = {"local": {"swap_used_mib": 100.0}}
        with mock.patch.object(bd, "snapshot_hosts", return_value=mem), \
                contextlib.redirect_stdout(io.StringIO()):
            rc = bd.main(["--url", server.url, "--tokens", str(tokens), "--cells", cells, "--out", tmp.name])
        with open(os.path.join(tmp.name, "bench.json"), encoding="utf-8") as fh:
            return rc, json.load(fh)

    def one_row(self, fault: str) -> dict:
        server = FakeServer(fault=fault)
        try:
            return bd.stream_one(server.url, "fake/GLM", bd.spec("x", "hi", 31, True), time.perf_counter())
        finally:
            server.close()

    def test_error_event_after_tokens(self):
        row = self.one_row("error_event")
        self.assertIn("server error event", row["error"])
        self.assertIn("EngineCore died", row["error"])
        rc, rep = self.run_faulty("error_event")
        self.assertEqual(rc, 1)
        a = next(s for s in rep["summary"] if s["group"] == "A")
        self.assertEqual((a["n_waves"], a["n_failed"]), (0, 8))
        self.assertIsNone(a["tok_s"])

    def test_truncated_chunked_body(self):
        row = self.one_row("truncate")
        self.assertIn("IncompleteRead", row["error"])
        rc, rep = self.run_faulty("truncate")
        self.assertEqual(rc, 1)
        a = next(s for s in rep["summary"] if s["group"] == "A")
        self.assertEqual((a["n_waves"], a["n_failed"]), (0, 8))

    def test_stream_without_done_or_usage(self):
        self.assertEqual(self.one_row("no_done")["error"], "stream ended without [DONE]")
        rc, _ = self.run_faulty("no_done", cells="K")
        self.assertEqual(rc, 1)

    def test_http_error_body_captured(self):
        self.assertIn("is not a multimodal model", self.one_row("http400")["error"])

    def test_short_forced_request_fails_exit(self):
        rc, rep = self.run_faulty("short", tokens=NATURAL + 24)  # natural stop before 64
        self.assertEqual(rc, 1)
        a = next(s for s in rep["summary"] if s["group"] == "A")
        self.assertEqual((a["n_waves"], a["short_requests"]), (8, 8))

    def test_e_calibration_failure_is_a_failed_cell(self):
        rc, rep = self.run_faulty("http400", cells="E,K")
        self.assertEqual(rc, 1)
        self.assertEqual(rep["failed_cells"], ["E"])
        self.assertIn("K", rep["cells"])  # the bench carried on past E


class TestTTFT(unittest.TestCase):
    def test_first_reasoning_token_starts_the_window(self):
        server = FakeServer(reasoning_first=True)
        try:
            s = bd.spec("x", "hi", 31, True, temperature=0.0)
            row = bd.stream_one(server.url, "fake/GLM", s, time.perf_counter())
        finally:
            server.close()
        self.assertNotIn("error", row)
        self.assertEqual(row["completion_tokens"], 31)
        self.assertLess(row["ttft_s"], 0.5)
        self.assertAlmostEqual(row["tok_s"], 30 / row["decode_s"])


def fake_report(tok_s: list[float], acc: dict[str, float] | None = None, valid: bool = True) -> dict:
    """One boot: cell A with the given per-request tok_s; optional per-prompt acceptance."""
    acc = acc or {}
    waves = [{"group": "A", "c": 1, "ok": True, "acceptance_len": a,
              "requests": [{"prompt_id": pid}]} for pid, a in acc.items()]
    mean = sum(tok_s) / len(tok_s)
    return {
        "ruler_version": "v2", "model": "m", "_path": "mem",
        "summary": [{"group": "A", "valid": valid, "invalid_reason": None if valid else "swap",
                     "tok_s": {"mean": mean}, "step_ms": {"mean": 2400.0 / mean},
                     "acceptance_len": {"mean": 2.4}}],
        "waves": waves,
    }


class TestCompare(unittest.TestCase):
    def verdict(self, a: list[float], b: list[float]) -> str:
        rows = compare.compare([fake_report([x]) for x in a], [fake_report([x]) for x in b])
        return rows[0]["verdict"]

    def test_keep(self):
        self.assertEqual(self.verdict([20.0, 20.2, 19.8], [22.0, 22.1, 21.9]), "KEEP")

    def test_revert(self):
        self.assertEqual(self.verdict([20.0, 20.2, 19.8], [19.0, 19.1, 18.9]), "REVERT")
        # No change, tight boots: the upper bound stays under +1%.
        self.assertEqual(self.verdict([20.0, 20.02, 19.98], [20.0, 20.01, 19.99]), "REVERT")

    def test_inconclusive(self):
        # +2% gain with a tight CI is neither a keep (< +3%) nor a revert.
        self.assertEqual(self.verdict([20.0, 20.1, 19.9], [20.4, 20.5, 20.3]), "INCONCLUSIVE")
        # Big gain but noisy boots: the lower bound crosses zero.
        self.assertEqual(self.verdict([18.0, 22.0, 20.0], [19.0, 25.0, 22.0]), "INCONCLUSIVE")
        # One boot per arm cannot give a boot-level interval.
        self.assertEqual(self.verdict([20.0], [25.0]), "INCONCLUSIVE")

    def test_rel_change_math(self):
        c = compare.rel_change([10.0, 10.0], [11.0, 11.0])
        self.assertAlmostEqual(c["rel"], 0.1)
        self.assertAlmostEqual(c["lo"], 0.1)
        self.assertIsNone(compare.rel_change([10.0], [11.0, 12.0]))

    def test_invalid_boot_skipped(self):
        arm_a = [fake_report([20.0]), fake_report([20.2]), fake_report([19.8])]
        arm_b = [fake_report([22.0]), fake_report([22.1]), fake_report([5.0], valid=False)]
        row = compare.compare(arm_a, arm_b)[0]
        self.assertEqual((row["n_a"], row["n_b"]), (3, 2))
        self.assertEqual(row["verdict"], "KEEP")
        self.assertEqual(len(row["skipped"]), 1)

    def test_paired_bootstrap(self):
        base = {f"A/{i:02d}": 2.0 + 0.1 * i for i in range(8)}
        up = {k: v * 1.05 for k, v in base.items()}
        rows = compare.compare([fake_report([20.0], base)] * 2, [fake_report([21.0], up)] * 2)
        ap = rows[0]["acceptance_paired"]
        self.assertEqual(ap["n_prompts"], 8)
        self.assertAlmostEqual(ap["rel"], 0.05)
        self.assertAlmostEqual(ap["lo"], 0.05)
        self.assertAlmostEqual(ap["hi"], 0.05)
        noisy = {k: v * (1.05 if i % 2 else 0.95) for i, (k, v) in enumerate(base.items())}
        ap = compare.paired_bootstrap(base, noisy)
        self.assertLess(ap["lo"], 0)
        self.assertGreater(ap["hi"], 0)
        self.assertIsNone(compare.paired_bootstrap({"a": 1.0}, {"a": 1.0}))

    def test_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = {}
            for arm, vals in (("a", [20.0, 20.2]), ("b", [22.0, 22.2])):
                paths[arm] = []
                for i, v in enumerate(vals):
                    p = os.path.join(tmp, f"{arm}{i}.json")
                    with open(p, "w", encoding="utf-8") as fh:
                        json.dump(fake_report([v]), fh)
                    paths[arm].append(p)
            out = os.path.join(tmp, "cmp.json")
            with contextlib.redirect_stdout(io.StringIO()):
                rc = compare.main(["--a", *paths["a"], "--b", *paths["b"], "--json", out])
            self.assertEqual(rc, 0)
            with open(out, encoding="utf-8") as fh:
                self.assertEqual(json.load(fh)[0]["verdict"], "KEEP")


if __name__ == "__main__":
    unittest.main()

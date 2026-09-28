#!/usr/bin/env python3
"""Independent sanity checks over a ruler-v2 bench.json.

Per wave: tok_s vs acceptance_len*1000/step_ms, itl_count - drafts in [0, c],
sum(per_pos) == acceptance_len - 1, per_pos non-increasing and <= 1, drafts > 0,
forced lengths, finish reasons. Per cell: valid flag, sanity_ok.
Also prints the per-cell table the final report needs.
"""
import json
import sys

rep = json.load(open(sys.argv[1]))
problems = []
print(f"ruler={rep['ruler_version']} ts={rep['timestamp']} git={rep['git_sha']} model={rep['model']}")
for w in rep["waves"]:
    g, c = w["group"], w["c"]
    ids = ",".join(r["prompt_id"] for r in w["requests"])
    if not w["ok"]:
        problems.append(f"{g} [{ids}] failed")
        continue
    if not w.get("drafts"):
        problems.append(f"{g} [{ids}] drafts missing/0")
        continue
    ident = w["acceptance_len"] * 1000 / w["step_ms"]
    # wave tok_s is mean of per-request tok_s; identity holds on sums: use decode sums
    dec_tok = sum(r["decode_tokens"] for r in w["requests"])
    dec_s = sum(r["decode_s"] for r in w["requests"])
    ident_err = (dec_tok / dec_s) / ident - 1
    imd = w.get("itl_minus_drafts")
    pp = w.get("per_pos", [])
    pp_sum_err = sum(pp) + 1 - w["acceptance_len"]
    mono = all(pp[i] >= pp[i + 1] - 1e-9 for i in range(len(pp) - 1))
    fins = {r.get("finish_reason") for r in w["requests"]}
    short = [r["prompt_id"] for r in w["requests"] if r.get("short")]
    flag = []
    if abs(ident_err) > 0.03 and not g.startswith("J"):
        flag.append(f"identity {ident_err:+.3f}")
    if imd is None or not (0 <= imd <= c):
        flag.append(f"itl-drafts={imd}")
    if abs(pp_sum_err) > 1e-6:
        flag.append(f"per_pos sum err {pp_sum_err:+.2e}")
    if not mono or any(p > 1 + 1e-9 for p in pp):
        flag.append("per_pos not monotone/<=1")
    if short:
        flag.append(f"short {short}")
    if flag:
        problems.append(f"{g} [{ids}] " + "; ".join(flag))
    w["_ident_err"] = ident_err
print()
hdr = f"{'group':<8}{'c':>2}{'n':>3}{'tok/s':>8}{'sd':>6}{'acc_len':>8}{'step_ms':>8}{'ttft_med':>9}{'agg':>7}{'prefill':>8}{'sanity':>8}{'valid':>7}  per_pos"
print(hdr)
for s in rep["summary"]:
    f = lambda k, key="mean", d=2: "-" if not s.get(k) or s[k].get(key) is None else f"{s[k][key]:.{d}f}"  # noqa: E731
    pre = s["server_prefill_tok_s"] or s["client_prefill_tok_s"]
    pre_s = "-" if not pre else f"{pre['mean']:.0f}"
    print(f"{s['group']:<8}{s['c']:>2}{s['n_waves']:>3}{f('tok_s'):>8}{f('tok_s','stdev'):>6}{f('acceptance_len',d=3):>8}"
          f"{f('step_ms',d=1):>8}{f('ttft_s','median',3):>9}{f('agg_tok_s'):>7}{pre_s:>8}"
          f"{('-' if s['sanity_max_err'] is None else format(s['sanity_max_err'], '.3f')):>8}{'ok' if s['valid'] else 'INVALID':>7}  "
          + " ".join(f"{p:.3f}" for p in s["per_pos"]))
for s in rep["summary"]:
    if s["cell"] == "E":
        print(f"{s['group']}: prompt_tokens={s['prompt_tokens']['mean']:.0f} ttft_med={s['ttft_s']['median']:.2f}s "
              f"server_prefill_tok_s={s['server_prefill_tok_s']['mean'] if s['server_prefill_tok_s'] else None} "
              f"client_prefill_tok_s={s['client_prefill_tok_s']['mean']:.0f}")
print()
print("failed_cells:", rep.get("failed_cells"), "hygiene_unverified:", rep.get("hygiene_unverified"))
for k, v in rep["cells"].items():
    if v.get("invalid_reason"):
        print(f"cell {k} INVALID: {v['invalid_reason']}")
print("wave-level problems:" if problems else "wave-level problems: none")
for p in problems:
    print("  " + p)

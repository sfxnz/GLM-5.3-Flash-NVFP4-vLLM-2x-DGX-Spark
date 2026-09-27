import json
import sys

E = '/home/sfxnz/projects/ai-lab/recipes/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark-opt-nvidia/evidence/e0-nvidia-v11/'
a = json.load(open(E + 'bench-1/bench.json'))
b = json.load(open(E + 'bench-2/bench.json'))
sa = {s['group']: s for s in a['summary']}
sb = {s['group']: s for s in b['summary']}
print("within-boot repeat: bench-1 (full, 14:03Z) vs bench-2 (fast gate, 14:23Z), same serve; delta = bench-2/bench-1 - 1")
print(f"{'group':<7}{'tok/s 1':>9}{'tok/s 2':>9}{'d tok/s':>9}{'step 1':>8}{'step 2':>8}{'d step':>8}"
      f"{'acc 1':>7}{'acc 2':>7}{'d acc':>8}{'ttft1':>7}{'ttft2':>7}")
for g in sb:
    x, y = sa[g], sb[g]

    def f(s, k):
        return s[k]['mean']

    def d(k):
        return 100 * (f(y, k) / f(x, k) - 1)

    print(f"{g:<7}{f(x, 'tok_s'):>9.2f}{f(y, 'tok_s'):>9.2f}{d('tok_s'):>+8.1f}%{f(x, 'step_ms'):>8.1f}{f(y, 'step_ms'):>8.1f}"
          f"{d('step_ms'):>+7.1f}%{f(x, 'acceptance_len'):>7.3f}{f(y, 'acceptance_len'):>7.3f}{d('acceptance_len'):>+7.1f}%"
          f"{x['ttft_s']['median']:>7.3f}{y['ttft_s']['median']:>7.3f}")
print()
print("greedy output identity (sha256 of streamed text) and per-prompt acceptance, c=1 cells A/B")
for cell in ('A', 'B'):
    wa = {w['requests'][0]['prompt_id']: w for w in a['waves'] if w['group'] == cell}
    wb = {w['requests'][0]['prompt_id']: w for w in b['waves'] if w['group'] == cell}
    same = sum(wa[p]['requests'][0]['sha256'] == wb[p]['requests'][0]['sha256'] for p in wa)
    print(f"{cell}: identical outputs {same}/{len(wa)}; per-prompt acc (1->2): "
          + ", ".join(f"{p[-2:]} {wa[p]['acceptance_len']:.3f}->{wb[p]['acceptance_len']:.3f}" for p in sorted(wa)))
for g in ('J@c1', 'J@c2', 'K@c1', 'K@c2', 'H'):
    ha = [r['sha256'] for w in a['waves'] if w['group'] == g for r in w['requests']]
    hb = [r['sha256'] for w in b['waves'] if w['group'] == g for r in w['requests']]
    print(f"{g}: distinct outputs bench-1 {len(set(ha))}/{len(ha)}, bench-2 {len(set(hb))}/{len(hb)}, shared {len(set(ha) & set(hb))}")

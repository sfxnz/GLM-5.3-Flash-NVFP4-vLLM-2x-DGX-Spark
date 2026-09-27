#!/usr/bin/env bash
# capture_serve.sh DIR: both ranks' docker args, env (tokens redacted), mounts and
# image, then an args/env parity check. Run once both containers exist.
set -u
D="$1"
C="${CONTAINER_NAME:-glm53-flash-nvfp4}"
red='s/(TOKEN=)[^",]*/\1<redacted>/g'
for node in head worker; do
  if [[ $node == head ]]; then run=(bash -c); else run=(ssh spark2); fi
  "${run[@]}" "docker inspect -f '{{json .Args}}' $C" > "$D/docker-args-$node.json"
  "${run[@]}" "docker inspect -f '{{json .Config.Env}}' $C" | sed -E "$red" > "$D/docker-env-$node.json"
  "${run[@]}" "docker inspect -f '{{json .Mounts}}' $C" > "$D/docker-mounts-$node.json"
  "${run[@]}" "docker inspect -f '{{.Image}} {{.Config.Image}} {{.State.StartedAt}}' $C" > "$D/docker-image-$node.txt"
done
python3 - "$D" <<'EOF' > "$D/docker-args-check.txt"
import json, sys
d = sys.argv[1]
def load(n):
    return (json.load(open(f"{d}/docker-args-{n}.json")), json.load(open(f"{d}/docker-env-{n}.json")),
            json.load(open(f"{d}/docker-mounts-{n}.json")), open(f"{d}/docker-image-{n}.txt").read().split())
def flags(args):
    out, i = {}, 1
    while i < len(args):
        a = args[i]
        if a.startswith("--"):
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                out[a] = args[i + 1]; i += 2
            else:
                out[a] = True; i += 1
        else:
            i += 1
    return out
H, W = load("head"), load("worker")
drop = {"--node-rank", "--host", "--port", "--chat-template", "--headless"}
for name, (args, env, mounts, img) in (("head", H), ("worker", W)):
    f = flags(args)
    e = dict(x.split("=", 1) for x in env)
    for k in ("--speculative-config", "--compilation-config", "--kv-cache-memory", "--max-num-seqs",
              "--max-model-len", "--moe-backend", "--linear-backend", "--limit-mm-per-prompt",
              "--mm-processor-cache-gb", "--override-generation-config", "--node-rank"):
        print(f"{name:6} {k:30} {f.get(k)}")
    for k in sorted(e):
        if k.startswith(("GLM53_", "MAX_JOBS", "FLASHINFER_", "VLLM_CACHE", "DG_", "TRITON_", "TILELANG_", "NCCL_IB_HCA")):
            print(f"{name:6} env {k}={e[k]}")
    print(f"{name:6} GLM53_* env count: {sum(k.startswith('GLM53_') for k in e)}")
    print(f"{name:6} async flag present: {any('async-scheduling' in a for a in args)}")
    print(f"{name:6} mounts: {[m['Source'] + ':' + m['Destination'] for m in mounts]}")
    print(f"{name:6} image: {img}")
fa = {k: v for k, v in flags(H[0]).items() if k not in drop}
fb = {k: v for k, v in flags(W[0]).items() if k not in drop}
print("args parity (minus node-rank/host/port/chat-template/headless):", "OK" if fa == fb else f"DIFF {set(fa.items()) ^ set(fb.items())}")
ea = {x for x in H[1] if not x.startswith("VLLM_HOST_IP=")}
eb = {x for x in W[1] if not x.startswith("VLLM_HOST_IP=")}
print("env parity (minus VLLM_HOST_IP):", "OK" if ea == eb else f"DIFF {sorted(ea ^ eb)}")
print("image parity:", "OK" if H[3][0] == W[3][0] else f"DIFF {H[3][0]} {W[3][0]}")
EOF
cat "$D/docker-args-check.txt"

#!/usr/bin/env bash
# Every command run for the GLM re-bench 20260902T204243Z, verbatim, in order. cwd = repo root.
~/projects/ai-lab/forge/gpu/lease.sh acquire pair --ttl 3h --unit rebench-glm
UTC=20260902T204243Z
mkdir -p evidence/rebench-$UTC
.cursor/skills/verify-glm53-flash/scripts/doctor.sh > evidence/rebench-$UTC/doctor-preboot.txt 2>&1; echo exit=$? >> evidence/rebench-$UTC/doctor-preboot.txt
mkdir -p .cursor/skills/verify-glm53-flash/.run-state
./run.sh > evidence/rebench-$UTC/run.log 2>&1
.cursor/skills/verify-glm53-flash/scripts/doctor.sh > evidence/rebench-$UTC/doctor.txt 2>&1; echo exit=$? >> evidence/rebench-$UTC/doctor.txt
printf 'started %s host=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(hostname -s)" > .cursor/skills/verify-glm53-flash/.run-state/started
{ echo "spark1:"; docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}'; echo "spark2:"; ssh -o BatchMode=yes spark2 "docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}'"; } > evidence/rebench-$UTC/containers-before.txt
.cursor/skills/verify-glm53-flash/scripts/smoke.sh > evidence/rebench-$UTC/smoke.txt 2>&1; echo exit=$? >> evidence/rebench-$UTC/smoke.txt
python3 .cursor/skills/verify-glm53-flash/scripts/count_probe.py > evidence/rebench-$UTC/count.txt 2>&1; echo exit=$? >> evidence/rebench-$UTC/count.txt
python3 .cursor/skills/verify-glm53-flash/scripts/thinking_off_probe.py > evidence/rebench-$UTC/thinking-off.txt 2>&1; echo exit=$? >> evidence/rebench-$UTC/thinking-off.txt
python3 .cursor/skills/verify-glm53-flash/scripts/tool_call_probe.py > evidence/rebench-$UTC/tool-call.txt 2>&1; echo exit=$? >> evidence/rebench-$UTC/tool-call.txt
python3 .cursor/skills/verify-glm53-flash/scripts/hermes_probe.py > evidence/rebench-$UTC/hermes.txt 2>&1; echo exit=$? >> evidence/rebench-$UTC/hermes.txt
cp -r .cursor/skills/verify-glm53-flash/artifacts/serve-smoke/20260902T210611Z evidence/rebench-$UTC/smoke-artifacts
python3 bench_decode.py > evidence/rebench-$UTC/bench.txt 2>&1
# bench.json = the SUMMARY JSON array extracted from bench.txt
python3 .cursor/skills/verify-glm53-flash/scripts/needle_probe.py --prompt-tokens 8192 > evidence/rebench-$UTC/needle-8192.txt 2>&1; echo exit=$? >> evidence/rebench-$UTC/needle-8192.txt
python3 .cursor/skills/verify-glm53-flash/scripts/needle_probe.py --prompt-tokens 20480 --concurrency 2 > evidence/rebench-$UTC/needle-20480-c2.txt 2>&1; echo exit=$? >> evidence/rebench-$UTC/needle-20480-c2.txt
docker logs --tail 500 glm53-flash-nvfp4 > evidence/rebench-$UTC/engine.log.tail 2>&1
ssh -o BatchMode=yes spark2 docker logs --tail 500 glm53-flash-nvfp4 > evidence/rebench-$UTC/engine-rank1.log.tail 2>&1
{ docker inspect -f 'rank0 OOMKilled={{.State.OOMKilled}} status={{.State.Status}}' glm53-flash-nvfp4; ssh -o BatchMode=yes spark2 "docker inspect -f 'rank1 OOMKilled={{.State.OOMKilled}} status={{.State.Status}}' glm53-flash-nvfp4"; } > evidence/rebench-$UTC/oomkilled.txt
free -g > evidence/rebench-$UTC/free-after-bench.txt
./stop.sh > evidence/rebench-$UTC/stop.txt 2>&1
{ echo "spark1:"; docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}'; echo "spark2:"; ssh -o BatchMode=yes spark2 "docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}'"; } > evidence/rebench-$UTC/containers-after.txt

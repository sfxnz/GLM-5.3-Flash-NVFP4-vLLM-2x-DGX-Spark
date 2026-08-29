# Serve start and stop

`./run.sh` brings up tensor-parallel 2 across `spark1` (rank 0, API on port 8000) and `spark2` (headless rank 1). `./stop.sh` removes the `glm53-flash-nvfp4` container on both nodes. Ready means `/v1/models` answers.

## Sub-features

- `start-head` starts rank 0 on the head Spark and waits until `/v1/models` answers.
- `start-worker` starts rank 1 on `spark2` before the head (auto SSH, or `ROLE=worker ./run.sh`).
- `start-image-guard` refuses to run when `glm53-sm121-v11` is missing instead of pulling stock vLLM.
- `stop-both` removes the named container locally and on `spark2`.

## How to get to it (user POV)

- On the head Spark: `./run.sh`.
- Manual split: `ROLE=worker ./run.sh` on `spark2`, then `ROLE=head ./run.sh` on `spark1`.
- Stop from the head: `./stop.sh`.
- MTP instead of DFlash2: `SPEC=mtp ./run.sh`.

## Driving it with verify-glm53

Preconditions:

- Doctor has been run.
- If `status=ready`, `loading`, or `mismatch`, or `owned_by_verify=0` with a container present: **do not drive this feature**. Report `verified-unreachable` with that doctor output. Starting would `docker rm -f` the lab serve; stopping would kill a serve this run did not start.
- Drive only when `status=missing` and you intend to own the instance. After a successful `./run.sh`, write `.cursor/skills/verify-glm53-flash/.run-state/started` as in the skill Launch section.

- **Image guard.** With `IMAGE` pointing at a tag that is not present, `./run.sh` exits non-zero and prints `Do not use stock vllm/vllm-openai on sm_121`. Do not actually run this against the lab default while the lab container exists.
- **Start.** From the repo root, `./run.sh`. The script prints `Ready → http://127.0.0.1:8000/v1`. Doctor then prints `status=ready`, `image` matching `glm53-sm121-v11`, `worker=up`.
- **Stop.** `.cursor/skills/verify-glm53-flash/scripts/cleanup.sh` (or `./stop.sh` when `owned_by_verify=1`). Doctor then prints `status=missing`. `docker ps` on both nodes has no `glm53-flash-nvfp4`.
- **Proof.** Save the `./run.sh` tail (ready line + `/v1/models` JSON), pre/post doctor dumps, and the stop transcript under `artifacts/serve-start-stop/<stamp>/`.

## Gotchas

- `./run.sh` always `docker rm -f`s `glm53-flash-nvfp4` on the local node before starting. That is why attach-or-refuse is mandatory.
- Unpinned `NCCL_IB_HCA` on GB10 can pick a DOWN HCA. Do not “fix” a start failure by unsetting `HCA`.
- First boot downloads/warm-loads weights: 15–20 minutes when the HF cache is warm, longer when not. `wait_ready` is the signal, not a fixed sleep.
- `SPEC=dflash2` needs the pinned draft snapshot; without `hf` and without that snapshot, `run.sh` exits before `docker run`.
- Cleanup of a verify-owned serve uses `./stop.sh`, not `kill` by process name.

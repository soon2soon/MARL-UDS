# MARL-UDS + AMAF (amaf branch)

Fork of [Zhiyu014/MARL-UDS](https://github.com/Zhiyu014/MARL-UDS) (Zhang
et al. 2023, *Water Research* 229:119498) with the AMAF agent
([soon2soon/AMAF](https://github.com/soon2soon/AMAF)) added as a new
strategy trained and evaluated under the paper's exact protocol.

## What's on this branch

- `storm/agent/amaf.py` — AMAF-DQN port (multi-head dueling + adaptive
  gate + uniform-anchored fusion), same interface as `agent.dqn.DQN`
- `storm/utils/config.yaml` — `AMAF` strategy entry (hyperparameters
  identical to the paper's DQN; `if_mac: True` per-valve heads
  8/7/6/7=28 outputs, matching the shipped pretrained DQN checkpoint)
- Seeded training: `train_astlingen.py` reads `SEED` (default 11)
- `docker/` — turnkey reproduction environment

## Quick start (Docker)

**Full install/run/resume/multi-seed guide: [`docker/RUN_GUIDE.md`](docker/RUN_GUIDE.md) (한국어)**

```bash
git clone https://github.com/soon2soon/MARL-UDS.git
cd MARL-UDS && git checkout amaf
docker build -t marl-uds:amaf -f docker/Dockerfile .

# Train AMAF, seed 11, results appear in ./storm/model/astlingen_AMAF
docker run --rm -v "$PWD/storm/model:/work/storm/model" \
    -e ALGO=AMAF -e SEED=11 marl-uds:amaf

# Zhang baseline for the same protocol
docker run --rm -v "$PWD/storm/model:/work/storm/model" \
    -e ALGO=DQN -e SEED=11 marl-uds:amaf
```

`ALGO` picks the strategy (`AMAF` | `DQN` | `VDN` | `IQL`); the
container edits `train:` in `utils/config.yaml` accordingly and seeds
Python/numpy/TF RNGs from `SEED`.

Notes:
- The image pins `numpy==1.18.5` (Zhang's soft-update code uses
  inhomogeneous `asarray`, which hard-errors on numpy >= 1.24).
- TF 2.3 wheels exist for x86-64 Linux; on Apple silicon we run
  TF 2.13 in the local `zhang` conda env instead (weights are
  load-compatible).
- GPU: not required — the paper's setup is CPU training.

## Protocol alignment (devnote summary)

| item | Zhang 2023 (reproduced) | this branch |
|---|---|---|
| engine | pyswmm 1.5.1 (EPA-SWMM 5.1) | same, unmodified |
| action space | per-valve presets 8/7/6/7 (sum 28) | same, `if_mac: True` |
| control interval | 5 min | same (env default) |
| training events | 50 measured events, 5–15 mm | same, unmodified |
| eval | weighted flooding perf vs BC | same harness, unmodified |
| verified DQN reduction vs BC | +9.90% (one eval event, pretrained ckpt) | re-runnable via `ALGO=DQN` |
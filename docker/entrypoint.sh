#!/bin/bash
# [amaf branch] Entry point: seed + select algorithm via env vars.
#   ALGO: AMAF | DQN | VDN | IQL  (default AMAF)
#   SEED: RNG seed                  (default 11)
#   EXTRA_ARGS: appended to the python call
set -e

ALGO="${ALGO:-AMAF}"
SEED="${SEED:-11}"

export PYTHONPATH=/work/storm
cd /work/storm

# Set algorithm in config: yaml 'train:' key picks the strategy section
python - "$ALGO" <<'PYEOF'
import re
import sys

algo = sys.argv[1]
path = '/work/storm/utils/config.yaml'
with open(path, 'r') as fh:
    text = fh.read()
# Only replace the top-level 'train:' directly under 'astlingen:'
text = re.sub(
    r"(?m)^astlingen:\n  train: \S+",
    f"astlingen:\n  train: {algo}",
    text, count=1)
with open(path, 'w') as fh:
    fh.write(text)
print(f'config train: -> {algo}')
PYEOF

exec python -u train_astlingen.py ${EXTRA_ARGS:-}
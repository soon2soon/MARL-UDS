#!/bin/bash
# [amaf branch] Entry point: seed + select scenario/algorithm via env vars.
#   SCENARIO: astlingen | chaohu  (default astlingen)
#   ALGO:  AMAF | DQN | VDN | IQL  (default AMAF)
#   SEED:  RNG seed                (default 11)
#   RESUME: 1 = resume from saved weights + replay memory (if_load: True)
#
# Results go to the seed-scoped working directory, e.g.
#   storm/model/astlingen_AMAF_seed11/   (host-mounted volume)
# so concurrent multi-seed containers never collide.
set -e

SCENARIO="${SCENARIO:-astlingen}"
ALGO="${ALGO:-AMAF}"
SEED="${SEED:-11}"
RESUME="${RESUME:-0}"

export PYTHONPATH=/work/storm
cd /work/storm

python - "$SCENARIO" "$ALGO" "$SEED" "$RESUME" <<'PYEOF'
import os
import sys

scenario, algo = sys.argv[1], sys.argv[2]
seed, resume = int(sys.argv[3]), sys.argv[4] == '1'
path = os.environ.get('CONFIG_PATH', 'utils/config.yaml')
lines = open(path).read().splitlines(keepends=True)

# Bound the scenario block (top-level keys have no leading space).
start = next(i for i, ln in enumerate(lines)
             if ln.rstrip() == scenario + ':')
end = next((i for i in range(start + 1, len(lines))
            if lines[i].strip() and not lines[i].startswith(' ')
            and lines[i].rstrip().endswith(':')), len(lines))

# 1) flip the strategy selector inside the scenario block only
for j in range(start + 1, min(start + 3, len(lines))):
    if lines[j].lstrip().startswith('train:'):
        lines[j] = '  train: %s\n' % algo
        break

# 2) seed-scope the strategy's working directory so concurrent seed
#    runs never collide (astlingen_AMAF -> astlingen_AMAF_seed11)
sec = '  %s:' % algo
for i in range(start + 1, end):
    if lines[i].rstrip() == sec:
        for j in range(i + 1, end):
            s = lines[j]
            if s.startswith('    cwd:'):
                val = s.split(':', 1)[1].split('#')[0].strip()
                lines[j] = s.replace(val, '%s_seed%d' % (val, seed), 1)
                print('cwd -> %s_seed%d' % (val, seed))
                break
            if s.strip() and not s.startswith('    '):
                break  # left the strategy section
        break

# 3) optional resume: flip if_load inside the same section only
if resume:
    for i in range(start + 1, end):
        if lines[i].rstrip() == sec:
            for j in range(i + 1, end):
                s = lines[j]
                if s.startswith('    if_load:'):
                    lines[j] = s.replace('if_load: False', 'if_load: True')
                    print('if_load -> True (resume)')
                    break
                if s.strip() and not s.startswith('    '):
                    break
            break

open(path, 'w').write(''.join(lines))
print('%s.train -> %s' % (scenario, algo))
PYEOF

if [ "$SCENARIO" = "chaohu" ]; then
    exec python -u train_chaohu.py
else
    exec python -u train_astlingen.py
fi
# [amaf branch] Dockerfile for MARL-UDS + AMAF experiments.
# Reproduces Zhang et al. 2023's original training stack exactly
# (their requirements.txt pins, incl. TF 2.3 wheels that exist for
# x86-64 Linux — unavailable on modern macOS arm64).
#
# Build:  docker build -t marl-uds:amaf .
# Run:    docker run --rm -v $PWD/storm/model:/work/storm/model \
#             -e ALGO=AMAF -e SEED=11 marl-uds:amaf
#
# The image sets utf-8, seeds from $SEED, and runs train_astlingen.py
# with the algorithm chosen by $ALGO (AMAF | DQN | VDN | IQL).

FROM python:3.8-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

# SWMM engine build deps + runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ make libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

# Install the exact 2021 stack first (layers cache well)
COPY requirements.txt /work/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    # numpy pin: Zhang's qagent soft-update uses numpy.asarray on
    # get_weights() (inhomogeneous), which hard-errors on numpy>=1.24.
    && pip install --no-cache-dir "numpy==1.18.5"

# Repo (storm/ package, config, network files).
# Pretrained checkpoints (storm/model, 227M) and results (storm/results)
# are excluded via .dockerignore — mount storm/model at runtime instead.
COPY . /work/

WORKDIR /work/storm

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
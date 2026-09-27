# 도커 설치 및 실행 가이드 — MARL-UDS + AMAF (amaf 브랜치)

연구용 PC(사무실)에서 Zhang 2023 프로토콜 그대로 AMAF 학습을
돌리기 위한 절차. 전체 흐름: 도커 설치 → 이미지 빌드 → 실행 →
결과 확인 → (필요시) 중단·재개 → 3시드 병렬.

---

## 0. 준비물 / 사양 권장

- Linux x86-64 (또는 Windows + WSL2, macOS Intel). Apple Silicon 맥은
  가능하지만 x86 이미지라 에뮬레이션으로 느립니다 (학습용 비권장).
- CPU 코어가 많을수록 좋습니다 (pyswmm 시뮬이 multiprocessing pool
  5프로세스로 병렬). 메모리 16GB 이상 권장 — replay buffer가
  최대 1,048,576 transition까지 자랍니다 (수십 GB까지 가능).
  디스크 여유: 시드당 model/ 수 GB + results.
- GPU 불필요 (Zhang 원 설정이 CPU 학습).

## 1. 도커 설치

### Ubuntu / Debian (사무실 PC 전형)

```bash
# 공식 스크립트 설치 (docker engine + compose plugin)
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER   # 그룹 추가 후 재로그인
newgrp docker                   # 또는 로그아웃/로그인
docker run hello-world          # 이 문구가 나오면 성공
```

회사망에서 get.docker.com이 막혀 있으면: 회사 미러 레지스트리가
있는지 IT에 확인 → `/etc/docker/daemon.json`에
`{"registry-mirrors": ["https://<회사미러>"]}` 추가 후
`sudo systemctl restart docker`.

### Windows

1. WSL2 설치: 관리자 PowerShell → `wsl --install` → 재부팅
2. Docker Desktop 설치: https://docs.docker.com/desktop/
   설정에서 "Use the WSL 2 based engine" 확인
3. PowerShell에서 `docker run hello-world` 확인

### root 권한이 없는 서버 (회사 클러스터 등)

rootless mode: https://docs.docker.jp/engine/security/rootless.html
또는 IT에 "docker 그룹 추가"만 요청하면 됩니다 (설치는 이미
되어 있는 경우가 많습니다: `docker --version` 으로 먼저 확인).

## 2. 소스 가져오기 + 이미지 빌드

```bash
git clone https://github.com/soon2soon/MARL-UDS.git
cd MARL-UDS
git checkout amaf

docker build -t marl-uds:amaf -f docker/Dockerfile .
```

- 첫 빌드: 약 10~30분 (pip 설치가 대부분; TF 2.3 휠 ~320MB 포함).
- 빌드가 막히면: (1) 회사망 프록시 → `docker build`에
  `--build-arg HTTP_PROXY=... HTTPS_PROXY=...` 전달,
  (2) pypi 미러 필요시 requirements 설치 줄 앞에
  `-i https://pypi.org/simple` 대신 사내 미러 사용.
- 이미지 크기: 약 2.5~3GB.
- 이 이미지는 **Zhang의 원본 스택 그대로**입니다 (TF 2.3,
  spektral 1.2.0, pyswmm 1.5.1, swmm-api 0.2.0.18.3, pystorms 1.0.0,
  numpy 1.18.5 핀). numpy 핀이 중요합니다: Zhang 코드의
  soft-update가 numpy>=1.24에서 터지는 버그가 있습니다
  (비균질 배열 asarray). 이미지가 이를 방지합니다.

## 3. 학습 실행

```bash
# AMAF, 시드 11 (기본 실험)
docker run --rm \
  -v "$PWD/storm/model:/work/storm/model" \
  -e ALGO=AMAF -e SEED=11 \
  --name amaf_s11 \
  marl-uds:amaf
```

- 결과는 호스트의 `storm/model/astlingen_AMAF_seed11/`에
  기록됩니다 (체크포인트 train/reward/eval 3종, 로그 CSV).
- 로그에서 `Training Reward at event ...`, `Sampling Complete`,
  `Upgrade Complete`가 반복 출력되면 정상 학습 중입니다.
- Zhang DQN 베이스라인도 동일 컨테이너로:
  `-e ALGO=DQN` → `storm/model/astlingen_DQN_graphconv_seed11/`.

### 시간 예상

1 에피소드 = 50개 강우 이벤트 샘플링 + 학습. PC 사양에 따라
에피소드당 수 분~십수 분. 총 5,000 에피소드가 원 설정이며
며칠 규모입니다. 논문 비교에는 전체가 필요하지만, 먼저
"곡선이 떨어지는지" 확인하려면 아래 §6의 중단 방법으로
몇백 에피소드만 봐도 됩니다.

## 4. 백그라운드 실행 (터미널 닫아도 유지)

```bash
docker run -d \
  -v "$PWD/storm/model:/work/storm/model" \
  -e ALGO=AMAF -e SEED=11 \
  --name amaf_s11 \
  marl-uds:amaf

docker logs -f amaf_s11        # 실시간 로그 보기 (Ctrl+C로 빠져나옴)
```

`--rm`을 빼면 컨테이너가 종료 후에도 남습니다 (디버깅 편의).
호스트 재부팅 시 자동 재시작을 원하면 `--restart unless-stopped` 추가.

## 5. 중단 / 재개

```bash
docker stop amaf_s11           # 중단 (체크포인트는 이미 저장된 것까지)
docker rm amaf_s11             # 컨테이너 삭제 (model/ 은 호스트에 안전)

# 재개: RESUME=1 은 if_load: True + 시드별 cwd 유지 → 저장된
# 가중치와 replay memory에서 이어서 학습
docker run -d \
  -v "$PWD/storm/model:/work/storm/model" \
  -e ALGO=AMAF -e SEED=11 -e RESUME=1 \
  --name amaf_s11 \
  marl-uds:amaf
```

주의: 재개는 "마지막 save 시점"부터입니다. 저장 시점은 3가지 —
(a) `save_gap`(100 에피소드)마다 정기 저장(가중치+메모리+로그),
(b) 최고 train reward/perf를 갱신할 때 `train/`·`reward/`,
(c) 최고 eval 성능일 때 `eval/`. 즉 중단 시 최대 100 에피소드치를
다시 돌릴 수 있습니다.

## 6. 학습 곡선 확인 / 조기 판정

```bash
# 에피소드별 성능 로그
ls storm/model/astlingen_AMAF_seed11/
# train/ 디렉터리에 CSV 로그가 쌓입니다 (Trainlogger 기록)
```

CSV에서 (reward, perf, loss) 컬럼을 그래프로 그려 확인.
판정 기준: Zhang DQN의 검증된 개선율 대비 (BC 대비
CSO 5.62~9.30% 감소 밴드, 우리 재현 체크포인트에서는 +9.90%).
AMAF 곡선이 이 밴드에 진입하는지가 1차 관찰 포인트입니다.

## 7. 3시드 병렬 (권장 프로토콜)

컨테이너 3개를 동시에 — 각자 다른 시드, 결과 디렉터리는 자동 분리
(엔트리포인트가 cwd를 `_seed<SEED>`로 만듭니다):

```bash
for SEED in 11 22 33; do
  docker run -d \
    -v "$PWD/storm/model:/work/storm/model" \
    -e ALGO=AMAF -e SEED=$SEED \
    --restart unless-stopped \
    --name amaf_s$SEED \
    marl-uds:amaf
done
docker ps                        # 3개 떠 있는지 확인
docker logs -f amaf_s11          # 하나 골라서 로그 관찰
```

경고: 3컨테이너 × (시뮬 pool 5프로세스 + TF) 입니다. 코어가
부족하면 시드를 순차 실행하세요:

```bash
docker run --rm -v "$PWD/storm/model:/work/storm/model" \
  -e ALGO=AMAF -e SEED=11 marl-uds:amaf \
  && docker run --rm -v "$PWD/storm/model:/work/storm/model" \
  -e ALGO=AMAF -e SEED=22 marl-uds:amaf \
  && docker run --rm -v "$PWD/storm/model:/work/storm/model" \
  -e ALGO=AMAF -e SEED=33 marl-uds:amaf
```

베이스라인 재현도 같은 방식: `-e ALGO=DQN` × 3시드.

## 7.5 실네트워크 시나리오 — Chaohu (실도시 합류식 배수망)

Zhang 리포에는 벤치마크(Astlingen) 외에 **실전 사례**가 포함되어
있습니다: Chaohu(중국 안후이성 차오후시) 실도시 합류식 하수망.
펌프장 2개(CC/JK)의 펌프 7개를 2개 멀티에이전트가 제어
(액션 9×6), 설계강우 시나리오 20개 학습. 학습된 베이스라인
체크포인트(VDN/DQN/IQL)도 리포에 포함되어 있어 비교 기준이
명확합니다. Zhang 2023 논문의 실전 검증 챕터가 바로 이 시나리오로,
"실네트워크에서 AMAF가 작동하는가"의 가장 빠른 검증 무대입니다.

```bash
# Chaohu에서 AMAF 학습
docker run -d \
  -v "$PWD/storm/model:/work/storm/model" \
  -e SCENARIO=chaohu -e ALGO=AMAF -e SEED=11 \
  --name chaohu_amaf_s11 \
  marl-uds:amaf

# Chaohu 베이스라인 (배포 체크포인트 보유)
docker run -d \
  -v "$PWD/storm/model:/work/storm/model" \
  -e SCENARIO=chaohu -e ALGO=VDN -e SEED=11 \
  --name chaohu_vdn_s11 \
  marl-uds:amaf
```

결과 디렉터리: `storm/model/chaohu_AMAF_seed11/` 등. 로그/재개
방식은 §4~§5와 동일합니다. Chaohu는 `train_chaohu.py`를 사용하며
엔트리포인트가 `SCENARIO=chaohu`일 때 자동으로 이 스크립트를
구동합니다.

추가 시나리오: 리포의 `storm/model/chaohu202209/`에 Zhang가
2022-09 버전 Chaohu 모델 추가 체크포인트가 있습니다 (논문 후속
업데이트). 첫 실험은 위의 표준 `chaohu_*` 체크포인트 세트로
하는 것을 권장합니다.

## 8. 문제 해결

| 증상 | 해결 |
|---|---|
| `docker: permission denied` | `sudo usermod -aG docker $USER` 후 재로그인 |
| 빌드 중 pip 다운로드 실패 | 회사망 프록시/미러 (§2 빌드 항목 참조) |
| 메모리 부족 (컨테이너 OOM kill) | replay buffer 상한 축소: config `max_capacity` (기본 2^20=1048576; 예: 524288) — 학습 품질과 트레이드오프, 기록 필요 |
| `Sampling Complete`가 안 나오고 느림 | 코어 경합. 병렬 컨테이너 수를 줄이거나 `processes: 5` (config)를 내리기 (0 또는 1이면 멀티프로세싱 끔) |
| 로그에 numpy 배열 에러 | 이미지가 맞는지 확인: `docker image inspect marl-uds:amaf` — numpy 1.18.5 핀이 없는 구 이미지라면 다시 빌드 |
| 재개가 0부터 시작 | `RESUME=1`을 줬는지 + `storm/model/..._seed11/`에 agent0.h5가 있는지 확인 |

## 9. 무엇을 어디까지 검증했는가 (정직 기록)

- 검증 완료 (Mac TF 2.13 대체 환경): AMAF 에이전트 빌드/순전파/
  act()/액션테이블 매핑/1 사이클 학습(그래디언트 정상, 유한 loss)/
  save-load 재현성(Q차 0.0); entrypoint config 치환(전략 전환·
  시드별 cwd·RESUME·타 전략 무손상, 3 케이스 실 config으로 통과);
  시드 주입 컴파일; Zhang 사전학습 DQN 재현(BC 대비 +9.90%,
  논문 밴드 상단과 일치).
- 검증 미완료 (PC에서 첫 빌드 시 확인 필요): docker 이미지 빌드
  자체, 컨테이너 내부 전체 학습 루프 (이 Mac에는 도커가 없어
  불가했습니다). 빌드 후 §3 로그가 나오면 그 시점부터가 진짜
  검증입니다 — 막히면 로그를 주세요.

---

요약 카드: `docker build -t marl-uds:amaf -f docker/Dockerfile .` →
`docker run -d -v "$PWD/storm/model:/work/storm/model" -e ALGO=AMAF -e SEED=11 --name amaf_s11 marl-uds:amaf` →
`docker logs -f amaf_s11`
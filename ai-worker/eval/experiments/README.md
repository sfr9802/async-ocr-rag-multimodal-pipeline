# eval/experiments — Optuna round-refinement 워크스페이스

이 디렉토리는 round 기반 Optuna 튜닝의 프로젝트 측 artifact 를 들고
있습니다. Optuna 오케스트레이션 자체 (sampler / pruner 생성, trial
디스패치, 번들 export, 스키마 검증, markdown 렌더링) 는 설치된
`optuna-round-refinement` skill 이 소유 —
`C:\Users\sfr99\.claude\skills\optuna-round-refinement\SKILL.md` 참조.

프로젝트 측은 4종의 artifact 만 기여:

1. [`ai-worker/eval/tune_eval.py`](../tune_eval.py) 의
   **`evaluate(params: dict) -> dict` 호출 가능** — 각 round config 가
   `evaluate: "eval.tune_eval:evaluate"` 로 참조.
2. **Round config** — `rounds/round_NN_config.json` 아래, skill 의
   `next_round_config.schema.json` 에 대해 검증됨.
3. **Round 분석** — `rounds/round_NN_analysis.md` 아래 (완료된 round
   당 1개, `/analyze-study` 흐름이 생성).
4. **Study 영수증** — 각 study 디렉토리 최상위:
   `FINAL_BEST.json` 와 `STUDY_SUMMARY.md`.

## 디렉토리 구조

```
eval/experiments/
├── README.md                ← 이 파일
├── active.yaml              ← legacy scripts/tune.py 탐색 공간 (별도 시스템)
├── rounds/                  ← 커밋: config + 분석
│   ├── round_01_config.json
│   ├── round_01_analysis.md
│   ├── round_02_config.json
│   └── ...
├── run_output/              ← gitignore: 실행별 skill 출력
│   ├── .gitkeep
│   ├── study_bundle.json
│   └── llm_input.md
└── studies/                 ← legacy scripts/tune.py study artefact (replay 용으로 유지)
```

## Round cadence

- **git 브랜치당 1 round**, 또는 최소한 **commit 당 1 round**.
  round_NN_config.json 이 입력; 분석 markdown + 다음 round_(NN+1)_config.json
  이 출력.
- 번들 (`study_bundle.json`) 과 렌더링된 `llm_input.md` 는
  `run_output/` 아래에 살고 gitignore 됨 — 매 실행마다 skill 이 재생성.
- Config (`rounds/round_NN_config.json`) 는 어떤 reviewer 든 round 를
  주도한 정확한 탐색 공간과 provenance 해시를 재현할 수 있도록 커밋됨.
- 분석 markdown (`rounds/round_NN_analysis.md`) 과 study 영수증
  (`FINAL_BEST.json`, `STUDY_SUMMARY.md`) 은 디스크의 SQLite study DB
  없이도 round-to-round 의 narrative 가 살아남도록 커밋됨.

## Round 주도

`ai-worker/` 에서:

```bash
python "C:/Users/sfr99/.claude/skills/optuna-round-refinement/scripts/round_runner.py" run \
    --config eval/experiments/rounds/round_01_config.json \
    --out-bundle eval/experiments/run_output/study_bundle.json \
    --out-llm-input eval/experiments/run_output/llm_input.md \
    --evaluate-search-path .
```

Runner 가 `eval.tune_eval:evaluate` 를 import 하고, 공유 retriever
번들에 대해 `n_trials` 개의 Optuna trial 을 주도하고, 스키마 검증된
번들과 렌더링된 LLM 입력을 작성한 뒤 제어를 돌려줌. Round 분석과 다음
round 의 config 는 번들만 가지고 동작하는 skill 의 분석가 prompt
(`/analyze-study`, `/propose-next`) 가 생성.

## 이 디렉토리가 들고 있지 **않은** 것

- Optuna sampler / pruner / suggest_* 코드 — skill 소유.
- 스키마 파일 — skill 소유 (`schemas/next_round_config.schema.json`,
  `schemas/study_bundle.schema.json`).
- 템플릿 — skill 소유 (`templates/llm_input.md`,
  `templates/round_report.md`).
- Eval harness 내부 — [`ai-worker/eval/harness/`](../harness/) 참조.

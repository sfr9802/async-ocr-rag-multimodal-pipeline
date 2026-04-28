# Optuna + Claude-interpreter 튜닝 loop — 계획

Optuna 가 수치 탐색을 주도하고 Claude (이후 Claude Code 세션에서) 가
interpreter 역할을 하는 — study 결과를 읽고, narrative 요약을 작성하고,
다음 round 의 탐색 공간을 사람 검토용 YAML diff 로 제안하는 — 로컬
하이퍼파라미터 튜닝 파이프라인의 구현 로드맵.

**핵심 원칙:** LLM 은 interpreter 이지 sampler 가 아닙니다. Optuna 가
trial 파라미터를 고르고, Claude 는 결과를 분석하고 정제를 제안합니다.
`active.yaml` 의 모든 mutation 은 사람의 승인을 통과합니다.

## Phase 1 — 핵심 Optuna driver (cheap params 만)

- `ai-worker/scripts/tune.py` — 기존 eval harness (`eval.run_eval`) 를
  감싸는 YAML 주도 Optuna runner.
- `ai-worker/eval/experiments/active.yaml` — `experiment_id`,
  `objective`, `search_space`, `optuna`, `_meta` 섹션을 가진 탐색 공간
  선언.
- 타깃 파라미터 (재임베딩 불필요):
  - `rag_top_k`
  - `rrf_k` (cross-modal)
  - `max_query_chars`, `short_query_words`, `max_fused_chunk_chars`
  - `excerpt_chars`
  - `ocr_min_confidence_warn`
  - `multimodal_max_vision_pages`
- trial 별 `cost_usd`, `latency_ms`, `secondary_metric_values`,
  `config_hash` 를 위한 `trial.set_user_attr`.
- SQLite study 저장:
  `ai-worker/eval/experiments/studies/<exp-id>/study.db`.

## Phase 2 — 시각화 + narrative stub

- `ai-worker/scripts/summarize_study.py` —
  `optuna.visualization.matplotlib.*` 경유 matplotlib PNG 생성
  (`optimization_history`, `param_importances`, 파라미터별 `slice`,
  top-2 `contour`).
- 명시적 Claude-narrative placeholder
  (`<!-- claude-narrative:top-trial-pattern -->`,
  `param-importances`, `next-direction`) 가 있는 `summary.md` stub.
- 라이브 브라우징을 위한 `optuna-dashboard`:
  `optuna-dashboard sqlite:///ai-worker/eval/experiments/studies/<exp>/study.db`.
- 다음을 다루는 `docs/tuning.md` 작성: 첫 study 시작, 대시보드 사용,
  cheap-vs-expensive 파라미터 구분, 사람 승인 게이트.

## Phase 3 — Claude Code 통합

- `.claude/commands/tune-round.md` — `scripts/tune.py` 를 감싸고,
  자동으로 summarize 호출, gitignore 된 `study.db` 를 제외한
  `studies/<id>/` 커밋.
- `.claude/commands/analyze-study.md` — study.db + `summary.md` 읽고,
  narrative placeholder 채우고, 채워진 summary 커밋.
- `.claude/commands/propose-next.md` — 새 `active.yaml` 을 **채팅 diff
  로만** emit (`active.yaml` 에 직접 `Edit` 안 함). 각 변경은 정당화를
  들고 와야 함.

## Phase 4 — 선택적 자동화

- 반복 tune → analyze 사이클을 스케줄링하는 `loop` skill.
- 매 세 번째 round 는 로컬 옵티멈 함정을 피하기 위해 자동으로
  `RandomSampler` wide-random round 를 주입.
- 사람 승인은 여전히 `active.yaml` mutation 을 게이팅.

## Phase 5 — 보류: expensive 파라미터

- `embedding_model`, chunk `MIN_CH/MAX_CH/OVERLAP`, query/passage
  prefix 를 탐색 공간에 추가.
- `(embedding_model, chunk_size, overlap)` 해시로 key 된 인덱스 캐시가
  `rag-data/<hash>/{faiss.index, build.json}` 에 필요 — trial 들이
  재빌드가 아니라 재사용.
- 2-phase study 패턴: 외부 expensive sweep 이 Pareto 후보를 선정,
  내부 cheap sweep 이 각각을 정제.

## 가드레일 (코드 + 명령 문서에 인코딩)

- `scripts/tune.py` 는 `--resume` 이 전달되지 않으면 기존 `study.db`
  덮어쓰기를 거부.
- `propose-next` 명령은 Claude 에게 명시적으로 diff 를 출력하라고
  지시, 절대 `active.yaml` 에 직접 `Edit` 호출하지 않음.
- `active.yaml._meta.created_by` 가 provenance 추적
  (`human` | `claude-proposed` | `claude-approved`).
- `.gitignore` 가 `study.db` + `plots/` 는 제외하지만 `summary.md` 와
  `config.yaml` 은 유지 (결정적 재생은 바이너리 study 파일이 아니라
  YAML 을 요구).

## 범위 외 (명시적)

- Agent SDK daemon — 슬래시 명령 + `loop` skill 만으로 충분.
- Weights & Biases / MLflow / 어떤 클라우드 통합 — 모두 로컬.
- GCS 스토리지 어댑터 — 튜닝과 무관.
- Expensive params 용 인덱스 캐싱 — Phase 5, 킥오프 task 에 없음.

## 우선순위 정렬

Phase 1 → Phase 2 → Phase 3 을 단일 킥오프 task 로. Phase 4 와
Phase 5 는 baseline loop 이 진짜 데이터에 대해 운동되고 나서 후속
세션에서 도착.

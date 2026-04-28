# 하이퍼파라미터 튜닝 — Optuna + Claude interpreter

이 파이프라인은 **Optuna TPE sweep + Claude narrative interpreter** loop
로 eval harness 를 튜닝합니다. 역할 분리는 의도적입니다:

- **Optuna** 가 수치 탐색을 담당. TPE sampler, seed 고정, 재현 가능.
- **Claude** (`/analyze-study` 와 `/propose-next` 세션에서) 가 study 를
  읽고, 사람이 읽을 수 있는 요약을 작성하고, 다음 탐색 공간을 YAML
  diff 로 제안.
- **사람의 승인** 이 `active.yaml` 의 모든 mutation 을 게이팅. Claude 는
  in-place 로 절대 수정하지 않음.

핵심 원칙: **LLM 은 interpreter 이지 sampler 가 아니다.**

## 디렉토리 구조

```
ai-worker/eval/experiments/
├── active.yaml                 # 현재 탐색 공간 — git 트래킹
├── studies/<experiment_id>/
│   ├── study.db                # Optuna SQLite — gitignore
│   ├── plots/*.png             # matplotlib 시각화 — gitignore
│   ├── summary.md              # 메트릭 표 + Claude narrative — 트래킹
│   └── config.yaml             # frozen 탐색 공간 스냅샷 — 트래킹
└── archive/                    # 과거 active.yaml 버전
```

`summary.md` + `config.yaml` 은 결정적 재생 기록을 줍니다: seed, 정확한
탐색 경계, 그리고 best trial. 바이너리 `study.db` 는 발견을 재현하는
데 필요하지 않음 — 누가 삭제해도 frozen `config.yaml` 에 대해
`scripts.tune` 을 다시 돌리면 같은 trial 들이 재생성됩니다.

## 첫 study 시작

`ai-worker/` 에서 (먼저 venv 활성화):

```bash
# 0. 개발 dep — 1회
pip install -r requirements.txt -r requirements-dev.txt

# 1. 기존 테스트가 여전히 통과하는지 확인
pytest tests/ -q

# 2. 초기 탐색 공간 검토
cat eval/experiments/active.yaml

# 3. 작은 budget 으로 스모크 테스트
python -m scripts.tune --experiment rag-cheap-sweep-v1 --n-trials 5

# 4. 요약 + plot 렌더링
python -m scripts.summarize_study --experiment rag-cheap-sweep-v1

# 5. 인터랙티브 탐색을 위해 브라우저에서 study 열기
optuna-dashboard sqlite:///eval/experiments/studies/rag-cheap-sweep-v1/study.db
```

진짜 round 에서는 `--n-trials 5` 를 빼고 `active.yaml` 의
`optuna.n_trials` (기본 50) 가 주도하게 두세요.

## Round-by-round loop

튜닝 1 round 는 순서대로 invoke 되는 3개의 Claude Code 슬래시 명령:

1. **`/tune-round <experiment-id>`** — `scripts.tune` 후
   `scripts.summarize_study` 실행, `summary.md` + `config.yaml` 커밋.
   아직 narrative 없음.

2. **`/analyze-study <experiment-id>`** — Claude 가 study + summary 를
   읽고 `summary.md` 의 세 narrative placeholder 를 채우고, 채워진
   summary 커밋.

3. **`/propose-next <experiment-id>`** — Claude 가 제안된 새
   `active.yaml` 을 변경마다 정당화 bullet 과 함께 통합 **diff 로 채팅
   에서만** 출력. 사람이 검토하고, 필요하면 수정하고, 수동으로 적용
   하고, `_meta.created_by` 를 `claude-approved` 로 뒤집은 뒤,
   다음 experiment ID 로 `/tune-round` 로 돌아옴.

### 매 세 번째 round 는 random-sampler round 여야 함

TPE 는 20-30 trial 후 좋아 보이는 영역으로 편향됩니다. 로컬 옵티멈에
머무르지 않게 하려면, **매 세 번째 round 는 wide exploration 을 위해
`RandomSampler` 를 사용해야** 합니다:

```bash
python -m scripts.tune --experiment <exp> --random-sampler
```

`/tune-round` 슬래시 명령은 과거 round 를 카운트하고 round 3, 6, 9,
... 에 자동으로 `--random-sampler` 를 주입 — 좁은 영역을 망치질해야
하면 명시적 인자로 override 가능.

## 언제 cheap vs. expensive 파라미터를 사용할지

### Phase 1 — cheap params (현재 sweep)

이것들은 재임베딩이나 인덱스 재빌드를 트리거하지 않습니다. trial 은
eval 실행 자체에 의해 묶임 (한국어 샘플에서는 초 단위):

- `rag_top_k`
- `cross_modal_rrf_k` (MULTIMODAL 모드, cross-modal 만)
- `ocr_min_confidence_warn`
- `multimodal_max_vision_pages`

capability 설정에는 살지만 `WorkerSettings` 에 환경변수로 아직 wire
되지 않은 Phase 2 knob:

- `max_query_chars`, `short_query_words` (fusion)
- `max_fused_chunk_chars` (multimodal capability)
- `excerpt_chars` (extractive generator)

`tune.py` 가 각각에 대해 환경변수를 설정해 YAML 이 자체 문서화 되도록
유지하지만, 누가 `app/core/config.py` 에 wire 할 때까지 eval 서브
프로세스는 무시. `active.yaml` 의 TODO 참조.

### Phase 2 — expensive params (보류)

이것들은 trial 마다 FAISS 인덱스를 재빌드. 인덱스 캐시
(`(embedding_model, chunk_size, overlap)` 으로 key 된
`rag-data/<hash>/{faiss.index, build.json}`) 가 빌드될 때까지 활성화
하지 마세요:

- `rag_embedding_model` — bge-m3 ↔ multilingual-e5-small 전환은 풀
  재임베딩.
- `rag_chunk_min_chars` / `rag_chunk_max_chars` / `rag_chunk_overlap`
  — 재 chunk + 재임베딩.
- `rag_embedding_prefix_query` / `rag_embedding_prefix_passage`.

이 phase 에 도달했을 때 패턴은:

1. 큰 재빌드 파라미터를 탐색 공간에 가진 외부 "expensive" study.
   작은 `n_trials` (≤ 20) 로 한 줌의 인덱스 후보를 Pareto-select.
2. 후보별로 Phase 1 파라미터를 튜닝하는 내부 "cheap" study.

전체 계획은 `docs/optuna-tuning-plan.md` Phase 5 참조.

## `active.yaml` 의 사람 승인 게이트

`active.yaml._meta.created_by` 가 provenance 를 담음:

| 값 | 의미 |
| --- | --- |
| `human` | 초기 상태 — 사람이 작성. |
| `claude-proposed` | `/propose-next` 가 draft (사람이 diff 를 적용하지 않으면 디스크에 닿지 않아야 함). |
| `claude-approved` | 사람이 Claude 제안을 검토하고 다음 round 를 그것으로 실행 중. |

`active.yaml` 을 변경하는 커밋이 `_meta.created_by:
claude-proposed` 로 보이면, 누군가 플래그를 뒤집지 않은 채 Claude
diff 를 받아들였다는 뜻 — 커밋을 reject 하고 플래그 업데이트를 요청
하세요. 게이트의 요점은 `claude-approved` 문자열이 사람이 diff 를
보았다는 증거라는 것입니다.

## 실행 중인 study 모니터링

### optuna-dashboard (인터랙티브, 로컬)

```bash
optuna-dashboard sqlite:///eval/experiments/studies/<exp>/study.db
```

자동 새로고침. `summarize_study.py` 가 emit 하는 같은 plot 에 trial
별 검사와 hyperparameter-relationship 뷰 추가.

### `study.db` 의 빠른 tail

```bash
python -c "
import optuna
s = optuna.load_study(
    study_name='<exp>',
    storage='sqlite:///eval/experiments/studies/<exp>/study.db',
)
for t in s.trials[-5:]:
    print(t.number, t.state.name, t.value, t.params)
"
```

## Resuming vs. restarting

- `--resume` 없는 `python -m scripts.tune --experiment <exp>` 는 기존
  `study.db` 덮어쓰기를 **거부**. 의도적임 — study 는 누적은 싸지만
  잃기는 고통스러움.
- 같은 study 에 trial 을 추가하려면 `--resume` 전달. frozen
  `config.yaml` 은 보존; 변경된 `active.yaml` 이 resume 시 효력을
  발휘하리라 기대하지 마세요 (재현 불가능해짐).
- 진정으로 restart: 그 experiment 의 `study.db` 삭제 (선택적으로
  `plots/` + `summary.md` 도), 그 다음 `--resume` 없이 재실행.

## 트러블슈팅

- **"active.yaml experiment_id does not match --experiment"**: 파일이
  다음 버전으로 bump 됐지만 CLI 플래그가 여전히 옛 것을 가리킴. CLI
  플래그를 일치시킴.
- `summarize_study` 의 **"No study.db at …"**: `scripts.tune` 이 그
  experiment 에 대해 실행되지 않았거나 studies root 가 잘못됨 (CWD 가
  `ai-worker/` 여야 함).
- **Param-importances plot 누락**: 완료된 trial 이 너무 적음. 실제로
  변동한 param 위에 적어도 한 줌의 trial 이 필요.
- **"eval subprocess exited N"**: 하부 eval harness 가 실패함 —
  `scripts.tune` 가 출력한 stderr tail 확인, 그 다음
  `python -m eval.run_eval <mode> --dataset <...>` 로 직접 디버그.

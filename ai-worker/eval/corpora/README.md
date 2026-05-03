# Retrieval 코퍼스

각 서브디렉토리는 retrieval-eval harness 가 인덱싱하고 eval query 를
점수 매기는 대상 문서 집합인 **retrieval 코퍼스** 입니다. 같은 코퍼스를
여러 query 셋으로 테스트하고 같은 query 셋을 나중에 다른 코퍼스에
다시 가리킬 수 있도록 코퍼스는 eval query
([`../eval_queries/`](../eval_queries/) 참조) 와 분리되어 있습니다.

현재 active Phase 7 기준선은 `namu-v4-structured-combined/` 입니다.
`anime_namu_v3*` 디렉토리는 historical reproduction / migration
provenance 용도로만 유지합니다.

## 디렉토리 구조

```
corpora/
├── namu-v4-structured-combined/
│   ├── pages_v4.jsonl          canonical v4 pages (gitignored 가능)
│   ├── chunks_v4.jsonl         structured v4 chunks (gitignored 가능)
│   ├── rag_chunks.jsonl        production retrieval / answerability join 기준
│   ├── split_manifest.json
│   ├── split_manifest.report.json
│   └── validation_report.json
├── anime_namu_v3/
│   ├── README.md               legacy v3 source 경로, 스키마, re-stage 지침
│   └── corpus.jsonl            (gitignore — 261 MB; 필요 시 stage)
└── README.md                   이 파일
```

## 규약

| 규칙 | 이유 |
|---|---|
| 각 코퍼스 디렉토리 안의 정규 파일명은 `corpus.jsonl` | 도구가 manifest 조회 없이 찾을 수 있음 |
| `.jsonl` 자체는 gitignore; `README.md` 는 커밋 | 코퍼스가 큼 (>100 MB 흔함) 그리고 재배포 불가능한 upstream 출처를 가짐 |
| 모든 코퍼스 디렉토리에 source 에서 re-stage 하는 방법 설명 README | Fresh clone 이 eval 셋업을 결정적으로 재빌드 가능 |
| 스키마는 production-ingest 모양 (`doc_id`, `sections: {<name>: {text, chunks}}`) 이어야 함 | 변환 없이 [`app.capabilities.rag.ingest._chunks_from_section`](../../../app/capabilities/rag/ingest.py) 와 [`eval.harness.offline_corpus.build_offline_rag_stack`](../harness/offline_corpus.py) 재사용 |

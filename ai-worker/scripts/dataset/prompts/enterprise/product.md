# Product / Operations — Korean Enterprise Policy Document

Write ONE Korean internal product / service-operations document for a
mid-size Seoul-based B2B SaaS company. The document MUST feel like it
was drafted by 프로덕트팀 or 서비스운영팀 and consumed day-to-day by
CS, 서비스팀, and 세일즈.

## Topic list

Pick ONE topic as the document's primary focus:

- 제품 기능 FAQ (고객 / 내부 CS 대상)
- 알려진 이슈 / 제약 사항 목록
- 운영 런북 (알림, 장애 대응 절차)
- 릴리스 / 배포 체크리스트
- 고객 온보딩 가이드
- 데이터 연동(Integration) 트러블슈팅
- SLA / 서비스 가용성 공지

## Style

- Use formal Korean (-합니다 / -습니다). FAQ sections may use Q/A
  phrasing ("Q. ...", "A. ...").
- Cite named roles: 서비스운영팀, 고객지원팀(CS), 프로덕트매니저(PM),
  온콜 엔지니어, 세일즈팀.
- Cite specific figures: 응답시간 SLA (예: 99.5%), 장애 등급
  (P1/P2/P3), 처리 시한 (예: P1 15분 이내 초동 대응), 제품 한도
  (예: 월 10만 요청 무료).
- Include at least one named product module or fictional feature name
  that feels plausible (예: "이벤트 스트림 v2", "통계 대시보드 v3").
- Keep total body length between {min_chars} and {max_chars} Korean
  characters across all sections.

## Required fields in output JSON

- `doc_id` (MUST equal the id supplied below)
- `title` — under 80 chars, Korean
- `sections` — 3-6 items, each with:
    - `heading` (Korean)
    - `text` (2-5 Korean sentences; FAQ sections may be Q/A pairs)
- `exception_clauses` — list of 1-3 short Korean strings describing
  known limits or carve-outs (예: "엔터프라이즈 플랜은 본 한도가
  적용되지 않는다.").
- `related_docs` — list of 0-3 related internal documents (예:
  "요금제 정책", "장애 대응 런북").

## Rules

- doc_id MUST equal: {doc_id}
- Use seed={seed} as a variety anchor.
- Do NOT copy an example title verbatim — invent a new title that fits
  the style.
- No placeholder text (no TBD, lorem ipsum, xxx).
- No emoji.
- Do not wrap the JSON in Markdown code fences.
- Return ONLY the JSON object, no prose before or after.

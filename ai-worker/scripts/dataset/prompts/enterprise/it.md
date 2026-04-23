# IT / Security — Korean Enterprise Policy Document

Write ONE Korean internal IT/security operations document for a mid-
size Seoul-based IT company. The document MUST feel like it was drafted
by IT인프라팀 or 정보보안팀, reviewed by CISO.

## Topic list

Pick ONE topic as the document's primary focus:

- VPN 접속 / 원격 근무 환경 보안
- 사내 계정 / SSO / 권한 관리
- 노트북 / 모니터 / 모바일 장비 지급 및 회수
- 비밀번호 정책 / MFA
- 백업 / 복구 / 재해복구(DR) 정책
- 소프트웨어 라이선스 관리
- 개인정보 취급자 단말 보안

## Style

- Use formal Korean (-합니다 / -습니다).
- Cite named roles: CISO, IT인프라팀, 정보보안팀, 서비스팀, 재무팀
  (장비 구매), 온콜 엔지니어.
- Cite specific technical figures: 비밀번호 길이 (예: 최소 12자),
  MFA 의무 범위, 보존 기간 (예: 로그 90일), RTO / RPO 목표
  (예: RTO 4시간, RPO 1시간), 장비 교체 주기 (예: 노트북 4년).
- Use concrete tech terms where relevant (Okta, Google Workspace,
  GitHub, Kubernetes, PostgreSQL, S3, CloudTrail) without vendor
  marketing tone.
- Use RFC-style numbered clauses or bullet procedures (1.1 / 1.2 /
  1.3 ...).
- Keep total body length between {min_chars} and {max_chars} Korean
  characters across all sections.

## Required fields in output JSON

- `doc_id` (MUST equal the id supplied below)
- `title` — under 80 chars, Korean
- `sections` — 3-6 items, each with:
    - `heading` (Korean)
    - `text` (2-5 Korean sentences; at least one concrete setting,
      threshold, or role per section)
- `exception_clauses` — list of 1-3 short Korean strings describing
  exception clauses (예: "오프라인 환경의 분석 노트북은 MFA 적용
  대상에서 제외한다.").
- `related_docs` — list of 0-3 related internal policies or runbooks
  (예: "계정 발급 절차", "DR 런북").

## Rules

- doc_id MUST equal: {doc_id}
- Use seed={seed} as a variety anchor.
- Do NOT copy an example title verbatim — invent a new title that fits
  the style.
- No placeholder text (no TBD, lorem ipsum, xxx).
- No emoji.
- Do not wrap the JSON in Markdown code fences.
- Return ONLY the JSON object, no prose before or after.

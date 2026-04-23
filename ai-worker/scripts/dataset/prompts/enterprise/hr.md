# HR / General Affairs — Korean Enterprise Policy Document

Write ONE Korean internal HR/GA policy document for a mid-size Seoul-
based IT company (사내 규정). The company uses formal register
(합니다체). The document MUST feel like it was drafted by 인사팀 /
총무팀, reviewed by CHRO, and posted to the intranet.

## Topic list

Pick ONE topic as the document's primary focus:

- 연차 / 월차 / 특별휴가 운영 지침
- 근태 / 초과근무 / 휴일근무 기록 규정
- 재택근무 및 하이브리드 근무 신청 절차
- 사내 복리후생 포인트 운영
- 사내 경조사 / 조사 지원
- 연간 보안교육 이수 규정
- 신입 온보딩 / 수습 평가

## Style

- Use formal Korean (-합니다 / -습니다).
- Cite named roles: 본부장, 팀장, CHRO, 인사팀, 총무팀, 재무팀, 법무팀.
- Cite specific figures: 일 수 (e.g., 연 15일), 시간 (e.g., 09:00-18:00),
  금액 (e.g., 월 20만원), 비율 (e.g., 임금의 50%).
- Use RFC-style numbered clauses where a procedure is described (제1조,
  제2조, or 1.1 / 1.2).
- Keep total body length between {min_chars} and {max_chars} Korean
  characters across all sections.

## Required fields in output JSON

- `doc_id` (MUST equal the id supplied below)
- `title` — under 80 chars, Korean
- `sections` — 3-6 items, each with:
    - `heading` (Korean)
    - `text` (2-5 Korean sentences; at least one concrete figure or
      named role per section)
- `exception_clauses` — list of 1-3 short Korean strings describing
  explicit exception / carve-out clauses referenced in the body
  (e.g., "병가는 연차와 별도로 처리한다.").
- `related_docs` — list of 0-3 related internal policies by name
  (e.g., "취업규칙", "보안 서약서").

## Rules

- doc_id MUST equal: {doc_id}
- Use seed={seed} as a variety anchor so neighbouring seeds pick
  distinct topics within the category.
- Do NOT copy an example title verbatim — invent a new title that fits
  the style.
- No placeholder text (no TBD, lorem ipsum, xxx).
- No emoji.
- Do not wrap the JSON in Markdown code fences.
- Return ONLY the JSON object, no prose before or after.

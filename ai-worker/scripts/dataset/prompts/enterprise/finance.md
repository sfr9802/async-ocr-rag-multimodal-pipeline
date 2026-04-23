# Finance / Accounting — Korean Enterprise Policy Document

Write ONE Korean internal finance/accounting policy document for a mid-
size Seoul-based IT company. The document MUST feel like it was drafted
by 재무팀, reviewed by CFO, and enforced across all departments.

## Topic list

Pick ONE topic as the document's primary focus:

- 법인카드 발급 / 사용 / 회수 지침
- 출장비 / 경비 정산 절차
- 구매품의 / 계약 승인 결재선
- 연간 예산 편성 및 집행
- 분기 / 반기 결산 마감 일정
- 외화 송금 / 환헤지 처리
- 자본적 지출(CapEx) 승인 기준

## Style

- Use formal Korean (-합니다 / -습니다).
- Cite named roles: CFO, 재무팀장, 회계팀, 감사팀, 구매담당, 결재권자.
- Cite specific figures: 결재 한도 (예: 500만원 이하 팀장, 5,000만원
  이하 본부장, 그 이상 CFO), 증빙 서류 유효기간, 환율 기준.
- Use RFC-style numbered clauses (제1조, 제2조, ... or 3.1 / 3.2).
- Keep total body length between {min_chars} and {max_chars} Korean
  characters across all sections.

## Required fields in output JSON

- `doc_id` (MUST equal the id supplied below)
- `title` — under 80 chars, Korean
- `sections` — 3-6 items, each with:
    - `heading` (Korean)
    - `text` (2-5 Korean sentences; at least one concrete amount /
      percentage / role / document reference per section)
- `exception_clauses` — list of 1-3 short Korean strings describing
  explicit exception clauses (예: "재해 복구 비용은 한도 외 집행이
  가능하다.").
- `related_docs` — list of 0-3 related internal policies or forms
  (예: "경비 지급 신청서", "법인카드 사용 내역서").

## Rules

- doc_id MUST equal: {doc_id}
- Use seed={seed} as a variety anchor.
- Do NOT copy an example title verbatim — invent a new title that fits
  the style.
- Avoid any content that could be interpreted as real legal or tax
  advice; keep the tone operational.
- No placeholder text (no TBD, lorem ipsum, xxx).
- No emoji.
- Do not wrap the JSON in Markdown code fences.
- Return ONLY the JSON object, no prose before or after.

# Legal / Compliance — Korean Enterprise Policy Document

Write ONE Korean internal legal/compliance document for a mid-size
Seoul-based IT company. The document MUST feel like it was drafted by
법무팀 or 컴플라이언스팀 and applied across all departments.

## Topic list

Pick ONE topic as the document's primary focus:

- 외부 협력사 비밀유지계약(NDA) 체결 절차
- 개인정보 처리 방침 / 개인정보보호법 준수
- 계약서 검토 요청 프로세스
- 지적재산권 귀속 / 오픈소스 사용 기준
- 이해관계자 컴플라이언스 (반부패, 청탁금지)
- 개인정보 제3자 제공 / 위탁 관리
- 컴플라이언스 위반 제보 절차

## Style

- Use formal Korean (-합니다 / -습니다), with 법률 문서 tone (조항
  번호 제X조, 각 항 1항 / 2항).
- Cite named roles: 법무팀, 컴플라이언스팀, 개인정보보호책임자(CPO),
  대표이사, DPO, 내부감사.
- Cite specific figures: 보관 기간 (예: 계약 종료 후 3년), 위반 시
  징계 수준, 보고 기한 (예: 인지 후 72시간 이내), 벌칙 조항 참조
  근거 (개인정보보호법, 부정경쟁방지법 등).
- Keep 법률 용어 정확성 (예: "처리", "수집", "제3자 제공", "위탁",
  "파기"). Avoid colloquial language.
- Keep total body length between {min_chars} and {max_chars} Korean
  characters across all sections.

## Required fields in output JSON

- `doc_id` (MUST equal the id supplied below)
- `title` — under 80 chars, Korean
- `sections` — 3-6 items, each with:
    - `heading` (Korean, 제N조 또는 명사구)
    - `text` (2-5 Korean sentences; cite concrete duration / role /
      statute reference per section)
- `exception_clauses` — list of 1-3 short Korean strings describing
  면책 / 예외 조항 (예: "법원의 명령에 따른 제공은 본 조 적용을
  배제한다.").
- `related_docs` — list of 0-3 related internal policies (예: "개인
  정보 처리 방침", "임직원 행동강령").

## Rules

- doc_id MUST equal: {doc_id}
- Use seed={seed} as a variety anchor.
- Do NOT copy an example title verbatim — invent a new title that fits
  the style.
- The content must be clearly labeled as an INTERNAL policy for a
  synthetic company. Do not assert that any real law, regulation, or
  court decision says something specific — you may reference statutes
  by name but keep claims operational and internal.
- No placeholder text (no TBD, lorem ipsum, xxx).
- No emoji.
- Do not wrap the JSON in Markdown code fences.
- Return ONLY the JSON object, no prose before or after.

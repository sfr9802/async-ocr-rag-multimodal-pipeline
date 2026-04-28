# Handoff: 글래스 콘솔 — `async-ocr-rag-multimodal-pipeline` 프론트엔드 비주얼 리프레시

## Overview

기존 `frontend/app/`(React 19 · Vite · Tailwind 3 · shadcn/ui) 콘솔의 비주얼 언어를 **light glass**로 교체합니다. 기능 변경 없음 — 헤더, 사이드바, 작업 제출 폼, 결과 뷰어의 시각 토큰과 패널 트리트먼트만 바뀝니다.

이전 디자인은 **warm parchment**(크림 종이 + 네이비 잉크 + 시그널 오렌지)였고, 새 디자인은 **light glass**(크림-라벤더 베이스 + 오로라 블리드 + 프로스티드 흰색 패널 + 잉크-바이올렛 잉크 + 바이올렛 액센트)입니다. 다크 모드도 같이 갱신됨 — `.dark` 스코프에 contained aurora + 저투명도 글래스가 들어갑니다.

## About the Design Files

이 번들의 파일들은 **디자인 참조용 HTML 프로토타입**입니다 — 의도된 비주얼과 동작을 보여주는 레퍼런스이지, 그대로 복붙할 프로덕션 코드가 아닙니다.

작업 방향: **이 디자인을 기존 레포의 환경(React 19 · Vite · Tailwind 3 · shadcn/ui · `lucide-react`)에 맞춰 재구현**합니다. 기존 컴포넌트 구조, shadcn 프리미티브, Tailwind 클래스 시스템을 유지하면서 색·표면·그림자 토큰만 바꾸는 식으로 가는 것이 가장 매끄럽습니다.

## Fidelity

**High-fidelity (hifi).** 색·타입·스페이싱·반경·그림자 모두 최종 값으로 잡혀 있습니다. 픽셀 단위 재현이 가능합니다.

## Files in this bundle

| Path | Purpose |
|------|---------|
| `design_refs/colors_and_type.css` | **토큰의 진실** — light + dark 두 스코프 전부, HSL/RGB/그림자/스페이싱 모두 포함 |
| `design_refs/ui_kit_console/` | 풀 콘솔 hi-fi 재현 (JSX 컴포넌트 + `index.html`) — 인터랙션 포함, mock job lifecycle 동작 |
| `design_refs/ui_kit_console/console.css` | 글래스 컴포넌트 스타일 (`.ds-app`, `.ds-card`, `.ds-sidebar`, 버튼, 인풋 등) |
| `design_refs/assets/logo.svg` | 새 브랜드 마크 (잉크 stroke + 바이올렛 닷) |
| `design_refs/DESIGN_SYSTEM.md` | 프로젝트 디자인 시스템 README — 콘텐츠 보이스, 비주얼 파운데이션 풀 문서 |

## 적용 전략

레포 구조:
```
frontend/app/
├── src/
│   ├── index.css               ← Tailwind 토큰 진입점, 여기 글래스 토큰 이식
│   ├── components/
│   │   ├── Header.tsx
│   │   ├── SubmitForm.tsx
│   │   ├── JobSidebar.tsx
│   │   ├── ResultViewer.tsx
│   │   └── ui/                 ← shadcn 프리미티브, 대부분 그대로 두면 토큰 따라감
```

**3-step 마이그레이션:**

1. **토큰 교체.** `frontend/app/src/index.css`의 `:root` / `.dark` 블록을 `design_refs/colors_and_type.css`의 RAW HSL TOKENS 섹션 + `.dark` 스코프로 통째 치환. 신규 토큰 추가:
   - `--glass-panel`, `--glass-panel-2`, `--glass-panel-3`, `--glass-panel-strong`
   - `--hairline`, `--hairline-2`, `--hairline-strong`
   - `--shadow-glass-panel`, `--shadow-glass-pop`, `--shadow-glass-button`
   - `--cap-mock`, `--cap-rag`, `--cap-ocr`, `--cap-multimodal` (값 변경)

2. **Tailwind config 확장.** `tailwind.config.js`에서 컬러는 이미 `hsl(var(--*))`로 매핑돼 있을 테니 그대로 동작. 추가로 `backgroundColor`, `borderColor`, `boxShadow` 키에 신규 글래스 토큰을 RGB 형식으로 노출:
   ```js
   extend: {
     backgroundColor: {
       'glass': 'rgb(var(--glass-panel))',
       'glass-2': 'rgb(var(--glass-panel-2))',
       'glass-3': 'rgb(var(--glass-panel-3))',
       'glass-strong': 'rgb(var(--glass-panel-strong))',
     },
     borderColor: {
       'hairline': 'rgb(var(--hairline))',
       'hairline-2': 'rgb(var(--hairline-2))',
       'hairline-strong': 'rgb(var(--hairline-strong))',
     },
     boxShadow: {
       'glass-panel': 'var(--shadow-glass-panel)',
       'glass-pop': 'var(--shadow-glass-pop)',
       'glass-button': 'var(--shadow-glass-button)',
     },
     borderRadius: {
       'xl-glass': 'var(--radius-xl)',  // 22px
     },
   }
   ```

3. **컴포넌트 패치.** 각 컴포넌트는 아래 "Per-component spec" 따라 클래스만 교체. 구조/JSX는 거의 그대로. shadcn 프리미티브(Button, Input, Card, Tabs)는 토큰 변경만으로 자동 따라옴.

## Design Tokens (전체 목록은 `design_refs/colors_and_type.css` 참조)

### Colors — Light

| Token | HSL | Hex | Use |
|-------|-----|-----|-----|
| `--background` | `252 33% 97%` | `#F5F3FB` | 앱 베이스 (오로라 위에 깔림) |
| `--foreground` | `245 35% 12%` | `#16142A` | 본문 텍스트, 잉크 |
| `--primary` | `250 38% 13%` | `#1A1530` | Primary CTA, 로고 stroke |
| `--accent` | `254 100% 61%` | `#6A3AFF` | 액센트 닷, focus ring, score fill, active tab |
| `--muted-foreground` | `245 18% 42%` | — | secondary 텍스트 |
| `--success` | `152 76% 30%` | `#128A4A` | 성공 pill (12% tint + 40% border) |
| `--warning` | `36 100% 39%` | `#C97A00` | 실행 중 pill (14% tint + 40% border) |
| `--destructive` | `354 64% 51%` | `#D23541` | 실패 pill (10% tint + 40% border) |

### Capability accents

| Token | Light | Dark |
|-------|-------|------|
| `--cap-mock` | neutral ink `245 18% 42%` | light slate `252 100% 70%` |
| `--cap-rag` | cyan `#0099C4` | light cyan `#5EE0FF` |
| `--cap-ocr` | violet `#6A3AFF` (= accent) | light violet `#B890FF` |
| `--cap-multimodal` | rose `#D4316F` | light rose `#FF8FB1` |

### Glass surfaces (RGB triplets, used with `rgb(var(--…))`)

| Token | Light | Dark |
|-------|-------|------|
| `--glass-panel` | `255 255 255 / 0.55` | `18 21 32 / 0.66` |
| `--glass-panel-2` | `255 255 255 / 0.4` | `255 255 255 / 0.025` |
| `--glass-panel-3` | `255 255 255 / 0.7` | `255 255 255 / 0.045` |
| `--glass-panel-strong` | `255 255 255 / 0.85` | `7 9 15 / 0.72` |
| `--hairline` | `22 20 42 / 0.08` | `255 255 255 / 0.09` |
| `--hairline-2` | `22 20 42 / 0.13` | `255 255 255 / 0.14` |
| `--hairline-strong` | `22 20 42 / 0.22` | `255 255 255 / 0.24` |

### Shadows

| Token | Value |
|-------|-------|
| `--shadow-glass-panel` | `0 1px 0 rgba(255,255,255,0.7) inset, 0 24px 60px -28px rgba(58,30,120,0.18), 0 2px 6px -2px rgba(58,30,120,0.08)` |
| `--shadow-glass-pop` | `0 12px 32px -10px rgba(58,30,120,0.22), 0 2px 6px -2px rgba(58,30,120,0.10)` |
| `--shadow-glass-button` | `0 6px 16px -8px rgba(26,21,48,0.28), 0 0 0 1px rgba(255,255,255,0.10) inset` |
| `--shadow-soft` | `0 1px 0 rgba(255,255,255,0.7) inset, 0 4px 14px -6px rgba(58,30,120,0.10), 0 2px 4px -2px rgba(58,30,120,0.06)` |

### Radii

`--radius-sm: 6px` · `--radius-md: 10px` (default) · `--radius-lg: 14px` · `--radius-xl: 22px` (메인 패널) · `999px` pill

### Typography

OS-네이티브 스택 그대로 유지 (변경 없음):
- `--font-sans`: `system-ui, -apple-system, "Segoe UI", "Apple SD Gothic Neo", "Malgun Gothic", "Noto Sans CJK KR", ...`
- `--font-mono`: `ui-monospace, SF Mono, Menlo, Consolas, ...`

스케일은 작고 타이트함: body 12.5–14px, eyebrow 10–11px mono uppercase tracking 0.14–0.18em, 최대 디스플레이 13.5px.

## Glass recipe — 어떻게 "글래스"로 만드는가

**핵심 4-요소 조합** (이게 글래스 언어의 전부):

1. **반투명 배경.** `background: rgb(var(--glass-panel))` (= `rgba(255,255,255,0.55)` light)
2. **Backdrop blur.** `backdrop-filter: saturate(150%) blur(18px)` (메인 패널) / `blur(14px)`–`blur(16px)` (작은 표면)
3. **하이라이트 inset + 페어트론 그림자.** `box-shadow: var(--shadow-glass-panel)` — 위쪽 1px 흰색 inset + 부드러운 바이올렛 틴트 드롭
4. **헤어라인 보더.** `border: 1px solid rgb(var(--hairline-2))`

**오로라 백그라운드** (앱 셸에 깔림 — 이게 backdrop-blur이 의미를 갖게 해주는 베이스):
```css
.ds-app {
  background:
    radial-gradient(120vw 80vh at 50% -20%, #ffe9f4 0%, transparent 60%),
    hsl(var(--background));
  overflow: hidden;
}
.ds-app::before, .ds-app::after {
  content: ""; position: fixed; border-radius: 50%;
  filter: blur(110px); pointer-events: none; z-index: 0;
  opacity: 0.55; mix-blend-mode: multiply;
  animation: ds-aurora-drift 24s ease-in-out infinite alternate;
}
.ds-app::before {
  width: 60vw; height: 60vw; left: -18vw; top: -22vw;
  background: radial-gradient(circle at 30% 30%, #c9b8ff 0%, #ffd5e8 55%, transparent 75%);
}
.ds-app::after {
  width: 65vw; height: 65vw; right: -22vw; bottom: -28vw;
  background: radial-gradient(circle at 70% 70%, #ffd1e0 0%, #b9e6ff 55%, transparent 75%);
  animation-direction: alternate-reverse; animation-duration: 30s;
}
```
다크모드: `mix-blend-mode: screen`, opacity 0.22, blob 색을 `#6a4dff`/`#ff5dc8`+`#5ee0ff`로 교체 (`design_refs/ui_kit_console/console.css` 그대로 복사).

## Per-component spec

각 컴포넌트의 시각 트리트먼트만 적습니다. JSX 구조와 props는 기존 그대로 유지.

### `Header.tsx` — 헤더

- 컨테이너: `position: sticky; top: 0; z-index: 10`
- bg: `rgb(var(--glass-panel-strong))` + `backdrop-filter: saturate(140%) blur(16px)`
- 하단 보더: `1px solid rgb(var(--hairline))`
- inner: max-width 1440px, padding `12px 28px`, flex justify-between
- 좌측: 로고 SVG 28px + 브랜드 텍스트 (이름 13.5px/600/-0.01em, 서브 mono 10.5px uppercase tracking 0.14em color muted-foreground)
- 우측: connection chip + ghost 아이콘 버튼들 (`Settings2`, `Sun`/`Moon`)

### Connection chip

- pill: `padding 4px 10px`, `border-radius 999px`, `border 1px solid rgb(var(--hairline-2))`, `background rgb(var(--glass-panel-3))`, `font-size 11px`
- online: `border-color hsl(var(--success)/0.4)`, `bg hsl(var(--success)/0.10)`, `color hsl(var(--success))`, dot 6px filled
- offline: 동일 패턴 destructive
- 라벨은 mono uppercase tracking 0.12em

### Buttons — `ui/button.tsx` (shadcn 프리미티브)

기본 base: `border-radius 8px`, `height 36px`, `padding 0 16px`, `font-size 13.5px / 500`, transition 0.15s.

| Variant | Background | Text | Border | Box-shadow |
|---------|------------|------|--------|------------|
| `default` (primary) | `hsl(var(--primary))` `#1A1530` | `#fff` | transparent | `var(--shadow-glass-button)` |
| `secondary` | `rgb(var(--glass-panel-3))` | `hsl(var(--foreground))` | `rgb(var(--hairline-2))` | — |
| `outline` | `rgb(var(--glass-panel-2))` | `hsl(var(--foreground))` | `rgb(var(--hairline-2))` | — |
| `ghost` | `transparent` | `hsl(var(--foreground))` | none | — |
| `destructive` | `hsl(var(--destructive))` | `#fff` | none | — |

Hover: primary는 `bg hsl(var(--primary)/0.92) + translateY(-1px)`. Outline은 `bg glass-panel-3 + border hairline-strong`.

> **Frosted ink primary 옵션** (탐색본에서 user가 좋아한 트리트먼트, 프로덕션엔 옵션):
> ```css
> background: rgba(26, 21, 48, 0.78);
> backdrop-filter: saturate(150%) blur(14px);
> box-shadow:
>   0 1px 0 rgba(255,255,255,0.18) inset,
>   0 6px 18px -8px rgba(26,21,48,0.35),
>   0 0 0 1px rgba(255,255,255,0.06) inset;
> ```

### Inputs — `ui/input.tsx`, `ui/textarea.tsx`

- bg: `rgb(var(--glass-panel-3))`
- border: `1px solid rgb(var(--hairline-2))`, radius 8px
- focus: `border-color hsl(var(--ring)/0.7)`, `box-shadow 0 0 0 3px hsl(var(--ring)/0.18)`, bg `glass-panel-strong`
- input height 36px, textarea padding 10px 12px line-height 1.6
- placeholder: `hsl(var(--muted-foreground))`

### `SubmitForm.tsx`

- 컨테이너 = 메인 글래스 패널 (위 recipe 그대로):
  ```css
  background: rgb(var(--glass-panel));
  backdrop-filter: saturate(150%) blur(18px);
  border: 1px solid rgb(var(--hairline-2));
  border-radius: 22px;
  box-shadow: var(--shadow-glass-panel);
  padding: 20px 22px;
  ```
- 헤더 행: 좌측 제목 (`새 작업` 15px/600) + 서브 (`역량을 고르고 입력을 채우세요…` 12.5px muted), 우측 BASE URL 인풋 (`width 280px`)
- Capability picker (4 타일, grid):
  - 타일 idle: `bg glass-panel-2`, `border hairline-2`, `radius 10px`, `padding 12px`
  - 타일 hover: `bg glass-panel-3`, `translateY(-1px)`
  - 타일 active: `bg glass-panel-strong`, `border-color hsl(var(--cap-X)/0.45)`, `box-shadow 0 0 0 1px hsl(var(--cap-X)/0.4) + var(--shadow-glass-pop)`
  - 안에 lucide 아이콘 16px + capability tag (mono 9.5px uppercase tracking 0.18em) + 이름 (14px/600)
- 풋: `border-top 1px dashed rgb(var(--hairline-strong))`, padding-top 12px. 좌측 키보드 힌트 `⌘ ↵` (kbd 스타일), 우측 primary 버튼 `작업 제출`

### Dropzone (`FileDropzone.tsx`)

- 컨테이너: `border: 1.5px dashed rgb(var(--hairline-strong))`, `bg rgb(var(--glass-panel-2))`, `radius 10px`, `padding 22px`, flex column center, gap 6px
- hover/drag: `bg glass-panel-3`, `border-color hsl(var(--ring)/0.6)`
- File row (업로드 후): `bg glass-panel-3`, border hairline-2, radius 10px, padding 10px 12px

### `JobSidebar.tsx`

- 컨테이너: 메인 글래스 패널 (320px width, `position: sticky; top: 80px`)
- 헤더 padding `16px 16px 12px`, border-bottom hairline
- 비어있는 상태: 36px 라운드 아이콘 (`bg glass-panel-3 + border hairline-2`) + 제목/본문
- Bucket label: mono 10px uppercase tracking 0.18em color muted/0.7
- 행 (`button`):
  - idle: transparent
  - hover: `bg glass-panel-2`
  - active: `bg glass-panel-3`, `border hairline-2`, `box-shadow var(--shadow-soft)`, 좌측 2.5px 액센트 rail (`bg hsl(var(--accent))`)
  - cap mono row + 본문 2-line clamp + 시간 mono row

### `ResultViewer.tsx`

- 컨테이너: 메인 글래스 패널
- 헤더 행: padding `16px 20px`, border-bottom hairline, 좌측 잡 ID + capability pill, 우측 status pill
- Timeline 스트립: padding `12px 20px`, `bg rgb(var(--glass-panel-2))`, border-bottom hairline
- 탭: padding `0 14px`, border-bottom hairline. 탭 자체 mono 11.5px uppercase tracking 0.14em. active는 `border-bottom 2px solid hsl(var(--accent))`
- 콘텐츠 영역 padding `18px 20px`
- **Top hit 카드**: `bg rgb(var(--glass-panel-3))`, border hairline-2, radius 12px, padding `14px 16px`. 답변 18px/600/-0.01em.
- **Score bar**: track `bg rgb(var(--hairline-2))` height 4px, fill `bg hsl(var(--accent))`
- **Hits 리스트**: 행 grid `24px 1fr 180px`, hover `bg rgb(var(--glass-panel-2))`
- **OCR text 블록**: mono 12.5px line-height 1.7, `bg glass-panel-2 + border hairline-2 + radius 10px + padding 14px 16px`
- **Multimodal 답변**: 14.5px line-height 1.55, 좀 더 진한 글래스 (`glass-panel-3 + radius 12px`)
- **Raw JSON**: max-height 360px scroll, mono 12px, `bg glass-panel-2 + border hairline-2 + radius 10px`

### Status pills

`padding 4px 10px`, `border-radius 999px`, `font-size 12px / 500`, `border 1px solid`, dot 6px.

각 변종은 12% tint bg + 40% border + full color text + dot full color. 상세 토큰은 위 Colors 표 참조.

### Job timeline (`JobTimeline.tsx`)

- 행: flex wrap, gap 6px 8px, mono 10px uppercase tracking 0.16em color muted
- Event = dot (6px) + label
  - empty: outline 1px muted/0.4
  - filled: muted/0.55
  - warning/success/destructive: full hsl
- Connector: 12px 1px 라인 (`bg rgb(var(--hairline-2))`) — dashed 변종은 transparent + border-top dashed
- Duration 라벨: mono 10.5px tabular-nums, no tracking
- Attempt pill: mono 9.5px uppercase tracking 0.18em, border hairline-2, bg glass-panel-3, radius 999px

## Interactions & Behavior

기능적 동작은 모두 **현재 구현 유지**. 시각만 바뀜. 다만 다음 모션 디테일은 글래스 언어 일부:

- **Aurora drift**: `.ds-app::before/::after`이 24s/30s alternate 무한 루프. CPU 부담 거의 없음 (transform + opacity만).
- **Tile/button hover**: `translateY(-1px)` + 그림자/보더 강도 증가. 0.15s ease.
- **Status pulse**: RUNNING 상태일 때 `pulse-soft` 키프레임 (1.6s ease-in-out, opacity 1→0.55→1). 이미 기존에 있던 토큰.
- **Spinner**: `Loader2` lucide + linear 1s 무한 회전 (Tailwind `animate-spin`).

## Responsive

기존 그대로. `@media (max-width: 960px)`에서 사이드바를 메인 위로 stack, capability grid 2열로 등.

## Dark mode

`.dark` 스코프 토큰만 적용하면 자동. 추가 작업 없음. 단:
- aurora blob 색이 light/dark에서 다름 — `console.css`의 `.dark .ds-app::before/::after` 블록 참조해서 그대로 복사
- 헤더 backdrop, 패널 배경 모두 토큰만으로 자동 전환

## Browser support

- `backdrop-filter`: Chrome 76+ / Safari 9+ / Firefox 103+. 미지원 환경에서는 RGBA bg가 그대로 보여서 깨지진 않지만 frosted 효과는 사라짐. 필요시 `@supports not (backdrop-filter: blur(1px))` fallback에서 opacity를 0.85로 올려 가독성 보강 가능.

## Assets

- `assets/logo.svg` — 새 브랜드 마크. 32×32 rounded-square (radius 9) + 잉크-바이올렛 stroke + violet 닷. 인라인 JSX 버전은 `Header.tsx` 안에 있을 텐데, 새 SVG로 교체 권장.
- 이미지/사진: 없음. 모든 비주얼은 CSS gradient + lucide 아이콘.

## Out of scope (이 핸드오프에서 다루지 않는 것)

- API 시그니처, job lifecycle 로직, polling 동작 — 변경 없음
- i18n / 카피 변경 — 모든 한글 카피 그대로 유지
- 신규 기능 — 비주얼 리프레시만

## 참조: parchment 백업

원래의 warm parchment 디자인은 `colors_and_type-parchment.css` + `ui_kits/console-parchment/`로 본 디자인 시스템 프로젝트에 보존돼 있습니다. 롤백이 필요하면 거기서 토큰을 다시 가져오면 됨.

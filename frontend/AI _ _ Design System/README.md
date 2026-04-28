# AI 파이프라인 콘솔 — Design System

A design system for the developer-facing **console** of the **async-ocr-rag-multimodal-pipeline** — a Korean-language internal operator UI for submitting AI jobs (MOCK / RAG / OCR / MULTIMODAL), monitoring their lifecycle, and inspecting artifacts.

The product is an **internal dev tool**, not a marketing surface. The design vocabulary reflects that: dense, monospaced where it earns its keep, status-driven, and quiet by default. Color is reserved for *meaning* (status, capability) rather than decoration.

---

## Sources

- **Repository:** `sfr9802/async-ocr-rag-multimodal-pipeline` (default branch `main`)
- **Frontend root:** `frontend/app/`
- **Token source of truth:** `frontend/app/src/index.css`
- **Component source:** `frontend/app/src/components/` — bespoke + `components/ui/` shadcn primitives
- **Stack:** React 19 · Vite · Tailwind 3 · shadcn/ui (Radix Primitives) · `lucide-react` icons

This design system *does not* assume a reader has access to the repo — every relevant rule is restated here.

---

## Index

| File / Folder                  | Purpose |
|--------------------------------|---------|
| `README.md`                    | This file. Overview, content fundamentals, visual foundations, iconography. |
| `colors_and_type.css`          | All color tokens (HSL + RGB aliases), type scale, shadow/radius/spacing tokens. |
| `SKILL.md`                     | Agent-skill manifest — instructions for Claude (or Claude Code) to design *for* this brand. |
| `assets/`                      | Logo SVG (recreated from `Header.tsx`), favicon, swatches. |
| `preview/`                     | Cards that populate the **Design System** tab. |
| `ui_kits/console/`             | Hi-fi click-thru recreation of the console — JSX components + `index.html`. |

> **No slide template was provided** by the source repo, so `slides/` is intentionally absent.

---

## Product context

**One product, one surface:** a single-page internal console at `frontend/app/`. The app shell is:

```
┌──────────────────────────────────────────────────────────────┐
│  [logo] AI 파이프라인 콘솔                  [● online]  ⚙  ◐  │  ← sticky header
│         async · ocr · rag · multimodal                       │
├──────────┬───────────────────────────────────────────────────┤
│ 최근     │   새 작업  ─ capability picker (4 cards)          │
│  · MOCK  │             prompt / file dropzone                │
│  · RAG   │             [⌘ ↵  전송]                            │
│  · OCR   │                                                   │
│          │   작업 패널 ─ status pill + timeline + tabs       │
│          │             (결과 / 아티팩트 / 원본)              │
└──────────┴───────────────────────────────────────────────────┘
```

The four core capabilities are first-class objects, each carrying its own accent hue:

| Capability      | Purpose                                                          | Accent (`--cap-*`) |
|-----------------|------------------------------------------------------------------|--------------------|
| **MOCK**        | Echo prompt as JSON — used to verify the full pipeline round-trip | slate              |
| **RAG**         | Question-answer over an indexed corpus (bge-m3 + FAISS)          | teal-blue          |
| **OCR**         | Text extraction from PNG / JPEG / PDF                            | violet             |
| **MULTIMODAL**  | Claude Vision over image + question                              | orange (= accent)  |

---

## Content fundamentals

The product UI is **Korean-first**. The repo's README and code comments are English; user-visible strings are Korean. Tone is **terse, technical, and second-person-implicit** — Korean's natural omission of pronouns is leveraged so copy reads as concise dev-tool labels rather than chat.

### Voice & tone

- **Terse and technical.** No marketing copy. No exclamation points. No emoji.
- **Honest about state.** The UI says exactly where a job is (`대기` / `큐 대기` / `실행 중` / `성공` / `실패` / `취소됨`) — never softens with "처리 중이에요" or similar.
- **Mono for machine-emitted strings.** Job IDs, capability names, byte counts, durations, error codes — all set in `JetBrains Mono` and treated as the user's eye is reading them as data.
- **Mid-dot separators (`·`)** chain related metadata: `application/json · 1.2 KB · a3f9c2…`. This is the brand's signature data-density move.

### Casing & punctuation

- Korean copy is **plain sentence-form**, no ALL CAPS.
- English-derived terms (capability names, status codes) stay **UPPERCASE** verbatim — `MOCK`, `RAG`, `IN_PROGRESS`, `SUCCEEDED`. They are surfaced in the UI through monospace + `letter-spacing: 0.06em–0.18em`.
- Eyebrow labels (e.g. `최근`, `기능`, `결과`) are uppercase-tracked monospace, used as quiet section markers above content blocks.
- Sentences omit the trailing period unless the line is a full grammatical sentence (e.g. tooltip help).

### "I" vs "you"

Korean copy mostly uses **plain declarative form** with no explicit subject — the system describes its own state, not a relationship to the user. Where a subject is needed, the system refers to itself impersonally ("워커", "core-api"), never "Claude" or "AI". The user is addressed implicitly via imperative ("전송", "비우기", "저장").

### Examples (verbatim from the codebase)

| Where                     | Korean                                       | English equivalent / rationale |
|---------------------------|----------------------------------------------|--------------------------------|
| Document title            | `AI 파이프라인 콘솔`                         | Plain product noun. No tagline. |
| Header eyebrow            | `async · ocr · rag · multimodal`            | Mid-dot signature; lowercase mono. |
| Connection chip (online)  | `core-api 온라인`                            | Names the service, not "system". |
| Connection chip (offline) | `오프라인`                                   | One word. Doesn't apologize. |
| Submit-form section       | `새 작업`                                    | Two words. No verb hiding. |
| Submit-form keyboard hint | `⌘ ↵ 전송`                                   | Keys + verb. No "Press to…". |
| Empty state title         | `선택된 작업 없음`                            | Stative — describes a fact. |
| Empty state body          | `위에서 작업을 제출하거나 최근 목록에서 선택하세요.` | Imperative, single sentence. |
| Polling caption           | `1.5초마다 폴링 · 작업이 종료 상태에 도달하면 출력이 표시됩니다.` | Tells the user the mechanism. |
| Sidebar storage note      | `로컬에 N건 저장됨`                          | Tells the user data lives client-side. |
| Sidebar empty             | `아직 작업이 없습니다`                        | Plain present tense. |
| Settings popover help     | `비워두면 같은 origin의 /api를 사용합니다…` | Long-form only when explaining a config. |

### What we **don't** do

- ❌ Emoji of any kind. The brand never reaches for them.
- ❌ Sentence-case English headings on Korean screens.
- ❌ Marketing verbs (`Discover`, `Unleash`, `Get started`).
- ❌ Title-case product nouns (`Submit Form`, `Job Panel`) — those are component names in code, *not* labels in the UI.
- ❌ Unicode arrows or icons inline in copy. Iconography lives in `lucide-react`, not in strings.

---

## Visual foundations

### Palette philosophy

The system pivots on a **warm parchment** background (`HSL 36 25% 97%` ≈ `#FAF7F2`) and **deep navy ink** foreground (`HSL 220 25% 10%` ≈ `#14181F`). Both have hue baked in — neither is a pure off-white nor pure charcoal. The result reads as a typographic dev-tool, not a generic SaaS dashboard.

A single **signal orange** (`HSL 24 92% 52%` ≈ `#F36A12`) is the only chromatic accent in the chrome. It appears on the logo dot, on the `Sparkles` glyph above section headers, on focus rings via `--cap-multimodal`, and as the MULTIMODAL capability hue. Use it sparingly — it should pop because the rest of the UI is restrained.

Semantic state colors (success / warning / destructive) are pinned to their conventional hues but **always on a 8–12% tint background with a 35% border**, never on full saturation. This keeps a dense status-heavy UI readable without screaming.

### Per-capability hues

The four `--cap-*` tokens drive a tiny visual moment: a 1.5 px dot in the sidebar, the icon color in the SubmitForm capability picker, and the 2 px ring on the active card. Outside these three places, capability hues do **not** color anything else.

### Typography

- **UI font:** the source CSS does not declare a font family — it inherits the browser system stack. We follow the same approach: `--font-sans` resolves to `system-ui` first, falling through to `-apple-system` / `Segoe UI` / `Apple SD Gothic Neo` / `Malgun Gothic` / `Noto Sans CJK KR`. Latin text on macOS becomes SF, on Windows becomes Segoe UI; Hangul resolves to whichever OS Korean face is installed. **No webfont is loaded** — the design follows each user's machine.
- **Mono font:** OS-native mono stack — `ui-monospace` / `SF Mono` / `Menlo` / `Consolas`. Used for job IDs, durations, byte counts, capability tags, status pills, eyebrows, and anywhere a value comes back from the API.
- **Scale is small and tight.** Body is `12.5–14px`. Eyebrows are `10–11px` mono uppercase with `0.14em–0.18em` tracking. There is no display heading larger than `13.5px` in the production UI.
- **Tabular numerals** (`font-variant-numeric: tabular-nums`) are used everywhere a number could change in place — durations, byte counts, attempt counts, char counters.
- **Letter-spacing is a brand signal.** Uppercase tracked mono is the system's way of saying "this is a label, not content."

### Backgrounds

- The app shell uses two soft radial gradients (`canvas-bg` utility) — one tinted with `--primary` from top-left, one with `--accent` from top-right, both at 4–5% opacity. **No** large hero gradient, no full-bleed photography, no patterns or textures.
- Cards sit on `--card` (pure white in light mode, `HSL 220 18% 10%` in dark) with a 1 px `--border` and a soft elevation shadow.
- Section headers within cards use `bg-muted/25–30` (tint of parchment) to demarcate without adding a stroke.

### Borders, shadows, radii

- **Border-radius scale:** 4 / 6 / 8 / 12 / 999 px. The default `--radius` is 8 px (cards, buttons, inputs). Pill radius (`999px`) is used for status badges and the connection chip.
- **Shadows** are uniformly soft, multi-layer, and warm-leaning (the shadow color is the foreground HSL at 2–8% — not a generic black). The `--shadow-soft` token is the workhorse for cards and popovers.
- **Borders are usually `--border` at 60–80% alpha**, applied via `border-border/60` etc. Hairlines, not heavy strokes.

### Animation

- **Sparing.** `transition-colors` for hover, `animate-pulse-soft` (1.6s ease-in-out, opacity 100→55→100) for live states (running spinner dot, polling indicator).
- **No bounces, no springs, no large translations.** A spinner spins, a status pulses, a hover changes color. That's the entire motion vocabulary.
- **`Loader2` from lucide** spins via Tailwind's `animate-spin` (linear, 1s) on running buttons / inline indicators.

### Hover / press states

- **Hover on buttons:** color shift to `--primary/90`, `--secondary/80`, or a `bg-foreground/[0.03]` tint. Never a transform.
- **Hover on history rows:** `bg-foreground/[0.03]` and a Trash icon fades in via `opacity-0 group-hover:opacity-100`.
- **Hover on artifact rows:** border darkens from `border-border/70` to `border-border`.
- **Press / active state:** the active sidebar item gets a 2.5 px primary-colored left bar (`bg-primary` rounded-r) plus card background and soft shadow — not a transform.
- **Focus:** a 1 px `--ring` outline (which is `--primary` in light, `--primary-gold` in dark) plus a 2-px ring expansion on inputs.

### Transparency & blur

- The sticky header uses `bg-background/85 backdrop-blur supports-[backdrop-filter]:bg-background/65` — Safari/Chrome get translucency over scrolling content; non-supporters get a slightly more opaque fallback.
- Popovers use `bg-popover` opaque, no blur.
- The "copy" button in the output preview uses `bg-background/80 backdrop-blur` to float over scrollable code without obscuring it.

### Density & layout

- **Max content width:** `1380px` (declared on header and main grid).
- **Sidebar:** `18rem` (288 px) on `lg+`, full-width above the breakpoint.
- **Padding scale inside cards:** `5` units (20 px) is standard; `6` (24 px) for the SubmitForm body; `3` (12 px) for header strips.
- **Grid gaps:** `5–6` (20–24 px) between major sections.
- **Touch targets:** ghost icon buttons are `7×7` to `8×8` units (28–32 px); primary buttons are `h-9` (36 px). Smaller than mobile minimum because this is a desktop-first internal tool.

### Imagery

The product ships **no photography**. The only visual asset is the SVG logo (a stylized "A" with an orange dot — recreated as `assets/logo.svg`). New designs that need imagery should default to **none**; if absolutely required, prefer monochrome line illustration over photography.

---

## Iconography

**Source:** `lucide-react@^1.11`. The icon set is loaded as a JS dependency, not as an SVG sprite. There is no custom icon font. There are **no PNG icons**. There is **no emoji**.

**Stroke style:** lucide's default — 2 px stroke, `stroke-linecap: round`, `stroke-linejoin: round`, no fill except where intrinsic. Icons sit on `1em × 1em` and inherit `currentColor`, so they take whatever foreground/muted/destructive color is in scope.

**Sizes used in the codebase:**

| Use                                  | Size class | px equivalent |
|--------------------------------------|------------|---------------|
| Inline with body copy                | `h-3 w-3`  | 12 px         |
| Button leading icon                  | `h-3.5 w-3.5` | 14 px      |
| Default lucide size                  | `h-4 w-4`  | 16 px         |
| Empty-state hero icon                | `h-5 w-5`  | 20 px         |

**Icons actually used** (from grepping the source):

`Activity`, `Brain`, `Check`, `CheckCircle2`, `CircleAlert`, `CircleDashed`, `CircleSlash`, `Copy`, `CornerDownLeft`, `Download`, `ExternalLink`, `FileText`, `FileWarning`, `Hash`, `ImageIcon`, `Inbox`, `Layers`, `Loader2`, `Moon`, `Settings2`, `Sparkles`, `Sun`, `Trash2`, `Upload`, `X`.

**For mocks, prototypes, and recreations:** load lucide via the CDN bundle so you can use the same set without a build step:

```html
<script src="https://unpkg.com/lucide@latest/dist/umd/lucide.min.js"></script>
<script>lucide.createIcons();</script>
<i data-lucide="activity"></i>
```

…or copy individual icon SVGs from <https://lucide.dev/icons>.

**Brand mark:** see `assets/logo.svg`. It is a 32×32 rounded-square with a stylized capital "A" stroked in `currentColor` plus a 2.4 px orange dot at the lower right (`hsl(var(--accent))`). The dot **is** the brand — keep it whenever the mark is reproduced. Do not invent alternative wordmarks.

---

## UI kits

### `ui_kits/console/`

A click-thru recreation of the entire app: header, sidebar, submit form, job panel, status badges, and timeline. Implemented as small JSX components composed in `index.html`. Includes a fake job lifecycle (PENDING → RUNNING → SUCCEEDED) so the user can submit a mock job and watch the timeline animate. See `ui_kits/console/README.md` for component map.

---

## Caveats

1. **System font, on purpose.** Neither the repo nor this design system pins a webfont. `--font-sans` and `--font-mono` resolve to the OS-native stack so the design matches what each user actually sees in their browser. If the team later picks a brand webfont, drop the files into `fonts/` and prepend it to `--font-sans` in `colors_and_type.css`.
2. **Logo recreation.** The brand mark in `Header.tsx` is inline JSX, not an exported SVG. `assets/logo.svg` is a faithful re-export of that JSX. If a designer-authored logo file exists elsewhere, swap it in.
3. **Icons via CDN.** The console bundles `lucide-react`. The UI-kit recreation links lucide via UMD CDN to match without a build step.

# SKILL.md — AI 파이프라인 콘솔 design system

This skill teaches Claude how to design *for* the **async-ocr-rag-multimodal-pipeline console** — the Korean-language internal dev tool whose visual language is captured in this design system.

## When to use this skill

Use this skill when the user asks for any of:

- A new screen, panel, or popover for the console.
- A redesign or variation of an existing console surface (sidebar, submit form, job panel, header).
- A monitoring / diagnostic view, admin tool, or developer utility that should match the console's voice.
- Marketing-adjacent material *for the project* (e.g. a pitch deck explaining the pipeline) — but **only** if you also borrow the type and color tokens from `colors_and_type.css`. The product itself is not a marketing surface.

If the user asks for a consumer-facing app, generic SaaS landing page, or anything that would benefit from photography or marketing tone — **do not use this skill**. Tell the user this brand is internal-only and ask if they have a different brand in mind.

## Hard rules — non-negotiable

1. **Korean-first copy.** Every user-visible string is Korean. English-derived terms (`MOCK`, `RAG`, `OCR`, `MULTIMODAL`, status codes like `IN_PROGRESS`) stay UPPERCASE verbatim and are set in mono. If the user writes English-only, ask which Korean copy to use, or default to the patterns documented in `README.md` § Content fundamentals.
2. **No emoji.** None. Anywhere. Use `lucide` icons via CDN.
3. **No marketing voice.** No "Discover", "Get started", "Unleash", exclamation points, or chatty empty states. Copy is terse, technical, and stative — see the verbatim examples table in `README.md`.
4. **Tokens come from `colors_and_type.css`.** Never invent new colors, font sizes, shadows, or radii. If you need a value that's not there, add it to that file with a justifying comment — don't hardcode.
5. **Reserve color for meaning.** Capability accents (`--cap-mock/rag/ocr/multimodal`) appear ONLY on the sidebar dot, capability-picker active ring, and the capability pill in the job header. State colors (`--success/warning/destructive`) only on status. Everything else is parchment + ink + signal orange.
6. **Mono for machine-emitted strings.** Job IDs, durations, byte counts, capability tags, status pills, eyebrows. Never set body prose in mono.
7. **Mid-dot separator (`·`).** Chain related metadata. Never use bullets, pipes, slashes, or em-dashes for the same job.
8. **Iconography is `lucide`.** 2 px stroke, `currentColor`. The complete set used by the product is enumerated in `README.md` § Iconography. Don't draw new SVG glyphs unless the user explicitly asks.

## Files to load before designing

```
README.md             # Content fundamentals, voice, examples, density rules
colors_and_type.css   # Token source of truth — read this every time
ui_kits/console/      # Working component recreations; lift styles + JSX from here
preview/              # Atomic component cards — visual reference for every component
assets/logo.svg       # Brand mark — use as-is, don't redraw
```

## Designing a new screen — checklist

1. **Layout.** Default to the existing app shell: sticky header (`Header.jsx`), 320 px sidebar on `lg+`, max content width 1440 px, `24 px` gaps between major regions.
2. **Card chrome.** New cards = `bg-card`, `border-border`, `--shadow-soft`, `border-radius: 8px`, padding `18–20px`. Optional muted strip (`bg-muted/30`) for the section eyebrow row.
3. **Section eyebrow.** Mono uppercase 10–11 px label with `letter-spacing: 0.14em–0.18em`, color `--muted-foreground`. Keep them quiet — they're navigation, not decoration.
4. **Buttons.** Use the five variants in `console.css`: primary / secondary / outline / ghost / destructive. Heights: `36 px` default, `30 px` small, `32×32` icon-only.
5. **Status.** Always render via `<StatusBadge status withDot? size />`. Never inline a custom status pill.
6. **Empty states.** Centered, 36 px circular muted icon tile, 13 px bold title (Korean), 12 px muted body sentence. One sentence max. Imperative if it asks for action.
7. **Animation.** `animate-pulse-soft` for live state, `animate-spin` (1 s linear) for loaders, `transition-colors` for hover. No transforms, no springs.
8. **Korean line-breaking.** Use `word-break: keep-all` on prose so Korean words don't split across lines. Already set on the body in `colors_and_type.css`.

## Designing for variations

When the user asks for "another version", expose variations as **Tweaks** inside the existing screen (see the platform's Tweaks protocol). Don't fork the file unless they explicitly ask. Good axes to tweak in this brand:

- **Density** — compact (sidebar 280 px, card padding 14 px) vs. comfortable (current 320 px / 18 px).
- **Mono usage** — full mono everywhere on data rows, vs. mono-only-on-IDs (current default).
- **Capability accent intensity** — 2 px ring + tinted bg (current) vs. just colored dot, vs. full-bleed colored card top stripe.
- **Status pill style** — icon variant vs. dot variant.

## Designing **outside** the product (decks, docs)

If the user wants a deck or doc *about* the pipeline:

- Title slides: parchment background, navy ink type, signal-orange dot accent. No photos.
- Body: same type system. Set code in JetBrains Mono. Use the mid-dot separator on data lines.
- Diagrams: 2 px stroke (matches lucide), navy on parchment, capability hues only when calling out a specific capability.
- Cover: the brand mark from `assets/logo.svg` at 64–96 px. No tagline.

## Anti-patterns to refuse

- A "hero gradient" or full-bleed photo background.
- Title-case English headings on Korean screens.
- Sentence-case ALL CAPS labels (e.g. "Submit Job"). Korean copy or all-caps mono codes only.
- A new accent color outside the four `--cap-*` hues + `--accent` orange.
- `border-radius` larger than 12 px outside of pills.
- Stroke widths other than 2 px (lucide's default).
- Drop-shadows with pure-black `rgba(0,0,0,_)`. Always derive shadow color from foreground HSL.

## When in doubt

Read `ui_kits/console/index.html` running in a browser. Match what you see — density, color frugality, mono accents, quiet animation. Then read the verbatim copy table in `README.md` and write Korean strings that fit that register.

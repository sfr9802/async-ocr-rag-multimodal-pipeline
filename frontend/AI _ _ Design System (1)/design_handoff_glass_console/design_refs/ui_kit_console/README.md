# UI kit — `ui_kits/console/`

Hi-fi, click-thru recreation of the **AI 파이프라인 콘솔** screen. Open `index.html` and you get a working demo: a fake job lifecycle (PENDING → QUEUED → RUNNING → SUCCEEDED) animates in real time, the sidebar persists to localStorage, dark mode toggles, and each capability picker swaps to its own form.

This is a *design recreation*, not the real app — there is no API call. The lifecycle is simulated by `setTimeout` inside `ConsoleScreen.jsx::submitJob`.

## Files

| File                  | Responsibility |
|-----------------------|---|
| `index.html`          | Loads tokens (`../../colors_and_type.css`), `console.css`, lucide UMD, React + Babel, then mounts `<ConsoleScreen/>`. |
| `console.css`         | All component styles for this kit. Sits *on top of* the global token stylesheet — never duplicates a token. |
| `Header.jsx`          | Sticky top bar — logo, eyebrow, connection chip, settings, theme toggle. |
| `JobSidebar.jsx`      | Local-history list, bucketed by recency. Active row gets a 2.5 px primary rail + card shadow. |
| `SubmitForm.jsx`      | Capability-aware composer. Renders one of four form bodies based on `capability` prop. |
| `CapabilityPicker.jsx`| 4-up tile picker. Active tile = 2 px tinted ring + card background. |
| `FileDropzone.jsx`    | Drop target with drag-hover state + selected-file row + remove button. |
| `JobTimeline.jsx`     | Pill-strip of lifecycle events. Connectors carry the duration between events; current event pulses. |
| `ResultViewer.jsx`    | Job header + status pill + timeline + tabs (`결과` / `원본`). Body switches on capability. |
| `StatusBadge.jsx`     | Status → KR-label + tone + icon mapping. The single source of truth for status display. |
| `ConsoleScreen.jsx`   | Top-level screen. Owns history state, theme, online ping, and the fake job lifecycle. |

## Component dependency order

`Header` ← `StatusBadge` ← `JobTimeline` ← `CapabilityPicker` ← `FileDropzone` ← `SubmitForm` ← `ResultViewer` ← `JobSidebar` ← `ConsoleScreen`

Each component exposes itself on `window` at the bottom of its file (Babel scripts don't share scope). `index.html` loads them in this order so they are defined before consumers.

## Re-using outside this kit

Every component reads only CSS custom properties from `colors_and_type.css` and class names defined in `console.css`. To lift a single component into a new design, copy the `.jsx` file plus the relevant `.ds-*` classes from `console.css` and you're done — no internal imports.

## Known shortcuts

- `online` is faked: it flips to `true` 700 ms after mount.
- The submit lifecycle is fixed at PENDING(0) → QUEUED(180 ms) → RUNNING(1.1 s) → SUCCEEDED(2.4–3.6 s). Real failures / cancellations only show up in the seeded history.
- `RAG` results, `OCR` text, and `MULTIMODAL` answers are hand-fabricated in `fabricateResult()` to mirror the shapes the real `core-api` returns.
- localStorage key is `console.demo.v1` — bump it when changing the persisted shape.

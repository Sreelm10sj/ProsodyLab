# ProsodyLab — Frontend Audit

> **Scope**: Read-only reconnaissance. No code was modified.
> **Method**: Source code extraction (Step 2 of web-design-engineer skill) + redesign-protocol.md §2.
> **Date**: 2026-08-30
> **Visual baseline screenshots**: Screenshots were requested but could not be captured — the Playwright browser driver CDN returned a 404. Screenshots should be captured manually from `http://localhost:5173/` (dev server is running) and saved to `docs/screenshots/`.

---

## 1. Structure

### 1.1 Full Component Tree

```
src/
+-- main.jsx                   Entry point; mounts <App> in React.StrictMode
+-- App.jsx                    Root component; owns all app-level state, orchestrates layout
+-- api.js                     HTTP layer; three named exports (checkHealth, transcribeAudio, generateAnnotation)
+-- index.css                  Global CSS + custom design tokens (@theme block + hand-authored plain CSS)
+-- App.css                    Leftover Vite scaffold CSS — NOT imported by App.jsx (dead code)
+-- components/
    +-- AudioInput.jsx          Tab-bar shell: delegates to Recorder (record) or file <input> (upload)
    +-- Recorder.jsx            MediaRecorder + Web Audio visualizer canvas (real-time frequency bars)
    +-- TranscriptView.jsx      Word-level stress display with karaoke highlight + pause badges + phrase markers
    +-- PhraseIntonation.jsx    Inline arrow badge rendered at phrase boundaries; CSS hover tooltip
    +-- ProsodyPanel.jsx        Collapsible panel: legend + zoom/view controls + PitchContour
    +-- PitchContour.jsx        Two-layer canvas (static + RAF playhead); scrollable, interactive, keyboard-seekable
    +-- AnnotationViewer.jsx    Structured annotation display: transcript, words table, pauses table,
                                intonation table, chronological timeline; download buttons (JSON + TXT)
```

**One-line purpose per file:**

| File | Purpose |
|---|---|
| `main.jsx` | React 19 entry — renders `<App>` into `#root` |
| `App.jsx` | App shell; holds all shared state and event handlers |
| `api.js` | Three fetch wrappers for `/api/health`, `/api/transcribe`, `/api/annotate` |
| `index.css` | Tailwind v4 `@import`, `@theme` token block, all hand-authored plain-CSS classes |
| `App.css` | Vite scaffold leftover — **not imported anywhere; dead code** |
| `AudioInput.jsx` | Tab-bar (Upload / Record) + selected-audio preview row + TRANSCRIBE submit button |
| `Recorder.jsx` | Browser MediaRecorder + requestAnimationFrame frequency-bar visualizer |
| `TranscriptView.jsx` | Renders each word: stress highlight, karaoke inversion, pause dots, phrase arrows; IPA toggle |
| `PhraseIntonation.jsx` | Stateless arrow badge + multi-field CSS hover tooltip |
| `ProsodyPanel.jsx` | Collapsible container; owns zoom (pixelsPerSec) + viewMode state; legend + PitchContour |
| `PitchContour.jsx` | Two stacked canvases (static contour + dynamic playhead via RAF); ResizeObserver; keyboard seek |
| `AnnotationViewer.jsx` | Read-only display of /api/annotate response with download buttons |

### 1.2 Routes / Pages / Navigation

**No router is present.** The app is a single-page, single-route application.
Navigation is purely state-driven section visibility — no URL changes, no `react-router-dom`.

| Trigger | What becomes visible |
|---|---|
| App load | Header + AudioInput (record tab active) + empty TranscriptView placeholder |
| File selected or recording complete | TRANSCRIBE button activates |
| TRANSCRIBE submitted | isLoading spinner replaces TranscriptView |
| API returns | TranscriptView with stress highlights; if pitch_contour or phrases non-empty ? ProsodyPanel |
| Generate Annotation clicked | isAnnotating spinner; AnnotationViewer mounts below |

### 1.3 State Management

All state lives in `App.jsx` using `useState` and `useRef`. No context, no store.

| State variable | Type | Purpose |
|---|---|---|
| `transcription` | string | Raw text from backend |
| `words` | Word[] | Word objects: start, end, stressed, pause_before, word |
| `phrases` | Phrase[] | Phrase objects: start, end, word_indices, pitch_stylized |
| `pitchContour` | PitchPoint[] | Raw F0 contour: {time, semitone} pairs |
| `pitchStylized` | PitchPoint[] | DP-MAE stylized F0: {time, semitone, P?, K?} |
| `utteranceMedianHz` | number/null | Reference pitch for Hz calculation |
| `showProsody` | boolean | Controls ProsodyPanel expand/collapse AND inline phrase arrows |
| `isLoading` / `isAnnotating` | boolean | Loading states for the two API calls |
| `error` / `annotationError` | string | Error message display |
| `health` | object | Backend status from GET /api/health |
| `audioUrl` / `audioBlob` / `audioFilename` | mixed | Held so annotation can re-submit the same audio |
| `annotation` | object | Full annotation response |
| `currentTime` / `duration` | number | Driven by `<audio>` onTimeUpdate / onLoadedMetadata |
| `audioRef` | ref | Points at `<audio>` DOM node for imperative seek |

`ProsodyPanel.jsx` holds `viewMode` ('both'|'raw'|'stylized') and `pixelsPerSec` (50/100/200/400).
`PitchContour.jsx` holds `containerWidth` (ResizeObserver-driven, for canvas sizing).

---

## 2. Current Visual System

### 2.1 Tailwind Configuration

**Tailwind v4** (via `@tailwindcss/vite` plugin). **No `tailwind.config.js`** — all customization lives in the `@theme {}` block inside `index.css`.

Custom tokens:

```css
/* Font families */
--font-sans:  'Inter', system fallbacks
--font-serif: 'Source Serif 4', Georgia fallbacks
--font-mono:  'IBM Plex Mono', ui-monospace fallbacks

/* Color palette */
--color-paper:        #F7F8FA   (cool off-white, page background)
--color-ink:          #1E2530   (near-black, primary text + borders)
--color-ink-light:    #4A5568   (medium gray, secondary text)
--color-ink-lighter:  #A0AEC0   (light gray, de-emphasized labels)
--color-highlight:    #F4C430   (highlighter yellow, accent)
```

No dark mode tokens. No custom spacing scale. No custom border-radius tokens.
Shadows are arbitrary Tailwind values (not token-ised):
- `shadow-[4px_4px_0_0_#1E2530]` — primary card shadow
- `shadow-[2px_2px_0_0_#F4C430]` — secondary (JSON download button)

### 2.2 Global CSS Classes Defined in index.css

| Class / selector | Purpose |
|---|---|
| `body` | font-sans, color-paper bg, color-ink text, antialiasing |
| `:focus-visible` | `outline: 2px solid var(--color-ink); offset: 2px` — site-wide focus ring |
| `.highlight-sweep` / `::before` | Stress highlight sweep animation (scaleX 0?1) |
| `.word-tooltip-container` / `.word-tooltip` | Word hover tooltip (CSS-only show/hide) |
| `.phrase-marker-container` / `.phrase-marker` | Inline intonation arrow badge |
| `.phrase-marker--rising/falling/level/insuf` | Per-pattern color (hardcoded hex, not CSS vars) |
| `.phrase-tooltip` | Multi-row hover tooltip for phrase markers |
| `.pitch-contour-container` | Canvas wrapper (height: 160px — overridden by inline style in JSX) |
| `.pitch-tooltip` | Fixed-position hover tooltip for pitch canvas |
| `.prosody-panel-body` / `--open` | Grid-template-rows height animation |
| `.prosody-panel-inner` | overflow:hidden clip target |
| `.prosody-chevron` | Chevron rotation container |
| `.sr-only` | Screen-reader-only utility |
| `.legend-grid` / `.legend-item` / `.legend-swatch-wrap` | Legend layout |
| `.legend-preview-stress/pause` / `.legend-arrow` | Legend swatch rendering |

**Rogue hardcoded hex values (outside the token system):**
- Phrase/contour pattern colors — `#059669`, `#3b82f6`, `#94a3b8`, `#9ca3af` — repeated verbatim in `index.css`, `ProsodyPanel.jsx` (LEGEND_ITEMS SVG), and `PitchContour.jsx` (PHRASE_COLORS/PHRASE_BG). Three sources, no single token.
- Canvas background `#F7F8FA` and `#ECEFF4` hardcoded in `PitchContour.jsx drawBackground()`.
- Canvas text color `#4A5568` hardcoded in `drawTimeAxis()`.

### 2.3 Typography — Actual Values from Code

| Element | Font | Size | Weight | Other |
|---|---|---|---|---|
| Logo "ProsodyLab" | mono | text-3xl (30px) | 700 bold | uppercase tracking-tighter |
| Logo sub-tagline | mono | text-xs (12px) | 500 medium | uppercase tracking-widest |
| Status indicator | mono | text-[10px] | 700 bold | uppercase tracking-wider |
| Transcript words | serif | text-[22px] | 400; 700 on stressed | leading-relaxed |
| Section headers (h2, h3) | mono | text-sm (14px) | 700 bold | uppercase tracking-widest |
| AnnotationViewer h2 | mono | text-xl (20px) | 700 bold | uppercase tracking-tight |
| Submit / action buttons | mono | text-sm (14px) | 700 bold | uppercase tracking-widest |
| Word tooltip | sans | 0.75rem (12px) | 500 | — |
| Phrase marker arrow | mono | 0.68rem (~10.9px) | 700 | — |
| Phrase tooltip rows | mono | 0.68rem | 400 / 600 | — |
| Pitch canvas time axis | literal "10px" IBM Plex Mono | — | — | Set in canvas ctx.font |
| Pitch Y-axis labels | mono | text-[10px] | — | DOM overlay spans |
| Table cells | mono | text-sm | 400 / 700 | — |
| Loading / error text | mono | text-sm | 700 bold | uppercase tracking-widest |

Three deliberate voices: **sans** (UI chrome), **serif** (transcript reading content), **mono** (data/labels/everything else). Mono dominates.

### 2.4 Component Libraries Installed

| Library | Version | Used for |
|---|---|---|
| `lucide-react` | `^1.25.0` | Upload, Play, Trash2, FileAudio, Mic, Square, ToggleLeft, ToggleRight, ChevronDown |

**No UI component library** (no shadcn, Radix, Headless UI, MUI, Ant Design, etc.).
All interactive components (tabs, toggle, collapsible, tooltips) are hand-rolled.
**No animation library** (no Framer Motion, GSAP, Lottie, Popmotion, React Spring).

---

## 3. Current Motion

### 3.1 Motion Inventory

| Animation | Trigger | Mechanism | File | Duration / Easing |
|---|---|---|---|---|
| Stress highlight sweep (`::before` scaleX 0?1) | Word mounts after API returns | CSS `@keyframes sweep` | `index.css` L39–64 | 0.25s cubic-bezier(0.1, 0.9, 0.2, 1) |
| Sweep reduced-motion fallback (opacity 0?1) | prefers-reduced-motion | CSS `@keyframes fade` | `index.css` L66–78 | 0.25s ease-out |
| Phrase tooltip show/hide | Hover on phrase marker | CSS transition: opacity + translateY | `index.css` L186, 203–207 | 0.2s (default ease) |
| Word tooltip show/hide | Hover on word | Same pattern | `index.css` L103, 118–122 | 0.2s (default ease) |
| Phrase marker hover opacity (0.55 ? 1.0) | Hover on badge | CSS transition: opacity | `index.css` L156 | 0.15s ease |
| ProsodyPanel collapse/expand | Toggle button click | CSS transition: grid-template-rows | `index.css` L329 | 0.30s cubic-bezier(0.4, 0, 0.2, 1) |
| Chevron rotation (0° ? 180°) | Same toggle | CSS transition: transform on .prosody-chevron | `index.css` L342 | 0.30s cubic-bezier(0.4, 0, 0.2, 1) |
| Collapse + chevron (reduced-motion) | prefers-reduced-motion | transition: none | `index.css` L346–353 | Instant |
| Loading accent bar pulse | isLoading spinner | Tailwind animate-pulse | `App.jsx` L163 | ~2s ease-in-out (Tailwind default) |
| Loading ping dot | isLoading spinner | Tailwind animate-ping | `App.jsx` L165 | ~1s linear (Tailwind default) |
| Backend status dot pulse | health.status === 'checking' | Tailwind animate-pulse | `App.jsx` L129 | Tailwind default |
| Annotation ping spinner | isAnnotating | Tailwind animate-ping | `App.jsx` L210 | Tailwind default |
| Content area fade (opacity 1 ? 0.5) | isLoading | Tailwind transition-opacity duration-300 | `App.jsx` L160 | 0.3s (default ease) |
| TRANSCRIBE button press offset | :active | active:translate-y-[2px] + shadow reduce | `AudioInput.jsx` L90 | Instant (no transition duration set) |
| Generate Annotation button hover offset | :hover | hover:translate-x/y[2px] + shadow reduce + transition-all | `App.jsx` L204 | Tailwind transition-all default (~0.15s ease) |
| Waveform frequency bar visualizer | While recording | requestAnimationFrame, canvas fillRect | `Recorder.jsx` L97–122 | Continuous RAF |
| Pitch playhead line + triangle | Audio playback | requestAnimationFrame, canvas stroke | `PitchContour.jsx` L339–389 | Continuous RAF |
| Zoom scroll preservation | pixelsPerSec state change | Imperative scrollLeft assignment | `PitchContour.jsx` L272–280 | Instant |
| Upload icon / label hover color | Hover on drop zone | Tailwind transition-colors | `AudioInput.jsx` L107–108 | Default |
| Tab button hover | Hover | Tailwind transition-colors | `AudioInput.jsx` L51, 61 | Default |
| Stop button hover | Hover | Tailwind transition-colors | `Recorder.jsx` L152 | Default |

### 3.2 Audit Findings — Animation Standards Concerns

> Observations only. No fixes applied.

| # | Finding | File | Severity |
|---|---|---|---|
| 1 | **`transition-all`** on Generate Annotation button hover + AnnotationViewer download buttons | `App.jsx` L204; `AnnotationViewer.jsx` L43, 49 | Moderate — transitions unknown properties, wastes compositor |
| 2 | **`animate-pulse` / `animate-ping` have no reduced-motion guard** | `App.jsx` L129, 163, 165, 210 | Moderate — these fire for users with prefers-reduced-motion: reduce; the two Tailwind default utilities do not auto-suppress in Tailwind v4 without explicit `motion-safe:` variant or `@media` override |
| 3 | **Tooltip transitions use default `ease` easing** — `transition: opacity 0.2s, transform 0.2s` with no curve specified | `index.css` L103, 186 | Minor — imprecise, not wrong |
| 4 | **`scaleX(0)` origin** — no `will-change` on the highlight-sweep `::before` | `index.css` L55 | Low — potential paint cost on long transcripts |
| 5 | **Stagger delay: `idx × 80ms`** — for a 50-word transcript the last word highlights ~4 seconds after render | `TranscriptView.jsx` L104 | UX concern — not a motion standards violation per se, but noticeable |
| 6 | **TRANSCRIBE `:active` has no `transition-duration`** — the shadow/translate change is instant | `AudioInput.jsx` L90 | Minor — abrupt feel on press |

---

## 4. Key Flows and Their UI Surfaces

### 4.1 Audio Upload / Record

**Components**: `AudioInput.jsx` (shell) + `Recorder.jsx` (record mode)

**Upload tab:**
- Dashed-border drop zone (`label` wrapping hidden `input[type=file]`). Upload icon + "Choose audio file" label. `focus-within:ring-2` on label.
- On file select ? stores `{blob, url, name}` in `selectedAudio`.

**Record tab (default):**
- Circular Mic button ? `getUserMedia()` ? `MediaRecorder` start.
- During recording: red "RECORDING MM:SS" badge + `animate-pulse` dot + frequency-bar canvas (yellow bars, 64px, `bg-black/5`) + Stop button.
- Stop ? `Blob('audio/webm')` ? same selectedAudio shape.

**Shared selected state:**
- FileAudio icon + filename + Trash2 clear button row.
- Full-width TRANSCRIBE button (yellow bg, ink border, 4px offset shadow). Click ? `handleAudioSubmit(blob, filename, url)`.

### 4.2 Transcription Display with Stress Highlighting

**Components**: `TranscriptView.jsx` + `PhraseIntonation.jsx`

- Words rendered as `<span>` flow in `font-serif text-[22px] leading-relaxed`.
- Each word inside `.word-tooltip-container` for hover behavior.
- **Stressed words**: `.highlight-sweep` class ? yellow `::before` sweep animation (delayed by `idx × 80ms`) + `font-bold`.
- **Non-stressed words**: no background; hover ? `hover:bg-black/5`.
- **Linguist View (IPA toggle)**: replaces sweep with a yellow ' prefix (font-sans, inline).
- **Hover tooltip**: mono 12px, shows "Stressed: Yes/No" + [start – end] timestamp.

### 4.3 Karaoke Sync

**Mechanism**: Pure React re-render. `currentTime` from `<audio onTimeUpdate>` compared per-word:

```js
const isPlaying = duration > 0 && currentTime > 0 &&
  currentTime >= startTime && currentTime < endTime;
// isPlaying ? className: 'bg-ink text-paper' (dark inversion)
// stressed && isPlaying ? sweep NOT applied (class excluded)
```

No tween, no animation library. onTimeUpdate fires ~4×/s; fast speech may skip briefly.

### 4.4 Pause Badges

- Computed per-word from `item.pause_before` (acoustic, preferred) or fallback `startTime - prevEndTime`.
- Rendered as `<span className="text-ink-light/50 mx-1 cursor-help tracking-widest" title="Pause: Xs (acoustic)">...</span>`.
- Visual: three ellipsis dots, very muted. Browser-default title tooltip for duration.

### 4.5 Phrase-Level Intonation Markers

- `phraseMarkerMap` in TranscriptView: Map from last-word-index ? phrase object.
- After the last word of each phrase: `<PhraseMarker phrase={…} />` if `showPhraseMarkers`.
- Arrow characters: ? (rising, green #059669), ? (falling, blue #3b82f6), ? (level, gray #94a3b8), ~ (insuf, gray transparent).
- Hover tooltip: pattern name, overall slope, terminal slope, voiced fraction, time span.
- Visibility coupled to `showProsody` — hiding ProsodyPanel also removes inline arrows.

### 4.6 PitchContour — Canvas Rendering Synced to Playback

**Architecture**: Two stacked `<canvas>` + scrollable `<div>` + fixed DOM Y-axis overlay.

| Layer | Ref | Content | Update trigger |
|---|---|---|---|
| Static | `staticRef` | Background, grid, phrase bands, raw/stylized contours, word boundaries, stress diamonds, border, time axis | `drawStatic` useCallback re-runs on data/zoom/size change |
| Dynamic | `dynamicRef` | Playhead vertical line + downward triangle | Every requestAnimationFrame (reads currentTimeRef directly) |

**Sizing:**
- Width = `max(containerWidth, ceil(duration × pixelsPerSec) + 60)` — scrollable for long audio.
- Height = `max(height prop, 450)`; height prop defaults to 500px. The `.pitch-contour-container` CSS `height: 160px` is overridden by inline `style={{ height: logicalH }}`.
- DPR-aware: `canvas.width = floor(logicalW × devicePixelRatio)`.

**Y scale**: Semitone relative to utterance median. Range from contour + stylized data ± padding, clamped to at least ±5 st.

**DOM Y-axis overlay**: 40px fixed div, `z-index:10`, `pointer-events:none`. Tick labels as absolute `<span>` elements (not drawn on canvas).

**Interaction surface:**
- Click on scroll div ? seek to time.
- Hover ? `handleMouseMove` ? DOM-mutates `tooltipRef.current.innerHTML` (bypasses React for perf).
- `ArrowLeft` / `ArrowRight` ? ±0.5s seek.
- `tabIndex={0}` on scroll div.

---

## 5. Protected Contracts

### 5.1 API Call Shapes (must not change)

```
GET  /api/health
     ? { status, device, model_loaded }

POST /api/transcribe  (FormData: file=Blob)
     ? { transcription, words, phrases?, pitch_contour?, pitch_stylized?, utterance_median_hz? }
     Note: ?? guards throughout App.jsx for backward compat with pre-Phase-4 backends

POST /api/annotate  (FormData: file=Blob)
     ? { metadata, transcription, words, pauses, intonation, timeline }
```

### 5.2 Component Prop Interfaces

| Component | Props contract |
|---|---|
| `AudioInput` | `onAudioSubmit(blob, filename, url)`, `onAudioClear()`, `isLoading` |
| `Recorder` | `onRecordingComplete({ blob, url, name })`, `isLoading` |
| `TranscriptView` | `transcription`, `words`, `phrases`, `showPhraseMarkers`, `currentTime`, `duration`, `onSeek(t)` |
| `PhraseMarker` | `phrase` (with `pitch_stylized.pattern`, `start`, `end`) |
| `ProsodyPanel` | `isExpanded`, `onToggle`, `words`, `phrases`, `pitchContour`, `pitchStylized`, `utteranceMedianHz`, `currentTime`, `duration`, `onSeek(t)` |
| `PitchContour` | `words`, `phrases`, `pitchContour`, `pitchStylized`, `viewMode`, `pixelsPerSec`, `height`, `utteranceMedianHz`, `currentTime`, `duration`, `onSeek(t)` |
| `AnnotationViewer` | `annotation` (full annotate shape), `filename` |

### 5.3 Keyboard / Focus Behavior

| Element | Behavior | Location |
|---|---|---|
| All interactive elements | :focus-visible 2px ink outline | `index.css` |
| ProsodyPanel toggle | focus-visible:ring-2 focus-visible:ring-ink focus-visible:ring-inset | `ProsodyPanel.jsx` |
| PitchContour scroll div | tabIndex=0; ArrowLeft/Right ±0.5s seek | `PitchContour.jsx` |
| ProsodyPanel collapsible body | `inert` attribute set/removed imperatively — blocks tab-into-hidden content | `ProsodyPanel.jsx` CollapsibleBody |
| AudioInput file label | focus-within:ring-2 | `AudioInput.jsx` |
| Tab buttons | focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-highlight | `AudioInput.jsx` |
| Clear / stop buttons | focus-visible:ring-2 focus-visible:ring-ink | `AudioInput.jsx`, `Recorder.jsx` |
| TRANSCRIBE button | focus-visible:ring-2 focus-visible:ring-ink focus-visible:ring-offset-2 | `AudioInput.jsx` |

### 5.4 Accessibility Attributes Already Present (must not be removed)

| Element | ARIA |
|---|---|
| ProsodyPanel toggle button | `aria-expanded`, `aria-controls="prosody-details"`, `id="prosody-toggle"` |
| ProsodyPanel body | `role="region"`, `aria-labelledby="prosody-toggle"`, `id="prosody-details"` |
| Both pitch canvases | `aria-hidden="true"` |
| Legend | `role="list"` / `role="listitem"` on items |
| Legend swatches | `aria-hidden="true"` |
| PhraseMarker wrapper | `aria-label="Intonation: {label}"` |
| PhraseMarker tooltip | `role="tooltip"` |
| Upload file input | `className="sr-only"` |
| Buttons | `aria-label` on: Remove audio, Start recording, Stop recording, Upload Audio Tab, Record Audio Tab, Toggle linguist view |

### 5.5 Other Contracts

- `showProsody` state is shared: controls both ProsodyPanel visibility AND `showPhraseMarkers` in TranscriptView. Redesigns must not silently decouple these without App.jsx awareness.
- `.pitch-contour-container` height: 160px in CSS is overridden by inline style. If the CSS class is removed, the inline `style={{ height: logicalH }}` must remain.
- Canvas draw code hardcodes `#F4C430` for stress diamonds — does not read CSS vars. Token changes won't auto-propagate here.

---

## 6. Dependencies — Animation and UI Related

From `package.json` (full picture):

| Package | Version | Redesign guidance |
|---|---|---|
| `lucide-react` | `^1.25.0` | **Already installed.** Use it. Do not introduce a second icon library. |
| `tailwindcss` | `^4.3.3` | v4 API — `@theme {}` block only, no `tailwind.config.js`. Tokens added here become Tailwind classes automatically. |
| `@tailwindcss/vite` | `^4.3.3` | Vite plugin — no additional config. |
| `react` + `react-dom` | `^19.2.7` | React 19 — concurrent features available. |
| `vite` | `^8.1.1` | |
| `@vitejs/plugin-react` | `^6.0.3` | |

**No animation library.** All motion is CSS or RAF. Before introducing Framer Motion / GSAP / etc., invoke the `pick-ui-library` skill.

**`App.css`** — 185 lines never imported. Safe to delete.

---

## 7. Visual Baseline — Screen-by-Screen Description

> Screenshots: not captured automatically (Playwright CDN unavailable). Capture manually from http://localhost:5173/ and save to docs/screenshots/.

### Screen 1 — Upload / Record (empty state)

Layout: centered column, `max-w-3xl`, `py-12 px-4`, `gap-8` between sections.

- **Header**: 4px bottom border (`border-b-4 border-ink`). Left: "ProsodyLab" wordmark — mono, 30px, bold, uppercase, tracking-tighter. "Lab" is in a yellow-bg / ink-text chip (`px-1 rounded-sm`). Right: status dot + "STATUS: ONLINE/OFFLINE/CHECKING..." in 10px mono bold uppercase.
- **AudioInput card**: `border-2 border-ink shadow-[4px_4px_0_0_#1E2530]` with a 6px black top bar (`h-1.5 bg-ink`). Tab bar (Upload | Record) with active tab as `bg-ink text-paper`. Record tab default ? centered circular Mic button in a white 180px+ content area.
- **TranscriptView placeholder**: Border + 4px offset shadow, centered 14px mono text "Record or drop a clip to see which words carry the stress." Thin 1px top bar (ink/20).
- No ProsodyPanel (no data).

**Visual character**: High contrast, printerly, brutalist-tool. Five-color palette: off-white bg, near-black ink, yellow accent (absent here — no data). Heavy offset shadows on cards. Very geometric, no decoration. Large whitespace between sections.

### Screen 2 — Loading State

- AudioInput disabled (button shows "ANALYZING...", opacity-70).
- Content area fades to opacity-50 + pointer-events-none.
- Loading placeholder: yellow `h-2 animate-pulse` top bar + "Listening for stress..." + `animate-ping` yellow dot. Contained in a bordered card with a larger `shadow-[4px_4px_0_0_#1E2530]`.

### Screen 3 — Transcript with Stress Highlights

- `<audio>` playback bar (black bg, yellow accent label "Master Playback", native audio controls inverted white-on-black).
- TranscriptView card: yellow `h-2` top bar, "TRANSCRIPT OUTPUT" header (14px mono bold), Linguist View toggle (ToggleRight/Left icon + label). Words at 22px Source Serif 4, leading-relaxed. Stressed words: yellow sweep highlight + bold. Pause dots: muted ellipsis. Phrase arrows: small colored bordered badges at phrase ends.
- ProsodyPanel below (if data): yellow top bar, "PROSODY DETAILS" + utterance median Hz, expand chevron.

**Visual character**: Stress highlights (yellow) become the primary visual accent. The serif transcript text is the center of gravity. The remaining chrome recedes. The design reads as a research instrument displaying real linguistic data.

### Screen 4 — Prosody Panel Expanded (Pitch Contour)

- **Legend**: 7 items in a flex-wrap row: Stress preview (yellow-bg "word"), Pause preview ("..."), Raw F0 line (SVG), Stylized F0 (3-segment colored SVG), Rising/Falling/Level arrow badges.
- **Controls**: Zoom segmented (50|100|200|400) + View segmented (Both|Raw|Stylized). Both groups: `bg-ink/5` pill, active = `bg-white shadow-sm`.
- **PitchContour canvas** (500px+ tall): 40px Y-axis overlay (semitone ticks). Inner plot with `#ECEFF4` bg. Grid lines light. Phrase bands subtle colored fills. Raw contour: thin gray line (rgba(30,37,48,0.30)). Stylized contour: 3px colored line per phrase. Stress diamond markers: yellow rgba(244,196,48,0.88). Time axis at bottom (10px monospace). Playhead: 1.5px ink vertical line + downward triangle. Horizontally scrollable at higher zoom levels.

**Visual character**: Data-instrument aesthetic. The canvas overwhelms in vertical space. The panel header (legend + controls) is compact and functional.

### Screen 5 — Annotation Viewer

- Full-width card, 6–8px padding, gap-8 between sections.
- **Header**: "ANNOTATION VIEWER" (20px mono bold) + "Generated from backend pipeline" sub-label + TXT/JSON download buttons. TXT: paper bg, ink border, 2px offset shadow. JSON: ink bg, yellow shadow.
- **Four sections**: Transcription text (18px serif in ink/5 bg box), Words table, Pauses table, Intonation table (all with black `bg-ink text-paper` theads, mono text, hover rows).
- **Timeline**: Scrollable list (max-h-96), word events tagged yellow / pause events tagged black mono chips.

**Visual character**: Consistent with overall system but denser. Tables have a strong data-terminal feel.

---

## 8. Visual Room Assessment per Surface

> Feeds the redesign classifier — not a classification itself.

| Surface | Free to move | Load-bearing (must not change) |
|---|---|---|
| Header / wordmark | Typography details, "Lab" chip treatment, status indicator layout | Text "ProsodyLab"; status dot + content |
| AudioInput card | Shadow, border, padding, tab styling, drop-zone appearance, button treatment | onAudioSubmit signature; file-type validation; disabled behavior; aria-labels |
| Recorder waveform | Canvas bg, bar color (hardcoded #F4C430), canvas size, surrounding layout | MediaRecorder logic; onRecordingComplete shape |
| TranscriptView | Font size, card padding, empty-state copy, stress background treatment, header layout | .highlight-sweep reduced-motion path; tooltip hover behavior; karaoke class logic; phrase marker insertion |
| Pause badges | Visual style (dots color, opacity, size) | cursor-help title tooltip; pause_before consumption logic |
| Phrase markers | Badge shape, colors (currently hardcoded — could be tokenised) | role="tooltip", aria-label, tooltip content fields |
| Master Playback bar | Container styling, label placement, time display format | `<audio>` element; onTimeUpdate / onLoadedMetadata; audioRef |
| ProsodyPanel header | Legend layout, control button group style, chevron treatment | aria-expanded, aria-controls, id attributes; inert behavior; .prosody-panel-body class names |
| PitchContour | Y-axis overlay styling, scroll container styling, canvas bg colors (JS-hardcoded) | Two-canvas architecture; RAF loop; keyboard handlers; tabIndex=0; zoom scroll preservation |
| AnnotationViewer | Section header styling, table styling, download button treatment | Download filenames; annotation response shape; table column order |
| Global tokens | All five @theme colors, three font families, arbitrary shadow values | :focus-visible ring; bg-paper on body |

---

## 9. Quality Debt Observations

**Preserve** (strengths to carry forward):
- Three-voice type system (sans / serif / mono) with clear, consistent roles
- Reduced-motion handling for the two most prominent animations (sweep, panel collapse + chevron)
- Accessibility wins: aria-expanded, inert, role="tooltip", role="region", sr-only, tabIndex on canvas, keyboard seek, aria-labels on all interactive elements
- ResizeObserver + DPR-aware canvas rendering
- DOM-ref tooltip mutation on pitch hover (correct perf pattern for 60fps canvas)
- Semantic `<section>` elements in AnnotationViewer

**Improve** (weak points a redesign should address):
- Phrase/canvas pattern colors are hardcoded in three separate files — converge to one source
- `transition: all` on two button interactions (Generate Annotation, AnnotationViewer downloads)
- `animate-pulse` / `animate-ping` lack reduced-motion guards
- Stagger `idx × 80ms` creates very late highlights for long transcripts
- `height: 160px` in `.pitch-contour-container` is misleading — overridden immediately by inline style
- `model_loaded` field received in health response but never displayed
- No `<meta name="description">` or OG tags in `index.html`

**Remove** (dead / broken):
- `App.css` — 185 lines of Vite scaffold, not imported anywhere
- CSS vars `--accent`, `--accent-bg`, `--accent-border`, `--border`, `--social-bg`, `--shadow` referenced only in `App.css` — never defined anywhere (dead references inside dead code)

---

*End of audit — ready to feed redesign classification.*

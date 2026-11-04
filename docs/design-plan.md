# ProsodyLab — Design Plan

> **Status**: Planning deliverable. No code written, no files modified, no dependencies installed.
> **Date**: 2026-08-30
> **Input**: docs/frontend-audit.md; web-design-engineer skill steps 2-3; redesign-protocol.md; design-calibration.md; failure-patterns.md; style-recipe catalog.

---

## Part 0 — Brainstorm: Before Locking a Direction

### Three candidate anchors (rejected before choosing)

The brief demands the direction be grounded in acoustic phonetics, not a generic default.
The three clusters flagged in the brief are checked first:

| Cluster | Verdict |
|---|---|
| Warm cream + terracotta serif | Reject. This is a food/lifestyle cluster. Prosody research is numeric, time-series, phonological — not warm. It would be a costume. |
| Near-black + acid-green / vermilion | Reject. This reads as developer-tool cosplay (terminal emulation). Phonetics is not a devtool; its users are linguists and speech scientists, not engineers. Acid-green evokes hacking, not speech. |
| Broadsheet hairline newspaper | Reject unless deliberately chosen. The existing app already leans into a printerly brutalist voice (offset ink shadows, uppercase mono). Arriving at a broadsheet unprompted would be a half-commit to the existing direction, not a decision. |

### What makes ProsodyLab visually different from all of those

Speech prosody analysis has three core visual artefacts that no other field has:

1. **Pitch contour lines** — a time-series F0 trace that undulates and breaks (unvoiced gaps). Visually: sinuous colored curves over a gridded plot. They look like an ECG or seismograph, but they are pitch.
2. **Stress marks** — annotating which syllables carry lexical emphasis. In linguistic notation: bold, underline, or the IPA diacritic.
3. **Intonation arrows** — the up, down, level markers that classify the pitch slope of a phrase. These are a compact, symbol-heavy visual language.

These three artefacts are not decorative — they *are* the product. Any palette and typography choice must make these artefacts the visual anchor, not compete with them.

### The chosen anchor: a new recipe — "Phonetics Workbench"

No single recipe from the 25-file catalog fits cleanly:
- `tufte-dataink` comes closest on the Analyze surface (data-ink, direct labels, minimal chrome) but its static editorial philosophy conflicts with the interactive/real-time aspects (recording orb, karaoke sync, pitch playhead).
- `bloomberg-terminal` fits the density of the pitch contour view but its amber-on-navy palette would overwhelm the yellow stress-highlight — two accent colors fighting each other.
- `field-io` fits the landing page motion brief but not the analytical surfaces.

**Resolution**: Build a new recipe called "Phonetics Workbench" that draws deliberately from:
- `tufte-dataink` — data-ink discipline, direct labels, sparse chrome, scientific typography base (for Analyze and Model Generation)
- `field-io` — generative/reactive motion *only on the landing page*
- The existing app's brutalist skeleton — kept as the structural voice for Analyze and Model Generation

This is an intentional two-school combination, not a drift. The school boundary is hard:
- Landing page = field-io motion layer on top of Phonetics Workbench palette.
- Analyze + Model Generation = tufte-dataink discipline, no generative motion.

---

## Part 1 — Design Read and Five Dials (per surface)

### Surface A — Landing Page

```yaml
Design Read:
  artifact: landing page (greenfield, net-new)
  audience: linguistics researchers, speech scientists, NLP practitioners;
            secondary: curious newcomers discovering the tool
  visual-language: phonetics-instrument editorial — scientific clarity
                   broken by one generative field moment in the hero
  mode: greenfield / extension (no prior contract)
  constraints: must link to Analyze and Model Generation; must share the same
               palette as the tool so the transition feels like entering
               a room, not navigating to a different site
  visual-variance: 7
  motion-intensity: 8
  information-density: 3
  asset-dependence: 2
  brand-fidelity: 4
```

Dial consequences:
- Variance 7 — off-grid hero composition; landing section breaks can break the column
- Motion 8 — interactive hover background is justified; GSAP-sequenced headline entry; orb teaser
- Density 3 — one dominant idea per viewport section; no card grids
- Asset dependence 2 — the interactive background is the asset; no photography needed
- Fidelity 4 — tokens must share ancestry with Analyze; typographic voice must feel related

### Surface B — Analyze (Redesign Preserve)

```yaml
Design Read:
  artifact: research tool / analytical instrument (redesign-preserve)
  audience: linguists, phoneticians, speech researchers; advanced users
  visual-language: phonetics workbench — tufte-inflected instrument;
                   high data-ink, minimal chrome, serif transcript,
                   mono data labels
  mode: redesign-preserve (sections 5.1-5.5 from audit are frozen)
  constraints: all API shapes, prop interfaces, keyboard/focus behavior,
               and ARIA attributes from audit section 5 are frozen;
               canvas draw code for pitch contour is contract-critical
  visual-variance: 4
  motion-intensity: 4
  information-density: 7
  asset-dependence: 1
  brand-fidelity: 9
```

Dial consequences:
- Variance 4 — stable grid, one asymmetric moment allowed (the orb)
- Motion 4 — orb (meaningful: real mic input), panel transitions; nothing cinematic
- Density 7 — pitch contour + transcript + legend can coexist; do not thin them out
- Fidelity 9 — existing token values are the foundation; evolution not overhaul

### Surface C — Model Generation (Extension, mock-only)

```yaml
Design Read:
  artifact: UI prototype / mock feature (extension on Analyze vocabulary)
  audience: same as Analyze; the user has already used Analyze
  visual-language: identical to Analyze — Phonetics Workbench tokens
  mode: extension (new feature using existing system)
  constraints: MOCK ONLY — no real API calls; static/sample data;
               must be clearly labeled as preview/beta
  visual-variance: 4
  motion-intensity: 3
  information-density: 6
  asset-dependence: 1
  brand-fidelity: 10
```


---

## Part 2 — The Phonetics Workbench Token System

### Why this is a new recipe, not a borrowed one

The existing app (from the audit) already has a strong skeleton:
- Paper/ink/highlight three-value contrast system — keep it
- Three-voice typography (sans / serif / mono) — keep it; only refine the faces
- Offset-shadow card vocabulary — evolve it, do not abandon it

What the existing system lacks:
- A display-level type face for the landing page hero
- A proper semantic color tier for the three pitch-pattern colors (rising/falling/level)
- A token structure that works across three surfaces without hardcoding the same hex in three files

### 2.1 Palette — Six Named Values

| Token name | Hex | Role |
|---|---|---|
| `--c-paper` | `#F5F4F0` | Page background. Warm white with the slightest chalk undertone — warmer than the current #F7F8FA which reads slightly cold and digital. |
| `--c-ink` | `#1C2128` | Primary text, borders, icons. A touch of blue-black — evokes academic print. |
| `--c-ink-mid` | `#4E5766` | Secondary text, muted labels. |
| `--c-ink-faint` | `#A8B0BE` | De-emphasized, placeholder, hairline rules. |
| `--c-stress` | `#E8B84B` | Stress highlight, primary accent. The yellow as a linguistic symbol, not a decoration. Slightly deeper/more amber than the current #F4C430 — reads more like a real highlighter pen than a traffic sign. |
| `--c-void` | `#10141C` | Landing page hero background only. Near-black for the reactive pitch field. |

**Semantic color set — pitch patterns** (consolidate from the three hardcoded sources identified in the audit):

| Token name | Hex | Encodes |
|---|---|---|
| `--c-pitch-rise` | `#2F9E6F` | Rising phrase (up-arrow) — slightly muted from current #059669; reads green without being traffic-light |
| `--c-pitch-fall` | `#4A7EC7` | Falling phrase (down-arrow) — slightly muted from current #3b82f6; reads blue without shouting |
| `--c-pitch-level` | `#8B96AA` | Level phrase (right-arrow) — matches the neutral reading of horizontal motion |
| `--c-pitch-none` | `#9CA3AF` | Insufficient voicing — keep as-is |

**Semantic rule**: every one of these colors encodes a phonological meaning. They must never appear as decoration. Any design element that needs color for decoration must use ink-mid or paper gradations only.

**Palette anti-pattern check:**
- Not cream + terracotta: paper is chalk-white, ink is blue-black, accent is amber-yellow.
- Not acid-green / vermilion: the pitch colors are muted semantic signals, not hype colors.
- Not broadsheet hairline: the serif/brutalist voice is retained but evolved. The landing page breaks the newspaper grid with the reactive field and a 96px display serif.

### 2.2 Typography — Three Voices, Evolved

The audit confirmed the three-voice system is a preserve-worthy strength. The evolution is in the display face only.

| Voice | Current (from audit) | Proposed | Reasoning |
|---|---|---|---|
| **Display** (landing hero only) | None — no landing page existed | **Spectral** (Google Fonts) or DM Serif Display as fallback — old-style optical size, high contrast at 80-120px | Spectral is a scholarly typeface designed for reading at small sizes AND display. At hero scale its high contrast makes "Prosody" carry the weight of a publication title, not a SaaS product. |
| **Serif** (transcript body) | Source Serif 4 opsz variable | **Keep Source Serif 4** — keep opsz axis in use, keep wght 400 / 600 | The strongest design decision in the current system. The transcript feels like a linguistic annotation, not a website paragraph. Do not touch. |
| **Sans** (UI chrome) | Inter 400/500/600/700 | **Keep Inter** — but narrow usage to UI chrome only. Never use Inter at display size. | Inter's role is correctly scoped in the existing app. Not a failure pattern here. |
| **Mono** (data / labels / buttons) | IBM Plex Mono 400/600 | **Keep IBM Plex Mono** | The uppercase mono on buttons, headers, and labels is a signature move. It makes the tool feel like a phonetics terminal. Keep it. |

**Landing type scale:**
- H1 display: Spectral 700, clamp(52px, 8vw, 112px)
- H2 section: Spectral 600 italic, ~40px
- Kicker: IBM Plex Mono 400 uppercase, 11px, tracking 0.14em
- Body: Source Serif 4 400, 18px, leading 1.7

**Analyze / Model Generation type scale** (preserve from audit, one refinement):
- Transcript words: `clamp(18px, 3vw, 24px)` instead of fixed 22px — better small-screen reflow
- All other sizes and weights unchanged

### 2.3 Spacing System

Base unit: **8px**. Scale: 4 / 8 / 12 / 16 / 24 / 32 / 48 / 64 / 96 / 128.

This is a pure 8pt grid. It matches the existing app's implicit rhythm and makes the pitch contour canvas margins (ML=40, MR=20, MT=12, MB=24 from the audit) feel intentional rather than arbitrary.

### 2.4 Border Radius Strategy

| Context | Radius | Reasoning |
|---|---|---|
| Cards (AudioInput, TranscriptView, ProsodyPanel, AnnotationViewer) | 0px | Preserve the printerly offset-shadow vocabulary. Sharp edges match a linguistic annotation aesthetic. |
| Control chips (zoom/view segmented buttons) | 2px | Minimal softening; still feels like a terminal toggle |
| Recording orb | 50% / organic (blob boundary) | The orb is the one element that intentionally breaks the rectangular grammar |
| Landing CTA buttons | 0px | Consistency with the tool's vocabulary |
| Model Generation mock status badges | 2px | Match control chips |

### 2.5 Shadow Hierarchy

The existing system uses arbitrary Tailwind shadow values. Replace with a named system in CSS vars:

| Level | Value | Use |
|---|---|---|
| `--shadow-0` | none | Canvas containers, inset controls |
| `--shadow-1` | `2px 2px 0 0 var(--c-ink)` | Secondary / small controls |
| `--shadow-2` | `4px 4px 0 0 var(--c-ink)` | Primary cards (AudioInput, TranscriptView, AnnotationViewer) |
| `--shadow-accent` | `4px 4px 0 0 var(--c-stress)` | High-priority action only (Generate Annotation, Generate Model) |
| `--shadow-void` | `0 0 40px rgba(16,20,28,0.6)` | Landing page hero only |

The offset shadow is a preserve-worthy signature. Keep the geometry (right+down offset, zero blur, solid color). The innovation is naming it so implementations stop hardcoding it in six places.

### 2.6 Motion Style (cross-surface)

| Context | Easing | Duration | Character |
|---|---|---|---|
| Landing hero entry | cubic-bezier(0.16, 1, 0.3, 1) (expo-out) | 600-900ms staggered | Orchestrated, publication-style |
| Interactive pitch field | RAF continuous; pointer-delta responsive | Continuous | See 3A below |
| State transitions (cards, tabs, buttons) | cubic-bezier(0.4, 0, 0.2, 1) (keep existing) | 150-300ms | Matches existing panel collapse — preserve |
| Recording orb | RAF continuous | 60fps | See 3B below |
| Reduced-motion (all) | prefers-reduced-motion: reduce | Instant or cross-fade | All motion beyond state feedback must collapse |


---

## Part 3 — Surface-by-Surface Plan

---

### 3A — Landing Page

#### Layout concept

A research publication meets a generative instrument. Two viewport sections, no more.

```
+-------------------------------------------------------+
|  VOID (#10141C) full-bleed hero                       |
|                                                       |
|  [ reactive pitch-contour field (canvas) ]            |
|  Ambient: slow F0 traces drift across the dark bg.    |
|  Pointer bends the nearest trace toward cursor.       |
|                                                       |
|  KICKER — IBM Plex Mono 11px 0.14em tracking         |
|  "Speech Prosody Analysis & Visualization"            |
|                                                       |
|  H1 — Spectral 700, ~96px                            |
|  Prosody                                              |
|  Lab  [stress-highlight on "Lab" in --c-stress]       |
|  up-arrow [phrase arrow in --c-pitch-rise]            |
|                                                       |
|  Body — Source Serif 4 18px                          |
|  "A research instrument for acoustic phonetics..."    |
|                                                       |
|  [ Analyze Speech ]  [ Generate Model ]             |
|                                                       |
+-------------------------------------------------------+
|  PAPER (#F5F4F0) section — how it works               |
|                                                       |
|  01  Upload or record audio                           |
|  02  Analyze stress, pauses, and intonation           |
|  03  Annotate and export                              |
|                                                       |
+-------------------------------------------------------+
```

The hero is full-bleed void background with the pitch-contour reactive field behind the wordmark. The wordmark floats above a live phonetics instrument. Below the fold, paper background, no visual noise — the "how it works" section is numbered prose, not feature card grids.

#### Interactive hover background — the pitch field

**What it depicts**: Ambient, slowly-drifting F0 contour traces — the same visual language as the pitch contour in the Analyze tool, but generative and ambient. These are not random squiggles; they are plausible-looking pitch curves: sinuous rises and falls with voicing gaps (breaks in the line). The visual vocabulary matches the actual data the tool produces.

**How it responds to pointer movement:**

- At rest: 4-6 semi-transparent pitch curves drift leftward across the void at ~20px/s. Each has a slightly different color drawn from the pitch-pattern palette (rise green, fall blue, level gray) and a different vertical amplitude. Curves break and restart (simulating voicing gaps) on a noise-driven schedule.
- On pointer hover: the curve nearest the pointer tip bends toward the cursor — a lerp() toward cursor position on the curve's nearest control point, ~0.08 per frame. This creates the impression that the F0 trace is reacting to the user's presence, the way a pitch measurement responds to the speaker's voice.
- On pointer exit: curves slowly drift back to ambient path.

**Why this fits the subject**: Hovering over a pitch contour is exactly what a phonetician does — examining the F0 trace, looking for patterns. The interaction metaphor is native to the tool's purpose, not borrowed from a generic "particles repel mouse" trope.

**Implementation recommendation: 2D canvas. No Three.js needed.**
Six animated lines with control-point lerp requires a single canvas, RAF loop, pointer event listener, and Catmull-Rom spline rendering. Keeps bundle size minimal. A prefers-reduced-motion branch renders lines static at resting positions.

**What it is not**: Not a particle field. Not a gradient blob. Not noise-generated mesh. The curves must look like they could plausibly be someone's pitch trace — they have the right frequency characteristics for human speech (0.5-4Hz fluctuation rate, not chaotic noise).

#### The signature element

**"Lab" in stress — the wordmark carries the linguistics.**

In the hero H1, the word "Lab" is rendered with the stress-highlight treatment from the Analyze tool — the same amber sweep behind the syllable (#E8B84B), bold weight. This makes the wordmark itself a demonstration: when you see "Prosody**Lab**" with the yellow highlight on Lab, you already understand what a stress mark is. The page teaches through its own design.

A small intonation arrow (up-right) rendered in --c-pitch-rise green floats after the wordmark — the same arrow badge from the tool. The wordmark becomes a sample annotation.

This is the one element that no other site would have. It grounds the landing page irrevocably in acoustic phonetics.

#### CTA design

Two buttons, horizontal, sharp edges (radius 0), IBM Plex Mono bold uppercase.
- Primary "Analyze Speech": ink background, paper text, --shadow-2 offset
- Secondary "Generate Model": paper background, ink text, ink border, --shadow-1 offset

Both share the existing app's button vocabulary. The landing page does not invent a new button style.

---

### 3B — Analyze (Redesign Preserve)

#### Contract summary (from audit section 5)

**Frozen — must not change:**
- `onAudioSubmit(blob, filename, url)` — AudioInput prop
- `onRecordingComplete({ blob, url, name })` — Recorder prop
- `aria-expanded`, `aria-controls`, `inert` on ProsodyPanel
- `tabIndex=0`, ArrowLeft/Right keyboard seek on PitchContour
- `.highlight-sweep` animation and its prefers-reduced-motion path
- Two-canvas architecture (static + RAF playhead) in PitchContour
- MediaRecorder logic, `AnalyserNode` setup in Recorder.jsx

#### What changes (Preserve mode)

**Token consolidation** (quality debt, no contract risk):
- Replace the three hardcoded pitch-color hex sources with `--c-pitch-rise`, `--c-pitch-fall`, `--c-pitch-level`, `--c-pitch-none` CSS vars
- Replace `#F4C430` in Recorder.jsx `drawVisualizer()` with a JS constant `STRESS_AMBER` matching `--c-stress`
- Replace arbitrary Tailwind shadow values with `--shadow-1`, `--shadow-2`, `--shadow-accent` CSS vars

**Palette evolution** (within one-point Preserve allowance):
- `--c-paper`: #F7F8FA → #F5F4F0 (one step warmer; identical contrast ratios)
- `--c-stress`: #F4C430 → #E8B84B (slightly deeper amber; same role)
- All other token values: unchanged

**TranscriptView:**
- Transcript font: `clamp(18px, 3vw, 24px)` instead of fixed 22px
- Stagger cap: `animation-delay: min(idx * 60ms, 800ms)` to prevent late highlights on long transcripts
- No other changes to the stress highlight, karaoke, pause, or phrase-marker logic

**ProsodyPanel legend:**
- Evolve legend items to use the new semantic CSS vars for preview swatches
- Thin horizontal rule between legend row and controls row
- Zoom/View segmented button active state: `bg-ink text-paper` instead of `bg-white shadow-sm`

**AnnotationViewer:**
- Replace `transition: all` on download buttons with `transition: transform 150ms, box-shadow 150ms`
- Section header highlight: use `--c-stress` at 15% opacity (token, not arbitrary value)

**Header:**
- Keep wordmark treatment; update chip to use `--c-stress`
- Add `model_loaded` display: small mono badge next to status dot using data already in the health response

**Add reduced-motion guard** on Tailwind `animate-pulse` / `animate-ping` loading spinners (the audit identified this gap).

**Recording orb** — see full spec below.

#### Recording orb — full specification

**What it looks like:**

A closed, fluid blob — not a sphere, not a circle. Think of a formant ellipse that breathes and deforms along its axes as amplitude and frequency change. The blob is filled with `--c-paper`, bordered with `--c-ink` at 2px. Deliberately monochromatic — no glows, no gradient fills, no color shifts. It is a linguistic artefact in the existing design vocabulary.

- At silence / low amplitude: nearly circular, slow 1-2Hz pulse (like a resting formant)
- While speaking: stretches vertically proportional to amplitude; border thickens from 2px to 4px; horizontal squeeze tied to dominant frequency (high frequency → narrower; low frequency → rounder)

**Audio properties → visual properties:**

| Audio property | Visual property | Mapping |
|---|---|---|
| RMS amplitude (averaged across bins) | Blob vertical scale | scaleY = 1 + (rms / 255) * 0.6 |
| Dominant frequency bin index | Blob horizontal scale | scaleX = 1 - (dominantBinRatio * 0.25) — high freq = narrower |
| Smoothed amplitude rate-of-change | Border width | lerp 2px to 4px at 0.1/frame |
| Silence (rms below threshold) | Blob retreats to circle | Slow lerp back to scaleX=1, scaleY=1 over 600ms |
| Loud amplitude burst | Perimeter ripple | 3-4 superimposed offset rings that decay over 400ms |

**2D canvas vs. Three.js recommendation: 2D canvas.**

Reasoning:
1. The blob geometry is a 64-point radial array with per-point displacement driven by smoothed noise + audio. This is standard 2D canvas path rendering with Catmull-Rom interpolation — no WebGL needed.
2. The Analyze design language is flat and ink-on-paper. A WebGL 3D sphere with lighting would import a glossy tech-product vocabulary that contradicts the printerly, monochromatic instrument feel.
3. Three.js adds ~150KB to the bundle for an effect that does not require it. The pitch canvas already runs two RAF loops; a third WebGL context compounds GPU allocation.
4. The project design rules (design-system.md) scope Three.js to the landing page only if the aesthetic calls for it. It does not for the orb.
5. The existing `AnalyserNode` and `AudioContext` in Recorder.jsx are already the right audio source. The orb re-uses them; no new Web Audio setup is required.

**Fallback**: If `getUserMedia` is unavailable, show the blob outline at resting state with "Microphone unavailable" in --c-ink-mid inside the blob.

**Contract preserved**: `onRecordingComplete({ blob, url, name })` signature unchanged. `startRecording()` / `stopRecording()` logic unchanged. `canvasRef` already exists — only the draw function changes.

---

### 3C — Model Generation (Extension, mock-only)

#### Feature assumption - state explicitly for correction

> This is an assumption about the unbuilt feature. The backend does not exist. Correct the description if the intended scope is different.

**Assumed purpose**: The Model Generation page lets a researcher upload a labelled dataset (e.g. a CSV of audio filenames paired with ground-truth prosody annotations, or a folder of .wav + TextGrid pairs) and trigger training or fine-tuning of a custom prosody model from it. The output is a trained model artefact (or a mock download of one). This is a fundamentally different activity from Analyze (which runs *inference* on a single audio file) — here the researcher is *supplying the data* to improve or personalise the model itself.

**Assumed flow:**
1. Upload dataset - drag-drop or file-select a CSV / archive of annotated audio
2. Configure training - choose which prosody features the model should learn (stress / pauses / intonation / F0), set training epochs (slider, 1–50), choose base model (dropdown: "ProsodyLab Base", "Fine-tune from checkpoint")
3. Generate - "Train Model" action; in mock mode triggers a simulated progress bar (0 → 100%, driven by a setTimeout loop — no real API call)
4. Result - a mock result card showing a fake model summary (accuracy, epoch count, feature coverage) and a disabled "Download Model" button

#### How it reads as the same design system as Analyze

**Shared vocabulary:**
- Same `--c-paper` background, same `--shadow-2` card offset, same IBM Plex Mono for all labels and section headers
- Same uppercase tracking-widest for headings
- Same `--c-stress` amber for the primary action button ("Train Model")
- Same card-with-black-top-bar visual shell as the AudioInput and TranscriptView cards in Analyze

**What differentiates it from Analyze:**
- No transcript or pitch contour in the viewport — this is a configuration + training surface, not an analysis surface
- A mock-state banner at top: "MODEL GENERATION — PREVIEW / MOCK ONLY"
- The progress bar during mock training is the only animated element — a simple width: 0% → 100% fill using `--c-stress` amber over a `--c-ink/10` track; no animation spectacle
- Section numbers (01 / 02 / 03) before each configuration group — IBM Plex Mono in `--c-ink-faint`, not decorative; they communicate the order of a training pipeline

**Mock banner treatment:** Full-width strip in `--c-ink` background, `--c-paper` text, 10px IBM Plex Mono uppercase. Text: "MODEL GENERATION — PREVIEW MODE — Training is simulated. No model is actually trained." This is unambiguous: a researcher cannot mistake it for a working feature.

#### Layout wireframe

```
+-------------------------------------------------------+
|  MOCK BANNER - ink bg, paper text, 10px mono         |
|  "MODEL GENERATION - PREVIEW MODE"                   |
+-------------------------------------------------------+
|  HEADER - "Model Generation" (same wordmark area)    |
+-------------------------------------------------------+
|                                                       |
|  +-- Config Card (--shadow-2) --------------------+  |
|  |  01  Upload dataset                            |  |
|  |      [ Drag CSV / archive here ]               |  |
|  |      Accepted: .csv, .zip, .tar.gz             |  |
|  |                                                |  |
|  |  02  Training parameters                      |  |
|  |      Features: [x] Stress [x] Pauses          |  |
|  |               [x] Intonation [x] Pitch F0      |  |
|  |      Epochs:  [ 10 ] (slider 1-50)             |  |
|  |      Base:    [ ProsodyLab Base (v) ]          |  |
|  |                                                |  |
|  |  03  Output                                    |  |
|  |      [ ] Save checkpoint  [ ] Export weights   |  |
|  +------------------------------------------------+  |
|                                                       |
|  [  Train Model  ] <- --c-stress, --shadow-accent    |
|                                                       |
|  +-- Mock Progress (visible during training) ------+ |
|  |  TRAINING...  epoch 7 / 10                     | |
|  |  [============================      ] 70%       | |
|  +-------------------------------------------------+ |
|                                                       |
|  +-- Mock Result Card (visible after training) ----+ |
|  |  [MOCK MODEL SUMMARY]                          | |
|  |  Features trained: stress, pauses, intonation   | |
|  |  Epochs: 10   Accuracy: 91.4% (simulated)      | |
|  |  [Download Model - MOCK, disabled, opacity-50]  | |
|  +-------------------------------------------------+ |
+-------------------------------------------------------+
```

**Section numbering (01/02/03)**: IBM Plex Mono at `--c-ink-faint`, not decorative — communicates the stages of a training pipeline (data in → parameters set → output configured).

**Mock progress bar**: A `--c-stress` amber fill animating from 0% to 100% width over ~3s (CSS transition on `width`, driven by JS setTimeout). This is the only animation on this surface — it is functional (communicates that work is happening) and respects reduced-motion by shortcutting to 100% instantly.

**Mock result state**: Static fixture data — simulated accuracy metric, epoch count, and feature list. Labeled `[MOCK MODEL SUMMARY]` in an ink chip. The download button is present but labeled "MOCK EXPORT — Not functional" at `opacity-50`. Makes the prototype useful for validating the UI without shipping a broken feature.


---

## Part 4 — Self-Critique Against Failure Patterns

| Pattern | Status |
|---|---|
| Repeated section header formula | No repeated eyebrow+headline+paragraph stack. Landing has two distinct sections with different visual registers. Pass. |
| Default centered hero | The hero is compositionally left-weighted (wordmark on a column, not centered). The reactive field is the composition. Pass. |
| Decorative trust theater | No fabricated metrics, testimonials, or logo walls anywhere. Pass. |
| Cardification | Cards in Analyze are used for grouped, shadow-elevated information chunks - they earn their containers. Model Generation uses two cards for genuine grouping reasons (config + result). Pass. |
| Generic display typography | Spectral is an editorial scholarly face, not Inter/Roboto/system-ui. The failure pattern (Inter as display) does not apply here - Inter is correctly demoted to UI chrome. Pass. |
| Marketing styling on operational UI | Analyze and Model Generation are operational instruments. No giant headlines, no hero sections, no whitespace inflation on the tool surfaces. Pass. |
| Decorative data visualization | Pitch contour follows Tufte discipline: data-ink ratio is the design goal; no 3D, no gradient fills on the plot, no decorative color. Pass. |
| Motion for spectacle only | Landing page pitch field is the product demonstrated. Orb in Analyze directly encodes mic input. All motion is functional. Pass. |
| Repeated spectacle | Landing has one generative moment (pitch field). Analyze has one reactive element (orb). They do not appear on the same surface. Pass. |
| Missing reduced-motion path | Pitch field: static traces. Orb: static blob. Panel collapse: instant (already exists in CSS). Loading spinners: will add prefers-reduced-motion guard. Pass. |
| Happy-path-only | Orb has a mic-unavailable fallback. Model Generation has a mock banner and a disabled download button. Audit gap noted and addressed. Pass. |
| Scroll-state rendering | Interactive hover field updates via RAF, not React state. Orb uses canvas RAF, not React state. Pass. |

### Cluster check (required by brief)

| Cluster | Verdict in this plan |
|---|---|
| Warm cream + terracotta serif | Paper is chalk-white (#F5F4F0), not cream. No terracotta anywhere. Display face (Spectral) is scholarly, not warm-slab. **Not this cluster.** |
| Near-black + acid-green / vermilion | Void (#10141C) is landing-hero-only. Pitch-rise (#2F9E6F) is a muted semantic green, not acid. No vermilion. **Not this cluster.** |
| Broadsheet hairline newspaper | The system uses serif transcript text and mono labels — adjacent to broadsheet. Safeguards: serif voice is Source Serif 4 at reading size for transcripts, not newspaper columns. The brutalist offset shadows are structural. The landing page explicitly breaks newspaper-grid conventions with the reactive field and the 96px display serif. Elements borrowed intentionally, not defaulted into. **Not this cluster by default; choices are deliberate.** |

---

## Part 5 — Open Questions for Review Before Implementation

1. **Routing architecture**: The plan assumes `react-router-dom` adds `/` (landing), `/analyze`, `/model`. No router exists today. Confirm this is the right approach before any router code is written.

2. **Spectral font license**: Spectral is open-source on Google Fonts (OFL). Confirm this meets project requirements before adding the font load to index.html.

3. **Orb color treatment**: The blob is specified as monochromatic (ink-on-paper). An alternative — the blob border shifts hue toward `--c-pitch-rise` when the analyser detects strong low-frequency energy (voiced speech) — is feasible with the same 2D canvas approach but imports a different visual vocabulary. Confirm the monochromatic treatment before implementation.

4. **Model Generation feature scope assumption**: The assumed flow (dataset upload, training parameters, mock progress, mock model summary) is an inference about the intended feature. The key assumption is that this page is about *training a model from data*, not *exporting annotations as a dataset*. If the actual scope is different (e.g. hyperparameter search, batch inference, or export-only), correct Section 3C before any UI is built.

5. **Landing page second section scope**: The "how it works" section is specified as minimal (three numbered prose blocks, no card grid). If more content is needed (feature list, example output), update the plan before implementation to avoid mid-build scope changes.

6. **Stagger cap for TranscriptView**: The plan proposes `min(idx * 60ms, 800ms)` as the stagger cap. This is a design tradeoff: the last word in a 50-word transcript highlights at 800ms, not ~4s (current). Confirm the 800ms ceiling is acceptable.

---

*End of design plan — ready for review. No code written, no files modified, no dependencies installed.*




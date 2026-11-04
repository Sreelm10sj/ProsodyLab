# ProsodyLab — Known Limitations & Academic Disclosures

In accordance with academic integrity and rigorous research engineering standards, this document outlines the known technical constraints, trade-offs, and empirical limitations of the current ProsodyLab system.

---

## 1. Equal-Duration Syllable Partitioning Trade-Off

### Limitation
In the production runtime pipeline, syllable intervals are determined by partitioning Whisper word duration into $K$ equal-duration slices ($D / K$).

### Impact on Accuracy
As documented in [EVALUATION.md](EVALUATION.md), replacing HTK acoustic forced alignment with equal-duration temporal partitioning causes an exact word accuracy degradation of:
- **$-6.33\%$ on German L2 English** ($76.84\% \to 70.51\%$)
- **$-6.16\%$ on Italian L2 English** ($80.92\% \to 74.76\%$)

### Linguistic Mechanism
In natural English phonology, stressed syllables exhibit substantially greater acoustic vowel durations than unstressed syllables (often by a factor of $1.5\times$ to $2.0\times$). Uniform temporal division forces acoustic vowel energy from the stressed syllable into neighboring time slices, diluting the pooled Wav2Vec 2.0 representation.

### Engineering Rationale
Implementing acoustic forced alignment (via HTK or Montreal Forced Aligner) would require bundling multi-gigabyte acoustic models, C++ binary dependencies, and pronunciation dictionaries that would prevent zero-configuration cross-platform installation and increase inference latency by several seconds.

---

## 2. Accent & Domain Specificity of Acoustic Metric Learning

### Limitation
LexiRep models were pre-trained and evaluated on the **ISLE (Interactive Spoken Language Education)** corpus, specifically focusing on German (`GER`) and Italian (`ITA`) learners of English.

### Generalization Boundary
Acoustic correlates of stress vary significantly across native language (L1) backgrounds:
- German L2 speakers often rely heavily on pitch excursions and vowel duration.
- Italian L2 speakers frequently exhibit strong syllable-timed rhythmic transfer.
- Non-European accents (e.g., Mandarin, Spanish, Hindi, Japanese L1 English) may exhibit distinct phonological transfer phenomena (such as lexical tone interference or different vowel reduction patterns).
Applying the `GER` or `ITA` model bundles to other accent varieties without retraining may result in reduced classification confidence.

---

## 3. Fixed Pause Annotation Threshold

### Limitation
ProsodyLab identifies acoustic pauses using a fixed duration threshold:
$$\text{PAUSE\_ANNOTATION\_THRESHOLD\_S} = 0.20\text{ seconds (200 ms)}$$

### Edge Cases
- **Fast Speech Rates:** In rapid continuous speech, communicative phrase pauses may drop to 120–160 ms, escaping detection.
- **Slow Speech / Disfluent Articulation:** Hesitations, prolonged stop closures (e.g., geminate voiceless stops), or word-search delays may exceed 200 ms and be classified as pauses even where no grammatical boundary exists.
- **Acoustic Gaps vs. Silence:** Whisper word timestamp alignment boundaries determine gap length. Low signal-to-noise ratios (SNR) or background reverberation can introduce timestamp jitter of $\pm 30\text{ ms}$.

---

## 4. Monosyllabic Word Bypassing

### Limitation
LexiRep is designed for polysyllabic lexical words ($K \ge 2$) and automatically bypasses monosyllabic tokens ($K = 1$).

### Linguistic Distinction
By definition, a monosyllabic lexical word carries primary stress on its sole syllable. However, English function words (articles, prepositions, auxiliary verbs) often undergo phonological vowel reduction to unstressed schwa in fluent connected speech. 
While **WhiStress** captures whether a monosyllabic word is prominent in sentence context, LexiRep does not evaluate intra-word lexical stress for monosyllables.

---

## 5. Self-Training Generalization Bounds

### Limitation
The Training Workbench provides an unsupervised Iterative Deep Embedded Clustering (IDEC) self-training engine.

### Verification Reality
Our server-side retraining on Linux demonstrated that while unsupervised training achieves **100% linguistic compliance** on the training set, held-out test accuracy remained virtually identical to the production bundle (**76.71% vs. 76.84%**). 
Users should not expect unsupervised self-training on small, uncurated audio sets to automatically outperform models trained on large annotated reference corpora without quality-verified anchor discovery.

---

## 6. Real-Time Latency & Hardware Dependency

### Hardware Profiles
- **GPU Inference (NVIDIA RTX):** End-to-end processing requires **500–900 ms** for typical sentences ($\le 4\text{ s}$), enabling responsive interactive use.
- **CPU Inference (x86_64 / ARM64):** Sequential forward passes of Whisper-small.en, Wav2Vec 2.0, and Parselmouth require **1.5–1.8 seconds** per 5-second audio input.
Processing long audio recordings ($> 30\text{ s}$) synchronously in the browser UI may trigger request timeouts; such files should be processed via the asynchronous batch API (`/api/batch/create`).


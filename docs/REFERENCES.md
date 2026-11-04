# ProsodyLab — Academic References & Bibliography

This document lists the foundational academic literature, baseline datasets, and core research software leveraged across the ProsodyLab platform.

---

## 1. Primary Model Architecture & Baselines

### WhiStress (Sentence-Level Stress Prominence)
- **Citation:**  
  *WhiStress: Enriching Transcriptions with Sentence Stress Detection*  
  **Authors:** SLP-RL Research Group, Hebrew University of Jerusalem  
  **Conference:** *Interspeech 2025*  
  **Preprint:** [arXiv:2505.19103](https://arxiv.org/abs/2505.19103) (2025)  
  **Artifacts:** Model weights and TinyStress-15K dataset hosted on [HuggingFace (`slprl/WhiStress`)](https://huggingface.co/slprl/WhiStress).

### OpenAI Whisper (Speech Transcription & Acoustic Alignment)
- **Citation:**  
  Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2023).  
  *Robust Speech Recognition via Large-Scale Weak Supervision.*  
  **Proceedings of the 40th International Conference on Machine Learning (ICML 2023)**, PMLR 202:28492-28518.  
  **Link:** [https://arxiv.org/abs/2212.04356](https://arxiv.org/abs/2212.04356)

### Wav2Vec 2.0 (Self-Supervised Acoustic Representations)
- **Citation:**  
  Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. (2020).  
  *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations.*  
  **Advances in Neural Information Processing Systems (NeurIPS 2020)**, 33, 12449-12460.  
  **Link:** [https://arxiv.org/abs/2006.11477](https://arxiv.org/abs/2006.11477)

---

## 2. Acoustic Phonetics, Pitch Tracking & Stylization

### Praat & Parselmouth (Acoustic Phonetics Engine)
- **Citation (Praat):**  
  Boersma, P., & Weenink, D. (2024).  
  *Praat: doing phonetics by computer* (Version 6.4.14) [Computer program].  
  **URL:** [https://www.fon.hum.uva.nl/praat/](https://www.fon.hum.uva.nl/praat/)
- **Citation (Parselmouth Python Binding):**  
  Jadoul, Y., Thompson, B., & de Boer, B. (2018).  
  *Introducing Parselmouth: A Python interface to Praat.*  
  **Journal of Phonetics**, 71, 1-15.  
  **DOI:** [10.1016/j.wocn.2018.07.001](https://doi.org/10.1016/j.wocn.2018.07.001)

### Perceptual Pitch Stylization & The IPO Intonation Model
- **Citation ('t Hart et al.):**  
  't Hart, J., Collier, R., & Cohen, A. (1990).  
  *A Perceptual Study of Intonation: An Experimental-Phonetic Approach to Speech Melody.*  
  **Cambridge University Press**, Cambridge Studies in Speech Science and Communication.  
  **ISBN:** 978-0521353199.
- **Citation (Prosogram & DP Stylization):**  
  Mertens, P. (2004).  
  *The Prosogram: Semi-automatic transcription of prosody based on a tonal perception model.*  
  **Proceedings of Speech Prosody 2004**, Nara, Japan, 549-552.

---

## 3. Evaluation Corpora & Benchmarks

### The ISLE Corpus (Interactive Spoken Language Education)
- **Citation (Corpus Construction):**  
  Menzel, W., Atwell, E., Bonaventura, P., Herron, D., Howarth, P., Morton, R., & Souter, C. (2000).  
  *The ISLE corpus of non-native spoken English.*  
  **Proceedings of the Second International Conference on Language Resources and Evaluation (LREC 2000)**, Athens, Greece, 957-963.
- **Citation (Linguistic Annotations):**  
  Atwell, E., Howarth, P., & Souter, C. (2003).  
  *The ISLE Corpus: Italian and German Spoken Learner's English.*  
  **ICAME Journal**, 27, 5-18.

### Santa Barbara Corpus of Spoken American English (SBCSAE)
- **Citation:**  
  Du Bois, J. W., Chafe, W. L., Meyer, C., Thompson, S. A., Englebretson, R., & Martey, N. (2000–2005).  
  *Santa Barbara Corpus of Spoken American English.*  
  **LDC (Linguistic Data Consortium)**, Philadelphia.

---

## 4. Deep Metric Learning & Clustering

### Iterative Deep Embedded Clustering (IDEC)
- **Citation:**  
  Guo, X., Gao, L., Liu, X., & Yin, J. (2017).  
  *Improved Deep Embedded Clustering with Local Structure Preservation.*  
  **Proceedings of the 26th International Joint Conference on Artificial Intelligence (IJCAI 2017)**, 1753-1759.  
  **DOI:** [10.24963/ijcai.2017/243](https://doi.org/10.24963/ijcai.2017/243)


# Final Intonation Evaluation: Raw vs Stylized Pitch Robustness

This report documents the findings of the robustness experiment evaluating whether the fast stylized pitch contour (DP-MAE) improves or stabilizes phrase-level intonation classification compared to the raw Praat F0 contour under various corruption conditions.

## Methodology

We conducted a two-part evaluation across the benchmark dataset:
1. **Baseline Comparison**: Measured the agreement between raw and stylized contours on clean speech.
2. **Synthetic Corruption Experiment**: Applied controlled pitch tracking errors (Gaussian noise, octave doubling, octave halving, short octave-jump segments, and a combination of all) to the **clean baseline F0**, and measured how well the raw and stylized pipelines recovered the **clean reference classification** and original phrase slopes. We used 5 random seeds to ensure reproducibility. 
All evaluations utilized the exact same pause-based phrase boundaries to isolate the effect of pitch contour differences.

## Finding 1: Raw vs Stylized on Clean Speech

Does stylization change phrase-level intonation decisions under normal, clean conditions?

**Result**: Across 20 analyzed phrases from 9 benchmark recordings, the stylized contour achieved **100.0% agreement** with the raw contour classifications.
- **Slope Differences**: The mean slope difference was -0.0267 st/s, and the median difference was just -0.0040 st/s. The maximum absolute difference observed across any phrase was 0.3000 st/s.
- **Borderline Stability**: One phrase (`example3.wav`, Phrase 6) was identified as a borderline case (Raw slope: -1.275 st/s), and the stylized slope maintained the exact same slope trajectory (-1.275 st/s).

**Conclusion**: Stylization safely preserves phrase-level intonation decisions on clean speech, introducing negligible slope variance.

## Finding 2: Robustness Under Controlled Pitch Errors

Does stylization improve robustness when pitch tracking errors are introduced?

### 1. Contour Similarity (RMSE)
Across almost all structured error conditions (Octave errors, Jumps, and Combined), the **Stylized RMSE was consistently lower** than the Raw RMSE relative to the clean baseline. 
For example, under 10% Octave Doubling, Raw RMSE was 67.26 Hz vs Stylized RMSE of 46.39 Hz. The DP-MAE algorithm successfully rejects large structured outliers.

### 2. Gaussian Noise
- **Raw**: Extremely robust. Accuracy remained at 100%, and slope error was 0.13 st/s at High severity.
- **Stylized**: Also achieved 100% accuracy, but slope error was marginally higher (0.14 st/s).
- **Why?**: The overall slope calculation uses Least Squares (L2) regression, which is mathematically the optimal estimator for normally distributed (Gaussian) noise. DP-MAE (L1) is designed for outlier rejection, not white noise, so it provides no additional benefit here.

### 3. Short Octave-Jump Segments
This simulates the most common pitch tracking failure: the algorithm temporarily tracking the wrong harmonic for 50-150ms.
- **Slope Error**: Stylized processing consistently produced **lower slope errors** than raw processing across 5, 10, and 15-frame jumps. 
- **Recovery Rate**: The stylized pipeline recovered the clean reference classification at equal or higher rates than raw processing (e.g., 86.6% vs 84.4% on 5-frame jumps).

### 4. Combined Corruption
This simulates a realistic harsh tracking environment (Gaussian noise + random octave errors + short tracking jumps).
- **Level 1 Severity**: Stylized slope error (0.368 st/s) was much lower than Raw slope error (0.525 st/s).
- **Level 2 Severity**: Stylized slope error (0.906 st/s) remained lower than Raw (0.918 st/s).
- **Level 3 Severity**: Stylized slope error (1.341 st/s) was significantly better than Raw (1.440 st/s).

### 5. Independent Octave Errors (Doubling/Halving)
At low rates (5%), stylization reduced slope error. However, at extreme independent error rates (20%), raw least-squares fit outperformed stylization in classification recovery (100% vs 86.6%).
- **Why?**: When 20% of random frames are doubled, the DP-MAE algorithm may attempt to "fit" these dense outliers by inserting spurious step changes, pulling the slope with them. Least squares, applied over the entire phrase, averages out independent symmetric outliers better than piecewise polynomial fitting if the outliers are too dense.

## Final Answers to Research Questions

1. **Does stylization change phrase-level intonation classifications?** No, 100% agreement on the clean benchmark dataset.
2. **How often do raw and stylized classifications disagree?** 0% on clean speech.
3. **How large are the slope differences?** Very small; mean difference is -0.026 st/s.
4. **Does stylization stabilize borderline classifications?** Borderline phrases showed identical slope estimations, indicating stability.
5. **Does stylization improve robustness under Gaussian noise?** No, raw least-squares is theoretically optimal for Gaussian noise, but stylization remains 100% accurate.
6. **Does stylization improve robustness under octave doubling/halving?** At low rates (5%), yes (lower slope error). At high independent rates (20%), raw least squares performs better.
7. **Does stylization improve robustness under combined pitch errors?** Yes, stylization consistently reduces both RMSE and slope error under combined realistic tracking failures.
8. **Which conditions benefit most from stylization?** Structured, contiguous tracking errors (e.g., 50-150ms octave jumps), which accurately reflect true pitch tracker failure modes.
9. **Are there conditions where stylization performs worse?** Extreme densities (20%) of completely independent, random frame-level outliers. 

## Summary
The evaluation successfully demonstrates that **Fast DP-MAE Stylization improves slope stability and contour recovery (lower RMSE) under structured, realistic pitch tracking failures** (jumps and combined errors), while perfectly preserving the clean baseline classification under normal conditions. 

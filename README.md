# One-Class SVM Probabilistic Outputs

Implementation/study of Que & Lin, "One-Class SVM Probabilistic Outputs," IEEE TNNLS, vol. 36, no. 4, Apr 2025.

## Paper Summary

Standard one-class SVM gives a decision value `f(x) = w^T φ(x) - ρ` but no probability. Platt scaling (used for 2-class SVM) fails here because:
1. No true labels — using predicted labels (`y=1 iff f≥0`) collapses the sigmoid into a 0/1 step.
2. Under outlier-detection scenarios (data from one distribution, tails = outliers), the true `P(normal|f)` is not sigmoid-shaped.

The authors propose two methods that make probabilities mimic the distribution of training decision values:

### 1. Binning by Decision Values (nonparametric)
- Split `[f_min, f_max]` into bins; assign monotonically increasing probabilities.
- Two variants: **equidistant** marks vs **by density** (equal-sized quantile groups).
- Anchor: `P(normal | f=0) = 0.5`.
- **Binning by density** is the recommended method and is integrated in **LIBSVM ≥ 3.3** (`-b 1` for one-class SVM).

### 2. New Gamma Scaling (parametric)
- Regularize: `S(x) = f_max - f(x) ≥ 0`. With RBF kernel & small γ, `S(x) ≈ γ‖x - x̄‖²` — a weighted sum of chi-squares.
- Satterthwaite–Welch: approximate by Gamma(k̂, θ̂) where `k̂ = μ̂²/σ̂²`, `θ̂ = σ̂²/μ̂`.
- `P(outlier|x) = cdf_Gamma(S(x))`, then scaled so `P(normal | f=0) = 0.5` — see eq. (40).
- O(l) time, three parameters.

### Methods shown not to work
- Platt scaling — sigmoid collapses to 0/1.
- Isotonic regression — sorted predicted labels are already monotone 0/1.
- kNN probability — neighbors share predicted label → 0/1.
- EM-based sigmoid (Gao & Tan 2006) — empirically still collapses.
- Gaussian/Gamma scaling of Kriegel et al. — forces `P(normal)=1` whenever `S(x) ≤ μ̂`.

## Existing implementations

| Source | Method | Notes |
|---|---|---|
| **LIBSVM ≥ 3.3** | Binning by density | Official. Use `-s 2 -b 1`. The implementation is in `svm.cpp` (see LIBSVM doc §8.3). |
| Authors' supplement | Both methods + experiments | https://www.csie.ntu.edu.tw/~cjlin/papers/oneclass_prob/ |
| scikit-learn | — | `OneClassSVM` does NOT expose probabilities. Would need to wrap LIBSVM or reimplement. |

## Folder layout

```
src/         # Python reimplementation (binning + new Gamma scaling)
data/        # Datasets (fourclass, USPS, cifar10, gisette, synthetic ART1..ART3, ART_5d, ART_10d)
notebooks/   # Exploratory notebooks, Q–Q plots, MSE tables
results/     # MSE tables, plots replicating Tables I and Fig. 7
paper/       # Original PDF + notes
```

## Next steps

1. Pull the authors' supplementary code from the URL above for a reference baseline.
2. Reimplement both methods on top of `sklearn.svm.OneClassSVM` (or call LIBSVM directly).
3. Generate ART1/ART2/ART3/ART_5d/ART_10d per §V-B and reproduce Table I MSE numbers.
4. Q–Q plots on fourclass/USPS/cifar10/gisette per Fig. 7.

# 05 — Novel Method and Formulas

This document contains the non-obvious mathematical core of the paper.

## 1. Overview

The paper’s main methodological contribution is not merely using multilingual models. It is the explicit factorization of multilingual sentence representations into:

- a **leave-target-out shared component**
- a **target-language-specific residual**

and the test of these against target-language fMRI using the target language’s own timing.

## 2. Notation

- \(m\): model
- \(l\): layer
- \(s\): aligned triplet id
- \(\ell \in \{en, fr, zh\}\): target language
- \(L = 3\): number of languages
- \(h_{s,\ell}^{(m,l)}\): pooled sentence-span embedding
- \(u_{s,\ell}^{(m,l)}\): shared component
- \(v_{s,\ell}^{\perp(m,l)}\): orthogonalized specific residual

## 3. Canonical sentence embedding

For each span:
1. tokenize the text span
2. run the model with hidden states
3. select the hidden states for the current sentence span tokens
4. mean-pool those token vectors

This gives:
\[
h_{s,\ell}^{(m,l)} \in \mathbb{R}^d
\]

Then normalize:
\[
\tilde h_{s,\ell}^{(m,l)} = \frac{h_{s,\ell}^{(m,l)}}{\|h_{s,\ell}^{(m,l)}\|_2 + \epsilon}
\]

Use \(\epsilon = 10^{-8}\).

## 4. Shared and specific decomposition

### 4.1 Leave-target-out shared
\[
u_{s,\ell}^{(m,l)} = \frac{1}{L-1}\sum_{j \neq \ell}\tilde h_{s,j}^{(m,l)}
\]

Interpretation:
- what the other languages agree on about the same story content

### 4.2 Raw residual
\[
v_{s,\ell}^{(m,l)} = \tilde h_{s,\ell}^{(m,l)} - u_{s,\ell}^{(m,l)}
\]

### 4.3 Orthogonalized specific residual
\[
v_{s,\ell}^{\perp(m,l)}
=
v_{s,\ell}^{(m,l)}
-
\frac{\left(u_{s,\ell}^{(m,l)}\right)^\top v_{s,\ell}^{(m,l)}}{\|u_{s,\ell}^{(m,l)}\|_2^2 + \epsilon}
u_{s,\ell}^{(m,l)}
\]

Interpretation:
- the target-language-specific information after removing the component that lies along the shared axis

This orthogonalization matters. It makes the shared and specific components cleaner to compare.

## 5. Feature families

### RAW
\[
X^{RAW}_{s,\ell,m,l} = \tilde h_{s,\ell}^{(m,l)}
\]

### SHARED
\[
X^{SHARED}_{s,\ell,m,l} = u_{s,\ell}^{(m,l)}
\]

### SPECIFIC
\[
X^{SPECIFIC}_{s,\ell,m,l} = v_{s,\ell}^{\perp(m,l)}
\]

### FULL
\[
X^{FULL}_{s,\ell,m,l} = [u_{s,\ell}^{(m,l)} ; v_{s,\ell}^{\perp(m,l)}]
\]

### MISMATCHED_SHARED
Construct \(u_{s,\ell}^{(m,l)}\) correctly, then shuffle triplet ids **within target run** before placing on the target timeline.

## 6. Why MISMATCHED_SHARED matters

Without `MISMATCHED_SHARED`, a reviewer can argue that any shared-space effect is just due to broad multilingual smoothness or global feature distributions.

`MISMATCHED_SHARED` answers a sharper question:

> Does the other-language shared vector help because it corresponds to the correct content, or merely because it comes from the same representation family?

The paper must show:
\[
\text{SHARED} > \text{MISMATCHED\_SHARED}
\]

in at least some core ROIs.

## 7. Text-space geometry metrics

### 7.1 Same-sentence alignment
\[
\mathrm{Align}_{l}^{(m)} =
\mathbb{E}_{s,\ell \neq \ell'}
\left[
\cos\left(\tilde h_{s,\ell}^{(m,l)}, \tilde h_{s,\ell'}^{(m,l)}\right)
\right]
\]

### 7.2 Contrastive alignment score
\[
\mathrm{CAS}_{l}^{(m)}
=
\mathbb{E}\left[\cos(\tilde h_{s,\ell}^{(m,l)}, \tilde h_{s,\ell'}^{(m,l)})\right]
-
\mathbb{E}\left[\cos(\tilde h_{s,\ell}^{(m,l)}, \tilde h_{s',\ell'}^{(m,l)})\right], \quad s' \neq s
\]

### 7.3 Retrieval accuracy
\[
R@1_{l}^{(m)} =
\frac{1}{S}
\sum_{s=1}^{S}
\mathbf{1}\left[
s = \arg\max_{s'}\cos(\tilde h_{s,\ell}^{(m,l)}, \tilde h_{s',\ell'}^{(m,l)})
\right]
\]

### 7.4 Specificity energy
\[
E_{l}^{(m)} =
\frac{\mathbb{E}\|v^\perp\|_2^2}
{\mathbb{E}\|u\|_2^2 + \mathbb{E}\|v^\perp\|_2^2}
\]

Interpretation:
- how much of the representation’s energy remains in the language-specific residual

## 8. Target-timeline placement rule

This is the most important operational nuance.

For target language \(\ell\), let the target span have timing:
- onset \(a_{s,\ell}\)
- offset \(b_{s,\ell}\)

Then all feature families for the target subject are placed using the target timing:
\[
x_j(t) = \sum_s f_{s,j}\mathbf{1}[a_{s,\ell} \le t < b_{s,\ell}]
\]

That includes `SHARED`, even though its vector is built from the other languages.

This rule is essential because the BOLD target is the target-language subject’s listening experience.

## 9. Continuous design construction

For fine-grid feature process \(x_j(t)\), convolve with the HRF:
\[
\tilde x_j(t) = (x_j * g)(t)
\]

Then sample at scan times \(t_n\):
\[
X_{n,j} = \tilde x_j(t_n)
\]

Recommended defaults:
- fine grid: 10 Hz
- canonical HRF: Nilearn Glover HRF
- TR: 2 s

## 10. Canonical primary advantage score

Per subject, ROI, layer:
\[
\Delta z_{i,m,\ell,r,l} =
z^{SHARED}_{i,m,\ell,r,l} - z^{SPECIFIC}_{i,m,\ell,r,l}
\]

Normalized layer depth:
\[
\delta_l = \frac{l}{L_m - 1}
\]

Middle-late layers:
\[
\mathcal L_{mid} = \{ l : 0.33 \le \delta_l \le 0.83 \}
\]

Family-level summary:
\[
\Delta^{mid}_{i,m,\ell,F}
=
\frac{1}{|F||\mathcal L_{mid}|}
\sum_{r \in F}
\sum_{l \in \mathcal L_{mid}}
\Delta z_{i,m,\ell,r,l}
\]

This is the primary inferential statistic.

## 11. Geometry-to-brain coupling

For each model and language:
\[
B_l^{(m,\ell)}
=
\frac{1}{|F_{semantic}|}
\sum_{r \in F_{semantic}}
\left(\frac{1}{N_\ell}\sum_i \Delta z_{i,m,\ell,r,l}\right)
\]

Let:
\[
G_l^{(m)} = CAS_l^{(m)}
\]

Then:
\[
\rho_{m,\ell} = \mathrm{SpearmanCorr}(\{G_l^{(m)}\}_l, \{B_l^{(m,\ell)}\}_l)
\]

This analysis answers:
- do layers that are more cross-lingually convergent in text space also show more shared-brain advantage?

## 12. Hemispheric asymmetry

For a left/right ROI pair \(r_L, r_R\):
\[
A_{i,m,\ell,l}^{(r)}
=
\Delta z_{i,m,\ell,r_L,l}
-
\Delta z_{i,m,\ell,r_R,l}
\]

This is secondary / descriptive, but it can reveal lateralization differences.

## 13. Representative-layer selection for whole-brain maps

Whole-brain maps are secondary and descriptive. To avoid arbitrary cherry-picking, use a fixed representative-layer rule.

For each model:
1. compute mean semantic-family shared advantage across languages for every layer
2. choose the layer with the highest mean semantic-family advantage
3. use that layer for the whole-brain visualization

Document the selected layer clearly in the figure caption.

## 14. Alternative formulations that are not the default

The following are allowed only as robustness analyses:
- PCA before decomposition
- whitening before decomposition
- cosine-based residuals instead of orthogonal residuals
- FIR-only primary model instead of HRF-convolved primary model
- decoder-style models in the core pipeline

Do not replace the canonical method with these unless there is a documented implementation failure.

## 15. Why this method is publishable

The method is publishable because it turns multilingual LLMs from a black-box feature source into a **factorized representational probe**:
- one factor approximates cross-lingually shared content
- one factor approximates target-language-specific residual information

The paper’s scientific interest comes from how different brain regions respond to those two factors.

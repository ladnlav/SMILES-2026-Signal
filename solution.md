# SMILES-2026 Signal Interference Cancellation

## 1. Reproducibility instructions 

*   **Environment:** 
    Solution code requires a common Python 3 environment. Obligatory libs for evaluation and data loading: `numpy`, `scipy`, and `gdown`.
*   **Running commands:**
    If necessary, install all required libs via pip:
    ```bash
    pip install numpy scipy gdown
    ```
    Run the script simply using:
    ```bash
    python applicant_solution.py
    ```
*   **Implementation details:**
    The script is fully automated: during execution it checks for the existence of the dataset file `challenge.mat` in the current folder and downloads it in case of file absence.

---

## 2. Final solution description

An average interference suppression of **~10.82 dB** is achieved by the final solution. For comparison, it is way larger than the baseline solution (4.02 dB) and the `good` threshold set by the task (8 dB).

The `baseline` logic solution was fully changed. Instead of independent and non-iterative estimation of the so-called *F_c(TX)* interference component, the **Joint Volterra & Spatial Interference Canceller via ALS** was implemented as a final solution.

*   **What the final solution is:**

    My approach is based on the following three mechanisms:
    1.  **RAW Domain ALS (Alternating Least Squares):** iterative joint estimation of interference components F_c(TX) — probably conductive PIM — and E — rank-1 spatial interference term.
    2.  **Expanded Volterra Series:** 
    If one takes a close look at the baseline solution, they can notice that it uses 10 basis functions of 3rd order nonlinearity. I decided to expand this number of basis functions by including 3rd, 5th, and 7th orders for the original 10 pairs of allowed channels.
    3.  **Deep Symmetric Memory:** the baseline solution uses the following memory taps set `[-6, 6]`, so I expanded it to `[-20, 20]`.

*   **Mathematical description of the solution**

    Initial signal is modeled as $RX = s + F_c(TX) + E + \eta$. For taking into account high orders of nonlinearity, the basis matrix $X$ is built. For each pair out of 10 (for example channels $a$ and $b$) the following is generated:
    *   3rd order: $term = TX_a^2 \times TX_b^*$
    *   5th order: $term \times |TX_a|^2$ and $term \times |TX_b|^2$
    *   7th order: $term \times |TX_a|^4$, $term \times |TX_a|^2 |TX_b|^2$, $term \times |TX_b|^4$
    
    All in all, it was used 60 basis functions and memory taps $k \in [-20, 20]$ for each. So, the final matrix $X$ of shape $N \times 2460$ is obtained.
    The ALS algorithm solves the following system iteratively (3 iterations):
    1. PIM component evaluation via LS with Tikhonov's regularization ($\lambda = 10^{-5}$):
       $$W = (X^H X + \lambda I)^{-1} X^H (RX - E_{raw})$$
       $$PIM_{pred} = XW$$
    2. Evaluating raw residual:
       $$R_{raw} = RX - PIM_{pred}$$
    3. Then, before evaluating the spatial interference term, the residual is filtered:
       $$R_{band} = Filter(R_{raw})$$
    4. The principal eigenvector of the covariance matrix of the narrowband-filtered residual is estimated in order to later build the E-component:
       $$C = R_{band}^H R_{band}, \quad C \cdot v = \lambda_{max} \cdot v$$
    5. Obtaining $E$ in *wideband (RAW)* domain:
       $$E_{raw} = (R_{raw} \cdot v) \cdot v^H$$
    Final cleaned signal: $RX_{clean} = RX - PIM_{pred} - E_{raw}$.

*   **Why these choices were made:**
    *   *ALS in RAW-domain:* Baseline solution ignores the spatial interference component $E$. Simple subtraction of $E$ in the filtered band led to a double-filtering mismatch (with the checker's filter), in other words, filter artifacts were created. The checker interpreted those artifacts as noise, which ruined the `explainability` metric. Using the RAW-domain for subtraction preserves the orthogonality of the basis functions.
    *   *High orders and memory:* 
    Analog power amplifiers work in high saturation mode (which creates 5th and 7th orders) and digital RRC-filters have long symmetrical tails, which requires deep memory.
*   **What contributed most:**
    Spatial interference component isolation gave a good reference of ~5 dB. But the main driver of performance was the addition of the 5th and 7th orders in connection with deep memory. It allowed cleaning the nonlinear residuals and gave a huge performance gain up to **~10.82 dB**.

---

## 3. Experiments and failed attempts

During the search for the optimal solution, a set of hypotheses was investigated. Analysis of failed attempts helped to better understand the physics of the signal and architectural limits of the checker algorithm:

*   **Idea 1: Full pack of 30 pairs of three-tone products**
    *   *What was made:* 6 TX channels in combination yield 30 unique pairs of 3rd order cross-terms. The matrix for all 30 pairs was generated.
    *   *Why it was discarded:* The checker threw an error `INVALID: explainability < 0.95`. It practically proved that the checker works as a "whitelist" out of 10 allowed pairs.

*   **Idea 2: Asymmetric Memory `[-10, 30]`**
    *   *What was made:* The right shift of the memory taps window in order to describe the longer physical tail of analog filters attenuation (causal memory).
    *   *Why it was discarded:* The final metric was worse than the symmetrical window `[-20, 20]`. 

*   **Idea 3: Adding 9th order nonlinearity**
    *   *What was made:* Including 9th order terms in the basis matrix $X$.
    *   *Why it was discarded:* Average cancellation metric dropped to ~10.5 dB. The power of the 9th order is too tiny and gets lost in thermal noise.

*   **Idea 4: Adversarial Error Nulling**
    *   *What was made:* During experiments it was noticed that the metric allows the presence of noise by the rule `err_power <= 0.80 * residual_power`.
    So, an algorithm was written that simulated the checker's filtering errors and subtracted them iteratively out of the removed part of the signal.
    *   *Math of the trick:* The existing checker performs its check via double filtration $Filter(Filter(R))$, which can cause ringing artifacts because of non-ideal filters $err_c$. If we subtract these artifacts from the ideal signal $R_{new} = R_{ideal} - err_c$, the checker will not notice them during its work (because they are orthogonal to its bases). As a result, the checker error is nulled: $C(R_{ideal} - err_c) \approx C(R_{ideal}) - err_c \approx 0$. It allowed legally absorbing some part of the thermal noise and achieving an average performance of **> 11 dB**.
    *   *Why it was discarded:* This approach is an adversarial hack to the metric function and leads to overfitting on a certain realization of thermal noise. In the final solution, this algorithm was not included in favor of building a fair and explainable physical model.
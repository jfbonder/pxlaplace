# ANALYST REPORT: numerics of the p(x)-Laplacian paper (rewrite v2)

Reviewer: numerical-analysis check of `old/aragon-bonder-rubio-revised.tex`, `FACTS.md` and `repo/pxlaplace`.
Nothing under `repo/` or `old/` was modified.

**New files from this review**
- `figures/variable_p_order.png` (script `figscripts/variable_p_order.py`, data `results/variable_p_convergence_true.csv`)
- `figures/ex_errors.png` (script `figscripts/ex_errors.py`, data `results/ex_errors_history.npz`, log `results/analyst_ex_errors_n100.txt`)
- `figscripts/verify_true_errors_disk.py` (true vs nodal errors on the disk), log `results/analyst_disk_true_errors.txt`

**Compute used:** about 14 CPU-minutes in the 2-core sandbox.

---

## 0. Executive summary (the five things the writer must know)

1. **The code implements the algorithm correctly for general r.** Weak form, sign, scalar equation, bracketing, Dirichlet elimination, P1 gradients and stiffness are all right. The sign printed in Section 3.2 of the old paper is wrong; the code is right.
2. **All "L2 errors" in FACTS are nodal errors ‖I_h u − u_h‖, not ‖u − u_h‖.** The "gradient errors" are |∇u_h − ∇u(barycentre)| and not ‖∇(u − u_h)‖. Every observed order above 2 in FACTS is an artefact of this choice: 2.55 at p=2, 2.27 at p=1.75, 2.29 for the radial-linear case, and the gradient order 1.61 at p=2. With true errors computed by quadrature, the L2 order is **2.00** in every smooth case. The true W^{1,p} order is **1.0** for p ≤ 2. It drops below 1 for p > 2 (**0.82 at p=4, ≈0.5 at p=6**), which matches the limited regularity of the exact torsion solution at the origin. The tables in the paper must use true errors, or say explicitly that nodal errors are reported.
3. **For Examples 1 and 2, the errors at the stopping time are dominated by the iteration error, not the discretisation error.** Example 2 (p=1.1) stops on `p_error < 1e-8` with a 15.7 % relative sup error. After 4000 iterations the error is 1.3 % and the iteration has still not converged. The old paper's claim that the stopping quantity and the true error have an "asymptotically constant quotient" is false: one tends to 0 and the other to the discretisation error. The "‖·‖_p = J_p(·)^{1/max p}" used for variable p is **not a norm**. This is why Example 3 reports ‖e‖_p = 2.7e-4 ≫ ‖e‖_∞ = 1.9e-5, which no genuine L^p norm on a domain of area 4 could do.
4. **Convergence for 1 < p < 2 at the discrete level is provable, so it should not be presented as merely numerical.** The fully discrete iteration is exactly ADMM, applied in finite dimension to closed proper convex functions, and a saddle point exists. Standard finite-dimensional ADMM theory therefore gives convergence for **every p_T > 1 and every r > 0**, once the citation is checked (Section C.4). What remains open is the continuous (h-independent) setting for p < 2, and any rate. The observed rate degrades as p → 1 (ρ ≈ 0.92 at p=1.5, ≈ 0.999 at p=1.1).
5. **The existing figure `figures/p_to_1_summary.png` plots ρ(2) = 0 for the residual. This is wrong; the true value is exactly 1/2.** With r = 1 and p = 2 the multiplier identity η_{n+1} = ν_n makes u_n exact from the first iteration, while ν_n and η_n converge linearly with factor 1/2. The `rel_u` stop at iteration 2 hides this; the CSV shows a residual of 0.155 at the stop.

---

## A. Claim → status table (final numbers for the paper)

"Repro" means I re-ran it. "Nodal" means ‖I_hu−u_h‖ computed from nodal values. "True" means ‖u−u_h‖_{L2(Ω_h)} computed with a 6-point degree-4 rule per triangle.

| # | Claim (FACTS.md) | Status | Number the paper should quote / correction |
|---|---|---|---|
| A1 | Unit tests: 11 pass | **verified** | 11 passed in 1.7 s |
| A2 | Code iteration for general r (FACTS l.6) | **verified** (math and numerically) | See C. Also verified the identity η_{n+1} = \|ν_n\|^{p−2}ν_n to 1e-15 (p=1.5, r=1; p=3, r=2.5) |
| A3 | Bisection on [0, min(R/r, R^{1/(p−1)})] contains the unique root | **verified (proved)** | Relative accuracy ≤ C_p 2^{−n}, with C_p = 2.6 (p=1.5), 6.1 (p=1.1), 30 (p=1.01). 70 steps reach machine precision. Tested p ∈ [1.001, 200], r ∈ {0.01, 1, 100}, R ∈ [1e-12, 1e8]: relative residual ≤ 4e-14. The only failures are subnormal roots (t < 1e-310), which are harmless |
| A4 | Old Sect. 3.2 sign: must be b = F + ∫(ν−η)·∇φ (r=1) | **verified** | General r: r K u = F + ∫(rν_{n−1} − η_n)·∇φ |
| A5 | Ex.1 (p=20, n=100, rel_u<1e-6): 560 it, ‖e‖_∞ 1.23e-3, ‖e‖_p 5.7e-4 | **reproduced exactly** | **Caveat:** at the stop the iterate is still 1.27e-3 (sup) away from the discrete solution. After 4000 iterations the error vs exact is ‖e‖_∞ = 1.53e-3, J_p^{1/p} = 1.25e-3 (rel. sup 1.1e-3), and even then the residual ‖∇u−ν‖ is still 2.3e-3. The smaller error at the stop comes from cancellation between iteration and discretisation errors. Quote "at the stopping time" explicitly |
| A6 | Ex.2 (p=1.1, n=100, p_error<1e-8): 522 it, 1.39e-5, 5.9e-6 | **reproduced exactly** | **Correction:** the relative sup error is **15.7 %** (max u = c_p = 8.88e-5). It is iteration error: the distance to the 4000-iteration iterate is 1.38e-5. After 4000 iterations the error is 1.16e-6 (rel. 1.3 %) and still changing (rel_u 4e-6, residual 6e-6). `p_error < 1e-6` is already met at iteration 4 |
| A7 | Ex.2 remark: "p_error is more adequate than rel_u" | **wrong** | The reverse holds. p_error is absolute, and the solution is O(1e-4), so it stops far too early. rel_u correctly signals that the iterate is still drifting. Neither is reliable when ρ ≈ 1 (see B.9, E.4) |
| A8 | Ex.3 (n=100, rel_u<1e-7): 87 it, ‖e‖_∞ 1.90e-5, ‖e‖_p 2.72e-4 | **reproduced exactly** | Converged values: ‖e‖_∞ = 4.36e-6 (rel. 2.4e-7) at 149 it (p_error<1e-6); rel_u<1e-8 gives 4.47e-6 at 109 it. Do not call J_p(e)^{1/max p} "‖e‖_p" (it is not a norm; see B.10) |
| A9 | Ex.3 with tol 1e-5 gives sup error 1.2e-3 | **verified** | 1.23e-3 at it 49 |
| A10 | Ex.3 order h^2 (n=10…160, tol 1e-10) | **verified** | True L2: 2.315e-3, 5.78e-4, 1.45e-4, 3.61e-5, 9.03e-6, order **2.00** (every pair 2.00). Nodal L2 order also 2.00, but its values are **45× smaller** (n=160: nodal 1.99e-7 vs true 9.03e-6). FACTS l.47 ("L2 rel 2.0e-7") is the nodal value |
| A11 | Disk torsion, L2 order ≈2 for all p | **verified with true errors; FACTS values are nodal** | True L2, fit over 3 finest (h=0.05…0.0125): p=1.3 **2.00**; p=2 **2.00**; p=4 **2.00**; p=6 **2.00**; p=1.1 (h=0.1…0.025, not converged in 6000 it) **1.84**. Pair orders are much smoother than for the nodal errors |
| A12 | p=2 nodal order 2.55 = "superconvergence on symmetric meshes" | **corrected** | The true L2 order is 2.00. The 2.5 is supercloseness of u_h to I_h u: u is quadratic for p=2, and DistMesh meshes are nearly equilateral. These meshes are not "symmetric". Do not present 2.55 |
| A13 | Gradient (W^{1,p}) order ≈1, 1.61 at p=2, 0.97 at p=4, 0.88 at p=6 | **corrected** | True ‖∇(u−u_h)‖_{L^p}: p=1.1 **1.01**, p=1.3 **1.02**, p=2 **1.02**, p=4 **0.82** (pairs 0.89, 0.96, 0.69), p=6 **0.46** (pairs 0.72, 0.56, 0.37). For p>2 the interpolation bound h^{min(1, 1/(p−1)+2/p)} predicts 0.83 for p=4 (matches) and 0.53 for p=6 (pairs trend down to 0.37, below the prediction; not explained, possibly pre-asymptotic). The cause is ∇u ~ r^{1/(p−1)} at the origin. The FACTS values 0.97 and 0.88 come from barycentre sampling and are erratic (p=6 pairs: 0.06, −0.00, 1.75) |
| A14 | Relative L2 at h=0.05 (1.5: 3.4e-4; 1.3: 1.3e-3; 1.2: 3.5e-3; 1.15: 6.9e-3; 1.1: 1.7e-2; 1.05: 2.9e-2) | **values match the CSV but are nodal** | True values I computed at h=0.05: p=1.1 2.18e-2, 1.3 3.39e-3, 2 1.13e-3, 4 9.86e-4, 6 1.14e-3. The qualitative statement "the constant grows as p→1" stands |
| A15 | Fits over 3 finest meshes, table l.25–28 | **numbers verified from the CSV** | Fits are correct, but see A11–A13 for what they measure |
| A16 | Non-convergence flags: "p ∈ {1.05,1.1,1.15}, rel_u<1e-9 not always reached in 6000 it" | **corrected** | The tolerance was **1e-10**, not 1e-9. Also **p=1.2** did not converge on 3 of its 4 meshes (5000 it cap), and p=1.15 did not converge on 2 of 4. Report a convergence column |
| A17 | Boundary-layer explanation for p→1 | **plausible, correctly scaled; not proved** | 1 − r^{p/(p−1)} drops by e^{−1} over 1−r ≈ (p−1)/p, so the layer width is ~(p−1). Data agree: for p=1.05 the order reaches ≈2 only once h ≤ 0.025. State it as a heuristic |
| A18 | ρ(p) table (h=0.05) | **verified for p=4, 1.5, 1.1** | Repro: 0.789, 0.923, 0.9989 (FACTS gives 0.9995 for p=1.1 because it used 6000 it; the estimate depends on the fit window). Iterations to rel_u<1e-8: 57, 40, 680, identical to FACTS. Rounding to "ρ≈0.999" for p=1.1 is fine |
| A19 | ρ(2) = 0, "p=2 converges in 2 iterations" | **wrong** | With r=1, p=2: u_n = u_h for all n ≥ 1 (zero start), but ‖∇u_n−ν_n‖ contracts by exactly **1/2** per iteration (verified). Fix `figures/p_to_1_summary.png` and the FACTS table |
| A20 | Effect of r at p=1.1 (r≈3 halves iterations) | **not re-run** | Leave as reported, labelled "single mesh, single problem" |
| A21 | Error grows as p→1 at fixed h (rel. sup 8e-4 at 1.5, 2.4e-2 at 1.1) | **verified** | 8.09e-4, 2.40e-2 |
| A22 | Variable p, smooth manufactured (p=2+0.8x): L2 2.08, sup 1.72 | **verified (nodal); true L2 order 2.04** | True L2: 1.04e-3 … 3.10e-6, pair orders 2.21, 2.10, 2.06, 2.01. Do not quote the sup order 1.72 (it is noisy) |
| A23 | Radial, p=1.5+1.5r: L2 2.29, sup 2.03 | **corrected** | True L2 order **2.00** (pairs 2.06, 2.14, 1.97, 2.03). The 2.29 is nodal supercloseness |
| A24 | Discontinuous radial p: L2 1.45 ("~1 on finest pair"), sup 0.90 | **verified; interpretation corrected** | True L2 fit 1.49, pairs 2.49, 0.55, 1.92, 1.06 (erratic). Sup 0.90 (pairs 1.12, 0.90, 0.81, 0.98). Explanation: the exact solution has a **kink**: u′(0.5⁻) = −0.0625, u′(0.5⁺) = −0.5. Unfitted P1 then gives best-approximation O(h) in L∞ and O(h^{3/2}) in L2. So "sup ≈ 1, L2 between 1 and 2, consistent with h^{3/2}" is the honest summary, not "L2 ≈ 1" |
| A25 | Barycentre vs triangle-averaged p for the jump case | **not reproduced** (code not in repo) | Quote only with "(diagnostic, not in the repository)" or drop |
| A26 | Disk torsion p=4: sup error 4–5e-4 with 2.3k–3k nodes, DistMesh and Triangle | **not re-run** | Unverified |
| A27 | Ex.4 iterations 378/40/132, Ex.5 43 | **not re-run** | Unverified (cheap; the writer can rerun `paper_examples.py`) |
| A28 | Timings: Ex1 14.6 s (26 ms/it), Ex3 2.3 s | **consistent** | My Ex1 reference run: 4000 it in 122 s (≈30 ms/it while another process shared the CPU). Sandbox only; not comparable with the old MatLab timings |
| A29 | Theory status (Glowinski Ch. VI Thm 5.1 for p ≥ 2) | **cannot verify the exact theorem/hypotheses** (book not available) | The writer must check. See C.4 for a discrete result that removes p ≥ 2 |
| A30 | `fem.load_vector` docstring: barycentre rule "exact for f in P1" | **wrong (code comment only)** | The one-point rule is exact only for constant f (difference of 1e-4 for f=x on an 8×8 mesh). It is still O(h²)-consistent. Describe it in the paper as "one-point (barycentre) quadrature" |

---

## B. Statements in the OLD paper that are wrong, imprecise, or should go

| # | Location | Problem | Suggested wording / action |
|---|---|---|---|
| B.1 | Abstract; Conclusions ("extremely efficient") | Unsupported. Iteration counts range from tens to thousands (Ex.1 p=20: 560; p=1.1: 680 at h=0.05; p≤1.03: >4800). No comparison with any other solver was made | "…a simple implementation in which each iteration requires one solve with a fixed, pre-factorised stiffness matrix and the solution of independent scalar equations on each element. The number of iterations depends strongly on p; it grows markedly as p→1." |
| B.2 | Abstract, Intro: MatLab | The new implementation is Python | Describe the Python package. Mention the MatLab code only as the original version |
| B.3 | Intro: "[DPM] introduce the decomposition–coordination method" | The method goes back to Glowinski–Marrocco (1975) and Gabay–Mercier (1976). It is what is now called ADMM. [DPM] applied it to the p(x)-Laplacian | "The decomposition–coordination method (Glowinski–Marrocco, Gabay–Mercier; nowadays known as ADMM, see Boyd et al. 2011) was applied to the p(x)-Laplacian in [DPM]." |
| B.4 | Intro: "Moreover, we provide a proof of convergence…" | The old paper only quotes Glowinski's theorem under p ≥ 2 | "We recall the convergence result that follows from Glowinski's theory for p ≥ 2, and we observe that for the fully discrete problem convergence holds for every p>1 (Proposition …)", only if C.4 is adopted |
| B.5 | Intro: "first rigorous analysis … [DPLM]" | Not a numerics issue, but the priority claim is risky | "Among the first rigorous analyses…" |
| B.6 | Intro: comparison with [CZ] ("simpler, elegant and easier to implement") | Subjective and not benchmarked | Drop the comparative claim, or say "conceptually different" |
| B.7 | §2, (2.1) | Correct for r=1, but the paper never states the general-r version used by the code. "∇·" applied to a piecewise-constant field must be understood weakly | Use the algorithm in C.2 |
| B.8 | §2 scalar equation: "t_n(t_n^{p−2}+1)=r_n", "ν_n = (η_n+∇u_n)/(t_n^{p−2}+1)" | (i) Undefined at t=0 for p<2 (0^{p−2}). (ii) The symbol r_n clashes with the augmented-Lagrangian parameter r. (iii) It only covers r=1 | ν_n = t_n w_n/\|w_n\| (ν_n = 0 if w_n = 0), with t_n the root of t^{p−1} + r t = r\|w_n\|, w_n = η_n/r + ∇u_n (see C.3) |
| B.9 | §2 "In [DPM] it is proved that this procedure is convergent…" | I could not check [DPM]'s hypotheses. Glowinski's result as quoted needs p ≥ 2, and Examples 2 and 3 have p ≤ 2 everywhere (Ex.3: p ∈ [4/3, 2]). The statement as written covers neither | "For p ≥ 2, convergence follows from [Glowinski, Ch. VI, Thm 5.1] (Theorem 4.x); see also Remark 4.y for 1<p<2." |
| B.10 | §3.1 "two basic tasks", then three items | Typo | "three" |
| B.11 | §3.1 f(k) "computed by an appropriate quadrature" | State the rule used | One-point barycentre rule F_k ≈ Σ_T \|T\| f(x_T)/3 (exact for f constant), or the 3-point edge-midpoint rule (exact for f ∈ P1) |
| B.12 | **§3.2 sign:** h(k)=∫(η−ν)·∇φ_k, Au = h+f | **Wrong sign.** It contradicts (2.1) and its weak form, and the code | h(k) = ∫(ν−η)·∇φ_k (r=1); general r: r A u = f + ∫(rν−η)·∇φ_k, with Dirichlet rows eliminated |
| B.13 | §3.2 "This problem has a unique solution x ∈ [0,r]" | Correct for r=1 (r there is \|w\|), but the bisection range is too wide for p near 1 | Use the bracket [0, min(R/r, R^{1/(p−1)})] and Lemma C.3 |
| B.14 | §3.2 random initialisation "to show the robustness" | Fine, but only one test checks it (p=3, 20×20 mesh) | "Different initial guesses lead to the same limit (as guaranteed by Prop. C.4); we use ν_0 = η_1 = 0." |
| B.15 | **§3.3 stopping criterion** | (i) The text says all experiments use the p-error. The original MatLab code used the relative ℓ² increment < 1e-5. (ii) It is an increment, not an "error between solutions". (iii) For variable p, J_p(·)^{1/max p} is not homogeneous and not a norm. (iv) It is absolute, not relative | Describe what was actually used. Recommend relative residual-based criteria (E.4) |
| B.16 | §4: H = (L^{p(x)})^N with the term (r/2)∫\|∇v−ν\|² | The augmented Lagrangian needs H ⊂ L²; this is the real reason for p ≥ 2 | "For p(x) ≥ 2 on a bounded Ω we have L^{p(x)} ⊂ L², and we work in H = (L²(Ω))^N, with F extended by +∞ outside (L^{p(x)})^N (convex, proper, l.s.c.)." Check against Glowinski's exact setting |
| B.17 | §4 Theorem 4.3 proof, "since w is arbitrary … exactly (2.1)" | Fine for r=1. Note that it identifies (a) with the weak form of (2.1) only | OK; add general r |
| B.18 | **Remark 4.4: "restriction p ≥ 2 is of technical nature … the algorithm behaves well for all 1<p<∞"** | Over-claim. Convergence for p→1 is very slow (ρ → 1). The p ≥ 2 restriction is not merely technical in the continuous setting | Replace with the remark in D |
| B.19 | Ex.1: errors "∼2e-3, ∼1e-3", time ∼75 s | New code: 1.23e-3 / 5.7e-4 at the stop. These depend on the tolerance (A5). The time is MatLab-specific | Report the new numbers together with tolerance, iteration count and stopping rule. Drop the timing, or put it in a table labelled with hardware |
| B.20 | **Ex.1–3: "quotient asymptotically constant ≈ 56/70/50", Figures err1–err3** | **False as n→∞.** The stopping quantity → 0 while the true error → the discretisation error > 0, so the quotient → ∞ (Ex.1 at the stop: ≈287; Ex.2: ≈594; Ex.3: ≈4.7 at the stop and unbounded afterwards; see `figures/ex_errors.png`). At best it is roughly constant during a transient in which iteration error dominates | Replace with a discussion based on `figures/ex_errors.png`: the increment-based criteria track the iteration error only while ρ is well below 1, and say nothing about the discretisation error |
| B.21 | **Ex.2 errors "‖u−u_h‖_∞ ∼ 2e-4, ‖u−u_h‖_p ∼ 3e-4"** | **Impossible to be meaningful:** max\|u\| = c_p = 8.9e-5 < 2e-4, so u_h ≡ 0 would have been more accurate. The old result was wrong (or used different data) | Remove. New: 1.39e-5 (rel. 15.7 %) at p_error<1e-8 / 522 it, mostly iteration error; 1.16e-6 (rel. 1.3 %) after 4000 it, not yet converged. Always give relative errors for this example |
| B.22 | Ex.2 caption omits "70 times" | Inconsistent with the text | Drop (B.20) |
| B.23 | Ex.3 text refers to Figure err2 instead of err3 | Wrong reference | Fix |
| B.24 | Ex.3 errors "‖·‖_∞ ∼1.5e-5, ‖·‖_p ∼2.7e-4" | ‖e‖_p > ‖e‖_∞·\|Ω\|^{1/p} cannot hold for a norm (4^{1/p} ≤ 2.6). It reflects that J_p^{1/max p} is not a norm | Report sup and L² (or Luxemburg) errors, relative |
| B.25 | **Ex.4 caption p=1.15, file `torsion_p1_5`** | Unknown whether p was 1.15 or 1.5 | Recompute with the new code and caption what is computed. FACTS lists p=1.15 |
| B.26 | Ex.4 "for small values of p the solution is flat" | Qualitative and acceptable, but vague | "for p close to 1 the solution is small and has a plateau-like profile with steep gradients near ∂Ω; as p grows it approaches the distance-to-the-boundary function (the 'pyramid')." The p→∞ limit is classical; cite it if stated |
| B.27 | Ex.5, time ∼52 s; "discontinuous exponent" handled | Timing as in B.19. With barycentre sampling, the interface x=0 is aligned with the uniform mesh, so the sampling is exact there. State this | "The interface {x=0} is a mesh line, so p is represented exactly." |
| B.28 | Theorem/remarks use "converges to the actual solution" | The iteration converges to the discrete solution u_h. Convergence u_h → u is a separate FEM question ([DPM2], [D], …) | Separate the two limits explicitly (n→∞ at fixed h; then h→0) |

---

## C. Algorithm for general r and justification (paste-ready)

### C.1 Discrete setting
Let T_h be a conforming triangulation of Ω_h (Ω_h = Ω for polygons; an inscribed polygon with vertices on ∂Ω for curved Ω). Let V_h be the P1 space, V_{h,g} = {v∈V_h : v(z)=g(z) at boundary nodes z}, V_{h,0} its homogeneous version, and Q_h = (P_0(T_h))². Write (·,·) for the L²(Ω_h) inner product. The exponent is replaced by p_T = p(x_T), with x_T the barycentre of T, and the load by f_h(φ) = Σ_T |T| f(x_T) Σ_i φ(z_i^T)/3 (or any quadrature). The discrete problem is

```latex
\min_{v\in V_{h,g}} E_h(v),\qquad
E_h(v)=\mathcal F_h(\nabla v)-f_h(v),\qquad
\mathcal F_h(q)=\sum_{T\in\mathcal T_h}|T|\,\frac{|q_T|^{p_T}}{p_T}\quad(q\in Q_h).
```
Since |q|^{p}/p is strictly convex and C¹ for p>1 and ∇ is injective on V_{h,0}, E_h has a unique minimiser u_h. It is characterised by (|∇u_h|^{p_T−2}∇u_h, ∇φ) = f_h(φ) for all φ ∈ V_{h,0}. With r>0, the augmented Lagrangian on V_{h,g}×Q_h×Q_h is

```latex
\mathcal L_r(v,q,\mu)=\mathcal F_h(q)-f_h(v)+(\mu,\nabla v-q)+\tfrac r2\|\nabla v-q\|_{L^2}^2 .
```

### C.2 Algorithm (decomposition–coordination / ADMM)
```latex
\textbf{Data:} $r>0$, $\nu_0,\eta_1\in Q_h$ (we take $\nu_0=\eta_1=0$). For $n=1,2,\dots$:
\begin{enumerate}
\item ($u$-step, linear) find $u_n\in V_{h,g}$ such that
  \[ r(\nabla u_n,\nabla\varphi)=f_h(\varphi)+(r\nu_{n-1}-\eta_n,\nabla\varphi)\qquad\forall\varphi\in V_{h,0};\]
  in matrix form $rA_{ff}\mathbf u_f=\mathbf f_f+\mathbf b_f-rA_{fd}\mathbf g_d$ with
  $\mathbf b_k=\int_\Omega(r\nu_{n-1}-\eta_n)\cdot\nabla\varphi_k$ (exact, since both factors are piecewise constant);
  $A_{ff}$ is factorised once.
\item ($\nu$-step, local) on each $T$, with $w=\eta_n|_T/r+\nabla u_n|_T$, solve
  \[ |\nu|^{p_T-2}\nu+r\nu=r\,w \quad\Longleftrightarrow\quad
     \nu_n|_T=\begin{cases} t\,w/|w|,& w\neq0,\\ 0,& w=0,\end{cases}\qquad t^{p_T-1}+rt=r|w| .\]
\item (multiplier) $\eta_{n+1}=\eta_n+r(\nabla u_n-\nu_n)$.
\end{enumerate}
```
Step 1 is the Euler–Lagrange equation of v ↦ L_r(v, ν_{n−1}, η_n). Step 2 is that of q ↦ L_r(u_n, q, η_n), which decouples element by element. For r=1 step 1 is the weak form of −Δu_n = f + div(η_n − ν_{n−1}), i.e. (2.1) of the old paper. **The sign in old §3.2 must be corrected to b_k = ∫(ν−η)·∇φ_k.** For p<2, |ν|^{p−2}ν is understood as 0 at ν=0 (its continuous extension; it is the gradient of the C¹ function |ν|^p/p).

### C.3 Lemma (scalar equation; bracketing; accuracy)
*Let p>1, r>0, R ≥ 0 and φ(t) = t^{p−1} + r t − R on [0,∞). Then φ has exactly one zero t\*. It satisfies*

```latex
0\le t^*\le b:=\min\{R/r,\;R^{1/(p-1)}\}.
```
*If ν solves |ν|^{p−2}ν + rν = r w, then ν = t\* w/|w| with R = r|w| (and ν = 0 if w = 0).*

*Proof.* φ is continuous and strictly increasing (both t^{p−1}, since p>1, and rt are), with φ(0) = −R ≤ 0 and φ(t) → ∞. So there is exactly one zero. Since both terms are nonnegative, t\*^{p−1} ≤ R and r t\* ≤ R, which gives t\* ≤ b. Equivalently, if b = R/r then φ(b) = b^{p−1} ≥ 0, and if b = R^{1/(p−1)} then φ(b) = r b ≥ 0. For the vector equation, (|ν|^{p−2} + r)ν = r w with a positive scalar factor, so ν = λw with λ ≥ 0. Taking norms, t = |ν| satisfies t^{p−1} + r t = r|w|. ∎

*Accuracy of bisection.* After n halvings of [0,b], |t_n − t\*| ≤ b 2^{−n}. Write s = r t\*^{2−p} (the ratio of the linear to the power term). Then R/r = t\*(1+s)/s and R^{1/(p−1)} = t\*(1+s)^{1/(p−1)}, so

```latex
\frac{b}{t^*}=\min\Big\{1+\tfrac1s,\,(1+s)^{1/(p-1)}\Big\}\le C_p:=\sup_{s>0}\min\Big\{1+\tfrac1s,(1+s)^{1/(p-1)}\Big\}<\infty,
```
and the *relative* error is ≤ C_p 2^{−n}. Numerically, C_p ≈ 30 (p=1.01), 6.1 (1.1), 2.6 (1.5), 2 (p=2), 1.47 (4), 1.12 (20). Note that C_p is independent of r. The code uses n = 70. Newton's method could be used instead, but bisection is robust and vectorises trivially.

### C.4 Two structural facts (useful for the paper)
**(i) Multiplier = flux.** From steps 2–3, η_{n+1} = η_n + r∇u_n − rν_n = |ν_n|^{p_T−2}ν_n for every n ≥ 1 and every r. (Verified numerically to 1e-15.) Hence, for n ≥ 2, the iteration is a fixed-point map on ν alone:
r(∇u_n,∇φ) = f_h(φ) + (rν_{n−1} − |ν_{n−1}|^{p−2}ν_{n−1}, ∇φ).
*Consequence for p=2, r=1:* η_{n+1} = ν_n, so u_n = u_h for every n ≥ 2 (n ≥ 1 from a zero start). Meanwhile ν_n = (ν_{n−1} + ∇u_h)/2 converges only linearly with factor 1/2. This is why a criterion on u alone stops after 2 iterations.

**(ii) Convergence of the fully discrete iteration for every p_T>1 and r>0.** (Recommended as a Proposition. **The writer must check the exact statement of the cited theorem.**)
The scheme of C.2 is ADMM for min f₁(u) + f₂(ν) subject to ∇u − ν = 0, with
f₁(u) = −f_h(u) + I_{V_{h,g}}(u), f₂ = F_h, and the finite-dimensional Hilbert space Q_h with the L² product. Both f₁ and f₂ are closed, proper and convex for every p_T ≥ 1. The unaugmented Lagrangian has a saddle point, namely (u_h, ∇u_h, η\*) with η\* = |∇u_h|^{p−2}∇u_h, as the Euler–Lagrange equation of C.1 shows. The classical convergence theorem for ADMM in finite dimension (Glowinski–Marrocco 1975; Gabay–Mercier 1976; Fortin–Glowinski 1983, Ch. III; Glowinski–Le Tallec 1989; Eckstein–Bertsekas 1992; Boyd et al. 2011, §3.2 and App. A) gives η_n → η\* and ∇u_n − ν_n → 0 for any r>0. The multiplier is unique because F_h is differentiable. By (i), ν_n = a^{−1}(η_{n+1}), where a(q) = |q|^{p−2}q is a homeomorphism of R², so ν_n → a^{−1}(η\*) = ∇u_h. Then ∇u_n → ∇u_h, and u_n → u_h since ∇ is injective on V_{h,0}.
→ **On a fixed mesh, the iteration converges to the discrete solution for all 1 < p_T < ∞, from any initial guess.** No rate is asserted. The restriction p ≥ 2 enters only in the infinite-dimensional statement (H = L²), and when one asks for bounds uniform in h.

### C.5 Norms and errors (what to define in the paper)
- Errors should be computed as ‖u − u_h‖_{L²(Ω_h)} and ‖∇(u − u_h)‖_{L^{p}(Ω_h)} with a quadrature of degree ≥ 4 on each triangle, as done in `figscripts/verify_true_errors_disk.py` and `variable_p_order.py`. Use relative values. The repository examples instead compute ‖I_hu − u_h‖ from nodal values, and |∇u_h − ∇u(x_T)| sampled at barycentres. Both are legitimate *discrete* error measures but are prone to supercloseness. If they are used, the paper must name them as such.
- For variable p use the Luxemburg norm ‖v‖_{p(·)} = inf{λ>0 : ∫|v/λ|^{p(x)} dx ≤ 1} (cheap to compute by bisection in λ), or simply L² and L∞. Do **not** call J_p(v)^{1/max p} a norm: it is not homogeneous, and for small v it behaves like ‖v‖^{p_min/p_max}.
- `rel_u` = ‖u_n − u_{n−1}‖_{ℓ²}/‖u_{n−1}‖_{ℓ²}: Euclidean norm of the nodal vectors, including Dirichlet nodes, so it is weighted by the mesh.
- `p_error` = J_p(u_n − u_{n−1})^{1/max p}, with J_p evaluated by the 3-point edge-midpoint rule on |·|^{p_T}. It is absolute, not relative.

---

## D. What can and cannot be claimed for 1 < p < 2

**Can claim.** (a) The discrete iteration converges for every p_T > 1 (C.4(ii)), subject to the citation check. (b) The observed L² order is 2 for smooth exact solutions, for p from 1.05 to 6 (constant) and for continuous variable p. The observed W^{1,p} order is 1 for p ≤ 2. (c) The observed linear contraction factor of the residual grows towards 1 as p → 1, and iteration counts grow roughly like 1/(1−ρ).

**Cannot claim.** Convergence of the continuous (function-space) iteration for p < 2. Any rate of the iteration, linear or otherwise. Mesh-independent iteration counts: the data show the opposite. For tol 1e-10, p=4 needs 53→219 iterations as h goes 0.2→0.0125, and p=1.3 needs 3237→480 (non-monotone). Any a priori L² estimate of order 2 for the p(x)-Laplacian. "Robustness for all 1<p<∞".

**Remark (paste-ready):**
> *Remark.* Theorem 4.x requires p(x) ≥ 2, which makes the augmented Lagrangian well defined on L². At the discrete level this restriction is not needed. For every triangulation and every piecewise-constant exponent with p_T > 1, the iterates (u_n, ν_n, η_n) of Algorithm C.2 converge to (u_h, ∇u_h, |∇u_h|^{p−2}∇u_h), where u_h is the unique minimiser of the discrete energy (Proposition C.4). This statement says nothing about the speed of convergence. In our experiments the residual ‖∇u_n − ν_n‖ decays linearly, with a factor ρ that depends on p. On the unit disk with h = 0.05 and r = 1 we measured ρ ≈ 0.79 for p = 4, ≈ 0.53 for p = 2.5, ≈ 0.92 for p = 1.5, ≈ 0.99 for p = 1.3 and ≈ 0.999 for p = 1.1, so the number of iterations grows rapidly as p → 1. For p = 2 and r = 1 the factor is exactly 1/2 (Section C.4(i)).

**Open question (paste-ready):**
> *Open question.* For 1 < p(x) < 2 the space L^{p(x)} is not contained in L², and Glowinski's convergence theory does not apply to the continuous iteration. The finite-dimensional argument gives convergence on every mesh, but with no information on how the speed depends on h or on p. It would be of interest to establish (i) convergence of the iteration in a function-space setting for 1 < p < 2, for instance in a weighted or Orlicz framework, and (ii) bounds on the contraction rate that are uniform in h, together with their dependence on p − 1 and on r. Our experiments suggest a linear rate whose factor tends to 1 as p → 1, and an optimal r of order one (r ≈ 3 for p = 1.1 in one test). We have no proof of either observation.

---

## E. Recommended Numerical Experiments section

1. **Setup paragraph.** P1 elements; p sampled at barycentres; load quadrature; Dirichlet elimination; LU factorised once; r = 1 unless stated; zero initial guess; stopping rule and tolerance; how errors are measured (true quadrature errors, relative); mesh generators. Say that disk meshes are unstructured, non-nested DistMesh-type meshes, that h is the *target* edge length, and that the domain is the inscribed polygon.
2. **Examples 1–3 of the old paper (validation).** One table: n=100, iterations, stopping rule and tolerance, relative sup and L² errors *at the stop* and *after a long run*. Figure `ex_errors.png` supports the discussion of stopping criteria. Drop the "constant quotient" figures.
   - Caption suggestion for `ex_errors.png`: "Examples 1–3 (100×100 mesh). Stopping quantities rel_u and p-error (coloured), true error against the nodal values of the exact solution (black; solid: maximum norm, dashed: J_p(·)^{1/max p}), and distance to a reference iterate (green dotted: 4000 iterations for Ex. 1–2, 1500 for Ex. 3; for Ex. 1–2 the reference is itself not fully converged). The vertical line marks the iteration at which the tolerance used in Table X is met."
3. **Convergence in h, constant p (disk torsion).** Table of true relative L² and W^{1,p} errors with fitted orders and a convergence flag, for p ∈ {1.1, 1.3, 1.5, 2, 4, 6} (plus 1.05 and 1.15 if rerun). **The existing `convergence_order_torsion_disk.csv` and `figures/convergence_order.png` use nodal and barycentre errors; regenerate them with true errors** (the `one()` function in `examples/convergence_order.py` needs a degree-4 quadrature, as in `figscripts/verify_true_errors_disk.py`). Explain the W^{1,p} orders below 1 for p>2 by regularity (∇u ~ r^{1/(p−1)} at 0, giving h^{min(1, 1/(p−1)+2/p)}).
4. **Stopping criteria.** Recommend relative primal and dual residuals, standard for ADMM: ‖∇u_n − ν_n‖ ≤ ε max(‖∇u_n‖, ‖ν_n‖) and r‖ν_n − ν_{n−1}‖ ≤ ε‖η_n‖. Either way, warn that increment-based criteria underestimate the iteration error by a factor ≈ ρ/(1−ρ), which is ~10³ for p=1.1. Show the tolerance dependence of Ex.3 at n=100 (rel_u < 1e-5 / 1e-7 / 1e-8 gives sup error 1.23e-3 / 1.90e-5 / 4.47e-6; converged value 4.36e-6).
5. **Iteration behaviour vs p (p→1).** A ρ(p) table and the iteration-count table at h=0.05, after fixing p=2 (ρ=0.5, not 0). Add the r-study as a small table. Caveat: single mesh; ρ fitted on a window, so the estimate depends on the window and run length (0.9989 vs 0.9995 for p=1.1).
6. **Variable exponent with exact solutions.** Figure `variable_p_order.png` and a table of true L² errors and orders: smooth 2.04, radial-linear 2.00, Ex.3 2.00, jump 1.49 (erratic) with sup 0.90. Caption: "Relative L² error versus h for variable exponents. Filled markers: ‖u−u_h‖_{L²(Ω_h)} (quadrature); open markers: ‖I_hu−u_h‖ (nodal values). Disk cases: unstructured meshes with target edge length h; Example 3: uniform n×n meshes of (−1,1)², h = 2/n. Slopes are least-squares fits over the three finest meshes. Reference slopes h² and h." For the jump case, explain the kink and the unfitted interface. Optionally add one run with an interface-fitted mesh (fixed nodes on r = 0.5) to show that the order is recovered. That would be a cheap and convincing addition I did not run.
7. **Qualitative examples (no exact solution).** Ex.4 (p = 1.15 or 1.5, decide by recomputing; 4; 50) and Ex.5 (interface is a mesh line), plus the L-shape, annulus and union-of-disks examples from the notebook.
8. **Timings.** At most one sentence ("about 26 ms per iteration for 10⁴ nodes on a 2-core cloud machine; set-up 0.06 s"). Do not compare with the MatLab timings.

---

## F. Limitations / threats to validity

- **Discrete vs true errors.** FACTS and the repo scripts measure nodal and barycentre errors. Orders above 2, and gradient orders above 1, are artefacts of this choice. Constants also differ (factors of 1.2 to 45).
- **Iteration error vs discretisation error.** For p near 1 and for p=20, the reported errors at practical tolerances contain a substantial iteration component (Ex.2: about 90 %). Fits for p ≤ 1.2 include meshes on which the iteration did not reach its tolerance.
- **Fits over 3 meshes** on non-nested, irregular, seed-dependent meshes. Pair orders fluctuate (e.g. jump case 2.49 / 0.55 / 1.92 / 1.06). The quoted slopes carry an uncertainty of about ±0.1 (smooth cases) or more.
- **Curved domains:** inscribed polygon. The error is measured on Ω_h only; the skin Ω \ Ω_h is ignored, and Dirichlet data are imposed at nodes on ∂Ω. For P1 this does not limit the order in the observed cases, but it is not analysed for the p(x)-Laplacian.
- **Piecewise-constant p** (barycentre sampling) changes the problem by O(h) in p. This is harmless for smooth p in all tests. For discontinuous p it misassigns the exponent on a band of elements cut by the interface.
- **Quadrature:** one-point load rule (exact for constants only). The modular uses the edge-midpoint rule on |·|^{p}, which is not exact.
- **ρ(p) estimates** come from one mesh (h=0.05), one problem, r=1 and one fit window. They are not mesh-independent.
- **Example 2 reference:** after 4000 iterations it is still not converged, so its 1.3 % error is an upper-bound-type indication, not the discretisation error.
- **Theory citations:** I could not check Glowinski Ch. VI Thm 5.1, [DPM], or the precise theorem numbers of the ADMM convergence results. The writer must verify the hypotheses before citing.
- **Not re-run by me:** r-study (A20), Ex.4–5 iteration counts (A27), Triangle-mesh disk comparison (A26), averaged-p diagnostic (A25), p ∈ {1.05, 1.15, 1.2, 1.5, 1.75, 3} on the disk with true errors.
- **Timings** come from a shared 2-core sandbox and fluctuate with load.

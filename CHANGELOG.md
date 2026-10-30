# Changelog

## 2.2.0

- Fix the degrees of freedom absorbed by fixed effects: levels that are redundant across fixed effects no longer count (one per connected component), and fixed effects nested in a cluster count only once, for the constant. With two or more fixed effects, this changes `dof_fes`, `dof_residual`, standard errors, the F-statistic, and adjusted R².
- Fix `F_kp`: it uses the first-stage degrees of freedom when the model is over-identified, applies a degrees-of-freedom correction under simple covariance (it was too high, by more with many fixed effects), and no longer depends on the order of the endogenous variables. `p_kp` changes accordingly.
- Require Vcov 0.8.4, which contains the two changes to the rank test, and FixedEffects 3.4.1.
- Improve the results display: show the dependent variable in the header, mark IV models in the title, format the F-test p-value like coefficient p-values, add a thousands separator to the number of observations, and warn on non-convergence instead of showing a Converged row (#284).

## 2.1.0

- Reduce compilation latency: formula parsing, schema and model matrices go through a vector-based term pipeline, so methods compile per term type rather than per formula shape or column names (#283).

## 2.0.0

### Breaking changes

- A standalone RHS slope that is exactly spanned by a continuous-slope fixed effect is now omitted from coefficient output. For example, `fe(id)*x` still includes group intercepts and group-specific slopes, but no longer reports the unidentified common `x` coefficient as a dropped `0` with `NaN` inference statistics. Code that relies on coefficient positions or names should account for the shorter coefficient vector.

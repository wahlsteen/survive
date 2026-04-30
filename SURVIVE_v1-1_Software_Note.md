# SURVIVE v1.1 — Software Note
### A Multi-Model Population Viability Analysis Tool for Conservation Planning

*Eric Wahlsteen | 2026*

## 1. Introduction

SURVIVE v1.1 is a Python-based population viability analysis (PVA) tool designed to assess the long-term viability of small or threatened populations under uncertainty. It integrates demographic and genetic sub-models within a unified framework, enabling practitioners to evaluate the relative importance of stochastic demography, genetic drift, immigration, and density dependence on population persistence over a user-defined time horizon (default 100 years).

The tool is designed for ecological consulting workflows in which species-specific parameters are stored in separate configuration files (e.g. `config_verbascum.py`), allowing rapid scenario switching without modifying the core model code. It currently supports the following life-history archetypes: monocarpic perennials (e.g. *Verbascum lychnitis*), iteroparous amphibians (e.g. *Bufo bufo*, *Lissotriton vulgaris*), and custom species.

SURVIVE v1.1 produces an Excel results table and a suite of standardised figures for each run. Run parameters are automatically logged to `run_settings.txt` for full reproducibility.

### 1.1 Design Philosophy

Rather than a single monolithic model, SURVIVE decomposes the viability question into four independent or semi-coupled sub-models (A, B, B_link, C), each emphasising a different ecological mechanism. Their outputs are combined in a configurable meta-model. This structure makes model assumptions explicit and facilitates communication with non-specialist stakeholders: each sub-model can be presented and defended independently.

## 2. Theoretical Framework

### 2.1 Population Viability Analysis

PVA is the formal process of estimating the probability that a population will persist for a specified period (Dennis et al. 1991; Beissinger & Westphal 1998). SURVIVE implements extinction risk as the complement of the quasi-extinction probability: the probability that census size N(t) falls to or below a quasi-extinction threshold Q before the time horizon T.

### 2.2 Stochastic Demographic Process

Population dynamics follow a geometric Brownian motion (GBM) with density dependence, additive immigration, and rare catastrophes. On the log scale the annual increment is:

$$\ln N(t+1) = \ln N(t) + \mu(t) + \sigma_\text{eff}(t) \cdot Z(t) + \ln\!\left(1 + I/N(t)\right)$$

where Z(t) ~ N(0,1) is annual environmental noise, I is immigrants per year, σ_eff is the effective environmental standard deviation (Section 2.5), and μ(t) is the density-regulated drift term.

**Word equation form:** `ln N(t+1) = ln N(t) + μ(t) + σ_eff(t)·Z(t) + ln(1 + I/N(t))`

Two density regulation models are available:

**Ricker:**

    μ(t) = r · (1 − N(t)/K) − ½σ²

**Theta-logistic:**

    μ(t) = r · (1 − (N(t)/K)^θ) − ½σ²

where r is the intrinsic rate of natural increase, K is carrying capacity, and θ controls the shape of density dependence (θ = 1 reduces to logistic). The −½σ² term is the Itô correction ensuring that the expected value of the multiplicative process matches r (Lande et al. 2003).

> **Practical note:** At σ = 0.25 and r = 0, the effective drift μ_eff = −0.031 yr⁻¹, meaning the population declines slowly even without catastrophes. Use r ≥ 0.02 for a genuinely neutral baseline.

Catastrophes are modelled as independent per-replicate Bernoulli draws each year:

    N(t+1) ← N(t+1) · (1 − severity)   with probability p_event per replicate per year

Default: p_event = 0.01, severity = 0.90. Each replicate draws independently, preventing artificial synchronisation across the ensemble.

### 2.3 Analytical Quasi-extinction Probability (Dennis et al. 1991)

For Model A, extinction probability is also computed analytically using the two-term first-passage formula for GBM:

    P_ext = Φ(z₁) + (Q/N₀)^(2μ/σ²) · Φ(z₂)

    z₁ = (ln(Q/N₀) − μ·T) / (σ√T)
    z₂ = (ln(Q/N₀) + μ·T) / (σ√T)

where Φ is the standard normal CDF. This two-term formulation accounts for the full first-passage probability, not just the terminal-time probability (Dennis et al. 1991), and is used for scenario scoring and sensitivity analysis.

### 2.4 Effective Population Size and Ne/Nc Ratios

The effective population size N_e governs the rate of genetic drift and inbreeding. SURVIVE uses three Ne/Nc ratios (default: 0.05, 0.10, 0.15). Ne appears in two contexts:

- **Ne_demo** = ratio × NC_POP: effective size for the demographic Ne-sensitivity model (Model C).
- **Ne_gen** = ratio × NC_METAPOP: effective size for genetic sub-models (B, B_link).

The distinction is important when the census population (NC_POP) and the local breeding unit relevant for genetics (NC_METAPOP) differ in scale.

### 2.5 Life-history Correction for Generational Ne

Annual Ne/Nc ratios must be converted to generational Ne for genetic models. The correction depends on the life-history archetype, controlled by the `NE_GENERATION_MODEL` flag.

#### 2.5.1 Monocarpic Perennials — Vitalis et al. (2004)

For monocarpic perennials (plants that live for multiple years as rosettes but reproduce only once), the generational effective size is:

    Ne_gen = ((2 − s) / 2) · N · T²

where N is the annual census of flowering adults, T is generation time in years, and s is the fraction of established rosettes that die before ever flowering (pre-reproductive mortality). The multiplier relative to the annual Ne baseline (ratio × N) is:

    multiplier = ((2 − s) / 2) · T

For *Verbascum lychnitis* (T = 4, s = 0.05): **multiplier ≈ 3.9×**. This means an annual Ne/Nc of 0.10 corresponds to a generational Ne/Nc of ~0.39, substantially reducing drift per generation.

> **Note on parametrisation:** s refers to mortality of *established* rosettes, not germination-to-flowering mortality. If NC_POP already counts only established, counted rosettes, s should be set low (0.0–0.2). Setting NC_POP = annual flowering adults (the correct parametrisation) and s = pre-reproductive mortality of established rosettes are the two separate steps.

#### 2.5.2 Annuals with Seed Bank — Nunney (2002)

For true annuals with seed dormancy, the seed bank buffers genetic drift, producing:

    Ne_gen ≈ N · T     multiplier = T

#### 2.5.3 Iteroparous Organisms — Nunney (1991)

For iteroparous organisms with overlapping generations and significant adult survival, no upward T-correction is applied (multiplier = 1.0). The Ne/Nc ratio is assumed to already capture adult-survival effects and sex-ratio skew. This is the correct setting for amphibians.

### 2.6 Genetic Drift: Wright-Fisher Model

Inbreeding accumulation over t generations in an isolated population:

    F_t = 1 − (1 − 1/(2Ne))^t

Heterozygosity at time T:

    H(T) = H₀ · (1 − F_T)

With immigration (Wright island model), inbreeding recurses across generations:

    F_{g+1} = (1 − m)² · F_g + (1 − (1−m)²) · 1/(2Ne)

where m = migrants per generation / Nc. Equilibrium inbreeding:

    F_eq = 1 / (4 · Ne · m + 1)

Stochastic simulation uses binomial sampling of 2Ne gametes per generation (Wright-Fisher model), with immigration modelled as allele-frequency mixing with a genetically diverse donor pool (donor p ~ Uniform(0.3, 0.7)).

### 2.7 Ne-sensitive Environmental Stochasticity

Small populations experience increased demographic fragility due to inbreeding depression and Allee effects. SURVIVE proxies this as inflation of environmental stochasticity when Ne falls below a species-specific target:

    σ_eff = σ_e · max(1, Ne_target / Ne)^α

where α (ALPHA) controls the curvature of the penalty (0.5–1.0; lower = gentler). This term enters Models C and B_link only; it does not affect genetic sub-models B or B_var.

## 3. Model Architecture

SURVIVE v1.1 comprises four sub-models and one meta-model. Each sub-model produces a viability indicator for each scenario combination (Ne/Nc ratio × immigration rate). The meta-model aggregates these into a single weighted viability index.

### 3.1 Sub-model A: Density-regulated Demography

Model A simulates population dynamics under Ricker (or theta-logistic) density dependence with additive immigration, environmental stochasticity σ_e (unmodified), and catastrophes. Ne does NOT enter Model A directly. This model asks: *would this population persist at this census size, with these growth parameters and this immigration, if genetic effects are ignored?*

Survival probability = 1 − mean(fraction of replicates reaching N ≤ Q), averaged over the (r, σ_e) uncertainty grid. Viability criterion: Survival(A) ≥ 0.95.

### 3.2 Sub-model B: Genetics-only

Model B evaluates genetic viability in isolation from demography. Three variants:

**B_const (analytical):** Inbreeding F(T) and heterozygosity H(T) from the Wright-Fisher recursion with constant Ne_gen. Pass/fail based on H(T) ≥ H_MIN_FRAC × H₀.

**B_var (stochastic, time-varying Ne/Nc):** Ne/Nc ratio changes from NE_RATIO_START to NE_RATIO_END over NE_RATIO_RELAX_YEARS via exponential relaxation:

    ratio(t) = ratio_end − (ratio_end − ratio_start) · exp(−t / τ)

Generational Ne at time t: `Ne_g(t) = ratio(t) × Nc × multiplier`. H(T) is the mean across 2000 stochastic replicates.

### 3.3 Sub-model B_link: Eco-genetic Coupling

B_link is the most mechanistically integrated sub-model. It first simulates demographic trajectories using C-style assumptions (Ne-sensitive σ_eff), then feeds the resulting population-size paths directly into the genetic drift simulation:

    Ne_g(t) = ratio(t) × Nc(t) × abundance_factor(t) × multiplier

where abundance_factor is a Michaelis-Menten saturation function:

    abundance_factor = Nc(t) / (Nc(t) + K/2)

This means drift intensifies when the population crashes demographically. Gene flow continues even in small populations, which is why B_link can show maintained H(T) even when Model C shows demographic collapse.

> **Interpretation note:** B_link asks "does H(t) stay above threshold along these demographic trajectories?" This is necessary but not sufficient for viability. A population can retain high H until demographic extinction. Always read B_link together with Model C.

### 3.4 Sub-model C: Ne-sensitive Demography

Identical to Model A in structure but uses σ_eff (Ne-inflated) rather than σ_e. This asks: *what is the extinction risk if demographic volatility increases as the population becomes genetically depauperate?*

Model C is the primary demographic viability indicator for management recommendations because it integrates both demographic and genetic-fragility signals into a single population trajectory.

### 3.5 Meta-model

The meta-model combines sub-model survivals as a weighted mean:

    META_survival = w_A · S(A) + w_B · S(B_link) + w_C · S(C)

Default weights: w_A = 0.34, w_B = 0.33, w_C = 0.33. S(B_link) is used (not S(B_const)) because it is the most mechanistically integrated genetic estimate. Viability criterion: META_survival ≥ 0.95.

### 3.6 Disturbance Model

An initial disturbance profile inflates σ_e exponentially and relaxes over time:

    disturb_factor(t) = f_end + (f_start − f_end) · exp(−t / τ_relax)

Set DISTURBANCE_SIGMA_FACTOR_START = 1.0 and DISTURBANCE_RELAX_YEARS = 0 to disable (baseline scenario).

## 4. Key Equations Summary

| Symbol / Equation | Description | Used in |
|-------------------|-------------|---------|
| `μ = r(1 − N/K) − ½σ²` | Annual drift, Ricker, Itô corrected | A, C, B_link |
| `P_ext = Φ(z₁) + (Q/N₀)^(2μ/σ²)Φ(z₂)` | Dennis et al. (1991) first-passage formula | A (analytical) |
| `σ_eff = σ_e · max(1, Ne_t/Ne)^α` | Ne-sensitive stochasticity inflation | C, B_link |
| `F_t = 1 − (1 − 1/2Ne)^t` | Inbreeding accumulation, isolated | B_const |
| `F_{g+1} = (1−m)²F_g + (1−(1−m)²)/(2Ne)` | Wright island model recursion | B_const, B_link |
| `F_eq = 1/(4·Ne·m + 1)` | Equilibrium inbreeding with migration | B_const, scoring |
| `mult = ((2−s)/2)·T` | Vitalis (2004) monocarpic multiplier | B, B_var, B_link |
| `mult = T` | Nunney (2002) seed-bank multiplier | B, B_var, B_link |
| `Ne_g = ratio(t)·Nc(t)·abund_factor·mult` | Generational Ne in eco-genetic model | B_link |
| `ratio(t) = ratio_end − Δratio·exp(−t/τ)` | Time-varying Ne/Nc recovery | B_var, B_link |
| `Score = w_g·F_min + w_d·P_ext` | Composite ranking score (lower = better) | Ranking table |

## 5. Output Figures and Interpretation

Each model run produces a standardised set of figures saved in a timestamped results folder.

### 5.1 Figure A — Density-regulated Survival
**File:** `mod_A_survival_density_demography.png`

Shows the fraction of simulated populations surviving above Q as a function of time, with 95% bootstrap confidence intervals. Lines represent different immigration rates; curves are averaged over the (r, σ_e) uncertainty grid. The horizontal dashed line marks the 95% survival target.

*How to read:* A curve sustained above 95% throughout the 100-year horizon indicates a demographically robust scenario. Model A is optimistic because Ne effects on σ are not included. Compare with Figure C for the full Ne-sensitive view.

### 5.2 Figure B — Genetic Heterozygosity
**File:** `mod_B_heterozygosity_genetics.png`

Plots mean H(t) over time for each combination of Ne/Nc ratio and immigration rate (B_var model), with bootstrap confidence intervals. The dotted horizontal line marks the 70% threshold (H_MIN_FRAC × H₀).

*How to read:* Curves above threshold throughout indicate sufficient genetic diversity retention. The Vitalis/Nunney generational Ne correction is applied here; without it, curves for monocarpic perennials systematically underestimate H(T) by a factor of ~3–4×.

### 5.3 Figure C — Ne-sensitive Survival
**File:** `mod_C_survival_ne_sensitive.png`

Analogous to Figure A but with σ_eff replacing σ_e. This is the most conservative demographic viability estimate. The year at which a curve crosses 0.95 (if any) indicates the expected time to critical demographic failure.

### 5.4 Figure B_link — Eco-genetic Coupling
**File:** `mod_Blink_eco_genetic_coupling.png`

Shows H(t) from the B_link model where demography and genetics are coupled. Unlike Figure B, heterozygosity here responds to population crashes. A curve that drops sharply before recovering indicates a period of demographic crisis during which drift is intense.

### 5.5 Sensitivity Plot
**File:** `sensitivity_sigma.png`

Shows how P(extinction) varies with σ_e for each Ne/Nc ratio. Identifies the environmental-variability tolerance of each scenario.

### 5.6 Genetic–Demographic Scatter
**File:** `combined_genetic_demographic_scatter.png`

Plots equilibrium inbreeding F_eq (x-axis; genetic risk) against P_ext with immigration (y-axis; demographic risk) for all scenario combinations. The ideal scenario is in the lower-left quadrant. Reference lines at F_eq = 0.05 and P_ext = 0.10 mark standard viability thresholds.

### 5.7 Interpreting Divergent Sub-model Outputs

It is common for sub-models to produce apparently contradictory results. The table below covers the most frequent patterns.

| Pattern | Ecological meaning | Management implication |
|---------|-------------------|------------------------|
| **B_link PASS, Model C FAIL** | Immigration maintains genetic diversity even as demographic stochasticity drives population size below Q. Genetic viability is necessary but not sufficient: without surviving individuals, gene flow cannot rescue the population. | Priority: demographic management (habitat quality, carrying capacity). Immigration alone cannot substitute for demographic recovery. |
| **Model A PASS, Model C FAIL** | Ne-sensitive stochasticity (σ_eff > σ_e) is the sole driver of failure. The population is demographically viable if Ne effects are ignored, but inbreeding depression inflates effective environmental variance enough to cause collapse. | Increase Ne via genetic rescue (immigration). The Ne_target parameter defines the management threshold. |
| **B_const PASS, B_link FAIL** | Constant-Ne analytics are overly optimistic: demographic crashes reduce Nc(t) and hence Ne_g(t), accelerating drift beyond what the constant-Ne model assumes. | Trust B_link over B_const for conservative genetic management advice. |
| **All models PASS at high Ne/Nc, FAIL at low Ne/Nc** | Expected pattern: both risks increase as Ne/Nc decreases. | Target Ne/Nc ≥ 0.10–0.15 for both the 50-rule (short-term viability) and the 500-rule (long-term evolutionary potential). |

## 6. User Manual

### 6.1 Requirements

Python 3.9+ with: `numpy pandas matplotlib openpyxl`

```bash
pip install numpy pandas matplotlib openpyxl
```

### 6.2 File Structure

```
survive_1_1.py              Main model script
config_verbascum.py         Species config — Verbascum lychnitis
config_bufo.py              Species config — Bufo bufo
config_lissotriton.py       Species config — Lissotriton vulgaris
RESULTS/                    Output folder (created automatically)
```

### 6.3 Quick Start

1. Open `survive_1_1.py` and set `SPECIES = "V"` (or "B", "L", "M").
2. Set `PROJECT_NAME`, `NC_POP`, `NC_METAPOP` in the PROJECT CONFIG block.
3. Set scenario parameters: `DISTURBANCE_*`, `NE_RATIO_*`, `META_WEIGHTS`.
4. Edit the species config file if needed.
5. Run: `python survive_1_1.py`
6. Results in `RESULTS/<PROJECT_NAME>_<SPECIES>_<timestamp>/`

### 6.4 Configuration Reference

| Parameter | Description | Typical range |
|-----------|-------------|---------------|
| `NC_POP` | Census population (demographic models A, C) | 100–2000 |
| `NC_METAPOP` | Local breeding census (genetics B, B_link); usually = K | = K |
| `K` | Carrying capacity (breeding adults) | 50–1000 |
| `NE_RATIOS` | Ne/Nc ratios to test | [0.05, 0.10, 0.15] |
| `GEN_TIME_YEARS` | (min, max) generation time in years | (4, 5) for *V. lychnitis* |
| `MIGRANTS_PER_YEAR` | Annual immigration rates to test | [1], [8], [1, 8, 24] |
| `R_GRID` | Mean annual growth rates (uncertainty envelope) | [0.02, 0.04] |
| `SIGMA_E_GRID` | Environmental stochasticity (SD on ln scale) | [0.15, 0.25] |
| `H0` | Starting heterozygosity (observed) | 0.6–0.8 |
| `H_MIN_FRAC` | Minimum acceptable H as fraction of H0 | 0.70 |
| `NE_TARGET` | Ne below which σ_eff inflates | 50 (plants), 100 (amphibians) |
| `ALPHA` | Curvature of Ne–stochasticity penalty | 0.5–1.0 |
| `TIME_HORIZON_YEARS` | Simulation duration | 100 |
| `QUASI_EXT_THRESHOLD` | Quasi-extinction threshold Q | 5 |
| `REPLICATES` | Monte Carlo replicates | 2000 |
| `DENSITY_MODEL` | `"ricker"` or `"theta-logistic"` | `"ricker"` |
| `NE_GENERATION_MODEL` | `"monocarpic"`, `"annual_seedbank"`, `"iteroparous"` | Species-dependent |
| `S_PREREPRODUCTIVE_MORTALITY` | Fraction dying before first reproduction (monocarpic) | 0.0–0.2 |
| `DISTURBANCE_SIGMA_FACTOR_START` | Initial σ_e multiplier (1.0 = no disturbance) | 1.0–2.0 |
| `DISTURBANCE_RELAX_YEARS` | Years until habitat stabilises (τ) | 0–10 |
| `NE_RATIO_START / END` | Ne/Nc at founding and at equilibrium | 0.05–0.15 |
| `META_WEIGHTS` | Dict: weights for A, B_link, C (must sum to 1) | {0.34, 0.33, 0.33} |

### 6.5 Species-specific Guidance

#### *Verbascum lychnitis* (monocarpic perennial)

- `NE_GENERATION_MODEL = "monocarpic"`
- `S_PREREPRODUCTIVE_MORTALITY = 0.05–0.20` for established rosettes
- `GEN_TIME_YEARS = (4, 5)`; `GEN_TIME_FOR_GENETICS = 4`
- `NC_POP = K = annual flowering breeders` (not total rosette census)
- Vitalis multiplier at T = 4, s = 0.05: **≈ 3.9× generational Ne inflation**
- `NE_TARGET = 50`; `ALPHA = 0.5`

#### *Bufo bufo* / *Lissotriton vulgaris* (iteroparous amphibians)

- `NE_GENERATION_MODEL = "iteroparous"`
- `S_PREREPRODUCTIVE_MORTALITY = 0.0`
- `NE_TARGET = 100`; `ALPHA = 0.7`
- `NC_METAPOP = breeding individuals at the local pond`; `NC_POP = regional census`

## 7. References

1. Beissinger, S.R. & Westphal, M.I. (1998). On the use of demographic models of population viability in endangered species management. *Journal of Wildlife Management* 62: 821–841.
2. Dennis, B., Munholland, P.L. & Scott, J.M. (1991). Estimation of growth and extinction parameters for endangered species. *Ecological Monographs* 61: 115–143.
3. Lande, R., Engen, S. & Sæther, B.-E. (2003). *Stochastic Population Dynamics in Ecology and Conservation*. Oxford University Press.
4. Nunney, L. (1991). The influence of age structure and fecundity on effective population size. *Proceedings of the Royal Society B* 246: 71–76.
5. Nunney, L. (2002). The effective size of annual plant populations: the interaction of a seed bank with fluctuating population size in maintaining genetic variation. *American Naturalist* 160: 195–204.
6. Vitalis, R., Glemin, S. & Olivieri, I. (2004). When genes go to sleep: the population genetic consequences of seed dormancy and monocarpic perenniality. *American Naturalist* 163: 295–311.
7. Wright, S. (1931). Evolution in Mendelian populations. *Genetics* 16: 97–159.

# SURVIVE v1.1

**A Multi-Model Population Viability Analysis Tool for Conservation Planning**

SURVIVE v1.1 evaluates the long-term viability of small or threatened populations by integrating stochastic demographic and genetic sub-models within a single configurable framework. It is designed for ecological consulting workflows requiring reproducible, scenario-based PVA.

---

## Features

- **Four sub-models** covering density-regulated demography (A), genetics-only drift (B / B_var), eco-genetic coupling (B_link), and Ne-sensitive demography (C)
- **Life-history corrections** for generational Ne: Vitalis et al. (2004) for monocarpic perennials, Nunney (2002) for annuals with seed banks, Nunney (1991) for iteroparous organisms
- **Configurable species** via separate config files (`config_verbascum.py`, `config_bufo.py`, etc.)
- **Reproducible outputs**: Excel results table, PNG figures, and `run_settings.txt` parameter log
- **Disturbance modelling**: time-decaying sigma inflation for newly created or restored habitats
- **API / GUI compatible**: `run_scenario()` entry point for programmatic use

---

## Requirements

```
Python 3.9+
numpy  pandas  matplotlib  openpyxl
```

```bash
pip install numpy pandas matplotlib openpyxl
```

---

## Quick Start

1. Set `SPECIES` at the top of `survive_1_1.py`:
   - `"V"` — *Verbascum lychnitis* (monocarpic perennial)
   - `"B"` — *Bufo bufo* (common toad)
   - `"L"` — *Lissotriton vulgaris* (smooth newt)
   - `"M"` — custom species (`config_my_own_species.py`)

2. Edit the PROJECT CONFIG block:
   ```python
   PROJECT_NAME  = "MyProject_baseline"
   NC_POP        = 636   # census population size (demographic models)
   NC_METAPOP    = 636   # local breeding census for genetics (usually = K)
   ```

3. Run:
   ```bash
   python survive_1_1.py
   ```

4. Results are saved to `RESULTS/<PROJECT_NAME>_<SPECIES>_<timestamp>/`

---

## Configuration

All ecological parameters are set in the species config file (e.g. `config_verbascum.py`).

### Key species-config parameters

| Parameter | Description | Default (V. lychnitis) |
|-----------|-------------|------------------------|
| `K` | Carrying capacity (breeding adults) | 230 |
| `NE_RATIOS` | Ne/Nc ratios to test | [0.05, 0.10, 0.15] |
| `GEN_TIME_YEARS` | Generation time range (min, max) in years | (4, 5) |
| `MIGRANTS_PER_YEAR` | Annual immigration rates to test | [8] |
| `R_GRID` | Intrinsic growth rate uncertainty envelope | [0.02, 0.04] |
| `SIGMA_E_GRID` | Environmental stochasticity (SD on ln scale) | [0.15, 0.25] |
| `H0` | Starting observed heterozygosity | 0.6 |
| `H_MIN_FRAC` | Minimum acceptable H as fraction of H0 | 0.70 |
| `NE_TARGET` | Ne below which σ_eff begins to inflate | 50 |
| `ALPHA` | Curvature of Ne–stochasticity penalty | 0.5 |
| `NE_GENERATION_MODEL` | Life-history archetype for generational Ne | `"monocarpic"` |
| `S_PREREPRODUCTIVE_MORTALITY` | Pre-reproductive mortality s (monocarpic only) | 0.05 |

### Key project-level overrides (in `survive_1_1.py`)

| Parameter | Description |
|-----------|-------------|
| `NC_POP` | Census size for demographic models A and C |
| `NC_METAPOP` | Local breeding census for genetics (B, B_link) |
| `DISTURBANCE_SIGMA_FACTOR_START` | Initial habitat instability multiplier on σ_e |
| `DISTURBANCE_RELAX_YEARS` | Years until habitat stabilises (τ) |
| `NE_RATIO_START / NE_RATIO_END` | Ne/Nc at founding and at long-term equilibrium |
| `META_WEIGHTS` | Dict: weights for sub-models A, B_link, C in meta-model |

---

## Sub-models

| Model | Mechanism | Question asked |
|-------|-----------|----------------|
| **A** | Density-dependent demography, no Ne feedback | Does the population persist demographically? |
| **B / B_var** | Genetic drift + immigration, no demography | Does genetic diversity survive over 100 years? |
| **B_link** | Eco-genetic coupling: demography → genetics | Does H(t) stay above threshold along realistic population trajectories? |
| **C** | Demography with Ne-sensitive σ inflation | Does demographic stability collapse as the population becomes genetically depauperate? |
| **META** | Weighted mean of A, B_link, C survivals | Overall viability score |

---

## Output Files

| File | Contents |
|------|----------|
| `extinction_scenarios.xlsx` | Full results table for all scenario combinations |
| `run_settings.txt` | Complete parameter log |
| `mod_A_survival_density_demography.png` | Sub-model A: density-regulated survival |
| `mod_B_heterozygosity_genetics.png` | Sub-model B_var: genetic heterozygosity over time |
| `mod_C_survival_ne_sensitive.png` | Sub-model C: Ne-sensitive survival |
| `mod_Blink_eco_genetic_coupling.png` | Sub-model B_link: eco-genetic coupling |
| `sensitivity_sigma.png` | Sensitivity of P(extinction) to σ_e |
| `combined_genetic_demographic_scatter.png` | Genetic vs demographic risk scatter |
| `disturbance_decay.png` | Disturbance relaxation diagnostic |

---

## Key Equations

**Demographic process (Ricker, Itô-corrected):**
```
μ(t) = r · (1 − N/K) − ½σ²
N(t+1) = N(t) · exp(μ + σ_eff · Z) + I
```

**Ne-sensitive stochasticity:**
```
σ_eff = σ_e · max(1, Ne_target / Ne)^α
```

**Genetic drift — Wright-Fisher accumulation:**
```
F_t = 1 − (1 − 1/(2·Ne))^t      H(T) = H0 · (1 − F_T)
```

**Wright island model with immigration (per generation):**
```
F_{g+1} = (1−m)² · F_g + (1−(1−m)²) / (2·Ne)
```

**Generational Ne correction — Vitalis et al. (2004) for monocarpic perennials:**
```
Ne_gen = ((2 − s) / 2) · N · T²     multiplier = ((2 − s) / 2) · T
```
*For V. lychnitis: s = 0.05, T = 4 → multiplier ≈ 3.9×*

**Generational Ne correction — Nunney (2002) for annuals with seed bank:**
```
multiplier = T
```

---

## Interpreting Divergent Results

| Pattern | Ecological meaning |
|---------|--------------------|
| B_link PASS, Model C FAIL | Gene flow maintains H even during demographic decline; but genetic rescue cannot substitute for demographic recovery |
| Model A PASS, Model C FAIL | Ne-sensitive stochasticity alone drives failure; increase Ne or reduce σ_e |
| B_const PASS, B_link FAIL | Constant-Ne assumption is too optimistic; demographic crashes accelerate drift |

---

## References

1. Dennis, B., Munholland, P.L. & Scott, J.M. (1991). Ecological Monographs 61: 115–143.
2. Nunney, L. (1991). Proc. R. Soc. B 246: 71–76.
3. Nunney, L. (2002). American Naturalist 160: 195–204.
4. Vitalis, R., Glemin, S. & Olivieri, I. (2004). American Naturalist 163: 295–311.
5. Lande, R., Engen, S. & Sæther, B.-E. (2003). Stochastic Population Dynamics in Ecology and Conservation. Oxford University Press.

---

## Author

Eric Wahlsteen | 2026  
Contact: eric.wahlsteen@gmail.com

# =============== CONFIG VERBASCUM LYCHNITIS ===============================================================================================
NE_RATIOS = [0.05, 0.1, 0.15]        # Ne / Nc
GEN_TIME_YEARS = (3, 5)               # min, max (years per generation)
MIGRANTS_PER_YEAR = [10]       # immigration (breeders/year) Anemochory with dispersal distance 10-500 m (Lososová et al 2023). 

TIME_HORIZON_YEARS = 100              # years
QUASI_EXT_THRESHOLD = 5               # quasi-extinction at/below this N
SURVIVAL_TARGET = 0.95                # survival threshold for plots and flagging (95%)

DENSITY_MODEL = "ricker"        # "ricker" or "theta-logistic"
K = 1000                         # carrying capacity (breeding adults)
THETA = 1.0                     # θ for theta-logistic; 1.0 reduces to logistic

H0 = 0.543               # starting observed heterozygosity (species/population) (Van Rossum et al. 2024)
H_MIN_FRAC = 0.70         # must retain at least 70% of starting heterozygosity by T
GEN_TIME_FOR_GENETICS = 4    # if None, uses GEN_TIME_YEARS[0] (shortest)

# Genetic extinction rule (independent model):
# we recurse F_t with migration over generations and test H_t = 1-F_t
GENETIC_EVAL = "with_migration"  # "isolated" | "with_migration" | "best_of_both"

# Demographic uncertainty grid (annual)
R_GRID = [-0.01, 0.02, 0.04]          # mean growth per year
SIGMA_E_GRID = [0.15, 0.25]          # env. stochasticity (SD on ln scale)

# Scoring weights (lower score = better)
SCORE_WEIGHTS = {"genetic": 0.5, "demographic": 0.5}

# === tie demography to Ne (50/500 intuition) ===
NE_TARGET = 50.0     # below this Ne, volatility ramps up
ALPHA = 0.5           # curvature of penalty; 0.5–1 is reasonable. lower = gentler.

# Monte Carlo for demography-with-immigration
REPLICATES = 2000
RANDOM_SEED = 42

# === Life-history model for generational Ne correction (genetics only) ===
# "monocarpic"      — Vitalis et al. (2004): Ne_gen = ((2-s)/2) * N * T²
#                     Use for biennials / monocarpic perennials with seed bank
# "annual_seedbank" — Nunney (2002): Ne_gen = N * T
#                     Use for true annuals with seed dormancy
# "iteroparous"     — Nunney (1991): no upward T-correction (ne_ratio captures this)
#                     Use for perennial plants with vegetative survival, amphibians, etc.
NE_GENERATION_MODEL = "iteroparous"
S_PREREPRODUCTIVE_MORTALITY = 0.05   # fraction of plants dying before first reproduction
                                    # (rosettes that never flower); only used if monocarpic
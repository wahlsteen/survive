# =============== CONFIG BUFO ===============================================================================================
NE_RATIOS = [0.05, 0.1, 0.15]        # Ne / Nc
GEN_TIME_YEARS = (3, 5)               # min, max (years per generation)
MIGRANTS_PER_YEAR = [8, 16, 24]       # immigration (breeders/year)

TIME_HORIZON_YEARS = 100              # years
QUASI_EXT_THRESHOLD = 5               # quasi-extinction at/below this N
SURVIVAL_TARGET = 0.95                # survival threshold for plots and flagging (95%)

DENSITY_MODEL = "ricker"        # "ricker" or "theta-logistic"
K = 200                         # carrying capacity (breeding adults)
THETA = 1.0                     # θ for theta-logistic; 1.0 reduces to logistic

H0 = 0.7                 # starting observed heterozygosity (species/population)
H_MIN_FRAC = 0.70         # must retain at least 70% of starting heterozygosity by T
GEN_TIME_FOR_GENETICS = 4    # if None, uses GEN_TIME_YEARS[0] (shortest)

# Genetic extinction rule (independent model):
# we recurse F_t with migration over generations and test H_t = 1-F_t
GENETIC_EVAL = "with_migration"  # "isolated" | "with_migration" | "best_of_both"

# Demographic uncertainty grid (annual)
R_GRID = [-0.02, 0.0, 0.03]           # mean growth per year
SIGMA_E_GRID = [0.15, 0.25]           # env. stochasticity (SD on ln scale)

# Monte Carlo for demography-with-immigration
REPLICATES = 2000
RANDOM_SEED = 42

# Scoring weights (lower score = better)
SCORE_WEIGHTS = {"genetic": 0.5, "demographic": 0.5}

# === tie demography to Ne (50/500 intuition) ===
NE_TARGET = 100.0     # below this Ne, volatility ramps up
ALPHA = 0.7           # curvature of penalty; 0.5–1 is reasonable. lower = gentler.

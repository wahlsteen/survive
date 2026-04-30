# survive_network.py
import math
import os
import re
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
import rasterio
from rasterio import features
from rasterio.windows import from_bounds
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

import matplotlib.pyplot as plt  # we'll use this later for per-patch plots
# You already import config_bufo / config_lissotriton etc.

# --- species switch ---
SPECIES = "L"   # same idea
if SPECIES == "B":
    import config_bufo as cfg
elif SPECIES == "L":
    import config_lissotriton as cfg
else:
    raise ValueError(f"Unknown SPECIES {SPECIES!r}")

# --- movement / connectivity parameters ---
MAX_DISPERSAL_DIST = 800.0        # meters
BETA_COST_DECAY    = 0.005        # 1/cost
EMIGRATION_RATE_PER_CAPITA = 0.2 # fraction of each patch leaving per year
MAX_EMIGRANTS_PER_PATCH    = 12    # hard cap per source patch per year
NMD_RASTER_PATH = "nmd_amph.tif"
USE_REAL_COST = True

LANDSCAPE_MODE = "infinite"  # "island" or "infinite"
# Settings for the infinite source
INFINITE_SOURCE_NAME = "MAINLAND"
INFINITE_SOURCE_DISTANCE_COST = 50.0  # effective cost-distance from mainland to each patch
INFINITE_SOURCE_DISPERSERS_PER_YEAR = 12.0  # constant emigrants available from mainland

# ---- bind cfg like you already do ----
NE_RATIOS                 = cfg.NE_RATIOS
GEN_TIME_YEARS            = cfg.GEN_TIME_YEARS
TIME_HORIZON_YEARS        = cfg.TIME_HORIZON_YEARS
QUASI_EXT_THRESHOLD       = cfg.QUASI_EXT_THRESHOLD
SURVIVAL_TARGET           = cfg.SURVIVAL_TARGET
DENSITY_MODEL             = cfg.DENSITY_MODEL
K                         = cfg.K
THETA                     = cfg.THETA
R_GRID                    = cfg.R_GRID
SIGMA_E_GRID              = cfg.SIGMA_E_GRID
REPLICATES                = cfg.REPLICATES
RANDOM_SEED               = cfg.RANDOM_SEED

# ---- project / biology knobs that network still needs ----
DISTURBANCE_SIGMA_FACTOR_START = 1.0
DISTURBANCE_SIGMA_FACTOR_END   = 1.0
DISTURBANCE_RELAX_YEARS        = 10
INIT_PROP_ADULT                = 1.0
INIT_FEMALE_FRAC               = 0.5

NE_RATIO_START = 0.25        # starting Ne/Nc for a young / inbred / skewed breeder pool
NE_RATIO_END   = 0.25       # long-term Ne/Nc once structure normalizes
NE_RATIO_RELAX_YEARS = 10   # how fast breeder structure recovers
H0 = cfg.H0                 # starting heterozygosity in the source metapop
H_MIN_FRAC = cfg.H_MIN_FRAC # viability threshold as fraction of H0
GEN_TIME_FOR_GENETICS = cfg.GEN_TIME_FOR_GENETICS or cfg.GEN_TIME_YEARS[0]

# --- output settings ---
RESULTS_BASE = r"C:\Users\EricWahlsteen\OneDrive - Calluna AB\Skrivbordet\Python_test\Survive\RESULTS_NETWORK"

# -----------------------------------------------------------------
# shared helpers we need in BOTH worlds (copy them here or import from survive_single):
# disturbance_sigma_multiplier, climate_modifiers, catastrophe_event
# -----------------------------------------------------------------
def make_results_dir(base_dir: str) -> str:
    """
    Create a unique run folder inside base_dir using timestamp.
    Returns the path.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(base_dir, f"run_{ts}")
    os.makedirs(outdir, exist_ok=True)
    return outdir

def disturbance_sigma_multiplier(t_year: int,
                                 start_factor: float,
                                 end_factor: float,
                                 relax_years: float) -> float:
    return end_factor + (start_factor - end_factor) * np.exp(-t_year / max(relax_years, 1e-9))

def ne_ratio_over_time(t_years: float,
                       start_ratio: float,
                       end_ratio: float,
                       relax_years: float) -> float:
    # smooth recovery of Ne/Nc through time
    return end_ratio - (end_ratio - start_ratio) * np.exp(-t_years / max(relax_years, 1e-9))

def breeder_efficiency(Nc_t, half_sat, max_eff=1.0):
    Nc_t = np.asarray(Nc_t, dtype=float)
    # replace negative/NaN/inf with 0 for safety
    Nc_t = np.where(~np.isfinite(Nc_t) | (Nc_t < 0), 0.0, Nc_t)
    denom = Nc_t + half_sat + 1e-9
    frac = np.divide(Nc_t, denom, out=np.zeros_like(Nc_t, dtype=float), where=(denom > 0))
    frac = np.clip(frac, 0.0, 1.0)
    return max_eff * frac


def climate_modifiers(t_years: int) -> dict:
    return {"r_delta": 0.0, "sigma_mult": 1.0, "K_mult": 1.0}

def catastrophe_event(rng, p_event=0.01, severity=0.9):
    if rng.random() < p_event:
        return (1.0 - severity)
    else:
        return 1.0
def genetics_from_demopaths_single_patch(
    Nc_paths_patch: np.ndarray,  # shape (replicates, T_years+1) for ONE patch
    migrants_per_year_est: float,
    rng_global,
    gen_time_years: float,
    H0: float,
    H_threshold_frac: float,
    ne_ratio_start: float,
    ne_ratio_end: float,
    ne_ratio_relax_years: float,
    K_scale: float,
):
    """
    Take the demography paths for ONE patch across replicates, and simulate
    eco-genetic drift+immigration with recovering Ne/Nc ratio.

    Returns:
      H_end_mean : mean heterozygosity at end across replicates
      pass_flag  : 1.0 if H_end_mean >= H_threshold_frac*H0
    """

    reps, T_plus1 = Nc_paths_patch.shape
    total_years = T_plus1 - 1
    gens = int(np.floor(total_years / max(gen_time_years, 1e-9)))

    # initial allele freq ~0.5
    p = np.full(reps, 0.5, dtype=float)
    scale = H0 / 0.5  # scale heterozygosity to match H0 at t=0

    # track heterozygosity over generations (not urgently needed for summary, but clear)
    for g in range(1, gens+1):
        this_year = int(g * gen_time_years)
        this_year = min(this_year, total_years)

        Nc_t = Nc_paths_patch[:, this_year].astype(float)

        # breeder structure recovery
        structural_start = ne_ratio_start * INIT_PROP_ADULT * (INIT_FEMALE_FRAC * (1.0 - INIT_FEMALE_FRAC)) * 4.0
        base_ratio = ne_ratio_over_time(
            t_years=g * gen_time_years,
            start_ratio=structural_start,
            end_ratio=ne_ratio_end,
            relax_years=ne_ratio_relax_years
        )

        abundance_factor = breeder_efficiency(
            Nc_t,
            half_sat=0.5 * K_scale,
            max_eff=1.0,
        )
        ratio_t = base_ratio * abundance_factor

        # candidate Ne
        Ne_g = ratio_t * Nc_t

        # sanitize Ne_g
        Ne_g = np.where(~np.isfinite(Ne_g) | (Ne_g < 1.0), 1.0, Ne_g)

        # migration during this generation
        migrants_this_year = migrants_per_year_est
        m_gen = migrants_this_year * gen_time_years
        with np.errstate(divide='ignore', invalid='ignore'):
            m_frac = np.where(
                Nc_t > 0,
                np.clip(m_gen / Nc_t, 0.0, 1.0),
                1.0
            )

        # gene flow: donor allele frequency ~random high-diversity source
        rng = np.random.default_rng(rng_global.integers(0, 2**32 - 1))
        donor_p = rng.uniform(0.3, 0.7, size=reps)
        p = (1.0 - m_frac) * p + m_frac * donor_p

        # drift: binomial sampling with Ne_g
        new_p = np.zeros_like(p)
        for i in range(reps):
            # guard against nonsense in allele freq
            if not np.isfinite(p[i]):
                p[i] = 0.5

            # cap Ne_g to avoid insane binomial sizes
            Ne_eff = Ne_g[i]
            if (not np.isfinite(Ne_eff)) or (Ne_eff < 1.0):
                Ne_eff = 1.0
            # ceiling: once Ne is huge, drift is basically zero anyway
            if Ne_eff > 1000.0:
                Ne_eff = 1000.0

            twoNe = 2.0 * Ne_eff  # diploid chromosome count
            if twoNe < 2.0:
                twoNe = 2.0
            if twoNe > 2000.0:
                twoNe = 2000.0  # hard cap for numpy

            twoNe_int = int(twoNe)

            new_p[i] = rng.binomial(n=twoNe_int, p=p[i]) / max(twoNe_int, 1)
        p = new_p

    # heterozygosity at the end
    H_end = (2.0 * p * (1.0 - p)) * scale
    H_end_mean = float(np.mean(H_end))

    H_threshold = H_threshold_frac * H0
    pass_flag = 1.0 if (H_end_mean >= H_threshold) else 0.0

    return H_end_mean, pass_flag

# -----------------------------------------------------------------
# network-specific data structures / math
# -----------------------------------------------------------------

@dataclass
class Patch:
    name: str
    x: float    # Easting
    y: float    # Northing
    Nc0: float  # start Nc at t=0

def load_patches_from_file(path: str) -> list[Patch]:
    """
    Expected tab/CSV file with columns:
    name    Northing    Easting    NC_METAPOP
    A       6189588     383288     50
    ...
    """
    df = pd.read_csv(path, sep=None, engine="python")
    # Normalize column names just in case:
    df.columns = [c.strip().lower() for c in df.columns]

    # try to guess column names
    # we want: name, northing (y), easting (x), nc_metapop
    name_col = "name"
    y_col    = "northing"
    x_col    = "easting"
    n_col    = "nc_metapop"

    patches = []
    for _, row in df.iterrows():
        patches.append(Patch(
            name = str(row[name_col]),
            y    = float(row[y_col]),
            x    = float(row[x_col]),
            Nc0  = float(row[n_col]),
        ))
    return patches

def build_euclid_matrix(patches: list[Patch]) -> np.ndarray:
    P = len(patches)
    D = np.zeros((P, P), dtype=float)
    for i in range(P):
        for j in range(P):
            dx = patches[i].x - patches[j].x
            dy = patches[i].y - patches[j].y
            D[i, j] = (dx**2 + dy**2) ** 0.5
    return D


def build_cost_matrix_least_cost(patches: list[Patch],
                                 friction_tif_path: str,
                                 buffer_dist: float = 2000.0) -> np.ndarray:
    """
    Compute pairwise least-cost distances between patches using a friction raster,
    but only loading a cropped window around the patches to avoid blowing up memory.

    buffer_dist: meters to extend around the convex hull of all patches so paths
                 that detour around obstacles are still captured.
                 Rule of thumb: >= MAX_DISPERSAL_DIST or a couple km.
    """

    # -------------------------------------------------
    # 1. Get bounding box of all patch coordinates
    # -------------------------------------------------
    xs = np.array([p.x for p in patches], dtype=float)
    ys = np.array([p.y for p in patches], dtype=float)

    minx = xs.min() - buffer_dist
    maxx = xs.max() + buffer_dist
    miny = ys.min() - buffer_dist
    maxy = ys.max() + buffer_dist

    with rasterio.open(friction_tif_path) as src:
        # Build window for just that bbox (in same CRS as patches!)
        win = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
        win = win.round_offsets().round_lengths()

        # Read only that window
        friction_fullres = src.read(1, window=win).astype(np.float32)
        transform_win = src.window_transform(win)
        res = src.res[0]  # assume square pixels
        nodata = src.nodata

    # clean nodata / bad values
    friction_fullres[~np.isfinite(friction_fullres) | (friction_fullres <= 0)] = 1e6

    # -------------------------------------------------
    # 2. Map each patch to row/col within this cropped window
    # -------------------------------------------------
    # Note: we must use transform_win (the transform for the window)
    rows, cols = rasterio.transform.rowcol(
        transform_win,
        xs.tolist(),
        ys.tolist(),
        op=float,
    )
    rows = np.array(rows, dtype=int)
    cols = np.array(cols, dtype=int)

    # sanity: if any patch falls outside the window (shouldn't happen), raise
    ny, nx = friction_fullres.shape
    bad = (
        (rows < 0) | (rows >= ny) |
        (cols < 0) | (cols >= nx)
    )
    if np.any(bad):
        raise RuntimeError("At least one patch fell outside cropped window. Increase buffer_dist.")

    # -------------------------------------------------
    # 3. Build 8-neighbour cost graph for JUST the cropped tile
    # -------------------------------------------------
    ny, nx = friction_fullres.shape
    # Each pixel index in the cropped tile will become a node 0..(ny*nx-1)
    indices = np.arange(ny * nx, dtype=np.int64).reshape(ny, nx)

    edges_i, edges_j, edges_w = [], [], []

    sqrt2 = np.sqrt(2.0)
    neighbors = [
        (-1,  0, res),
        ( 1,  0, res),
        ( 0, -1, res),
        ( 0,  1, res),
        (-1, -1, res*sqrt2),
        (-1,  1, res*sqrt2),
        ( 1, -1, res*sqrt2),
        ( 1,  1, res*sqrt2),
    ]

    for di, dj, step in neighbors:
        # valid "from" pixels that have a neighbor in direction (di,dj)
        i_from = np.arange(max(0, -di), min(ny, ny - di), dtype=np.int64)
        j_from = np.arange(max(0, -dj), min(nx, nx - dj), dtype=np.int64)

        if i_from.size == 0 or j_from.size == 0:
            continue

        i_to = i_from + di
        j_to = j_from + dj

        idx_from = indices[np.ix_(i_from, j_from)]
        idx_to   = indices[np.ix_(i_to,   j_to  )]

        # movement cost between cells
        cost_local = (
            friction_fullres[np.ix_(i_from, j_from)] +
            friction_fullres[np.ix_(i_to,   j_to  )]
        ) / 2.0 * step

        edges_i.append(idx_from.ravel())
        edges_j.append(idx_to.ravel())
        edges_w.append(cost_local.ravel())

    edges_i = np.concatenate(edges_i)
    edges_j = np.concatenate(edges_j)
    edges_w = np.concatenate(edges_w)

    graph = csr_matrix(
        (edges_w, (edges_i, edges_j)),
        shape=(ny*nx, ny*nx),
    )

    # -------------------------------------------------
    # 4. Run least-cost paths between patch pixels
    # -------------------------------------------------
    patch_nodes = indices[rows, cols]  # linear graph index for each patch point
    n_patches = len(patches)
    C_cost = np.full((n_patches, n_patches), np.inf, dtype=np.float64)

    for i, start_node in enumerate(patch_nodes):
        dist, _ = dijkstra(
            csgraph=graph,
            directed=False,
            indices=start_node,
            return_predecessors=True,
        )
        for j, end_node in enumerate(patch_nodes):
            if i == j:
                continue
            C_cost[i, j] = dist[end_node]

    return C_cost

def build_dummy_cost_matrix_from_distance(D_euclid: np.ndarray,
                                          road_multiplier: float = 3.0) -> np.ndarray:
    return D_euclid * road_multiplier

def compute_pairwise_matrices(patches: list[Patch],
                              C_cost_matrix: np.ndarray,
                              D_euclid_matrix: np.ndarray,
                              max_disp_dist: float,
                              beta_cost: float) -> np.ndarray:
    P = len(patches)
    Pmat = np.zeros((P, P), dtype=float)

    for i in range(P):
        for j in range(P):
            if i == j:
                continue
            d_ij = D_euclid_matrix[i, j]
            C_ij = C_cost_matrix[i, j]

            if np.isinf(C_ij):
                pij = 0.0
            elif d_ij > max_disp_dist:
                pij = 0.0
            else:
                pij = np.exp(-beta_cost * C_ij)
                pij = max(0.0, min(1.0, pij))
            Pmat[i, j] = pij
    return Pmat

def add_mainland_donor(patches: list[Patch],
                       Pmat: np.ndarray,
                       source_name: str,
                       source_cost: float,
                       beta_cost: float,
                       max_disp_dist: float,
                       dispersers_per_year: float):
    """
    Extend the patch list and Pmat with a virtual 'mainland' source population.

    Rules:
    - Mainland can send migrants to every real patch.
    - Real patches CANNOT send migrants to mainland.
    - Mainland does not need coordinates or Nc0 in the demographic loop
      because we won't simulate its demography internally; it's external.

    Implementation:
    - We'll append a new Patch with Nc0 = dispersers_per_year just for bookkeeping.
    - We'll build a new (P+1)x(P+1) Pmat_ext:
        * Pmat_ext[0:P, 0:P] = Pmat
        * Pmat_ext[mainland, mainland] = 0
        * Pmat_ext[real, mainland] = 0  (no dispersal into mainland)
        * Pmat_ext[mainland, real_j] = p_mainland_j
    - p_mainland_j uses the same exponential decay:
        p = exp(-beta_cost * source_cost)
      unless max_disp_dist == 0 or whatever.

    Returns:
        patches_ext (list[Patch] of length P+1),
        Pmat_ext (np.ndarray of shape (P+1,P+1)),
        mainland_idx (int)
    """

    P = len(patches)
    mainland_idx = P

    # clone Pmat into larger matrix
    Pmat_ext = np.zeros((P+1, P+1), dtype=float)
    Pmat_ext[0:P, 0:P] = Pmat

    # success probability from mainland to each patch j
    # we treat source_cost like distance cost; same rule as other connections
    if max_disp_dist is None:
        allow_main = 1.0
    else:
        # if you want to forbid if too far, apply logic here.
        allow_main = 1.0

    p_mainland = allow_main * np.exp(-beta_cost * source_cost)
    p_mainland = max(0.0, min(1.0, p_mainland))

    for j in range(P):
        Pmat_ext[mainland_idx, j] = p_mainland

    # no dispersal into mainland
    for i in range(P):
        Pmat_ext[i, mainland_idx] = 0.0
    Pmat_ext[mainland_idx, mainland_idx] = 0.0

    # make a Patch object so downstream code still works, but:
    # x,y can be NaN, Nc0 stores "how many emigrants it can supply per year"
    mainland_patch = Patch(
        name=source_name,
        x=np.nan,
        y=np.nan,
        Nc0=dispersers_per_year,
    )

    patches_ext = patches + [mainland_patch]
    return patches_ext, Pmat_ext, mainland_idx

def emigration_supply(Nc_vec: np.ndarray,
                      emigration_rate: float,
                      max_emigrants: float) -> np.ndarray:
    E = emigration_rate * Nc_vec
    E = np.minimum(E, max_emigrants)
    E = np.clip(E, 0.0, None)
    return E

def immigration_from_connectivity(Nc_vec: np.ndarray,
                                  Pmat: np.ndarray,
                                  emigration_rate: float,
                                  max_emigrants: float) -> np.ndarray:
    P = Nc_vec.shape[0]
    E = emigration_supply(Nc_vec, emigration_rate, max_emigrants)
    Lambda = np.zeros((P, P), dtype=float)
    for i in range(P):
        for j in range(P):
            if i == j:
                continue
            Lambda[i, j] = E[i] * Pmat[i, j]
    My_vec = np.sum(Lambda, axis=0)
    return My_vec

def immigration_with_mainland(Nc_vec: np.ndarray,
                              Pmat: np.ndarray,
                              emigration_rate: float,
                              max_emigrants: float,
                              mainland_idx: int,
                              mainland_emigrants_per_year: float) -> np.ndarray:
    """
    Same as immigration_from_connectivity, but with a special mainland node:
    - mainland_idx is the row in Pmat for the mainland.
    - mainland sends a fixed number of emigrants each year
      (mainland_emigrants_per_year), not emigration_rate * Nc.
    - mainland does not receive immigrants.
    """
    P = Nc_vec.shape[0]

    # normal patches
    E_normal = emigration_supply(Nc_vec, emigration_rate, max_emigrants)

    # overwrite mainland row supply with fixed number
    E_normal[mainland_idx] = mainland_emigrants_per_year

    Lambda = np.zeros((P, P), dtype=float)
    for i in range(P):
        for j in range(P):
            if i == j:
                continue
            Lambda[i, j] = E_normal[i] * Pmat[i, j]

    My_vec = np.sum(Lambda, axis=0)
    return My_vec


def estimate_mean_immigrants_per_patch(Nc_vec_mean: np.ndarray,
                                       Pmat: np.ndarray,
                                       emigration_rate: float,
                                       max_emigrants: float) -> np.ndarray:
    """
    Deterministic expectation:
    E_i = min(emigration_rate * Nc_i, max_emigrants)
    lambda_ij = E_i * Pmat[i,j]
    My_j = sum_i lambda_ij
    Do this using Nc_vec_mean (mean Nc across replicates and time),
    just to get a ballpark yearly immigrants rate for genetics.
    """
    E = emigration_supply(Nc_vec_mean, emigration_rate, max_emigrants)
    Lambda = E[:, None] * Pmat  # outer product, shape (P,P)
    np.fill_diagonal(Lambda, 0.0)
    My_est = np.sum(Lambda, axis=0)
    return My_est

def simulate_network_demography(
    Nc0_vec: np.ndarray,
    Q: float,
    r: float,
    sigma_e_base: float,
    T_years: int,
    Pmat: np.ndarray,
    rng: np.random.Generator,
    disturb_start: float,
    disturb_end: float,
    disturb_relax: float,
    emigration_rate: float,
    max_emigrants: float,
    K: float,
    model: str,
    theta: float,
    landscape_mode: str,
    mainland_idx: int | None,
    mainland_emigrants_per_year: float | None,
):
    P = Nc0_vec.shape[0]
    Nc_t = Nc0_vec.astype(float).copy()
    Nc_time = np.zeros((T_years + 1, P), dtype=float)
    Nc_time[0, :] = Nc_t

    for t in range(int(T_years)):
        # migration step
        if landscape_mode == "infinite":
            My_vec = immigration_with_mainland(
                Nc_vec=Nc_t,
                Pmat=Pmat,
                emigration_rate=emigration_rate,
                max_emigrants=max_emigrants,
                mainland_idx=mainland_idx,
                mainland_emigrants_per_year=mainland_emigrants_per_year,
            )
        else:
            My_vec = immigration_from_connectivity(
                Nc_vec=Nc_t,
                Pmat=Pmat,
                emigration_rate=emigration_rate,
                max_emigrants=max_emigrants,
            )

        # environment this year
        mult_t = disturbance_sigma_multiplier(
            t,
            start_factor=disturb_start,
            end_factor=disturb_end,
            relax_years=disturb_relax
        )
        climate = climate_modifiers(t)
        sigma_t = sigma_e_base * mult_t * climate["sigma_mult"]
        breeder_penalty = INIT_PROP_ADULT if t == 0 else 1.0
        r_eff = (r + climate["r_delta"]) * breeder_penalty
        K_eff = K * climate["K_mult"]

        Z = rng.standard_normal(P)

        mu_vec = np.zeros(P, dtype=float)
        for j in range(P):
            Nj = Nc_t[j]
            if model == "theta-logistic":
                dens_term = r_eff * (1.0 - ((Nj / max(K_eff, 1e-9)) ** theta))
            else:
                dens_term = r_eff * (1.0 - (Nj / max(K_eff, 1e-9)))
            mu_vec[j] = dens_term - 0.5 * (sigma_t ** 2)


        growth_term = mu_vec + sigma_t * Z
        growth_term = np.clip(growth_term, -10.0, 5.0)

        Nc_next = Nc_t * np.exp(growth_term) + My_vec

        cat_mult = catastrophe_event(rng)
        Nc_next *= cat_mult

        # clean/sanitize
        Nc_next = np.where(~np.isfinite(Nc_next) | (Nc_next < 0), 0.0, Nc_next)

        Nc_next = np.minimum(Nc_next, 5.0 * K_eff)

        Nc_t = Nc_next
        Nc_time[t+1, :] = Nc_t

    return Nc_time

def run_network_replicates(
    patches,
    Pmat,
    r,
    sigma_e,
    rng_global,
    replicates,
    T_years,
    Q,
    disturb_start,
    disturb_end,
    disturb_relax,
    emigration_rate,
    max_emigrants,
    K,
    model,
    theta,
    landscape_mode,
    mainland_idx,
    mainland_emigrants_per_year,
):
    """
    Run the whole spatial system 'replicates' times with the same (r, sigma_e),
    return:
      - Nc_paths_all: array (replicates, T_years+1, P)
      - survived_mask: array (replicates, P) with 1 if patch never dropped to/below Q
    """
    P = len(patches)
    Nc0_vec = np.array([p.Nc0 for p in patches], dtype=float)

    Nc_paths_all = np.zeros((replicates, T_years + 1, P), dtype=float)
    survived_mask = np.zeros((replicates, P), dtype=float)

    for rep in range(replicates):
        rng = np.random.default_rng(rng_global.integers(0, 2**32 - 1))

        Nc_time = simulate_network_demography(
            Nc0_vec=Nc0_vec,
            Q=Q,
            r=r,
            sigma_e_base=sigma_e,
            T_years=T_years,
            Pmat=Pmat,
            rng=rng,
            disturb_start=disturb_start,
            disturb_end=disturb_end,
            disturb_relax=disturb_relax,
            emigration_rate=emigration_rate,
            max_emigrants=max_emigrants,
            K=K,
            model=model,
            theta=theta,
            landscape_mode=landscape_mode,
            mainland_idx=mainland_idx,
            mainland_emigrants_per_year=mainland_emigrants_per_year,
        )

        Nc_paths_all[rep, :, :] = Nc_time

        # survival flag per patch = never dropped to/below Q
        survived_mask[rep, :] = (Nc_time.min(axis=0) > Q).astype(float)

    return Nc_paths_all, survived_mask

def plot_network_map(patches,
                     surv_prob,
                     Pmat,
                     results_dir,
                     title="Metapopulation network"):
    """
    Make a simple map-style plot:
    - ponds as circles colored by survival probability
    - lines between ponds with thickness ~ connectivity Pmat[i,j]
    Saves PNG to results_dir.

    Mainland (infinite donor) has x,y = NaN, so we skip plotting it.
    """

    # figure out which patches have real coordinates (not mainland)
    xs_all = np.array([p.x for p in patches], dtype=float)
    ys_all = np.array([p.y for p in patches], dtype=float)
    surv_all = np.array(surv_prob, dtype=float)

    valid_mask = np.isfinite(xs_all) & np.isfinite(ys_all)

    xs = xs_all[valid_mask]
    ys = ys_all[valid_mask]
    surv_arr = surv_all[valid_mask]

    # We'll also need to map from "compressed" plotted indices back to original indices
    # so we can pull Pmat[i,j] but skip NaN coords.
    valid_idx = np.where(valid_mask)[0]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal', adjustable='box')

    # draw pairwise connections among only valid patches
    if valid_idx.size > 1:
        # find max P among valid pairs to scale linewidths
        subP = Pmat[np.ix_(valid_idx, valid_idx)]
        finite_mask = np.isfinite(subP)
        if np.any(finite_mask):
            max_p = np.max(subP[finite_mask])
        else:
            max_p = 1.0
        if max_p <= 0:
            max_p = 1.0

        for a_i, i in enumerate(valid_idx):
            for a_j, j in enumerate(valid_idx):
                if i == j:
                    continue
                w = Pmat[i, j]
                if not np.isfinite(w) or w <= 0:
                    continue
                lw = 0.5 + 3.5 * (w / max_p)
                ax.plot([xs_all[i], xs_all[j]],
                        [ys_all[i], ys_all[j]],
                        color="gray",
                        alpha=0.4,
                        linewidth=lw,
                        zorder=1)

    # scatter the real patches
    sc = ax.scatter(xs, ys,
                    c=surv_arr,
                    cmap="inferno",
                    vmin=0.0,
                    vmax=1.0,
                    s=120,
                    edgecolor="black",
                    linewidth=0.8,
                    zorder=2)

    # labels for real patches
    for k_plot, i in enumerate(valid_idx):
        ax.text(xs_all[i] + 5, ys_all[i] + 5,
                f"{patches[i].name}\n{surv_all[i]:.2f}",
                fontsize=8,
                color="white",
                ha="left",
                va="bottom",
                bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", boxstyle="round,pad=0.2"),
                zorder=3)

    # colorbar
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Survival probability over horizon")

    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(title)

    # Bounds
    if xs.size > 0 and ys.size > 0:
        span_x = xs.max() - xs.min()
        span_y = ys.max() - ys.min()
        pad = max(20.0, 0.05 * max(span_x, span_y))
        ax.set_xlim(xs.min() - pad, xs.max() + pad)
        ax.set_ylim(ys.min() - pad, ys.max() + pad)

    out_png = os.path.join(results_dir, "map_network.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved network map to {out_png}")

def main_network():
    rng_master = np.random.default_rng(RANDOM_SEED)
    # make output dir for this run
    results_dir = make_results_dir(RESULTS_BASE)
    print(f"Results directory: {results_dir}")

    # 1. load patches
    patches = load_patches_from_file("localities.txt")
    P = len(patches)

    # 2. build distance matrices
    D_euclid = build_euclid_matrix(patches)

    if USE_REAL_COST:
        try:
            C_cost = build_cost_matrix_least_cost(
                patches,
                NMD_RASTER_PATH,
                buffer_dist=MAX_DISPERSAL_DIST * 2.0,  # generous buffer around all ponds
            )
        except NotImplementedError:
            print("WARNING: build_cost_matrix_least_cost not implemented yet, falling back to dummy cost.")
            C_cost = build_dummy_cost_matrix_from_distance(D_euclid, road_multiplier=3.0)
    else:
        C_cost = build_dummy_cost_matrix_from_distance(D_euclid, road_multiplier=3.0)

    Pmat = compute_pairwise_matrices(
        patches=patches,
        C_cost_matrix=C_cost,
        D_euclid_matrix=D_euclid,
        max_disp_dist=MAX_DISPERSAL_DIST,
        beta_cost=BETA_COST_DECAY,
    )

    mainland_idx = None
    mainland_emigrants = None

    if LANDSCAPE_MODE == "infinite":
        patches, Pmat, mainland_idx = add_mainland_donor(
            patches=patches,
            Pmat=Pmat,
            source_name=INFINITE_SOURCE_NAME,
            source_cost=INFINITE_SOURCE_DISTANCE_COST,
            beta_cost=BETA_COST_DECAY,
            max_disp_dist=MAX_DISPERSAL_DIST,
            dispersers_per_year=INFINITE_SOURCE_DISPERSERS_PER_YEAR,
        )
        mainland_emigrants = INFINITE_SOURCE_DISPERSERS_PER_YEAR
        # Note: patches list is now length P+1, and Nc0_vec in the sim will include the mainland row.
        # That's fine: mainland will drift demographically in the sim for now.
        # If we want the mainland completely static demographically, we can later freeze Nc_t[mainland_idx].

    # Pick one (r, sigma_e) combo for now.
    r = R_GRID[0]
    sigma_e = SIGMA_E_GRID[0]

    # A. Run spatial demography many times
    Nc_paths_all, survived_mask = run_network_replicates(
        patches=patches,
        Pmat=Pmat,
        r=r,
        sigma_e=sigma_e,
        rng_global=rng_master,
        replicates=REPLICATES,
        T_years=TIME_HORIZON_YEARS,
        Q=QUASI_EXT_THRESHOLD,
        disturb_start=DISTURBANCE_SIGMA_FACTOR_START,
        disturb_end=DISTURBANCE_SIGMA_FACTOR_END,
        disturb_relax=DISTURBANCE_RELAX_YEARS,
        emigration_rate=EMIGRATION_RATE_PER_CAPITA,
        max_emigrants=MAX_EMIGRANTS_PER_PATCH,
        K=K,
        model=DENSITY_MODEL,
        theta=THETA,
        landscape_mode=LANDSCAPE_MODE,
        mainland_idx=mainland_idx,
        mainland_emigrants_per_year=mainland_emigrants,
    )
    # Nc_paths_all shape: (REPLICATES, T+1, P)
    # survived_mask shape: (REPLICATES, P)

    # Survival probability per patch
    surv_prob_per_patch = survived_mask.mean(axis=0)  # length P

    # Mean Nc over time + reps for immigrant estimate
    Nc_vec_mean = Nc_paths_all.mean(axis=(0,1))  # average Nc over all reps & all years, length P
    My_est_vec = estimate_mean_immigrants_per_patch(
        Nc_vec_mean=Nc_vec_mean,
        Pmat=Pmat,
        emigration_rate=EMIGRATION_RATE_PER_CAPITA,
        max_emigrants=MAX_EMIGRANTS_PER_PATCH,
    )

    # B. Genetics for each patch using its demography paths
    patch_results = []
    for j, patch in enumerate(patches):
        Nc_paths_j = Nc_paths_all[:, :, j]  # (reps, T+1)
        migrants_est_j = My_est_vec[j]

        H_end_mean_j, pass_flag_j = genetics_from_demopaths_single_patch(
            Nc_paths_patch=Nc_paths_j,
            migrants_per_year_est=migrants_est_j,
            rng_global=rng_master,
            gen_time_years=GEN_TIME_FOR_GENETICS,
            H0=H0,
            H_threshold_frac=H_MIN_FRAC,
            ne_ratio_start=NE_RATIO_START,
            ne_ratio_end=NE_RATIO_END,
            ne_ratio_relax_years=NE_RATIO_RELAX_YEARS,
            K_scale=K,  # or NC_METAPOP-style ref size. For now use K.
        )

        patch_results.append({
            "patch": patch.name,
            "Nc_start": patch.Nc0,
            "Nc_mean": float(Nc_vec_mean[j]),
            "Nc_end_mean": float(Nc_paths_all[:, -1, j].mean()),
            "survival_prob": float(surv_prob_per_patch[j]),
            "immigrants_per_year_est": float(migrants_est_j),
            "H_end_mean": float(H_end_mean_j),
            "H_threshold_frac": float(H_MIN_FRAC),
            "genetic_viable": bool(pass_flag_j >= 1.0),
        })
    # ---- SYSTEM-LEVEL METRICS ----

    # 1. Probability that any patch persists
    # survived_mask shape: (REPLICATES, P)
    any_survive = (survived_mask.sum(axis=1) >= 1).mean()

    # 2. Probability that >=2 patches persist together (functional network)
    two_or_more_survive = (survived_mask.sum(axis=1) >= 2).mean()

    # 3. Identify main donor patch
    # We'll approximate mean emigration per patch from Nc_vec_mean
    E_mean = emigration_supply(
        Nc_vec=Nc_vec_mean,
        emigration_rate=EMIGRATION_RATE_PER_CAPITA,
        max_emigrants=MAX_EMIGRANTS_PER_PATCH,
    )
    top_donor_idx = int(np.argmax(E_mean))
    top_donor_name = patches[top_donor_idx].name
    top_donor_rate = float(E_mean[top_donor_idx])

    system_row = {
        "any_patch_survives_prob": float(any_survive),
        "two_or_more_patches_survive_prob": float(two_or_more_survive),
        "top_donor_patch": top_donor_name,
        "top_donor_emigration_rate_est": top_donor_rate,
    }
    df_system = pd.DataFrame([system_row])
    print("\n=== System-level summary ===")
    print(df_system.to_string(index=False))

    df_out = pd.DataFrame(patch_results)
    print("\n=== Network summary per patch ===")
    print(df_out.to_string(index=False))

    # Optional: save to CSV for now (mirror your single-pop habit of writing outputs)
    out_csv = os.path.join(results_dir, "network_patch_summary.csv")
    df_out.to_csv(out_csv, index=False)

    system_csv = os.path.join(results_dir, "network_system_summary.csv")
    df_system.to_csv(system_csv, index=False)

    print(f"\nSaved patch summary to {out_csv}")
    print(f"Saved system summary to {system_csv}")
    # Map: visualize network
    plot_network_map(
        patches=patches,
        surv_prob=surv_prob_per_patch,
        Pmat=Pmat,
        results_dir=results_dir,
        title="Metapopulation survival & connectivity",
    )


if __name__ == "__main__":
    main_network()

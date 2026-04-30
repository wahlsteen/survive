# survive_network.py
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
import rasterio
from rasterio import features
from rasterio.windows import from_bounds
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from pyproj import Transformer
import folium
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

# --- species switch ---
SPECIES = "L"   # L=Lissotriton; B=Bufo 
if SPECIES == "B":
    import config_bufo as cfg
elif SPECIES == "L":
    import config_lissotriton as cfg
else:
    raise ValueError(f"Unknown SPECIES {SPECIES!r}")

PATCH_FILE = r"C:\Users\EricWahlsteen\OneDrive - Calluna AB\Skrivbordet\Python_test\Survive\localitiesVON0022.txt"

# --- movement / connectivity parameters ---
MAX_DISPERSAL_DIST = 600.0        # meters
EMIGRATION_RATE_PER_CAPITA = 0.1 # fraction of each patch leaving per year
MAX_EMIGRANTS_PER_PATCH    = 20    # hard cap per source patch per year
NMD_RASTER_PATH = "nmd_amph.tif" #nmd_amph.tif or None 
USE_REAL_COST = True

LANDSCAPE_MODE = "island"  # "island" or "infinite"
# Settings for the infinite source
INFINITE_SOURCE_NAME = "MAINLAND"
INFINITE_SOURCE_DISTANCE_COST = 50.0  # effective cost-distance from mainland to each patch
INFINITE_SOURCE_DISPERSERS_PER_YEAR = 10.0  # constant emigrants available from mainland

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

NE_RATIO_START = 0.2        # starting Ne/Nc for a young / inbred / skewed breeder pool
NE_RATIO_END   = 0.2     # long-term Ne/Nc once structure normalizes
NE_RATIO_RELAX_YEARS = 20   # how fast breeder structure recovers
H0 = cfg.H0                 # starting heterozygosity in the source metapop
H_MIN_FRAC = cfg.H_MIN_FRAC # viability threshold as fraction of H0
GEN_TIME_FOR_GENETICS = cfg.GEN_TIME_FOR_GENETICS or cfg.GEN_TIME_YEARS[0]
# genetics scaling knobs
BREEDER_HALF_SAT_FRAC = 0.2  # was implicitly 0.5; try 0.2 first, # ~20% of K to get near-max effective breeders
# catastrophe_event(rng, p_event=0.01, severity=0.5)
BARRIER_SHP = "None" # None or barrier.shp (name of shapefile with barriers) 

# --- output settings ---
RESULTS_BASE = r"C:\Users\EricWahlsteen\OneDrive - Calluna AB\Skrivbordet\Python_test\Survive\RESULTS_NETWORK"

# -----------------------------------------------------------------
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
            half_sat=BREEDER_HALF_SAT_FRAC * K_scale,
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
import json
from textwrap import indent

def _jsonify(obj):
    """Make numpy/scalars serializable for JSON."""
    try:
        import numpy as _np  # local import to avoid global dependency in json dumps
        if isinstance(obj, (_np.integer,)):
            return int(obj)
        if isinstance(obj, (_np.floating,)):
            return float(obj)
        if isinstance(obj, (_np.ndarray,)):
            return obj.tolist()
    except Exception:
        pass
    if isinstance(obj, (set,)):
        return sorted(list(obj))
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8", errors="replace")
    # NYTT: om något råkar vara en funktion/klass/objekt → repr-sträng
    if callable(obj):
        return repr(obj)
    # Sista utväg: str()
    return str(obj)

def collect_cfg_params(cfg_module):
    """
    Grab ALL UPPERCASE attributes from the active species config module.
    """
    out = {}
    for k, v in vars(cfg_module).items():
        if k.isupper():
            out[k] = v
    return out

def collect_main_params():
    """
    Grab key knobs defined in the main script (globals).
    Keep it explicit so it's stable and readable.
    """
    return {
        # species/source
        "SPECIES": SPECIES,
        # movement / connectivity
        "MAX_DISPERSAL_DIST": MAX_DISPERSAL_DIST,
        "EMIGRATION_RATE_PER_CAPITA": EMIGRATION_RATE_PER_CAPITA,
        "MAX_EMIGRANTS_PER_PATCH": MAX_EMIGRANTS_PER_PATCH,
        "NMD_RASTER_PATH": NMD_RASTER_PATH,
        "USE_REAL_COST": USE_REAL_COST,
        "LANDSCAPE_MODE": LANDSCAPE_MODE,
        "INFINITE_SOURCE_NAME": INFINITE_SOURCE_NAME,
        "INFINITE_SOURCE_DISTANCE_COST": INFINITE_SOURCE_DISTANCE_COST,
        "INFINITE_SOURCE_DISPERSERS_PER_YEAR": INFINITE_SOURCE_DISPERSERS_PER_YEAR,
        # disturbance / genetics scaffolding in this file
        "DISTURBANCE_SIGMA_FACTOR_START": DISTURBANCE_SIGMA_FACTOR_START,
        "DISTURBANCE_SIGMA_FACTOR_END": DISTURBANCE_SIGMA_FACTOR_END,
        "DISTURBANCE_RELAX_YEARS": DISTURBANCE_RELAX_YEARS,
        "INIT_PROP_ADULT": INIT_PROP_ADULT,
        "INIT_FEMALE_FRAC": INIT_FEMALE_FRAC,
        "NE_RATIO_START": NE_RATIO_START,
        "NE_RATIO_END": NE_RATIO_END,
        "NE_RATIO_RELAX_YEARS": NE_RATIO_RELAX_YEARS,
        "H0": H0,
        "H_MIN_FRAC": H_MIN_FRAC,
        "GEN_TIME_FOR_GENETICS": GEN_TIME_FOR_GENETICS,
        "BREEDER_HALF_SAT_FRAC": BREEDER_HALF_SAT_FRAC,
        "BARRIER_SHP": BARRIER_SHP,
        "RESULTS_BASE": RESULTS_BASE,
        # grids / seeds bound from cfg in this file (kept for convenience)
        "NE_RATIOS": NE_RATIOS,
        "GEN_TIME_YEARS": GEN_TIME_YEARS,
        "TIME_HORIZON_YEARS": TIME_HORIZON_YEARS,
        "QUASI_EXT_THRESHOLD": QUASI_EXT_THRESHOLD,
        "SURVIVAL_TARGET": SURVIVAL_TARGET,
        "DENSITY_MODEL": DENSITY_MODEL,
        "K": K,
        "THETA": THETA,
        "R_GRID": R_GRID,
        "SIGMA_E_GRID": SIGMA_E_GRID,
        "REPLICATES": REPLICATES,
        "RANDOM_SEED": RANDOM_SEED,
    }

def write_run_manifest(
    results_dir: str,
    cfg_module,
    patch_file_path: str,
    patches: list,
    derived: dict | None = None
):
    """
    Write a complete manifest of this run:
      - timestamp, script name
      - which cfg module, and all its UPPERCASE values
      - main-script parameters
      - patch file path + dump of the patch table used
      - derived values (beta_cost, connectivity summary, etc.)
      - JSON + pretty TXT
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1) dump patches used as CSV (exact inputs that entered the model)
    import csv
    patches_csv = os.path.join(results_dir, "patches_used.csv")
    with open(patches_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=",")
        w.writerow(["name", "Easting", "Northing", "Nc0"])
        for p in patches:
            w.writerow([p.name, p.x, p.y, p.Nc0])

    # 2) build manifest dict
    manifest = {
        "timestamp": now,
        "script": os.path.basename(__file__) if "__file__" in globals() else "survive_network_2.py",
        "species_config_module": getattr(cfg_module, "__name__", str(cfg_module)),
        "patch_file_path": patch_file_path,
        "outputs_dir": results_dir,
        "config_module_params": collect_cfg_params(cfg_module),
        "main_script_params": collect_main_params(),
        "derived": derived or {},
        "artifacts": {
            "patches_used_csv": os.path.abspath(patches_csv),
        },
    }

    # 3) JSON
    json_path = os.path.join(results_dir, "run_manifest.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, default=_jsonify, indent=2, ensure_ascii=False)

    # 4) Human-readable TXT
    txt_path = os.path.join(results_dir, "run_manifest.txt")
    def _pp(d, pad=""):
        lines = []
        for k in sorted(d.keys()):
            v = d[k]
            if isinstance(v, dict):
                lines.append(f"{pad}{k}:")
                lines.append(_pp(v, pad + "  "))
            else:
                lines.append(f"{pad}{k}: {v}")
        return "\n".join(lines)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== Survive-Network Run Manifest ===\n")
        f.write(f"time: {now}\n")
        f.write(f"script: {manifest['script']}\n")
        f.write(f"species_config_module: {manifest['species_config_module']}\n")
        f.write(f"patch_file_path: {patch_file_path}\n")
        f.write(f"outputs_dir: {results_dir}\n\n")
        f.write("[config_module_params]\n")
        f.write(indent(_pp(manifest["config_module_params"]), "  "))
        f.write("\n\n[main_script_params]\n")
        f.write(indent(_pp(manifest["main_script_params"]), "  "))
        f.write("\n\n[derived]\n")
        f.write(indent(_pp(manifest["derived"]), "  "))
        f.write("\n\n[artifacts]\n")
        f.write(indent(_pp(manifest["artifacts"]), "  "))
        f.write("\n")

    print(f"Saved run manifest:\n  - {json_path}\n  - {txt_path}\n  - {patches_csv}")

def genetics_time_series_for_patch(
    Nc_paths_patch: np.ndarray,   # shape (replicates, T_years+1)
    migrants_per_year_est: float,
    rng_global: np.random.Generator,
    gen_time_years: float,
    H0: float,
    H_threshold_frac: float,
    ne_ratio_start: float,
    ne_ratio_end: float,
    ne_ratio_relax_years: float,
    K_scale: float,
):
    """
    Returns (years, H_mean_series) where:
      years: np.array of generation checkpoints in years (0, G, 2G, ...)
      H_mean_series: mean heterozygosity across replicates at each checkpoint
    """
    reps, T_plus1 = Nc_paths_patch.shape
    total_years = T_plus1 - 1
    gens = int(np.floor(total_years / max(gen_time_years, 1e-9)))
    if gens < 1:
        # trivial horizon; just return start H
        return np.array([0.0]), np.array([H0], dtype=float)

    # initial allele freq ~ 0.5, scale ensures H(t=0) = H0
    p = np.full(reps, 0.5, dtype=float)
    scale = H0 / 0.5

    years = np.arange(0, gens + 1, dtype=int) * gen_time_years
    years[-1] = min(years[-1], total_years)

    # store H (include t=0)
    H_series = [float(np.mean(2.0 * p * (1.0 - p) * scale))]

    for g in range(1, gens + 1):
        this_year = int(g * gen_time_years)
        this_year = min(this_year, total_years)

        Nc_t = Nc_paths_patch[:, this_year].astype(float)

        structural_start = ne_ratio_start * INIT_PROP_ADULT * (INIT_FEMALE_FRAC * (1.0 - INIT_FEMALE_FRAC)) * 4.0
        base_ratio = ne_ratio_over_time(
            t_years=g * gen_time_years,
            start_ratio=structural_start,
            end_ratio=ne_ratio_end,
            relax_years=ne_ratio_relax_years
        )
        abundance_factor = breeder_efficiency(
            Nc_t,
            half_sat=BREEDER_HALF_SAT_FRAC * K_scale,
            max_eff=1.0,
        )
        ratio_t = base_ratio * abundance_factor

        Ne_g = ratio_t * Nc_t
        Ne_g = np.where(~np.isfinite(Ne_g) | (Ne_g < 1.0), 1.0, Ne_g)

        # migrants per generation
        m_gen = migrants_per_year_est * gen_time_years
        with np.errstate(divide='ignore', invalid='ignore'):
            m_frac = np.where(Nc_t > 0, np.clip(m_gen / Nc_t, 0.0, 1.0), 1.0)

        rng = np.random.default_rng(rng_global.integers(0, 2**32 - 1))
        donor_p = rng.uniform(0.3, 0.7, size=reps)
        p = (1.0 - m_frac) * p + m_frac * donor_p

        # drift
        new_p = np.zeros_like(p)
        for i in range(reps):
            pi = p[i] if np.isfinite(p[i]) else 0.5
            Ne_eff = Ne_g[i]
            if (not np.isfinite(Ne_eff)) or (Ne_eff < 1.0):
                Ne_eff = 1.0
            if Ne_eff > 1000.0:
                Ne_eff = 1000.0
            twoNe = int(np.clip(2.0 * Ne_eff, 2.0, 2000.0))
            new_p[i] = rng.binomial(n=twoNe, p=pi) / max(twoNe, 1)
        p = new_p

        H_series.append(float(np.mean(2.0 * p * (1.0 - p) * scale)))

    return years.astype(float), np.array(H_series, dtype=float)



import fiona
from shapely.geometry import shape, mapping
from rasterio.features import rasterize

def rasterize_barriers_to_multiplier(
    barrier_shp_path: str,
    transform_win,
    out_shape,                  # (ny, nx) of the cropped friction window
    default_mult: float = 1.0,  # where no barrier: multiply by this
    line_buffer_m: float = 5.0, # buffer lines so they become visible at raster scale
    attr_mult_field: str | None = None,  # e.g. "cost_mult" in the shapefile
    fallback_barrier_mult: float = 10.0  # used if attr not present
) -> np.ndarray:
    """
    Returns a multiplier raster (same size as friction window).
    Pixels covered by barrier geometry get a multiplier (from attribute or fallback),
    others get default_mult (usually 1.0).
    """
    ny, nx = out_shape
    if not os.path.exists(barrier_shp_path):
        return np.full((ny, nx), default_mult, dtype=np.float32)

    # Read all features and build (geom, value) pairs in raster coords
    geoms_vals = []
    with fiona.open(barrier_shp_path, "r") as src:
        # assume barrier CRS matches the friction raster; reproject if needed
        for feat in src:
            g = shape(feat["geometry"])
            # If it's a LineString/Multiline, buffer so it covers pixels:
            if g.geom_type in ("LineString", "MultiLineString"):
                g = g.buffer(line_buffer_m, cap_style=2, join_style=2)
            # Attribute-based multiplier?
            if attr_mult_field is not None and attr_mult_field in (feat["properties"] or {}):
                val = float(feat["properties"][attr_mult_field])
                if not np.isfinite(val) or val <= 0:
                    val = fallback_barrier_mult
            else:
                val = fallback_barrier_mult
            geoms_vals.append((mapping(g), val))

    if not geoms_vals:
        return np.full((ny, nx), default_mult, dtype=np.float32)

    # Rasterize: burn the multiplier value where barrier exists; elsewhere default_mult
    mult_raster = rasterize(
        shapes=geoms_vals,
        out_shape=(ny, nx),
        transform=transform_win,
        fill=default_mult,
        dtype="float32",
        all_touched=True
    )
    # Clean
    mult_raster[~np.isfinite(mult_raster)] = default_mult
    mult_raster = np.maximum(mult_raster, 0.0)
    return mult_raster

def calibrate_beta_from_costs(C_cost, target_P=0.3, quantile=0.25):
    """
    Choose β so that exp(-β * C*) ≈ target_P at a representative cost C*.
    We use a lower quantile of non-inf costs as 'typical near-neighbor' cost.
    """
    costs = C_cost[np.isfinite(C_cost) & (C_cost > 0)]
    if costs.size == 0:
        return 0.001  # fallback
    C_star = np.quantile(costs, quantile)
    C_star = max(C_star, 1e-9)
    beta = -np.log(target_P) / C_star
    return float(beta)

def project_to_wgs84(xs_m, ys_m, epsg_from="EPSG:3006"):
    """
    Convert arrays of coordinates in a projected CRS (meters)
    to lat/lon (WGS84).
    Returns (lats, lons) as numpy arrays.
    """
    transformer = Transformer.from_crs(epsg_from, "EPSG:4326", always_xy=True)
    lons, lats = transformer.transform(xs_m, ys_m)
    return np.array(lats), np.array(lons)

import matplotlib.cm as cm
import matplotlib.colors as mcolors

def survival_to_hexcolor(values, vmin=0.0, vmax=1.0):
    """
    Map survival probabilities (0-1) to hex colors smoothly
    from red (0) → yellow (0.5) → green (1).
    Returns list of "#RRGGBB" strings.
    """
    # Define a custom continuous colormap
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "green_yellow_red", [(0.0, "red"), (0.5, "yellow"), (1.0, "green")]
    )
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    rgba = cmap(norm(values))
    return np.array([mcolors.to_hex((r, g, b), keep_alpha=False) for (r, g, b, a) in rgba])

def add_barrier_overlay_to_folium(fmap, barrier_shp_path, color="#ff4444", weight=3, opacity=0.8):
    """
    Overlay the barrier shapefile on a Folium map for visualization.
    Assumes barrier shapefile uses the same CRS as the patches (EPSG:3006).
    """
    if not barrier_shp_path or not os.path.exists(barrier_shp_path):
        return
    import fiona
    from shapely.geometry import shape
    for feat in fiona.open(barrier_shp_path, "r"):
        g = shape(feat["geometry"])
        # handle polygons and lines
        if g.geom_type == "Polygon":
            coords = list(g.exterior.coords)
        elif g.geom_type in ("LineString", "LinearRing"):
            coords = list(g.coords)
        else:
            continue
        xs = np.array([c[0] for c in coords])
        ys = np.array([c[1] for c in coords])
        lats, lons = project_to_wgs84(xs, ys, epsg_from="EPSG:3006")
        folium.PolyLine(
            locations=list(zip(lats.tolist(), lons.tolist())),
            color=color,
            weight=weight,
            opacity=opacity,
        ).add_to(fmap)


def export_arcgis_map(patches, surv_prob, Pmat, results_dir, 
                      tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                      attr="Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
                      paths_xy=None):
    """
    Generate a satellite-style Folium map (ArcGIS World Imagery) with patches
    colored by survival probability and dispersal links scaled by Pmat.
    Saves both HTML and PNG in results_dir.
    """

    # Filter finite coordinates (ignore mainland NaN)
    xs = np.array([p.x for p in patches], dtype=float)
    ys = np.array([p.y for p in patches], dtype=float)
    surv_arr = np.array(surv_prob, dtype=float)
    names = [p.name for p in patches]
    valid_mask = np.isfinite(xs) & np.isfinite(ys)
    if not np.any(valid_mask):
        print("No finite coordinates for mapping.")
        return

    xs, ys, surv_arr = xs[valid_mask], ys[valid_mask], surv_arr[valid_mask]
    names = [n for (n, v) in zip(names, valid_mask) if v]

    # Reproject to lat/lon
    lats, lons = project_to_wgs84(xs, ys, epsg_from="EPSG:3006")

    # Colorize survival probs
    colors = survival_to_hexcolor(surv_arr)

    # Center map
    center_lat, center_lon = np.mean(lats), np.mean(lons)

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles=tiles,
        attr=attr,
        control_scale=True,
    )

    # Draw dispersal lines (gray, thickness = Pmat)
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) > 1:
        subP = Pmat[np.ix_(valid_idx, valid_idx)]
        max_p = np.nanmax(subP[np.isfinite(subP)]) or 1.0
        for a_i, i in enumerate(valid_idx):
            for a_j, j in enumerate(valid_idx):
                if i == j:
                    continue
                w = Pmat[i, j]
                if not np.isfinite(w) or w <= 0:
                    continue
                weight = 1.0 + 5.0 * (w / max_p)
                lat_i, lon_i = project_to_wgs84(np.array([patches[i].x]), np.array([patches[i].y]))
                lat_j, lon_j = project_to_wgs84(np.array([patches[j].x]), np.array([patches[j].y]))
                folium.PolyLine(
                    locations=[(lat_i[0], lon_i[0]), (lat_j[0], lon_j[0])],
                    color="steelblue",
                    weight=weight,
                    opacity=0,
                ).add_to(fmap)
    # Optional: real least-cost polylines
    if paths_xy:
        # compute max_p for linewidth scaling
        max_p = np.nanmax(Pmat[np.isfinite(Pmat)]) if np.isfinite(Pmat).any() else 1.0
        max_p = max(max_p, 1e-9)
        for (i, j), xy in paths_xy.items():
            pij = Pmat[i, j]
            if not np.isfinite(pij) or pij <= 0:
                continue
            # reproject polyline to lat/lon
            xs_line = np.array([x for x, y in xy])
            ys_line = np.array([y for x, y in xy])
            lat_line, lon_line = project_to_wgs84(xs_line, ys_line, epsg_from="EPSG:3006")
            weight = 1.0 + 5.0 * (pij / max_p)
            folium.PolyLine(
                locations=list(zip(lat_line.tolist(), lon_line.tolist())),
                color="#00BFFF",   # deepskyblue
                weight=weight,
                opacity=0.7,
            ).add_to(fmap)

    # Add patch circles
    for name, lat, lon, color, sp in zip(names, lats, lons, colors, surv_arr):
        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            color="black",
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=f"{name}: {sp:.2f}",
            tooltip=f"{name}: {sp:.2f}",
        ).add_to(fmap)
    add_barrier_overlay_to_folium(fmap, BARRIER_SHP)
    html_path = os.path.join(results_dir, "map_network_arcgis.html")
    fmap.save(html_path)
    print(f"Saved ArcGIS map HTML to {html_path}")

    # Screenshot map to PNG
    options = Options()
    options.headless = True
    options.add_argument("--window-size=1200,1000")
    driver = webdriver.Chrome(options=options)

    try:
        driver.get("file://" + os.path.abspath(html_path))
        time.sleep(4)
        png_path = os.path.join(results_dir, "map_network_arcgis.png")
        driver.save_screenshot(png_path)
        print(f"Saved ArcGIS map PNG to {png_path}")
    finally:
        driver.quit()

def export_folium_map(patches,
                      surv_prob,
                      Pmat,
                      results_dir,
                      map_tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                      map_attr="© OpenStreetMap contributors"):
    """
    Create a pretty basemap (Leaflet/folium) with:
      - background tiles (can be satellite if you swap tiles URL),
      - ponds as circles colored by survival probability (inferno),
      - labels with name + survival prob,
      - dispersal links as polylines with thickness ~ Pmat[i,j].

    Saves HTML and PNG into results_dir.
    """

    # 1. Filter out donor 'MAINLAND' which has NaN coords
    xs_all = np.array([p.x for p in patches], dtype=float)
    ys_all = np.array([p.y for p in patches], dtype=float)
    surv_all = np.array(surv_prob, dtype=float)
    names_all = [p.name for p in patches]

    valid_mask = np.isfinite(xs_all) & np.isfinite(ys_all)
    xs = xs_all[valid_mask]
    ys = ys_all[valid_mask]
    surv_arr = surv_all[valid_mask]
    names = [names_all[i] for i in np.where(valid_mask)[0]]

    # bail out if nothing to plot
    if xs.size == 0:
        print("No finite patch coordinates to map.")
        return

    # 2. Reproject to lat/lon for folium (assume EPSG:3006 -> EPSG:4326)
    lats, lons = project_to_wgs84(xs, ys, epsg_from="EPSG:3006")

    # 3. Pick a center for the map (mean of sites)
    center_lat = float(np.mean(lats))
    center_lon = float(np.mean(lons))

    # 4. Survival colors
    colors = survival_to_hexcolor(surv_arr)

    # 5. Build folium map
    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,  # adjust to taste
        tiles=map_tiles,
        attr=map_attr,
        control_scale=True,
    )

    # 6. Draw dispersal links between valid patches
    # We'll again restrict to valid_idx pairs, and reproject each segment.
    valid_idx = np.where(valid_mask)[0]

    # compute max connection prob among valid patches for linewidth scaling
    if len(valid_idx) > 1:
        subP = Pmat[np.ix_(valid_idx, valid_idx)]
        finite_mask = np.isfinite(subP)
        if np.any(finite_mask):
            max_p = float(np.max(subP[finite_mask]))
        else:
            max_p = 1.0
        if max_p <= 0:
            max_p = 1.0
    else:
        max_p = 1.0

    for a_i, i in enumerate(valid_idx):
        for a_j, j in enumerate(valid_idx):
            if i == j:
                continue
            w = Pmat[i, j]
            if not np.isfinite(w) or w <= 0:
                continue
            # linewidth scaling (roughly 1-6 px)
            weight = 1.0 + 5.0 * (w / max_p)
            # segment endpoints in lat/lon
            lat_i, lon_i = project_to_wgs84(
                np.array([xs_all[i]]),
                np.array([ys_all[i]]),
                epsg_from="EPSG:3006",
            )
            lat_j, lon_j = project_to_wgs84(
                np.array([xs_all[j]]),
                np.array([ys_all[j]]),
                epsg_from="EPSG:3006",
            )
            folium.PolyLine(
                locations=[(lat_i[0], lon_i[0]), (lat_j[0], lon_j[0])],
                color="gray",
                weight=weight,
                opacity=0.5,
            ).add_to(fmap)

    # 7. Draw patches
    for k, (lat, lon, col_hex, name, sp) in enumerate(zip(lats, lons, colors, names, surv_arr)):
        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            color="black",
            weight=1,
            fill=True,
            fill_color=col_hex,
            fill_opacity=0.9,
            popup=f"{name}: survival {sp:.2f}",
            tooltip=f"{name}: {sp:.2f}",
        ).add_to(fmap)

        # optional label as a small text-like marker
        folium.map.Marker(
            [lat, lon],
            icon=folium.DivIcon(
                html=f"""<div style="
                        font-size:10px;
                        color:white;
                        background-color:rgba(0,0,0,0.5);
                        padding:2px 4px;
                        border-radius:3px;
                        border:1px solid black;">
                        {name} {sp:.2f}
                        </div>"""
            ),
        ).add_to(fmap)

    # 8. Save HTML
    add_barrier_overlay_to_folium(fmap, BARRIER_SHP)
    html_path = os.path.join(results_dir, "map_network.html")
    fmap.save(html_path)
    print(f"Saved folium map HTML to {html_path}")

    # 9. Screenshot the map to PNG using headless Chrome
    png_path = os.path.join(results_dir, "map_network.png")

    options = Options()
    options.headless = True
    options.add_argument("--window-size=1200,1200")
    driver = webdriver.Chrome(options=options)

    try:
        driver.get("file://" + os.path.abspath(html_path))
        time.sleep(4)  # give tiles time to load
        driver.save_screenshot(png_path)
        print(f"Saved folium map PNG to {png_path}")
    finally:
        driver.quit()

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


def build_cost_matrix_least_cost(
    patches: list[Patch],
    friction_tif_path: str,
    buffer_dist: float = 4000.0,
    barrier_shp_path: str | None = None,    # NEW
    barrier_mode: str = "multiply",         # "multiply" or "block"
    barrier_value: float = 10.0,            # multiplier if multiply; ignored for block
    line_buffer_m: float = 5.0,             # for line barriers
    attr_mult_field: str | None = None      # e.g., "cost_mult" in shapefile
) -> np.ndarray:
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
    if barrier_shp_path:
        barrier_mult = rasterize_barriers_to_multiplier(
            barrier_shp_path=barrier_shp_path,
            transform_win=transform_win,
            out_shape=friction_fullres.shape,
            default_mult=1.0,
            line_buffer_m=line_buffer_m,
            attr_mult_field=attr_mult_field,
            fallback_barrier_mult=barrier_value
        )
        if barrier_mode == "multiply":
            friction_fullres = friction_fullres * barrier_mult
        elif barrier_mode == "block":
            # turn any barrier pixel into “nearly impassable”
            block_mask = barrier_mult > 1.0
            friction_fullres[block_mask] = 1e9
        else:
            raise ValueError(f"Unknown barrier_mode: {barrier_mode}")
        
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

    # We ALSO return the cropped friction raster + its transform
    # so we can use it as a basemap for plotting.
    patch_rc = np.column_stack([rows, cols])  # shape (P, 2) with (row, col)
    return C_cost, friction_fullres, transform_win, patch_rc

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

def plot_network_map(patches, surv_prob, Pmat, results_dir, title="Metapopulation network", paths_xy=None):
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
    valid_idx = np.where(valid_mask)[0]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal', adjustable='box')

    # --- draw least-cost paths first (so nodes are on top) ---
    if paths_xy:
        max_p = np.nanmax(Pmat[np.isfinite(Pmat)]) if np.isfinite(Pmat).any() else 1.0
        max_p = max(max_p, 1e-9)
        for (i, j), xy in paths_xy.items():
            pij = Pmat[i, j]
            if not np.isfinite(pij) or pij <= 0:
                continue
            lw = 0.5 + 3.5 * (pij / max_p)
            ax.plot(
                [x for x, y in xy],
                [y for x, y in xy],
                color="deepskyblue",
                alpha=0.7,
                linewidth=lw,
                zorder=1,
            )

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
                    cmap=mcolors.LinearSegmentedColormap.from_list(
                        "green_yellow_red", [(0.0, "red"), (0.5, "yellow"), (1.0, "green")]
                    ),
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

def plot_connectivity_heatmap(Pmat, patches, results_dir, label="current"):
    """
    Plot a heatmap of the connectivity matrix (Pmat) and save as PNG.
    Rows and columns are patches in the same order as 'patches'.
    """
    import seaborn as sns  # lightweight visual
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    names = [p.name for p in patches]
    P = np.array(Pmat, dtype=float)
    P = np.where(np.isfinite(P), P, 0.0)  # replace inf/nan with 0

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        P,
        ax=ax,
        cmap="viridis",
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={"label": "Connection probability (Pij)"},
        xticklabels=names,
        yticklabels=names,
    )
    ax.set_title(f"Connectivity matrix ({label})")
    ax.set_xlabel("Receiving patch")
    ax.set_ylabel("Source patch")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    out_path = os.path.join(results_dir, f"connectivity_heatmap.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved connectivity heatmap → {out_path}")

def extract_least_cost_paths_for_top_links(
    friction_win: np.ndarray,
    transform_win,
    patch_rc: np.ndarray,      # shape (P,2): (row,col) per patch within the window
    Pmat: np.ndarray,
    top_frac: float = 1,    # keep top 20% strongest connections
    min_prob: float | None = None,
    max_pairs: int = 50        # hard cap to keep it quick
):
    """
    Compute least-cost polylines for the strongest subset of links.

    Returns: dict {(i,j): [(x0,y0),(x1,y1),...]} in map coords (same CRS as raster).
    """

    # 0) Guard
    if friction_win is None or transform_win is None or patch_rc is None:
        return {}

    ny, nx = friction_win.shape
    res = abs(transform_win.a)  # pixel size (assumes square pixels)

    # 1) choose candidate pairs by probability
    P = Pmat.shape[0]
    pairs = []
    for i in range(P):
        for j in range(P):
            if i == j:
                continue
            pij = Pmat[i, j]
            if not np.isfinite(pij) or pij <= 0:
                continue
            pairs.append((pij, i, j))
    if not pairs:
        return {}

    pairs.sort(reverse=True)  # by pij descending
    if min_prob is not None:
        pairs = [(p,i,j) for (p,i,j) in pairs if p >= min_prob]
    else:
        k = max(1, int(len(pairs) * top_frac))
        pairs = pairs[:k]
    if len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]

    # 2) Build 8-neighbour graph ONCE from friction_win
    indices = np.arange(ny * nx, dtype=np.int64).reshape(ny, nx)
    edges_i, edges_j, edges_w = [], [], []
    sqrt2 = np.sqrt(2.0)
    neighbors = [(-1,0,res),(1,0,res),(0,-1,res),(0,1,res),
                 (-1,-1,res*sqrt2),(-1,1,res*sqrt2),(1,-1,res*sqrt2),(1,1,res*sqrt2)]
    F = friction_win.astype(np.float32).copy()
    F[~np.isfinite(F) | (F <= 0)] = 1e6

    for di, dj, step in neighbors:
        i_from = np.arange(max(0, -di), min(ny, ny - di), dtype=np.int64)
        j_from = np.arange(max(0, -dj), min(nx, nx - dj), dtype=np.int64)
        if i_from.size == 0 or j_from.size == 0:
            continue
        i_to = i_from + di
        j_to = j_from + dj
        idx_from = indices[np.ix_(i_from, j_from)]
        idx_to   = indices[np.ix_(i_to,   j_to  )]
        cost = (F[np.ix_(i_from, j_from)] + F[np.ix_(i_to, j_to)]) / 2.0 * step
        edges_i.append(idx_from.ravel()); edges_j.append(idx_to.ravel()); edges_w.append(cost.ravel())

    edges_i = np.concatenate(edges_i); edges_j = np.concatenate(edges_j); edges_w = np.concatenate(edges_w)
    graph = csr_matrix((edges_w, (edges_i, edges_j)), shape=(ny*nx, ny*nx))

    # 3) Run Dijkstra per unique source in pairs (return predecessors)
    from collections import defaultdict
    pairs_by_src = defaultdict(list)
    for pij, i, j in pairs:
        pairs_by_src[i].append((pij, j))

    # mapping (i,j) -> list of (x,y)
    paths_xy = {}

    for i_src, lst in pairs_by_src.items():
        r0, c0 = patch_rc[i_src]
        start_node = indices[r0, c0]
        dist, pred = dijkstra(csgraph=graph, directed=False, indices=start_node, return_predecessors=True)

        for pij, j_tgt in lst:
            r1, c1 = patch_rc[j_tgt]
            end_node = indices[r1, c1]
            if not np.isfinite(dist[end_node]):
                continue  # no path

            # backtrack
            path_nodes = []
            cur = end_node
            while cur != -9999 and cur != start_node:
                path_nodes.append(cur)
                cur = pred[cur]
            path_nodes.append(start_node)
            path_nodes.reverse()

            # convert nodes -> (row,col) -> (x,y)
            rc = np.column_stack(np.unravel_index(np.array(path_nodes, dtype=np.int64), (ny, nx)))
            # pixel center coords
            # x = a*col + b*row + c; y = d*col + e*row + f
            a,b,c,d,e,f = transform_win.a, transform_win.b, transform_win.c, transform_win.d, transform_win.e, transform_win.f
            rows = rc[:,0].astype(float) + 0.5  # center
            cols = rc[:,1].astype(float) + 0.5
            xs = a*cols + b*rows + c
            ys = d*cols + e*rows + f

            paths_xy[(i_src, j_tgt)] = list(zip(xs.tolist(), ys.tolist()))

    return paths_xy

def plot_heterozygosity_over_time(
    Nc_paths_all: np.ndarray,   # (replicates, T+1, P)
    patches: list[Patch],
    migrants_per_year_est_vec: np.ndarray,  # length P
    results_dir: str,
):
    """
    Plots mean heterozygosity per patch at generation checkpoints over the whole horizon.
    Saves: heterozygosity_over_time.png
    """
    reps, T_plus1, P = Nc_paths_all.shape
    rng_master = np.random.default_rng(RANDOM_SEED)

    plt.figure(figsize=(9, 6))
    any_years = None
    for j, patch in enumerate(patches):
        years, H_mean = genetics_time_series_for_patch(
            Nc_paths_patch=Nc_paths_all[:, :, j],
            migrants_per_year_est=float(migrants_per_year_est_vec[j]),
            rng_global=rng_master,
            gen_time_years=GEN_TIME_FOR_GENETICS,
            H0=H0,
            H_threshold_frac=H_MIN_FRAC,
            ne_ratio_start=NE_RATIO_START,
            ne_ratio_end=NE_RATIO_END,
            ne_ratio_relax_years=NE_RATIO_RELAX_YEARS,
            K_scale=K,
        )
        any_years = years
        plt.plot(years, H_mean, label=patch.name, linewidth=1.8)

    # threshold
    H_min = H_MIN_FRAC * H0
    plt.axhline(H_min, linestyle="--", linewidth=1.2, label=f"H threshold ({H_MIN_FRAC:.2f}×H₀)")

    plt.xlabel("Year")
    plt.ylabel("Heterozygosity (H)")
    plt.title("Heterozygosity over time (mean across replicates)")
    # place legend outside
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0, fontsize=8)
    plt.tight_layout()
    out_path = os.path.join(results_dir, "heterozygosity_over_time.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved heterozygosity time-series → {out_path}")

def plot_Nc_over_time(
    Nc_paths_all: np.ndarray,   # (replicates, T+1, P)
    patches: list[Patch],
    results_dir: str,
):
    """
    Plots mean Nc per patch over all years (0..T). One line per patch, no CI.
    Saves: Nc_over_time.png
    """
    mean_Nc = Nc_paths_all.mean(axis=0)   # (T+1, P)
    T_plus1, P = mean_Nc.shape
    years = np.arange(T_plus1)

    plt.figure(figsize=(9, 6))
    for j, patch in enumerate(patches):
        plt.plot(years, mean_Nc[:, j], label=patch.name, linewidth=1.8)

    # quasi-extinction threshold (horizontal line)
    plt.axhline(QUASI_EXT_THRESHOLD, linestyle="--", linewidth=1.2, label=f"Quasi-extinction (Q={QUASI_EXT_THRESHOLD})")

    plt.xlabel("Year")
    plt.ylabel("Nc (census size)")
    plt.title("Census size (Nc) over time (mean across replicates)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0, fontsize=8)
    plt.tight_layout()
    out_path = os.path.join(results_dir, "Nc_over_time.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Nc time-series → {out_path}")

def main_network():
    rng_master = np.random.default_rng(RANDOM_SEED)
    # make output dir for this run
    results_dir = make_results_dir(RESULTS_BASE)
    print(f"Results directory: {results_dir}")

    # 1. load patches
    patches = load_patches_from_file(PATCH_FILE)
    P = len(patches)

    # 2. build distance matrices
    D_euclid = build_euclid_matrix(patches)

    friction_win = None
    transform_win = None
    patch_rc = None

    if USE_REAL_COST:
        try:
            C_cost, friction_win, transform_win, patch_rc = build_cost_matrix_least_cost(
                patches,
                NMD_RASTER_PATH,
                buffer_dist=MAX_DISPERSAL_DIST * 4.0,  # generous buffer around all ponds
                barrier_shp_path=BARRIER_SHP,   # enable overlay
                barrier_mode="multiply",        # "multiply" = raises cost; "block" = effectively impossible
                barrier_value=50.0,             # 15× cost where barrier present (tune per scenario)
                line_buffer_m=5.0,              # makes line features bite at raster scale
                attr_mult_field="cost_mult"     # optional: read per-feature multiplier from attribute
            )
        except NotImplementedError:
            print("WARNING: build_cost_matrix_least_cost not implemented yet, falling back to dummy cost.")
            C_cost = build_dummy_cost_matrix_from_distance(D_euclid, road_multiplier=1.0)
        except Exception as e:
            print(f"WARNING: cost-distance failed ({e}), falling back to dummy cost.")
            C_cost = build_dummy_cost_matrix_from_distance(D_euclid, road_multiplier=1.0)
    else:
        C_cost = build_dummy_cost_matrix_from_distance(D_euclid, road_multiplier=1.0)

    beta_cost = calibrate_beta_from_costs(C_cost, target_P=0.25, quantile=0.10)

    Pmat = compute_pairwise_matrices(
        patches=patches,
        C_cost_matrix=C_cost,
        D_euclid_matrix=D_euclid,
        max_disp_dist=MAX_DISPERSAL_DIST,
        beta_cost=beta_cost,
    )
    def summarize_connectivity(Pmat):
        v = Pmat[np.isfinite(Pmat)]
        v = v[v > 0]
        if v.size == 0:
            return "No positive links."
        return (
            f"P>0 links: {v.size}, mean={v.mean():.3f}, "
            f"median={np.median(v):.3f}, 90th={np.quantile(v,0.9):.3f}, max={v.max():.3f}"
        )

    conn_str = summarize_connectivity(Pmat)
    print("[CONNECTIVITY]", conn_str)
    lost = int(np.sum((np.isfinite(C_cost)) & (C_cost > 0) & (Pmat == 0.0)))
    print(f"[CONNECTIVITY] pairs with P=0 after barrier: {lost}")

    derived_bits = {
        "beta_cost": float(beta_cost),
        "connectivity_summary": str(conn_str),  # <-- nu en sträng
        "pairs_with_P0_after_barrier": lost,
        "cost_mode": ("least_cost" if USE_REAL_COST else "euclid_dummy"),
        "barrier_file": (BARRIER_SHP if BARRIER_SHP and BARRIER_SHP != "None" else None),
        "tile_path": (NMD_RASTER_PATH if USE_REAL_COST else None),
    }
    write_run_manifest(
        results_dir=results_dir,
        cfg_module=cfg,                # the active species config
        patch_file_path=PATCH_FILE,    # exactly what was read
        patches=patches,               # dump the table we used
        derived=derived_bits
    )
        # Visualize connectivity matrix
    plot_connectivity_heatmap(Pmat, patches, results_dir, label="with_barrier")


    # Build least-cost polylines for strongest links (top 20% or set min_prob)
    paths_xy = {}
    if (friction_win is not None) and (transform_win is not None) and (patch_rc is not None):
        paths_xy = extract_least_cost_paths_for_top_links(
            friction_win=friction_win,
            transform_win=transform_win,
            patch_rc=patch_rc,
            Pmat=Pmat,
            top_frac=1,   # tweak: e.g., 0.10 for top 10%
            min_prob=0.0,   # or set e.g. 0.05 to draw only links >= 5%
            max_pairs=50000     # safety cap for speed
        )

    mainland_idx = None
    mainland_emigrants = None

    if LANDSCAPE_MODE == "infinite":
        patches, Pmat, mainland_idx = add_mainland_donor(
            patches=patches,
            Pmat=Pmat,
            source_name=INFINITE_SOURCE_NAME,
            source_cost=INFINITE_SOURCE_DISTANCE_COST,
            beta_cost=beta_cost,
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
    out_xlsx = os.path.join(results_dir, "network_patch_summary.xlsx")
    df_out.to_excel(out_xlsx, index=False, engine="openpyxl")

    system_xlsx = os.path.join(results_dir, "network_system_summary.xlsx")
    df_system.to_excel(system_xlsx, index=False, engine="openpyxl")

    print(f"\nSaved patch summary to {out_xlsx}")
    print(f"Saved system summary to {system_xlsx}")

    # Map: visualize network
# Scientific (plain) plot

    plot_network_map(
        patches=patches,
        surv_prob=surv_prob_per_patch,
        Pmat=Pmat,
        results_dir=results_dir,
        title="Metapopulation survival & connectivity",
        paths_xy=paths_xy,
    )

    export_arcgis_map(
        patches=patches,
        surv_prob=surv_prob_per_patch,
        Pmat=Pmat,
        results_dir=results_dir,
        paths_xy=paths_xy,  # << here
    )

    plot_heterozygosity_over_time(
        Nc_paths_all=Nc_paths_all,
        patches=patches,
        migrants_per_year_est_vec=My_est_vec,
        results_dir=results_dir,
    )

    plot_Nc_over_time(
        Nc_paths_all=Nc_paths_all,
        patches=patches,
        results_dir=results_dir,
    )

if __name__ == "__main__":
    main_network()

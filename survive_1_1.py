import os
import re

# --- SPECIES SWITCH ---
# "B"           = Bufo bufo (common toad) — amphibian with long generation time, high adult survival, and strong Ne/Nc disparity
# "L"           = Lissotriton vulgaris (smooth newt) — amphibian with shorter generation time, moderate adult survival, and moderate Ne/Nc disparity
# "V"           = Verbascum sp. (mullein) — plant with short generation time, high adult survival, and weak Ne/Nc disparity
# "H"           = Helichrysum sp. (everlasting) — plant with short generation time, low adult survival, and strong Ne/Nc disparity
# "M"           = My Own Species — custom species configuration
SPECIES = "V"

if SPECIES == "B":
    import config_bufo as cfg
elif SPECIES == "L":
    import config_lissotriton as cfg
elif SPECIES == "V":
    import config_verbascum as cfg
elif SPECIES == "H":
    import config_helichrysum as cfg
elif SPECIES == "M":
    import config_my_own_species as cfg
else:
    raise ValueError(f"Unknown SPECIES {SPECIES!r}")

K = cfg.K

# =============== PROJECT CONFIG ===============================================================================================
PROJECT_NAME = "C260159_Verbascum_Landskrona_vitalis_0-scen"         # used for results folder naming
ENGLISH = True        # True = English figure text; False = Swedish figure text

# CENSUS FOR THE TOTAL POPULATION
# i.e. the number of individuals that contribute to density dependence and demographic stochasticity in the 
# demographic models (A and C).
NC_POP =     636      

# Census for meta population, breeders locally (pond etc.). 
# his is the number that matters for genetics (B), and also for Ne-sensitive demography (C) since it affects sigma_e.
NC_METAPOP = K                                      

# INITIAL DISTURBANCE
# can be set to imitate a new, artificial environment (new dug pond, new habitat etc.) 
# that is expected to be more variable than a mature, established habitat. 
# This is implemented as a temporary inflation of sigma_e in the early years of the simulation, 
# which then relaxes back to the normal Ne-dependent value over time. The idea is that a new pond might have more variable 
# conditions (e.g. water levels, temperature, food availability) until it "settles in" and develops more stable microhabitats, vegetation, etc. 
# This is a simple way to capture that without having to model the complex ecological succession explicitly.
DISTURBANCE_SIGMA_FACTOR_START = 1.0   # multiplier on sigma_e in year 0 (freshly dug pond, unstable) sensitive range 1.5–2.0
DISTURBANCE_SIGMA_FACTOR_END   = 1.0   # multiplier on sigma_e once environment is mature/stable
DISTURBANCE_RELAX_YEARS        = 0    # how fast conditions "settle", in years

# INITAL BOTTLENECK / FOUNDER EFFECT 
# i.e. the initial Ne/Nc ratio at the time of founding or reintroduction. 
# This can be set to a low value to reflect a strong founder effect, and then it can either stay low (if the breeding structure remains skewed) 
# or it can recover over time (if the population grows and breeding structure normalizes). 
# This affects the early genetic dynamics and also the early demographic variability if using the Ne-sensitive demography model (C).
NE_RATIO_START = 0.15      # founder effective ratio Ne/Nc in year 0
NE_RATIO_END   = 0.15      # healthy long-term effective ratio Ne/Nc
NE_RATIO_RELAX_YEARS = 0   # how fast breeding structure "normalizes"

# DEMOGRAPHIC STRUCTURE
INIT_PROP_ADULT = 1       # fraction of N0 that are sexually mature breeders ex. 0.5, 0.6, 0.7 or 1 for hermaphrodites like plants.
INIT_FEMALE_FRAC = 0.5    # sex-ratio correction factor: 0.5 = hermaphrodite (no Ne penalty); <0.5 or >0.5 = skewed ratio for species with separate sexes


# Meta-model weights (must sum to 1)
META_WEIGHTS = {
    "demography_density": 0.34,
    "genetics_only": 0.33,
    "ne_sensitive_demography": 0.33,
}
SCORE_WEIGHTS = {"genetic": 0.5, "demographic": 0.5}

# Now bind all config values from cfg into local names for convenience.
NE_RATIOS                 = cfg.NE_RATIOS
GEN_TIME_YEARS            = cfg.GEN_TIME_YEARS
MIGRANTS_PER_YEAR         = cfg.MIGRANTS_PER_YEAR
TIME_HORIZON_YEARS        = cfg.TIME_HORIZON_YEARS
QUASI_EXT_THRESHOLD       = cfg.QUASI_EXT_THRESHOLD
SURVIVAL_TARGET           = cfg.SURVIVAL_TARGET
DENSITY_MODEL             = cfg.DENSITY_MODEL
THETA                     = cfg.THETA
H0                        = cfg.H0
H_MIN_FRAC                = cfg.H_MIN_FRAC
GEN_TIME_FOR_GENETICS     = cfg.GEN_TIME_FOR_GENETICS
GENETIC_EVAL              = cfg.GENETIC_EVAL
R_GRID                    = cfg.R_GRID
SIGMA_E_GRID              = cfg.SIGMA_E_GRID
REPLICATES                = cfg.REPLICATES
RANDOM_SEED               = cfg.RANDOM_SEED
NE_TARGET                 = cfg.NE_TARGET
ALPHA                     = cfg.ALPHA
NE_GENERATION_MODEL       = getattr(cfg, "NE_GENERATION_MODEL", "iteroparous")
S_PREREPRODUCTIVE_MORTALITY = getattr(cfg, "S_PREREPRODUCTIVE_MORTALITY", 0.0)

import math
import itertools
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize
from datetime import datetime


#=================================== DEFINITIONS ============================================================================
def _coerce_listlike(val):
    """
    Accept things like:
      [0.1,0.2,0.3]
      "0.1 0.2 0.3"
      "0.1,0.2,0.3"
    and return a list of numeric strings.
    """
    if isinstance(val, (list, tuple)):
        return list(val)
    if isinstance(val, str):
        # split on comma or whitespace
        raw = [piece.strip() for piece in re.split(r"[,\s]+", val) if piece.strip() != ""]
        return raw
    return [val]

def _ne_gen_multiplier(gen_time_years: float) -> float:
    """
    Scaling factor to convert annual Ne (ratio × Nc) to generational Ne.
    Applied ONLY to genetics sub-models (B, B_var, Blink).
    Does NOT affect demographic models (A, C) or sigma_effective().

    "monocarpic"      — Vitalis et al. (2004): Ne_gen = ((2-s)/2) * N * T²
                        factor = ((2-s)/2) * T  relative to annual-Ne baseline
    "annual_seedbank" — Nunney (2002): Ne_gen = N * T
                        factor = T
    "iteroparous"     — Nunney (1991): no upward T-correction; factor = 1.0
                        (ne_ratio already captures adult-survival effects)
    """
    if NE_GENERATION_MODEL == "monocarpic":
        return ((2.0 - S_PREREPRODUCTIVE_MORTALITY) / 2.0) * gen_time_years
    elif NE_GENERATION_MODEL == "annual_seedbank":
        return float(gen_time_years)
    else:  # "iteroparous" or unrecognised — safe default
        return 1.0

def _run_full_pipeline(result_dir, rng):
    """
    This is basically the body of main() from 'rows = []' down to the final plots.
    It assumes all globals like NC_POP etc. are already updated, and
    it uses result_dir instead of recomputing it.
    """
    rows = []
    gt_min, gt_max = GEN_TIME_YEARS
    T = TIME_HORIZON_YEARS

    grid = list(itertools.product(R_GRID, SIGMA_E_GRID))

    for ratio in NE_RATIOS:
        # Effective sizes for the two “worlds”:
        Ne_demo = ratio * NC_POP       # core / established pop (for demography models A and C)
        Ne_gen  = ratio * NC_METAPOP   # local pond / metapop unit (for genetics model B)

        # Generational Ne correction (Vitalis / Nunney) — genetics sub-models only.
        # ne_gen_mult_* converts annual-scale Ne to generational Ne for the given generation time.
        ne_gen_mult_min = _ne_gen_multiplier(gt_min)
        ne_gen_mult_max = _ne_gen_multiplier(gt_max)
        Ne_gen_min = Ne_gen * ne_gen_mult_min   # corrected Ne for min-gen-time scenarios
        Ne_gen_max = Ne_gen * ne_gen_mult_max   # corrected Ne for max-gen-time scenarios

        # Genetics
        generations_minGT = T / gt_min
        generations_maxGT = T / gt_max
        Ft_iso_minGT = inbreeding_after_t_generations(Ne_gen_min, generations_minGT)
        Ft_iso_maxGT = inbreeding_after_t_generations(Ne_gen_max, generations_maxGT)

        # Demography WITHOUT immigration (analytical), but now using Ne-dependent sigma
        pexts_noI = []
        for r, se in grid:
            se_eff = sigma_effective(se, Ne_demo)
            pext = quasi_ext_prob_diffusion(NC_POP, QUASI_EXT_THRESHOLD, r, se_eff, T)
            pexts_noI.append(pext)
        mean_pext_noI = float(np.mean(pexts_noI))

        for My in MIGRANTS_PER_YEAR:
            # Genetics with immigration
            H_threshold = H_MIN_FRAC * H0
            Mgen_minGT = My * gt_min
            Mgen_maxGT = My * gt_max
            Feq_minGT = equilibrium_inbreeding_with_migration(Ne_gen_min, Mgen_minGT, NC_METAPOP)
            Feq_maxGT = equilibrium_inbreeding_with_migration(Ne_gen_max, Mgen_maxGT, NC_METAPOP)

            # ===== Independent Model A: Demography with density dependence (Ne-independent) =====
            demog_density_pexts = []
            for r, se in grid:
                demog_density_pexts.append(
                    simulate_demography_density_independent(
                        N0=NC_POP,
                        Q=QUASI_EXT_THRESHOLD,
                        r=r,
                        sigma_e=se,
                        T_years=T,
                        immigrants_per_year=My,
                        K=K,
                        model=DENSITY_MODEL,
                        theta=THETA,
                        replicates=REPLICATES,
                        rng=rng,
                        disturb_start=DISTURBANCE_SIGMA_FACTOR_START,
                        disturb_end=DISTURBANCE_SIGMA_FACTOR_END,
                        disturb_relax=DISTURBANCE_RELAX_YEARS,
                    )
                )
            demog_density_pext = float(np.mean(demog_density_pexts))
            demog_density_surv = 1.0 - demog_density_pext
            demog_density_flag = demog_density_surv >= SURVIVAL_TARGET

            # ===== Independent Model B: Genetics-only =====
            gen_time_for_genetics = GEN_TIME_FOR_GENETICS or GEN_TIME_YEARS[0]
            # generational Ne correction for this specific gen-time setting
            ne_gen_mult_gfg = _ne_gen_multiplier(gen_time_for_genetics)
            Ne_gen_gfg = Ne_gen * ne_gen_mult_gfg

            gens_minGT = T / gen_time_for_genetics

            # Isolated drift only
            Ft_iso_T = inbreeding_after_t_generations(Ne_gen_gfg, gens_minGT)
            H_iso_T = H0 * (1.0 - Ft_iso_T)

            # Drift + immigration
            Ft_mig_T = F_with_migration_over_time(
                Ne=Ne_gen_gfg,
                Nc=NC_METAPOP,
                migrants_per_year=My,
                gen_time_years=gen_time_for_genetics,
                years=T,
                F0=0.0
            )
            H_mig_T = H0 * (1.0 - Ft_mig_T)

            # Choose evaluation rule
            if GENETIC_EVAL == "isolated":
                H_for_rule = H_iso_T
            elif GENETIC_EVAL == "best_of_both":
                H_for_rule = max(H_iso_T, H_mig_T)
            else:  # "with_migration"
                H_for_rule = H_mig_T

            genetics_flag = (H_for_rule >= H_threshold)
            genetics_survival = 1.0 if genetics_flag else 0.0

            # ===== New: Genetics-only with variable Ne/Nc ratio over time (B_var) =====
            H_var_T, surv_flag_var = genetics_viability_variable_ratio(
                Nc=NC_METAPOP,
                migrants_per_year=My,
                gen_time_years=gen_time_for_genetics,
                years=T,
                H0=H0,
                H_threshold_frac=H_MIN_FRAC,
                replicates=REPLICATES,
                rng=rng,
                ne_ratio_start=NE_RATIO_START,
                ne_ratio_end=NE_RATIO_END,
                ne_ratio_relax_years=NE_RATIO_RELAX_YEARS,
                ne_gen_multiplier=ne_gen_mult_gfg,
            )
            genetics_var_flag = bool(surv_flag_var >= 1.0)

            # ===== Independent Model C: Ne-sensitive demography (using extinction-time CDF) =====
            # We estimate survival as P(TTE > T), i.e. fraction of populations not extinct by horizon T.
            # We do this by simulating full trajectories once per (r, se) combo, collecting the
            # distribution of first-extinction times, and then averaging across the grid.

            surv_fracs = []
            for r, se in grid:
                se_eff = sigma_effective(se, Ne_demo)
                times = simulate_time_to_extinction(
                    N0=NC_POP,
                    Q=QUASI_EXT_THRESHOLD,
                    r=r,
                    sigma_e_eff_base=se_eff,
                    T_years=T,
                    immigrants_per_year=My,
                    replicates=REPLICATES,
                    rng=rng,
                    disturb_start=DISTURBANCE_SIGMA_FACTOR_START,
                    disturb_end=DISTURBANCE_SIGMA_FACTOR_END,
                    disturb_relax=DISTURBANCE_RELAX_YEARS,
                )
                survived = np.mean(times > T)
                surv_fracs.append(survived)

            ne_sensitive_surv = float(np.mean(surv_fracs))
            ne_sensitive_pext = 1.0 - ne_sensitive_surv
            ne_sensitive_flag = ne_sensitive_surv >= SURVIVAL_TARGET

            # ===== Coupled eco-genetics: B_link (C -> B) =====
            # We want demography paths under C-style assumptions,
            # and then we feed those paths into genetics.
            # We'll average across (r, se) grid like we do elsewhere.

            H_end_all = []

            for r, se in grid:
                # C-style sigma inflation from Ne_demo
                se_eff = sigma_effective(se, Ne_demo)

                # draw replicate demographic paths
                paths, extinct_time = simulate_demography_paths_Cstyle(
                    N0=NC_METAPOP,                # IMPORTANT: genetics is at pond scale
                    Q=QUASI_EXT_THRESHOLD,
                    r=r,
                    sigma_e_eff_base=se_eff,
                    T_years=T,
                    immigrants_per_year=My,
                    replicates=REPLICATES,
                    rng=rng,
                    disturb_start=DISTURBANCE_SIGMA_FACTOR_START,
                    disturb_end=DISTURBANCE_SIGMA_FACTOR_END,
                    disturb_relax=DISTURBANCE_RELAX_YEARS,
                )

                # run genetics on those demographic paths
                (
                    H_end,
                    H_mean_curve_link,
                    H_lo_curve_link,
                    H_hi_curve_link,
                    t_years_link,
                ) = simulate_genetics_on_demography_paths(
                    Nc_paths=paths,
                    gen_time_years=gen_time_for_genetics,
                    H0=H0,
                    migrants_per_year=My,
                    rng=rng,
                    ne_ratio_start=NE_RATIO_START,
                    ne_ratio_end=NE_RATIO_END,
                    ne_ratio_relax_years=NE_RATIO_RELAX_YEARS,
                    K_scale=NC_METAPOP,   # scale 'healthy' breeder census
                    ne_gen_multiplier=ne_gen_mult_gfg,
                )

                H_end_all.append(H_end)

            # concatenate across all (r,se) scenarios, then summarize
            if len(H_end_all) > 0:
                H_end_all = np.concatenate(H_end_all, axis=0)
                H_link_T_mean = float(np.mean(H_end_all))
            else:
                H_link_T_mean = float("nan")

            H_threshold = H_MIN_FRAC * H0
            genetics_link_flag = (H_link_T_mean >= H_threshold)
            genetics_link_survival = 1.0 if genetics_link_flag else 0.0


            # ===== Meta-model: weighted combination of survivals =====
            wA =META_WEIGHTS .get("demography_density", 1/3)
            wB = META_WEIGHTS.get("genetics_only", 1/3)
            wC = META_WEIGHTS.get("ne_sensitive_demography", 1/3)
            # normalize just in case
            wsum = max(1e-9, (wA + wB + wC))
            wA, wB, wC = wA/wsum, wB/wsum, wC/wsum
            meta_survival = wA * demog_density_surv + wB * genetics_survival + wC * ne_sensitive_surv
            meta_flag = meta_survival >= SURVIVAL_TARGET


            # Demography WITH immigration (simulation), with Ne-dependent sigma
            pexts_withI = []
            for r, se in grid:
                se_eff = sigma_effective(se, Ne_demo)
                pext_sim = simulate_demography_with_immigration(
                    N0=NC_POP,
                    Q=QUASI_EXT_THRESHOLD,
                    r=r,
                    sigma_e_eff=se_eff,
                    T_years=T,
                    immigrants_per_year=My,
                    replicates=REPLICATES,
                    rng=rng
                )
                pexts_withI.append(pext_sim)
            mean_pext_withI = float(np.mean(pexts_withI))

            meets_flag = (1.0 - mean_pext_withI) >= SURVIVAL_TARGET
            score = score_scenario(Ft_iso_minGT, Feq_minGT, mean_pext_noI, mean_pext_withI)

            viability_eval = evaluate_viability(
                demog_A_survival = demog_density_surv,
                genetic_B_const_survival = genetics_survival,
                genetic_B_var_survival   = float(surv_flag_var),
                genetic_Blink_survival   = genetics_link_survival,
                demog_C_survival = ne_sensitive_surv,
                surv_threshold = SURVIVAL_TARGET,
            )
            meta_survival = viability_eval["META_survival"]
            meta_flag     = viability_eval["META_pass"]


            rows.append(ScenarioRow(
                Nc=NC_POP,
                Ne_ratio=ratio,
                Ne=Ne_demo,
                gen_time_min=gt_min,
                gen_time_max=gt_max,
                migrants_per_year=My,
                migrants_per_generation_minGT=Mgen_minGT,
                migrants_per_generation_maxGT=Mgen_maxGT,
                Ft_isolated_minGT=Ft_iso_minGT,
                Ft_isolated_maxGT=Ft_iso_maxGT,
                Feq_migr_minGT=Feq_minGT,
                Feq_migr_maxGT=Feq_maxGT,
                mean_pext_noI=mean_pext_noI,
                mean_pext_withI=mean_pext_withI,
                score=score,
                meets_survival_target=bool((1.0 - mean_pext_withI) >= SURVIVAL_TARGET),

                demog_density_pext=demog_density_pext,
                demog_density_survival=demog_density_surv,
                demog_density_meets_target=bool(demog_density_flag),

                # B_const (original standalone genetics)
                genetic_H_isolated_T=H_iso_T,
                genetic_H_migration_T=H_mig_T,
                genetic_viability_survival=genetics_survival,
                genetics_survival_meets=bool(genetics_flag),

                # B_var (time-recovering Ne/Nc ratio, no demography feedback)
                genetic_H_variable_T=H_var_T,
                genetic_viability_variable_survival=float(surv_flag_var),
                genetics_variable_survival_meets=bool(genetics_var_flag),

                # B_link (eco-genetic coupling: demography -> genetics)
                genetic_H_linked_T=H_link_T_mean,
                genetic_viability_linked_survival=genetics_link_survival,
                genetics_linked_survival_meets=bool(genetics_link_flag),

                # C (Ne-sensitive demography)
                ne_sensitive_pext=ne_sensitive_pext,
                ne_sensitive_survival=ne_sensitive_surv,
                ne_sensitive_meets_target=bool(ne_sensitive_flag),

                meta_survival=meta_survival,
                meta_meets_target=bool(meta_flag),
            ))


    df = pd.DataFrame([asdict(r) for r in rows])
    df_sorted = df.sort_values("score", ascending=True).reset_index(drop=True)

    # Print and save
    pd.set_option("display.float_format", lambda x: f"{x:0.3f}")
    print("\n=== Top-ranked scenarios (lower score = better) ===")
    cols_view = [
        "Ne_ratio",
        "Ne",
        "migrants_per_year",

        # Submodel A
        "demog_density_survival",
        "demog_density_meets_target",

        # Submodel B_const
        "genetic_viability_survival",
        "genetics_survival_meets",
        "genetic_H_migration_T",

        # Submodel B_var
        "genetic_viability_variable_survival",
        "genetics_variable_survival_meets",
        "genetic_H_variable_T",

        # Submodel B_link (C → B)
        "genetic_viability_linked_survival",
        "genetics_linked_survival_meets",
        "genetic_H_linked_T",

        # Submodel C
        "ne_sensitive_survival",
        "ne_sensitive_meets_target",

        # Meta
        "meta_survival",
        "meta_meets_target",
    ]

    print(df_sorted[cols_view].head(10).to_string(index=False))
    xlsx_path = os.path.join(result_dir, "extinction_scenarios.xlsx")

    # --- Add model letters to key column headers before saving ---
    rename_map = {
        "demog_density_survival":        f"A_survival_Nc{NC_POP}",
        "demog_density_meets_target":    f"A_meets_target_Nc{NC_POP}",

        "genetic_viability_survival":            f"B_const_survival_Nc{NC_METAPOP}",
        "genetics_survival_meets":               f"B_const_meets_target_Nc{NC_METAPOP}",
        "genetic_H_migration_T":                 f"B_const_H_migration_T_Nc{NC_METAPOP}",

        "genetic_viability_variable_survival":   f"Bvar_survival_Nc{NC_METAPOP}",
        "genetics_variable_survival_meets":      f"Bvar_meets_target_Nc{NC_METAPOP}",
        "genetic_H_variable_T":                  f"Bvar_H_T_Nc{NC_METAPOP}",

        "genetic_viability_linked_survival":   f"Blink_survival_Nc{NC_METAPOP}",
        "genetics_linked_survival_meets":      f"Blink_meets_target_Nc{NC_METAPOP}",
        "genetic_H_linked_T":                  f"Blink_H_T_Nc{NC_METAPOP}",


        "ne_sensitive_survival":         f"C_survival_Nc{NC_POP}",
        "ne_sensitive_meets_target":     f"C_meets_target_Nc{NC_POP}",

        "meta_survival":                 "META_survival",
        "meta_meets_target":             "META_meets_target",
    }
    df_sorted.rename(columns=rename_map, inplace=True)
    for col in ["score", "meets_survival_target"]:
        if col in df_sorted.columns:
            df_sorted.drop(columns=col, inplace=True)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df_sorted.to_excel(writer, index=False, sheet_name="Results")

    print(f"\nSaved all scenarios to {xlsx_path}")

    # --------- FIGURES (two) ---------
    print("\nGenerating graphs...")
    out_dir = os.path.dirname(xlsx_path)
    os.makedirs(out_dir, exist_ok=True)


    # ---------- FIGURE: Sensitivity of extinction risk to sigma_e ----------
    print("Building sigma_e sensitivity curves ...")
    sigma_vals = np.linspace(0.15, 0.50, 10)  # adjust range if you like
    r_vals = R_GRID                          # average across your r-uncertainty

    nrows = 1
    ncols = len(NE_RATIOS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8*ncols, 4.2), sharey=True)
    if ncols == 1:
        axes = [axes]

    for ax, ratio in zip(axes, NE_RATIOS):
        Ne = ratio * NC_POP
        for My in MIGRANTS_PER_YEAR:
            risks = []
            for se in sigma_vals:
                # average over r-grid, WITH immigration, using Ne-dependent sigma
                vals = []
                for r in r_vals:
                    se_eff = sigma_effective(se, Ne)
                    p = simulate_demography_with_immigration(
                        N0=NC_POP, Q=QUASI_EXT_THRESHOLD, r=r, sigma_e_eff=se_eff,
                        T_years=TIME_HORIZON_YEARS, immigrants_per_year=My,
                        replicates=REPLICATES, rng=np.random.default_rng(RANDOM_SEED+777)
                    )
                    vals.append(p)
                risks.append(float(np.mean(vals)))
            ax.plot(sigma_vals, risks, label=f"Mig/yr={My}")
        ax.set_xlabel("Environmental SD on ln scale (σₑ)")
        ax.set_title(f"Ne/Nc = {ratio:.2f} (Ne={Ne:.0f})")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(f"P(extinction ≤ Q) by {TIME_HORIZON_YEARS} years")
    axes[-1].legend(title="Immigrants", fontsize=8)
    fig_sigma = os.path.join(out_dir, "sensitivity_sigma.png")
    plt.tight_layout()
    plt.savefig(fig_sigma, dpi=150)
    plt.close()
    print(f"Saved sensitivity plot: {fig_sigma}")

    # ---------- FIGURE: Combined genetic–demographic risk scatter ----------
    print("Building genetic–demographic risk scatter ...")
    fig, ax = plt.subplots(figsize=(7.5, 6))

    # marker per Ne_ratio
    marker_shapes = ["o", "s", "D", "^", "v", "<", ">"]
    markers = {ratio: marker_shapes[i % len(marker_shapes)] for i, ratio in enumerate(NE_RATIOS)}

    # color by migrants/year
    cmap = plt.cm.get_cmap("plasma")
    colors = {My: cmap(i / max(1, len(MIGRANTS_PER_YEAR)-1)) for i, My in enumerate(sorted(MIGRANTS_PER_YEAR))}


    for _, row in df.iterrows():
        x = row["Feq_migr_minGT"]        # genetic risk proxy (lower better)
        y = row["mean_pext_withI"]       # demographic risk (lower better)
        mrk = markers.get(row["Ne_ratio"], "o")
        col = colors.get(row["migrants_per_year"], "gray")
        ax.scatter(x, y, s=70, marker=mrk, edgecolor="black", linewidth=0.5, color=col)

    # helpers: marginal "good zone" lines
    ax.axvline(0.05, color="gray", linestyle="--", linewidth=1, label="F_eq = 0.05")
    ax.axhline(0.10, color="gray", linestyle=":", linewidth=1, label="P_ext = 0.10")

    ax.set_xlabel("Equilibrium inbreeding F_eq (migration, gen time = min)")
    ax.set_ylabel(f"P(extinction ≤ Q) at {TIME_HORIZON_YEARS} years (with immigration)")
    ax.set_title("Genetic (x) vs Demographic (y) risk per scenario")

    # legends
    from matplotlib.lines import Line2D
    leg1 = [Line2D([0],[0], marker=markers[r], color="w", label=f"Ne/Nc={r:.2f}",
                   markerfacecolor="lightgray", markeredgecolor="black", markersize=9)
            for r in NE_RATIOS]
    leg2 = [Line2D([0],[0], marker="o", color=c, label=f"Mig/yr={m}", markeredgecolor="black", markersize=9)
            for m, c in colors.items()]

    legend1 = ax.legend(handles=leg1, title="Ne/Nc", loc="upper right")
    ax.add_artist(legend1)
    ax.legend(handles=leg2, title="Immigration", loc="lower right")

    ax.grid(alpha=0.25)
    fig_scatter = os.path.join(out_dir, "combined_genetic_demographic_scatter.png")
    plt.tight_layout()
    plt.savefig(fig_scatter, dpi=150)
    plt.close()
    print(f"Saved scatter: {fig_scatter}")

    # FIGURE A: Submodel A (density-dependent demography, Ne-independent)
    print("Building Submodel A figure: density-dependent demography survival with CI ...")

    TS_dem = np.arange(0, TIME_HORIZON_YEARS + 1, 5)
    rng_plot = np.random.default_rng(RANDOM_SEED + 1001)

    plt.figure(figsize=(8, 5))

    for My in MIGRANTS_PER_YEAR:
        mean_surv_series = []
        lo_surv_series = []
        hi_surv_series = []

        for Tcur in TS_dem:
            if Tcur == 0:
                # At time 0 survival is trivially 1 with no uncertainty
                mean_surv_series.append(1.0)
                lo_surv_series.append(1.0)
                hi_surv_series.append(1.0)
                continue

            # collect replicate-level survival indicators across r, sigma grid
            all_survived = []
            for r, se in grid:
                extinct_flags = simulate_demography_density_independent(
                    N0=NC_POP,
                    Q=QUASI_EXT_THRESHOLD,
                    r=r,
                    sigma_e=se,
                    T_years=int(Tcur),
                    immigrants_per_year=My,
                    K=K,
                    model=DENSITY_MODEL,
                    theta=THETA,
                    replicates=REPLICATES,
                    rng=rng_plot,
                    disturb_start=DISTURBANCE_SIGMA_FACTOR_START,
                    disturb_end=DISTURBANCE_SIGMA_FACTOR_END,
                    disturb_relax=DISTURBANCE_RELAX_YEARS,
                    return_extinct_array=True,  # << NEW
                )
                # extinct_flags is length REPLICATES, 1 if extinct, 0 if alive
                survived_flags = 1.0 - extinct_flags  # 1 if alive
                all_survived.append(survived_flags)

            all_survived = np.concatenate(all_survived, axis=0)

            mean_s, lo_s, hi_s = bootstrap_ci(all_survived, n_boot=1000, rng=rng_plot)
            mean_surv_series.append(mean_s)
            lo_surv_series.append(lo_s)
            hi_surv_series.append(hi_s)

        # draw mean line
        plt.plot(
            TS_dem,
            mean_surv_series,
            label=(f"Immigration {My}/yr" if ENGLISH else f"Invandring {My}/år"),
            linewidth=2
        )
        # draw CI ribbon
        plt.fill_between(
            TS_dem,
            lo_surv_series,
            hi_surv_series,
            alpha=0.2
        )

    # 95% reference line
    plt.axhline(SURVIVAL_TARGET, color="black", linestyle="--", linewidth=1)
    plt.text(
        TIME_HORIZON_YEARS,
        SURVIVAL_TARGET + 0.003,
        f"{int(SURVIVAL_TARGET*100)}%",
        ha="right", va="bottom", fontsize=9, color="black"
    )

    plt.xlabel("Year" if ENGLISH else "År")
    plt.ylabel("Survival probability (1 − extinction risk)" if ENGLISH else "Överlevnadssannolikhet (1 − utdöenderisk)")
    plt.title(f"Sub-model A: Demography (Nc≈{NC_POP}), density regulation without Ne effects" if ENGLISH else f"Submodell A: Demografi (Nc≈{NC_POP}), täthetsreglering utan Ne-effekter")
    plt.legend(fontsize=8)
    plt.tight_layout()

    fig_A = os.path.join(out_dir, "mod_A_survival_density_demography.png")
    plt.savefig(fig_A, dpi=150)
    plt.close()
    print(f"Saved Submodel A figure: {fig_A}")

    # FIGURE B: Submodel B (genetics-only, stochastic drift + CI)
    print("Building Submodel B figure: genetic heterozygosity with drift CI ...")

    gen_time_for_plot = int(GEN_TIME_YEARS[0]) if GEN_TIME_FOR_GENETICS is None else GEN_TIME_FOR_GENETICS
    rng_gen = np.random.default_rng(RANDOM_SEED + 1500)

    plt.figure(figsize=(9, 6))

    # We'll assign each Ne_ratio a color, and each migration rate a linestyle.
    color_map = plt.cm.viridis  # stable, perceptually nice
    ratio_colors = {
        ratio: color_map(i / max(1, len(NE_RATIOS) - 1))
        for i, ratio in enumerate(sorted(NE_RATIOS))
    }

    linestyles_pool = ["-", "--", ":", "-."]  # cycle if >4 migration rates
    style_map = {
        My: linestyles_pool[i % len(linestyles_pool)]
        for i, My in enumerate(sorted(MIGRANTS_PER_YEAR))
    }

    # Loop over Ne ratio and migrants/year
    for ratio in NE_RATIOS:
        Ne = ratio * NC_METAPOP
        for My in MIGRANTS_PER_YEAR:

            # Simulate stochastic drift with immigration My
            H_traj, t_years = simulate_heterozygosity_drift_recoveringNe(
                Nc=NC_METAPOP,
                migrants_per_year=My,
                gen_time_years=gen_time_for_plot,
                years=TIME_HORIZON_YEARS,
                H0=H0,
                replicates=REPLICATES,
                rng=rng_gen,
                ne_ratio_start=NE_RATIO_START,            # <- project-level start
                ne_ratio_end=NE_RATIO_END,                # <- project-level end
                ne_ratio_relax_years=NE_RATIO_RELAX_YEARS,
                ne_gen_multiplier=_ne_gen_multiplier(gen_time_for_plot),
            )
            # Bootstrap CI across replicates at each recorded time point
            mean_curve = []
            lo_curve = []
            hi_curve = []
            for col in range(H_traj.shape[1]):
                m, lo, hi = bootstrap_ci(H_traj[:, col], n_boot=1000, rng=rng_gen)
                mean_curve.append(m)
                lo_curve.append(lo)
                hi_curve.append(hi)

            lbl = (f"Ne/Nc={ratio:.2f}, Imm={My}/yr" if ENGLISH else f"Ne/Nc={ratio:.2f}, Inv={My}/år")
            colr = ratio_colors[ratio]
            ls   = style_map[My]

            # Plot mean line
            plt.plot(
                t_years,
                mean_curve,
                linestyle=ls,
                linewidth=1.8,
                color=colr,
                label=lbl,
            )

            # Shaded CI
            plt.fill_between(
                t_years,
                lo_curve,
                hi_curve,
                color=colr,
                alpha=0.15,
            )

    # Horizontal line for viability threshold: keep this, it's biologically meaningful
    H_threshold = H_MIN_FRAC * H0
    plt.axhline(H_threshold, color="black", linestyle=":", linewidth=1)
    plt.text(
        TIME_HORIZON_YEARS,
        H_threshold + 0.01,
        (f"{int(H_MIN_FRAC*100)}% of H0" if ENGLISH else f"{int(H_MIN_FRAC*100)}% av H0"),
        ha="right", va="bottom", fontsize=9, color="black"
    )

    plt.xlabel("Year" if ENGLISH else "År")
    plt.ylabel("Heterozygosity H(t)" if ENGLISH else "Heterozygositet H(t)")
    plt.title(f"Sub-model B: Genetic variation (Nc≈{NC_METAPOP} ~ K={K}, re-established local population)" if ENGLISH else f"Submodell B: Genetisk variation (Nc≈{NC_METAPOP} ~ K={K}, återetablerad lokal population)")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    fig_B = os.path.join(out_dir, "mod_B_heterozygosity_genetics.png")
    plt.savefig(fig_B, dpi=150)
    plt.close()
    print(f"Saved Submodel B figure: {fig_B}")

    # --- FIGURE C: Submodel C (Ne-sensitive demography, CDF-based survival) ---
    print("Building Submodel C figure (CDF-based survival) ...")

    TS_plot = np.arange(0, TIME_HORIZON_YEARS + 1, 2)
    rng_plot2 = np.random.default_rng(RANDOM_SEED + 2002)

    plt.figure(figsize=(9, 6))
    rng_plot2 = np.random.default_rng(RANDOM_SEED + 2002)

    # --- Colour and style maps (match Submodel B and Blink) ---
    color_map = plt.cm.viridis
    ratio_colors = {
        ratio: color_map(i / max(1, len(NE_RATIOS) - 1))
        for i, ratio in enumerate(sorted(NE_RATIOS))
    }
    linestyles_pool = ["-", "--", ":", "-."]
    style_map = {
        My: linestyles_pool[i % len(linestyles_pool)]
        for i, My in enumerate(sorted(MIGRANTS_PER_YEAR))
    }

    for ratio in NE_RATIOS:
        Ne_demo = ratio * NC_POP
        colr = ratio_colors[ratio]

        for My in MIGRANTS_PER_YEAR:
            ls = style_map[My]
            all_times_list = []
            for r, se in grid:
                se_eff = sigma_effective(se, Ne_demo)
                times = simulate_time_to_extinction(
                    N0=NC_POP,
                    Q=QUASI_EXT_THRESHOLD,
                    r=r,
                    sigma_e_eff_base=se_eff,
                    T_years=TIME_HORIZON_YEARS,
                    immigrants_per_year=My,
                    replicates=REPLICATES,
                    rng=rng_plot2,
                    disturb_start=DISTURBANCE_SIGMA_FACTOR_START,
                    disturb_end=DISTURBANCE_SIGMA_FACTOR_END,
                    disturb_relax=DISTURBANCE_RELAX_YEARS,
                )
                all_times_list.append(times)
            all_times = np.concatenate(all_times_list, axis=0)

            mean_curve, lo_curve, hi_curve = [], [], []
            for tcur in TS_plot:
                survived_flags = (all_times > tcur).astype(float)
                m, lo, hi = bootstrap_ci(survived_flags, n_boot=1000, rng=rng_plot2)
                mean_curve.append(m)
                lo_curve.append(lo)
                hi_curve.append(hi)

            t_fail = time_to_threshold_cross(
                curve_y = mean_curve,
                curve_t = TS_plot,
                threshold = SURVIVAL_TARGET,
                direction = "down",
            )

            label = (f"Ne/Nc={ratio:.2f}, Imm={My}/yr" if ENGLISH else f"Ne/Nc={ratio:.2f}, Inv={My}/år")
            plt.plot(
                TS_plot,
                mean_curve,
                color=colr,
                linestyle=ls,
                linewidth=1.8,
                label=label,
            )
            plt.fill_between(
                TS_plot,
                lo_curve,
                hi_curve,
                color=colr,
                alpha=0.15,
            )

    plt.axhline(SURVIVAL_TARGET, color="black", linestyle="--", linewidth=1)
    plt.text(
        TIME_HORIZON_YEARS,
        SURVIVAL_TARGET + 0.02,
        f"{int(SURVIVAL_TARGET*100)}%",
        ha="right", va="bottom", fontsize=9, color="black"
    )

    plt.xlabel("Year" if ENGLISH else "År")
    plt.ylabel("Survival probability (1 − extinction time CDF)" if ENGLISH else "Överlevnadssannolikhet (1 − CDF för utdöendetid)")
    plt.title(f"Sub-model C: Demography with Ne-sensitive environmental stochasticity (Nc≈{NC_POP})" if ENGLISH else f"Submodell C: Demografi med Ne-känslig miljöstokasticitet (Nc≈{NC_POP})")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()

    fig_C = os.path.join(result_dir, "mod_C_survival_ne_sensitive.png")
    plt.savefig(fig_C, dpi=150)
    plt.close()
    print(f"Saved Submodel C figure: {fig_C}")

    # --- FIGURE Blink: eco-genetic coupling (C→B) ---
    print("Building Submodel Blink figure: eco-genetic coupling heterozygosity ...")

    plt.figure(figsize=(9, 6))
    rng_link = np.random.default_rng(RANDOM_SEED + 2500)

    # --- Colour and style maps just like Submodel B ---
    color_map = plt.cm.viridis
    ratio_colors = {
        ratio: color_map(i / max(1, len(NE_RATIOS) - 1))
        for i, ratio in enumerate(sorted(NE_RATIOS))
    }
    linestyles_pool = ["-", "--", ":", "-."]
    style_map = {
        My: linestyles_pool[i % len(linestyles_pool)]
        for i, My in enumerate(sorted(MIGRANTS_PER_YEAR))
    }

    for ratio in NE_RATIOS:
        Ne_demo = ratio * NC_POP
        colr = ratio_colors[ratio]
        for My in MIGRANTS_PER_YEAR:
            ls = style_map[My]
            H_end_all = []
            mean_curve_link = None
            t_years_link = None

            for r, se in grid:
                se_eff = sigma_effective(se, Ne_demo)
                paths, _ = simulate_demography_paths_Cstyle(
                    N0=NC_METAPOP,
                    Q=QUASI_EXT_THRESHOLD,
                    r=r,
                    sigma_e_eff_base=se_eff,
                    T_years=TIME_HORIZON_YEARS,
                    immigrants_per_year=My,
                    replicates=REPLICATES,
                    rng=rng_link,
                    disturb_start=DISTURBANCE_SIGMA_FACTOR_START,
                    disturb_end=DISTURBANCE_SIGMA_FACTOR_END,
                    disturb_relax=DISTURBANCE_RELAX_YEARS,
                )
                (
                    H_end,
                    H_mean_curve_link,
                    H_lo_curve_link,
                    H_hi_curve_link,
                    t_years_link,
                ) = simulate_genetics_on_demography_paths(
                    Nc_paths=paths,
                    gen_time_years=GEN_TIME_FOR_GENETICS,
                    H0=H0,
                    migrants_per_year=My,
                    rng=rng_link,
                    ne_ratio_start=NE_RATIO_START,
                    ne_ratio_end=NE_RATIO_END,
                    ne_ratio_relax_years=NE_RATIO_RELAX_YEARS,
                    K_scale=NC_METAPOP,
                    ne_gen_multiplier=_ne_gen_multiplier(GEN_TIME_FOR_GENETICS),
                )
                # compute recovery time for this (ratio, My, r, se) combo
                H_threshold_curve = H_MIN_FRAC * H0
                t_recover = time_to_threshold_cross(
                    curve_y = H_mean_curve_link,
                    curve_t = t_years_link,
                    threshold = H_threshold_curve,
                    direction = "up",
                )
                # store these if you want later export
                # (You can accumulate min/median across grid for reporting.)

                H_end_all.append(H_end)

            if H_mean_curve_link is None:
                continue

            label = (f"Ne/Nc={ratio:.2f}, Imm={My}/yr" if ENGLISH else f"Ne/Nc={ratio:.2f}, Inv={My}/år")

            # Mean line & CI ribbon with viridis colour
            plt.plot(
                t_years_link,
                H_mean_curve_link,
                color=colr,
                linestyle=ls,
                lw=1.8,
                label=label,
            )
            plt.fill_between(
                t_years_link,
                H_lo_curve_link,
                H_hi_curve_link,
                color=colr,
                alpha=0.15,
            )

    H_threshold = H_MIN_FRAC * H0
    plt.axhline(H_threshold, color="black", linestyle=":", linewidth=1)
    plt.text(
        TIME_HORIZON_YEARS,
        H_threshold + 0.01,
        (f"{int(H_MIN_FRAC*100)}% of H0" if ENGLISH else f"{int(H_MIN_FRAC*100)}% av H0"),
        ha="right", va="bottom", fontsize=9, color="black"
    )

    plt.xlabel("Year" if ENGLISH else "År")
    plt.ylabel("Heterozygosity H(t)" if ENGLISH else "Heterozygositet H(t)")
    plt.title("Sub-model Blink: Eco-genetic coupling (demography → genetics)" if ENGLISH else "Submodell Blink: Ekogenetisk koppling (demografi → genetik)")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    fig_Blink = os.path.join(result_dir, "mod_Blink_eco_genetic_coupling.png")
    plt.savefig(fig_Blink, dpi=150)
    plt.close()
    print(f"Saved Submodel Blink figure: {fig_Blink}")


    # --- Disturbance decay diagnostic plot (separate figure) ---
    years = np.arange(0, 30)
    factors = [disturbance_sigma_multiplier(
                y,
                start_factor=DISTURBANCE_SIGMA_FACTOR_START,
                end_factor=DISTURBANCE_SIGMA_FACTOR_END,
                relax_years=DISTURBANCE_RELAX_YEARS
            )
            for y in years]

    plt.figure(figsize=(5, 3))
    plt.plot(years, factors, lw=2)
    plt.xlabel("Years since establishment" if ENGLISH else "År sedan anläggning")
    plt.ylabel("σ_e multiplier" if ENGLISH else "σₑ-multiplikator")
    plt.title("Initial disturbance → stabilisation" if ENGLISH else "Initial störning → stabilisering")
    plt.grid(alpha=0.3)
    fig_disturb = os.path.join(result_dir, "disturbance_decay.png")
    plt.tight_layout()
    plt.savefig(fig_disturb, dpi=150)
    plt.close()
    print(f"Saved disturbance plot: {fig_disturb}")

    # Notes
    print("\nNotes:")
    print(f"- Ne-dependent volatility: sigma_eff = sigma_e * max(1, (NE_TARGET/Ne))**{ALPHA}. "
          f"With NE_TARGET={NE_TARGET}, Ne<100 amplifies risk; Ne≥100 behaves like baseline.")
    print("- Survival curves: one line per (Ne_ratio × migrants/year). Expect lower Ne to sit below higher Ne at the same immigration.")
    print("- Adjust NE_TARGET or ALPHA if you want a gentler/stronger Ne effect (e.g., ALPHA=0.5 gentler, 1.0 stronger).")

##################################################################################################################################
##################################################################################################################################
def ne_ratio_over_time(t_years: float,
                        start_ratio: float,
                        end_ratio: float,
                        relax_years: float) -> float:
    return end_ratio - (end_ratio - start_ratio) * np.exp(-t_years / max(relax_years, 1e-9))

def simulate_heterozygosity_drift_recoveringNe(
    Nc: float,
    migrants_per_year: float,
    gen_time_years: float,
    years: int,
    H0: float,
    replicates: int,
    rng: np.random.Generator,
    ne_ratio_start: float,
    ne_ratio_end: float,
    ne_ratio_relax_years: float,
    ne_gen_multiplier: float = 1.0,
):
    """
    Stochastic drift + immigration, BUT Ne is allowed to increase over time
    as the breeding system normalizes (genetic disturbance heals).

    Returns:
        H_traj: shape (replicates, gens+1)
        t_years_vec: array of times in years at each generation
    """

    if Nc <= 0:
        t_years_vec = np.arange(0, years+1, gen_time_years, dtype=float)
        return np.zeros((replicates, len(t_years_vec))), t_years_vec

    gens = int(np.floor(years / max(gen_time_years, 1e-9)))

    # migrants per generation
    m_gen = migrants_per_year * gen_time_years
    # migration fraction per generation
    m = m_gen / Nc
    m = max(0.0, min(1.0, m))

    # initial allele frequency ~0.5 as before
    p0 = 0.5
    scale = H0 / 0.5

    H_traj = np.zeros((replicates, gens+1), dtype=float)
    p = np.full(replicates, p0, dtype=float)

    # record generation 0
    H_traj[:, 0] = (2.0 * p * (1.0 - p)) * scale

    for g in range(1, gens+1):
        t_now_years = g * gen_time_years

        # update effective size for this generation
        ratio_t = ne_ratio_over_time(
            t_now_years,
            start_ratio=ne_ratio_start,
            end_ratio=ne_ratio_end,
            relax_years=ne_ratio_relax_years
        )
        Ne_g = ratio_t * Nc * ne_gen_multiplier
        Ne_g = max(2.0, Ne_g)  # avoid nonsense like Ne<2 in binomial

        # migration first
        if m > 0.0:
            donor_p = rng.uniform(0.3, 0.7, size=p.shape) # genetically rich donor pool, not identical to us
            p = (1.0 - m) * p + m * donor_p #0.5  # 0.5 is immigrant source frequency

        # drift with current Ne_g
        draws = rng.binomial(n=int(2.0 * Ne_g), p=p)
        p = draws / np.maximum(int(2.0 * Ne_g), 1)

        H_now = (2.0 * p * (1.0 - p)) * scale
        H_traj[:, g] = H_now

    t_years_vec = np.arange(0, gens+1, dtype=float) * gen_time_years
    return H_traj, t_years_vec

def simulate_heterozygosity_drift(
    Ne: float,
    Nc: float,
    migrants_per_year: float,
    gen_time_years: float,
    years: int,
    H0: float,
    replicates: int,
    rng: np.random.Generator,
):
    """
    Stochastic Wright–Fisher-style drift + optional immigration.
    We simulate a single biallelic locus per replicate.

    Returns:
        H_traj: np.ndarray with shape (replicates, n_timepoints)
                where columns correspond to times [0, gen_time_years, 2*gen_time_years, ...]
        t_years: np.ndarray of the years those columns correspond to.
    Assumptions:
      - Immigration introduces allele frequency ~0.5 (max diversity source),
        mixed at rate m per generation = migrants/gen / Nc.
      - We start with allele freq p0 consistent with H0 (we choose p0 = 0.5 for max H0).
    """

    if Ne <= 0 or Nc <= 0:
        # degenerate case: instantly inbred
        t_years = np.arange(0, years+1, gen_time_years, dtype=float)
        H_traj = np.zeros((replicates, len(t_years)))
        return H_traj, t_years

    # number of (whole) generations we simulate
    gens = int(np.floor(years / max(gen_time_years, 1e-9)))
    # migrants per generation
    m_gen = migrants_per_year * gen_time_years  # migrants/gen
    # migration fraction per generation
    m = m_gen / Nc  # fraction of breeders replaced by immigrants
    m = max(0.0, min(1.0, m))

    # Initial allele frequency.
    # If H0 = 2 p (1-p), solve p(1-p)=H0/2. Max H0 is 0.5 at p=0.5.
    # If H0 is realistic (0.6-0.7 in your notes), that's actually multilocus
    # expected heterozygosity across many loci. We'll approximate by starting p close to 0.5.
    # That means heterozygosity starts near H0 (scaled).
    # We'll just map: start_heterozygosity = H0, and turn that into p0 ~= 0.5.
    # For simplicity we initialize all replicates at p0 = 0.5.
    p0 = 0.5

    # allocate output
    # We will record at generation 0,1,2,...,gens
    H_traj = np.zeros((replicates, gens+1), dtype=float)
    # initialize allele freq per replicate
    p = np.full(replicates, p0, dtype=float)

    # record generation 0 heterozygosity, scaled to match H0
    # our p0=0.5 gives base heterozygosity H_base0 = 2*0.5*0.5 = 0.5
    # so scale to match user's H0 by ratio (H0 / 0.5)
    scale = H0 / 0.5
    H_traj[:, 0] = (2.0 * p * (1.0 - p)) * scale

    # iterate generations
    for g in range(1, gens+1):
        # gene flow first: mix p with migrant allele freq (assume 0.5 for "good" large donor pool)
        if m > 0.0:
            p = (1.0 - m) * p + m * 0.5

        # genetic drift: multinomial (binomial at 1 locus)
        # sample 2*Ne gametes
        draws = rng.binomial(n=int(max(2.0 * Ne, 2)), p=p)
        p = draws / np.maximum(int(max(2.0 * Ne, 2)), 1)

        # record heterozygosity after this generation
        H_now = (2.0 * p * (1.0 - p)) * scale
        H_traj[:, g] = H_now

    # map generation index to calendar years
    t_years = np.arange(0, gens+1, dtype=float) * gen_time_years
    return H_traj, t_years

import numpy as np

def bootstrap_ci(data, n_boot=1000, ci=95, rng=None):
    """Return mean and (low, high) percentile interval for an array-like sample."""
    data = np.asarray(data)
    rng = rng or np.random.default_rng()
    means = [rng.choice(data, size=len(data), replace=True).mean() for _ in range(n_boot)]
    lower = np.percentile(means, (100 - ci) / 2)
    upper = np.percentile(means, 100 - (100 - ci) / 2)
    return float(data.mean()), float(lower), float(upper)

def disturbance_sigma_multiplier(t_year: int,
                                 start_factor: float,
                                 end_factor: float,
                                 relax_years: float) -> float:
    """
    Time-dependent environmental volatility multiplier for new/restored ponds.
    High at t=0, decays toward end_factor over ~relax_years.

    f(t) = end + (start - end) * exp(-t / relax)
    """
    return end_factor + (start_factor - end_factor) * np.exp(-t_year / max(relax_years, 1e-9))

def climate_modifiers(t_years: int) -> dict:
    """
    Hook for climate forcing.
    Returns modifiers for this specific year t_years.
    Working theory (placeholder):
      r_delta: additive change to intrinsic growth rate r (negative under drying)
      sigma_mult: multiplicative inflation of environmental variance (more extremes)
      K_mult: fractional reduction of effective carrying capacity / habitat quality
    Later you can drive these from SMHI / RCP time series instead of constants.
    """
    # simple placeholder version: stable climate (no effect)
    return {
        "r_delta": 0.0,
        "sigma_mult": 1.0,
        "K_mult": 1.0,
    }


def write_run_settings(
    result_dir: str,
    species: str,
    project_name: str,
    Nc: int,
    NC_METAPOP: int,
    cfg_module,
    disturbance_start: float,
    disturbance_end: float,
    disturbance_relax: float,
    meta_weights: dict,
):
    """
    Dump the parameters used in this run to a human-readable .txt file
    in the result directory. This captures:
      - which species config we used,
      - project-level overrides (like Nc and disturbance profile),
      - and all the core cfg settings (time horizon, K, etc.).
    """
    settings_path = os.path.join(result_dir, "run_settings.txt")

    # pull simple attributes from cfg_module (species-level config)
    cfg_items = {}
    for key, val in cfg_module.__dict__.items():
        if key.startswith("_"):
            continue  # skip __name__, __file__, etc.
        if callable(val):
            continue  # skip functions
        if isinstance(val, (int, float, str, list, tuple, dict)):
            cfg_items[key] = val

    with open(settings_path, "w", encoding="utf-8") as f:
        f.write("RUN SETTINGS\n")
        f.write("=============================\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Species config: {species}\n")
        f.write(f"PROJECT_NAME: {project_name}\n")
        f.write(f"Nc_pop (census size): {Nc}\n")
        f.write(f"NC_METAPOP (genetics at local) = {NC_METAPOP}\n")

        f.write("\n--- Project-specific overrides (this run) ---\n")
        f.write(f"DISTURBANCE_SIGMA_FACTOR_START = {disturbance_start}\n")
        f.write(f"DISTURBANCE_SIGMA_FACTOR_END   = {disturbance_end}\n")
        f.write(f"DISTURBANCE_RELAX_YEARS        = {disturbance_relax}\n")
        f.write(f"META_WEIGHTS                   = {meta_weights}\n")

        f.write("\n--- Species config parameters (cfg) ---\n")
        for key in sorted(cfg_items.keys()):
            f.write(f"{key} = {cfg_items[key]}\n")

    print(f"Saved run settings to {settings_path}")

def migrants_schedule(t_years: int, My_constant: float) -> float:
    """
    Immigration / supplementation per year.
    For now we just return the constant My_constant (backward compatible).
    Later you can do:
        if t_years < 5: return pulse_size
        else: return background_rate
    or drive from a management plan.
    """
    return My_constant

def catastrophe_event(rng, p_event=0.01, severity=0.9, size=None):
    """
    Rare hit that suddenly removes a large fraction of N in that year.
    p_event: probability this year of catastrophe (independent per replicate).
    severity: fraction lost if catastrophe happens (0.9 => lose 90%).
    size: number of replicates. If provided, returns a per-replicate array so
          each replicate draws its own independent catastrophe. If None, returns
          a scalar (legacy behaviour, correlated across all replicates).
    Returns multiplier array of shape (size,), or scalar.
    """
    if size is None:
        # Legacy scalar path — kept for safety but not used in simulations
        if rng.random() < p_event:
            return (1.0 - severity)
        else:
            return 1.0
    hits = rng.random(size) < p_event
    return np.where(hits, 1.0 - severity, 1.0)


def simulate_demography_density_independent(
    N0: float,
    Q: float,
    r: float,
    sigma_e: float,
    T_years: int,
    immigrants_per_year: float,
    K: float,
    model: str,
    theta: float,
    replicates: int,
    rng: np.random.Generator,
    disturb_start: float,
    disturb_end: float,
    disturb_relax: float,
    return_extinct_array: bool = False,
):
    """
    Demography without explicit Ne feedback.
    Density dependence (theta-logistic or Ricker).
    Disturbance volatility that relaxes.
    Climate forcing (r_delta, sigma_mult, K_mult).
    Immigration schedule that can pulse.
    Rare catastrophes.
    Age/sex structure suppresses first-year growth if few actual breeders.
    """

    N = np.full(replicates, float(N0))
    extinct = np.zeros(replicates, dtype=bool)

    for t in range(int(T_years)):
        alive = ~extinct
        if not np.any(alive):
            break

        # Early-pond volatility factor
        mult_t = disturbance_sigma_multiplier(
            t,
            start_factor=disturb_start,
            end_factor=disturb_end,
            relax_years=disturb_relax
        )

        # Climate forcing this year
        climate = climate_modifiers(t)
        sigma_t = sigma_e * mult_t * climate["sigma_mult"]

        # Breeder penalty first year: if mostly juveniles, effective growth is lower
        breeder_penalty = INIT_PROP_ADULT if t == 0 else 1.0
        r_eff = (r + climate["r_delta"]) * breeder_penalty

        # Carrying capacity this year
        K_eff = K * climate["K_mult"]

        # Density regulation term for alive replicates
        Z = rng.standard_normal(alive.sum())
        N_alive = N[alive]

        if model == "theta-logistic":
            dens_term = r_eff * (1.0 - np.power(np.clip(N_alive / max(K_eff, 1e-9), 0.0, np.inf), theta))
        else:  # "ricker"
            dens_term = r_eff * (1.0 - N_alive / max(K_eff, 1e-9))

        mu = dens_term - 0.5 * (sigma_t ** 2)

        # Immigration pulse this year
        I_t = max(0.0, float(migrants_schedule(t, immigrants_per_year)))

        # Update population
        N_new = N_alive * np.exp(mu + sigma_t * Z) + I_t
        N_new = np.clip(N_new, 0.0, None)

        # Catastrophe check (e.g. drought crash) — independent per replicate
        N_new *= catastrophe_event(rng, size=int(alive.sum()))

        # Write back
        N[alive] = N_new

        # Track quasi-extinction
        extinct |= (N <= Q)

    if return_extinct_array:
        # return 1 if extinct, 0 if survived
        return extinct.astype(float)
    else:
        return float(extinct.mean())



def F_with_migration_over_time(
    Ne: float,
    Nc: float,
    migrants_per_year: float,
    gen_time_years: float,
    years: int,
    F0: float
) -> float:
    """
    Recurse F over generations with migration (island model) + drift per gen:
      F_{g+1} = (1 - m)^2 * F_g + (1 - (1 - m)^2) * (1/(2Ne))
    where m = migrants/gen / Nc. We map years -> generations with gen_time_years.
    Returns F_T at the end of the horizon (closest generation).
    """
    if Ne <= 0 or Nc <= 0:
        return 1.0
    gen_time = gen_time_years
    gens = int(np.floor(years / max(gen_time, 1e-9)))
    m_gen = migrants_per_year * gen_time  # migrants/gen
    m = max(0.0, m_gen) / Nc
    one_minus_m2 = (1.0 - m) ** 2
    F = float(np.clip(F0, 0.0, 1.0))
    for _ in range(gens):
        F = one_minus_m2 * F + (1.0 - one_minus_m2) * (1.0 / (2.0 * Ne))
    return float(np.clip(F, 0.0, 1.0))


# ---------- math helpers ----------
def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ---------- genetics ----------
def inbreeding_after_t_generations(Ne: float, generations: float) -> float:
    """
    Expected inbreeding coefficient accumulated over t generations
      F_t = 1 - (1 - 1/(2Ne))^t
    """
    if Ne <= 0:
        return 1.0
    return 1.0 - pow(1.0 - 1.0 / (2.0 * Ne), generations)


def equilibrium_inbreeding_with_migration(Ne: float, migrants_per_generation: float, Nc: float) -> float:
    if Ne <= 0 or Nc <= 0:
        return 1.0
    m = max(0.0, migrants_per_generation) / Nc
    Nem = Ne * m
    return 1.0 / (4.0 * Nem + 1.0)

def breeder_efficiency(Nc_t, half_sat, max_eff=1.0):
    """
    Returns an array in [0, max_eff].
    When Nc_t is very small, efficiency ~0 (few effective breeders).
    Once Nc_t >> half_sat, efficiency ~max_eff (breeder pool 'saturated').
    """
    Nc_t = np.asarray(Nc_t, dtype=float)
    return max_eff * (Nc_t / (Nc_t + half_sat + 1e-9))


# ---------- demography ----------
def sigma_effective(sigma_e: float, Ne: float) -> float:
    """
    Inflate environmental variance when Ne is small (simple proxy for
    greater demographic fragility and inbreeding-related performance noise).
    When Ne >= NE_TARGET → sigma_eff ~ sigma_e.
    When Ne <  NE_TARGET → sigma_eff increases smoothly.
    """
    if Ne <= 0:
        return float("inf")
    factor = max(1.0, (NE_TARGET / Ne)) ** ALPHA
    return sigma_e * factor


def quasi_ext_prob_diffusion(N0: float, Q: float, r: float, sigma_e_eff: float, T_years: float) -> float:
    """
    Probability of quasi-extinction: P(min_{0<=t<=T} N(t) <= Q).
    Uses the two-term first-passage formula from Dennis et al. (1991)
    for geometric Brownian motion with drift mu = r - 0.5*sigma^2:

      P_ext = Phi(z1) + (Q/N0)^(2*mu/sigma^2) * Phi(z2)

    where
      z1 = (log(Q/N0) - mu*T) / (sigma*sqrt(T))
      z2 = (log(Q/N0) + mu*T) / (sigma*sqrt(T))   [= z1 + 2*|log(Q/N0)|/(sigma*sqrt(T))]

    The original single-term version computed only Phi(z1), which is
    P(N(T) <= Q) -- the terminal-time probability -- and therefore
    systematically underestimates extinction risk.

    Reference: Dennis B, Munholland PL, Scott JM (1991) Estimation of
    growth and extinction parameters for endangered species.
    Ecological Monographs 61:115-143.
    """
    if N0 <= 0 or Q <= 0 or T_years <= 0 or sigma_e_eff <= 0:
        return 1.0
    if N0 <= Q:
        return 1.0
    mu = r - 0.5 * sigma_e_eff * sigma_e_eff
    sigma_sqrt_T = sigma_e_eff * math.sqrt(T_years)
    log_ratio = math.log(Q / N0)          # negative (Q < N0)
    z1 = (log_ratio - mu * T_years) / sigma_sqrt_T
    z2 = (log_ratio + mu * T_years) / sigma_sqrt_T
    # Exponent for the reflection term; clamp to avoid overflow
    if sigma_e_eff > 0:
        exp_pow = 2.0 * mu * log_ratio / (sigma_e_eff ** 2)
        reflection = math.exp(min(exp_pow, 700.0))
    else:
        reflection = 0.0
    p = normal_cdf(z1) + reflection * normal_cdf(z2)
    return min(max(p, 0.0), 1.0)


def simulate_demography_with_immigration(N0: float,
                                         Q: float,
                                         r: float,
                                         sigma_e_eff: float,
                                         T_years: int,
                                         immigrants_per_year: float,
                                         replicates: int,
                                         rng: np.random.Generator) -> float:
    """
    Multiplicative environmental noise with additive immigrants:
      N_{t+1} = N_t * exp(r - 0.5*sigma^2 + sigma*Z_t) + I
    Z_t ~ N(0,1).
    """
    if replicates <= 0:
        return float("nan")

    N = np.full(replicates, float(N0))
    extinct = np.zeros(replicates, dtype=bool)

    mu = r - 0.5 * (sigma_e_eff ** 2)
    I = max(0.0, float(immigrants_per_year))

    for _ in range(int(T_years)):
        alive = ~extinct
        if not np.any(alive):
            break
        Z = rng.standard_normal(alive.sum())
        N_alive = N[alive] * np.exp(mu + sigma_e_eff * Z) + I
        N_alive = np.clip(N_alive, 0.0, None)
        N[alive] = N_alive
        extinct |= (N <= Q)

    return extinct.mean()


# ---------- scoring ----------
def score_scenario(F_isolated_worst: float, F_eq_migr: float, pext_mean_noI: float, pext_mean_I: float) -> float:
    g = min(max(min(F_isolated_worst, F_eq_migr), 0.0), 1.0)
    d = min(max(pext_mean_I, 0.0), 1.0)
    w_g = SCORE_WEIGHTS.get("genetic", 0.5)
    w_d = SCORE_WEIGHTS.get("demographic", 0.5)
    return w_g * g + w_d * d



# ---------- run scenarios ----------
@dataclass
class ScenarioRow:
    Nc: int
    Ne_ratio: float
    Ne: float
    gen_time_min: float
    gen_time_max: float
    migrants_per_year: float
    migrants_per_generation_minGT: float
    migrants_per_generation_maxGT: float
    Ft_isolated_minGT: float
    Ft_isolated_maxGT: float
    Feq_migr_minGT: float
    Feq_migr_maxGT: float
    mean_pext_noI: float
    mean_pext_withI: float
    score: float
    meets_survival_target: bool

    demog_density_pext: float
    demog_density_survival: float
    demog_density_meets_target: bool

    # --- constant-ratio genetics (old B) ---
    genetic_H_isolated_T: float
    genetic_H_migration_T: float
    genetic_viability_survival: float   # 1.0 or 0.0
    genetics_survival_meets: bool

    # --- variable-ratio genetics (new B_var) ---
    genetic_H_variable_T: float        # mean H(T) under recovering Ne/Nc
    genetic_viability_variable_survival: float  # 1.0 or 0.0
    genetics_variable_survival_meets: bool

    # --- eco-genetic coupling: demography → genetics (B_link) ---
    genetic_H_linked_T: float                         # mean H(T) from coupled eco-genetics
    genetic_viability_linked_survival: float          # 1.0 or 0.0
    genetics_linked_survival_meets: bool

    # --- Ne-sensitive demography (C) ---
    ne_sensitive_pext: float
    ne_sensitive_survival: float
    ne_sensitive_meets_target: bool

    # --- meta ---
    meta_survival: float
    meta_meets_target: bool


def simulate_time_to_extinction(
    N0: float,
    Q: float,
    r: float,
    sigma_e_eff_base: float,
    T_years: int,
    immigrants_per_year: float,
    replicates: int,
    rng: np.random.Generator,
    disturb_start: float,
    disturb_end: float,
    disturb_relax: float,
):
    """
    Return array of first-extinction times (years). If never extinct by T_years,
    return T_years+1.
    This version includes:
      - time-varying disturbance
      - climate forcing
      - breeder penalty in first year
      - time-varying immigration schedule
      - catastrophes
    """

    if rng is None:
        rng = np.random.default_rng()

    N = np.full(replicates, float(N0))
    TTE = np.full(replicates, int(T_years) + 1, dtype=int)

    for t in range(int(T_years)):
        alive_mask = (TTE == T_years + 1)
        if not np.any(alive_mask):
            break

        # disturbance + climate
        mult_t = disturbance_sigma_multiplier(
            t,
            start_factor=disturb_start,
            end_factor=disturb_end,
            relax_years=disturb_relax
        )
        climate = climate_modifiers(t)

        sigma_t = sigma_e_eff_base * mult_t * climate["sigma_mult"]

        breeder_penalty = INIT_PROP_ADULT if t == 0 else 1.0
        r_eff = (r + climate["r_delta"]) * breeder_penalty

        Z = rng.standard_normal(alive_mask.sum())
        N_alive = N[alive_mask]

        mu = r_eff - 0.5 * (sigma_t ** 2)

        I_t = max(0.0, float(migrants_schedule(t, immigrants_per_year)))

        N_new = N_alive * np.exp(mu + sigma_t * Z) + I_t
        N_new = np.clip(N_new, 0.0, None)

        # catastrophe — independent per replicate
        N_new *= catastrophe_event(rng, size=int(alive_mask.sum()))

        N[alive_mask] = N_new

        just_gone = (N_new <= Q)
        TTE[np.where(alive_mask)[0][just_gone]] = t + 1

    return TTE

def simulate_demography_paths_Cstyle(
    N0: float,
    Q: float,
    r: float,
    sigma_e_eff_base: float,
    T_years: int,
    immigrants_per_year: float,
    replicates: int,
    rng: np.random.Generator,
    disturb_start: float,
    disturb_end: float,
    disturb_relax: float,
):
    """
    C-style demography (Ne-inflated sigma) but we KEEP the whole Nc path for each replicate.
    Includes:
      - disturbance decay
      - climate forcing
      - breeder penalty at founding
      - pulsed immigration
      - catastrophes
    """

    if rng is None:
        rng = np.random.default_rng()

    N = np.full(replicates, float(N0))
    paths = np.zeros((replicates, int(T_years) + 1), dtype=float)
    paths[:, 0] = N

    extinct_time = np.full(replicates, int(T_years) + 1, dtype=int)

    for t in range(int(T_years)):
        alive_mask = (extinct_time == T_years + 1)
        if not np.any(alive_mask):
            paths[:, t+1] = paths[:, t]
            continue

        mult_t = disturbance_sigma_multiplier(
            t,
            start_factor=disturb_start,
            end_factor=disturb_end,
            relax_years=disturb_relax
        )
        climate = climate_modifiers(t)

        sigma_t = sigma_e_eff_base * mult_t * climate["sigma_mult"]

        breeder_penalty = INIT_PROP_ADULT if t == 0 else 1.0
        r_eff = (r + climate["r_delta"]) * breeder_penalty

        Z = rng.standard_normal(alive_mask.sum())
        N_alive = N[alive_mask]

        mu = r_eff - 0.5 * (sigma_t ** 2)

        I_t = max(0.0, float(migrants_schedule(t, immigrants_per_year)))

        N_new = N_alive * np.exp(mu + sigma_t * Z) + I_t
        N_new = np.clip(N_new, 0.0, None)

        # catastrophe shock — independent per replicate
        N_new *= catastrophe_event(rng, size=int(alive_mask.sum()))

        # update N
        N[alive_mask] = N_new

        # mark first time they cross Q
        just_gone = (N_new <= Q)
        extinct_time[np.where(alive_mask)[0][just_gone]] = t + 1

        # after extinction, hold at Q so genetics sees "effectively gone"
        N[N <= Q] = np.minimum(N[N <= Q], Q)

        paths[:, t+1] = N

    return paths, extinct_time

def simulate_genetics_on_demography_paths(
    Nc_paths: np.ndarray,
    gen_time_years: float,
    H0: float,
    migrants_per_year: float,
    rng: np.random.Generator,
    ne_ratio_start: float,
    ne_ratio_end: float,
    ne_ratio_relax_years: float,
    K_scale: float,
    ne_gen_multiplier: float = 1.0,
):
    """
    Eco-genetic coupling ("B_link"):
    Run drift+immigration where each replicate's census size Nc_t comes from a
    demographic trajectory (C-style). Ne each generation depends on BOTH:
      (i) time-since-foundation healing of breeding structure (ne_ratio_over_time),
      (ii) current Nc_t relative to carrying capacity (abundance penalty if tiny).

    Arguments:
        Nc_paths: array (replicates, T_years+1) from simulate_demography_paths_Cstyle
        gen_time_years: years / generation (scalar; we reuse GEN_TIME_FOR_GENETICS)
        H0: baseline heterozygosity at t=0
        migrants_per_year: immigrants per calendar year (My)
        rng: np.random.Generator
        ne_ratio_start/end/relax: parameters controlling how Ne/Nc recovers
        K_scale: reference "good" pond census (use NC_METAPOP or K)

    Returns:
        H_end: heterozygosity at final time (per replicate, shape [replicates])
        H_traj_mean: mean H(t) over reps at each recorded generation
        H_traj_lo, H_traj_hi: bootstrap CI bands (same length as H_traj_mean)
        t_years_vec: vector of generation times in years
    """

    reps, T_plus1 = Nc_paths.shape
    total_years = T_plus1 - 1  # because Nc_paths covers 0..T
    gens = int(np.floor(total_years / max(gen_time_years, 1e-9)))

    # start all allele freqs ~0.5, scale H by H0 just like simulate_heterozygosity_drift_recoveringNe
    p = np.full(reps, 0.5, dtype=float)
    scale = H0 / 0.5

    # record H through time (generation steps)
    H_mat = np.zeros((reps, gens+1), dtype=float)
    H_mat[:, 0] = (2.0 * p * (1.0 - p)) * scale

    for g in range(1, gens+1):
        this_year = int(g * gen_time_years)
        this_year = min(this_year, total_years)

        Nc_t = Nc_paths[:, this_year].astype(float)  # current census size per replicate

        # adjust starting ratio based on breeder availability (sex ratio, maturity)
        # working theory: fewer effective breeders in year 0 -> lower start ratio.
        structural_start = ne_ratio_start * INIT_PROP_ADULT * (INIT_FEMALE_FRAC * (1.0 - INIT_FEMALE_FRAC)) * 4.0
        # Explanation: max of x*(1-x) is 0.25 at x=0.5; multiply by 4 rescales so 0.5/0.5 ->1.0

        base_ratio = ne_ratio_over_time(
            t_years = g * gen_time_years,
            start_ratio = structural_start,
            end_ratio   = ne_ratio_end,
            relax_years = ne_ratio_relax_years
        )

        # abundance factor in [0,1], softer penalty via sqrt
        abundance_factor = breeder_efficiency(
            Nc_t,
            half_sat = 0.5 * K_scale,    # half-saturation ~ half of (healthy) pond size
            max_eff  = 1.0,
        )
        ratio_t = base_ratio * abundance_factor

        Ne_g = ratio_t * Nc_t * ne_gen_multiplier
        Ne_g = np.maximum(2.0, Ne_g)  # avoid nonsense

        # migrants this generation = sum of yearly migrants across that gen window
        # simple approx: use migrants at this_year
        migrants_this_year = migrants_schedule(this_year, migrants_per_year)
        m_gen = migrants_this_year * gen_time_years

        with np.errstate(divide='ignore', invalid='ignore'):
            m_frac = np.where(
                Nc_t > 0,
                np.clip(m_gen / Nc_t, 0.0, 1.0),
                1.0
            )

        # mix allele frequencies with "donor" pool (diverse source)
        donor_p = rng.uniform(0.3, 0.7, size=reps)
        p = (1.0 - m_frac) * p + m_frac * donor_p

        # genetic drift for each replicate separately
        draws = []
        for i in range(reps):
            # binomial(2*Ne_g[i], p[i])
            twoNe = int(max(2.0 * Ne_g[i], 2.0))
            draws.append(rng.binomial(n=twoNe, p=p[i]) / max(twoNe, 1))
        p = np.array(draws, dtype=float)

        H_now = (2.0 * p * (1.0 - p)) * scale
        H_mat[:, g] = H_now

    # summarize across replicates
    H_end = H_mat[:, -1]

    # bootstrap CI across replicates at each generation step
    mean_curve = []
    lo_curve = []
    hi_curve = []
    for col in range(H_mat.shape[1]):
        m, lo, hi = bootstrap_ci(H_mat[:, col], n_boot=1000, rng=rng)
        mean_curve.append(m)
        lo_curve.append(lo)
        hi_curve.append(hi)

    t_years_vec = np.arange(0, gens+1, dtype=float) * gen_time_years

    return (
        H_end,
        np.array(mean_curve),
        np.array(lo_curve),
        np.array(hi_curve),
        t_years_vec,
    )

def genetics_viability_variable_ratio(
    Nc: float,
    migrants_per_year: float,
    gen_time_years: float,
    years: int,
    H0: float,
    H_threshold_frac: float,
    replicates: int,
    rng: np.random.Generator,
    ne_ratio_start: float,
    ne_ratio_end: float,
    ne_ratio_relax_years: float,
    ne_gen_multiplier: float = 1.0,
):
    """
    Genetic viability under a time-varying Ne/Nc ratio.
    We simulate drift + immigration where Ne_g(t) = Nc * ne_ratio_over_time(t) * ne_gen_multiplier,
    i.e. breeder structure 'recovers' toward a healthier ratio.
    ne_gen_multiplier applies Vitalis/Nunney generational Ne correction when set.

    Returns:
        H_mean_T: mean heterozygosity across replicates at final time
        surv_flag: 1.0 if H_mean_T >= H_threshold (fraction of H0), else 0.0
    """
    H_traj, t_years = simulate_heterozygosity_drift_recoveringNe(
        Nc=Nc,
        migrants_per_year=migrants_per_year,
        gen_time_years=gen_time_years,
        years=years,
        H0=H0,
        replicates=replicates,
        rng=rng,
        ne_ratio_start=ne_ratio_start,
        ne_ratio_end=ne_ratio_end,
        ne_ratio_relax_years=ne_ratio_relax_years,
        ne_gen_multiplier=ne_gen_multiplier,
    )

    # last column = heterozygosity at final time point for each replicate
    H_T = H_traj[:, -1]
    H_mean_T = float(np.mean(H_T))

    H_threshold = H_threshold_frac * H0
    surv_flag = 1.0 if (H_mean_T >= H_threshold) else 0.0
    return H_mean_T, surv_flag

def evaluate_viability(
    demog_A_survival: float,
    genetic_B_const_survival: float,
    genetic_B_var_survival: float,
    genetic_Blink_survival: float,
    demog_C_survival: float,
    surv_threshold: float,
):
    """
    Centralized decision logic.
    Returns dict of pass/fail by submodel, plus a meta score and pass.
    You can change weights here instead of hunting through main().
    """
    # choose which genetic channel should count toward meta; Blink is the eco-genetic truth,
    # so we elevate it.
    meta_survival = (demog_A_survival + genetic_Blink_survival + demog_C_survival) / 3.0

    return {
        "A_pass": demog_A_survival >= surv_threshold,
        "B_const_pass": genetic_B_const_survival >= surv_threshold,
        "B_var_pass": genetic_B_var_survival >= surv_threshold,
        "Blink_pass": genetic_Blink_survival >= surv_threshold,
        "C_pass": demog_C_survival >= surv_threshold,
        "META_survival": meta_survival,
        "META_pass": meta_survival >= surv_threshold,
    }

def time_to_threshold_cross(curve_y, curve_t, threshold, direction="down"):
    """
    Returns the first time t where the curve crosses a threshold.
    direction="down": first time y < threshold
    direction="up":   first time y >= threshold
    None if never crosses.
    """
    for y, t in zip(curve_y, curve_t):
        if direction == "down" and y < threshold:
            return t
        if direction == "up" and y >= threshold:
            return t
    return None



def main():
    rng = np.random.default_rng(RANDOM_SEED)
    # --- Create timestamped results folder ---
    base_results_dir = r"C:\Users\EricWahlsteen\OneDrive - Calluna AB\Skrivbordet\Python_test\Survive\RESULTS"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # safe slug for folder name
    project_slug = re.sub(r"[^A-Za-z0-9_-]+", "_", (PROJECT_NAME or "").strip()) or "RUN"
    result_dir = os.path.join(base_results_dir, f"{project_slug}_{SPECIES}_{ts}")
    os.makedirs(result_dir, exist_ok=True)
    print(f"\nResults folder: {result_dir}")
    
    # save run settings for reproducibility
    write_run_settings(
        result_dir=result_dir,
        species=SPECIES,
        project_name=PROJECT_NAME,
        Nc=NC_POP,
        NC_METAPOP=NC_METAPOP,
        cfg_module=cfg,
        disturbance_start=DISTURBANCE_SIGMA_FACTOR_START,
        disturbance_end=DISTURBANCE_SIGMA_FACTOR_END,
        disturbance_relax=DISTURBANCE_RELAX_YEARS,
        meta_weights=META_WEIGHTS,
    )


    rows = []
    gt_min, gt_max = GEN_TIME_YEARS
    T = TIME_HORIZON_YEARS

    grid = list(itertools.product(R_GRID, SIGMA_E_GRID))

    for ratio in NE_RATIOS:
        # Effective sizes for the two “worlds”:
        Ne_demo = ratio * NC_POP       # core / established pop (for demography models A and C)
        Ne_gen  = ratio * NC_METAPOP   # local pond / metapop unit (for genetics model B)

        # Generational Ne correction (Vitalis / Nunney) — genetics sub-models only.
        # ne_gen_mult_* converts annual-scale Ne to generational Ne for the given generation time.
        ne_gen_mult_min = _ne_gen_multiplier(gt_min)
        ne_gen_mult_max = _ne_gen_multiplier(gt_max)
        Ne_gen_min = Ne_gen * ne_gen_mult_min   # corrected Ne for min-gen-time scenarios
        Ne_gen_max = Ne_gen * ne_gen_mult_max   # corrected Ne for max-gen-time scenarios

        # Genetics
        generations_minGT = T / gt_min
        generations_maxGT = T / gt_max
        Ft_iso_minGT = inbreeding_after_t_generations(Ne_gen_min, generations_minGT)
        Ft_iso_maxGT = inbreeding_after_t_generations(Ne_gen_max, generations_maxGT)

        # Demography WITHOUT immigration (analytical), but now using Ne-dependent sigma
        pexts_noI = []
        for r, se in grid:
            se_eff = sigma_effective(se, Ne_demo)
            pext = quasi_ext_prob_diffusion(NC_POP, QUASI_EXT_THRESHOLD, r, se_eff, T)
            pexts_noI.append(pext)
        mean_pext_noI = float(np.mean(pexts_noI))

        for My in MIGRANTS_PER_YEAR:
            # Genetics with immigration
            H_threshold = H_MIN_FRAC * H0
            Mgen_minGT = My * gt_min
            Mgen_maxGT = My * gt_max
            Feq_minGT = equilibrium_inbreeding_with_migration(Ne_gen_min, Mgen_minGT, NC_METAPOP)
            Feq_maxGT = equilibrium_inbreeding_with_migration(Ne_gen_max, Mgen_maxGT, NC_METAPOP)

            # ===== Independent Model A: Demography with density dependence (Ne-independent) =====
            demog_density_pexts = []
            for r, se in grid:
                demog_density_pexts.append(
                    simulate_demography_density_independent(
                        N0=NC_POP,
                        Q=QUASI_EXT_THRESHOLD,
                        r=r,
                        sigma_e=se,
                        T_years=T,
                        immigrants_per_year=My,
                        K=K,
                        model=DENSITY_MODEL,
                        theta=THETA,
                        replicates=REPLICATES,
                        rng=rng,
                        disturb_start=DISTURBANCE_SIGMA_FACTOR_START,
                        disturb_end=DISTURBANCE_SIGMA_FACTOR_END,
                        disturb_relax=DISTURBANCE_RELAX_YEARS,
                    )
                )
            demog_density_pext = float(np.mean(demog_density_pexts))
            demog_density_surv = 1.0 - demog_density_pext
            demog_density_flag = demog_density_surv >= SURVIVAL_TARGET

            # ===== Independent Model B: Genetics-only =====
            gen_time_for_genetics = GEN_TIME_FOR_GENETICS or GEN_TIME_YEARS[0]
            # generational Ne correction for this specific gen-time setting
            ne_gen_mult_gfg = _ne_gen_multiplier(gen_time_for_genetics)
            Ne_gen_gfg = Ne_gen * ne_gen_mult_gfg

            gens_minGT = T / gen_time_for_genetics

            # Isolated drift only
            Ft_iso_T = inbreeding_after_t_generations(Ne_gen_gfg, gens_minGT)
            H_iso_T = H0 * (1.0 - Ft_iso_T)

            # Drift + immigration
            Ft_mig_T = F_with_migration_over_time(
                Ne=Ne_gen_gfg,
                Nc=NC_METAPOP,
                migrants_per_year=My,
                gen_time_years=gen_time_for_genetics,
                years=T,
                F0=0.0
            )
            H_mig_T = H0 * (1.0 - Ft_mig_T)

            # Choose evaluation rule
            if GENETIC_EVAL == "isolated":
                H_for_rule = H_iso_T
            elif GENETIC_EVAL == "best_of_both":
                H_for_rule = max(H_iso_T, H_mig_T)
            else:  # "with_migration"
                H_for_rule = H_mig_T

            genetics_flag = (H_for_rule >= H_threshold)
            genetics_survival = 1.0 if genetics_flag else 0.0

            # ===== New: Genetics-only with variable Ne/Nc ratio over time (B_var) =====
            H_var_T, surv_flag_var = genetics_viability_variable_ratio(
                Nc=NC_METAPOP,
                migrants_per_year=My,
                gen_time_years=gen_time_for_genetics,
                years=T,
                H0=H0,
                H_threshold_frac=H_MIN_FRAC,
                replicates=REPLICATES,
                rng=rng,
                ne_ratio_start=NE_RATIO_START,
                ne_ratio_end=NE_RATIO_END,
                ne_ratio_relax_years=NE_RATIO_RELAX_YEARS,
                ne_gen_multiplier=ne_gen_mult_gfg,
            )
            genetics_var_flag = bool(surv_flag_var >= 1.0)

            # ===== Independent Model C: Ne-sensitive demography (using extinction-time CDF) =====
            # We estimate survival as P(TTE > T), i.e. fraction of populations not extinct by horizon T.
            # We do this by simulating full trajectories once per (r, se) combo, collecting the
            # distribution of first-extinction times, and then averaging across the grid.

            surv_fracs = []
            for r, se in grid:
                se_eff = sigma_effective(se, Ne_demo)
                times = simulate_time_to_extinction(
                    N0=NC_POP,
                    Q=QUASI_EXT_THRESHOLD,
                    r=r,
                    sigma_e_eff_base=se_eff,
                    T_years=T,
                    immigrants_per_year=My,
                    replicates=REPLICATES,
                    rng=rng,
                    disturb_start=DISTURBANCE_SIGMA_FACTOR_START,
                    disturb_end=DISTURBANCE_SIGMA_FACTOR_END,
                    disturb_relax=DISTURBANCE_RELAX_YEARS,
                )
                survived = np.mean(times > T)
                surv_fracs.append(survived)

            ne_sensitive_surv = float(np.mean(surv_fracs))
            ne_sensitive_pext = 1.0 - ne_sensitive_surv
            ne_sensitive_flag = ne_sensitive_surv >= SURVIVAL_TARGET

            # ===== Coupled eco-genetics: B_link (C -> B) =====
            # We want demography paths under C-style assumptions,
            # and then we feed those paths into genetics.
            # We'll average across (r, se) grid like we do elsewhere.

            H_end_all = []

            for r, se in grid:
                # C-style sigma inflation from Ne_demo
                se_eff = sigma_effective(se, Ne_demo)

                # draw replicate demographic paths
                paths, extinct_time = simulate_demography_paths_Cstyle(
                    N0=NC_METAPOP,                # IMPORTANT: genetics is at pond scale
                    Q=QUASI_EXT_THRESHOLD,
                    r=r,
                    sigma_e_eff_base=se_eff,
                    T_years=T,
                    immigrants_per_year=My,
                    replicates=REPLICATES,
                    rng=rng,
                    disturb_start=DISTURBANCE_SIGMA_FACTOR_START,
                    disturb_end=DISTURBANCE_SIGMA_FACTOR_END,
                    disturb_relax=DISTURBANCE_RELAX_YEARS,
                )

                # run genetics on those demographic paths
                (
                    H_end,
                    H_mean_curve_link,
                    H_lo_curve_link,
                    H_hi_curve_link,
                    t_years_link,
                ) = simulate_genetics_on_demography_paths(
                    Nc_paths=paths,
                    gen_time_years=gen_time_for_genetics,
                    H0=H0,
                    migrants_per_year=My,
                    rng=rng,
                    ne_ratio_start=NE_RATIO_START,
                    ne_ratio_end=NE_RATIO_END,
                    ne_ratio_relax_years=NE_RATIO_RELAX_YEARS,
                    K_scale=NC_METAPOP,   # scale 'healthy' breeder census
                    ne_gen_multiplier=ne_gen_mult_gfg,
                )

                H_end_all.append(H_end)

            # concatenate across all (r,se) scenarios, then summarize
            if len(H_end_all) > 0:
                H_end_all = np.concatenate(H_end_all, axis=0)
                H_link_T_mean = float(np.mean(H_end_all))
            else:
                H_link_T_mean = float("nan")

            H_threshold = H_MIN_FRAC * H0
            genetics_link_flag = (H_link_T_mean >= H_threshold)
            genetics_link_survival = 1.0 if genetics_link_flag else 0.0


            # ===== Meta-model: weighted combination of survivals =====
            wA =META_WEIGHTS .get("demography_density", 1/3)
            wB = META_WEIGHTS.get("genetics_only", 1/3)
            wC = META_WEIGHTS.get("ne_sensitive_demography", 1/3)
            # normalize just in case
            wsum = max(1e-9, (wA + wB + wC))
            wA, wB, wC = wA/wsum, wB/wsum, wC/wsum
            meta_survival = wA * demog_density_surv + wB * genetics_survival + wC * ne_sensitive_surv
            meta_flag = meta_survival >= SURVIVAL_TARGET


            # Demography WITH immigration (simulation), with Ne-dependent sigma
            pexts_withI = []
            for r, se in grid:
                se_eff = sigma_effective(se, Ne_demo)
                pext_sim = simulate_demography_with_immigration(
                    N0=NC_POP,
                    Q=QUASI_EXT_THRESHOLD,
                    r=r,
                    sigma_e_eff=se_eff,
                    T_years=T,
                    immigrants_per_year=My,
                    replicates=REPLICATES,
                    rng=rng
                )
                pexts_withI.append(pext_sim)
            mean_pext_withI = float(np.mean(pexts_withI))

            meets_flag = (1.0 - mean_pext_withI) >= SURVIVAL_TARGET
            score = score_scenario(Ft_iso_minGT, Feq_minGT, mean_pext_noI, mean_pext_withI)

            viability_eval = evaluate_viability(
                demog_A_survival = demog_density_surv,
                genetic_B_const_survival = genetics_survival,
                genetic_B_var_survival   = float(surv_flag_var),
                genetic_Blink_survival   = genetics_link_survival,
                demog_C_survival = ne_sensitive_surv,
                surv_threshold = SURVIVAL_TARGET,
            )
            meta_survival = viability_eval["META_survival"]
            meta_flag     = viability_eval["META_pass"]


            rows.append(ScenarioRow(
                Nc=NC_POP,
                Ne_ratio=ratio,
                Ne=Ne_demo,
                gen_time_min=gt_min,
                gen_time_max=gt_max,
                migrants_per_year=My,
                migrants_per_generation_minGT=Mgen_minGT,
                migrants_per_generation_maxGT=Mgen_maxGT,
                Ft_isolated_minGT=Ft_iso_minGT,
                Ft_isolated_maxGT=Ft_iso_maxGT,
                Feq_migr_minGT=Feq_minGT,
                Feq_migr_maxGT=Feq_maxGT,
                mean_pext_noI=mean_pext_noI,
                mean_pext_withI=mean_pext_withI,
                score=score,
                meets_survival_target=bool((1.0 - mean_pext_withI) >= SURVIVAL_TARGET),

                demog_density_pext=demog_density_pext,
                demog_density_survival=demog_density_surv,
                demog_density_meets_target=bool(demog_density_flag),

                # B_const (original standalone genetics)
                genetic_H_isolated_T=H_iso_T,
                genetic_H_migration_T=H_mig_T,
                genetic_viability_survival=genetics_survival,
                genetics_survival_meets=bool(genetics_flag),

                # B_var (time-recovering Ne/Nc ratio, no demography feedback)
                genetic_H_variable_T=H_var_T,
                genetic_viability_variable_survival=float(surv_flag_var),
                genetics_variable_survival_meets=bool(genetics_var_flag),

                # B_link (eco-genetic coupling: demography -> genetics)
                genetic_H_linked_T=H_link_T_mean,
                genetic_viability_linked_survival=genetics_link_survival,
                genetics_linked_survival_meets=bool(genetics_link_flag),

                # C (Ne-sensitive demography)
                ne_sensitive_pext=ne_sensitive_pext,
                ne_sensitive_survival=ne_sensitive_surv,
                ne_sensitive_meets_target=bool(ne_sensitive_flag),

                meta_survival=meta_survival,
                meta_meets_target=bool(meta_flag),
            ))


    df = pd.DataFrame([asdict(r) for r in rows])
    df_sorted = df.sort_values("score", ascending=True).reset_index(drop=True)

    # Print and save
    pd.set_option("display.float_format", lambda x: f"{x:0.3f}")
    print("\n=== Top-ranked scenarios (lower score = better) ===")
    cols_view = [
        "Ne_ratio",
        "Ne",
        "migrants_per_year",

        # Submodel A
        "demog_density_survival",
        "demog_density_meets_target",

        # Submodel B_const
        "genetic_viability_survival",
        "genetics_survival_meets",
        "genetic_H_migration_T",

        # Submodel B_var
        "genetic_viability_variable_survival",
        "genetics_variable_survival_meets",
        "genetic_H_variable_T",

        # Submodel B_link (C → B)
        "genetic_viability_linked_survival",
        "genetics_linked_survival_meets",
        "genetic_H_linked_T",

        # Submodel C
        "ne_sensitive_survival",
        "ne_sensitive_meets_target",

        # Meta
        "meta_survival",
        "meta_meets_target",
    ]

    print(df_sorted[cols_view].head(10).to_string(index=False))

    xlsx_path = os.path.join(result_dir, "extinction_scenarios.xlsx")
    # --- Add model letters to key column headers before saving ---
    rename_map = {
        "demog_density_survival":        f"A_survival_Nc{NC_POP}",
        "demog_density_meets_target":    f"A_meets_target_Nc{NC_POP}",

        "genetic_viability_survival":            f"B_const_survival_Nc{NC_METAPOP}",
        "genetics_survival_meets":               f"B_const_meets_target_Nc{NC_METAPOP}",
        "genetic_H_migration_T":                 f"B_const_H_migration_T_Nc{NC_METAPOP}",

        "genetic_viability_variable_survival":   f"Bvar_survival_Nc{NC_METAPOP}",
        "genetics_variable_survival_meets":      f"Bvar_meets_target_Nc{NC_METAPOP}",
        "genetic_H_variable_T":                  f"Bvar_H_T_Nc{NC_METAPOP}",

        "genetic_viability_linked_survival":   f"Blink_survival_Nc{NC_METAPOP}",
        "genetics_linked_survival_meets":      f"Blink_meets_target_Nc{NC_METAPOP}",
        "genetic_H_linked_T":                  f"Blink_H_T_Nc{NC_METAPOP}",


        "ne_sensitive_survival":         f"C_survival_Nc{NC_POP}",
        "ne_sensitive_meets_target":     f"C_meets_target_Nc{NC_POP}",

        "meta_survival":                 "META_survival",
        "meta_meets_target":             "META_meets_target",
    }
    df_sorted.rename(columns=rename_map, inplace=True)
    for col in ["score", "meets_survival_target"]:
        if col in df_sorted.columns:
            df_sorted.drop(columns=col, inplace=True)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df_sorted.to_excel(writer, index=False, sheet_name="Results")

    print(f"\nSaved all scenarios to {xlsx_path}")

    # --------- FIGURES (two) ---------
    print("\nGenerating graphs...")
    out_dir = os.path.dirname(xlsx_path)
    os.makedirs(out_dir, exist_ok=True)


    # ---------- FIGURE: Sensitivity of extinction risk to sigma_e ----------
    print("Building sigma_e sensitivity curves ...")
    sigma_vals = np.linspace(0.15, 0.50, 10)  # adjust range if you like
    r_vals = R_GRID                          # average across your r-uncertainty

    nrows = 1
    ncols = len(NE_RATIOS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8*ncols, 4.2), sharey=True)
    if ncols == 1:
        axes = [axes]

    for ax, ratio in zip(axes, NE_RATIOS):
        Ne = ratio * NC_POP
        for My in MIGRANTS_PER_YEAR:
            risks = []
            for se in sigma_vals:
                # average over r-grid, WITH immigration, using Ne-dependent sigma
                vals = []
                for r in r_vals:
                    se_eff = sigma_effective(se, Ne)
                    p = simulate_demography_with_immigration(
                        N0=NC_POP, Q=QUASI_EXT_THRESHOLD, r=r, sigma_e_eff=se_eff,
                        T_years=TIME_HORIZON_YEARS, immigrants_per_year=My,
                        replicates=REPLICATES, rng=np.random.default_rng(RANDOM_SEED+777)
                    )
                    vals.append(p)
                risks.append(float(np.mean(vals)))
            ax.plot(sigma_vals, risks, label=f"Mig/yr={My}")
        ax.set_xlabel("Environmental SD on ln scale (σₑ)")
        ax.set_title(f"Ne/Nc = {ratio:.2f} (Ne={Ne:.0f})")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(f"P(extinction ≤ Q) by {TIME_HORIZON_YEARS} years")
    axes[-1].legend(title="Immigrants", fontsize=8)
    fig_sigma = os.path.join(out_dir, "sensitivity_sigma.png")
    plt.tight_layout()
    plt.savefig(fig_sigma, dpi=150)
    plt.close()
    print(f"Saved sensitivity plot: {fig_sigma}")

    # ---------- FIGURE: Combined genetic–demographic risk scatter ----------
    print("Building genetic–demographic risk scatter ...")
    fig, ax = plt.subplots(figsize=(7.5, 6))

    # marker per Ne_ratio
    marker_shapes = ["o", "s", "D", "^", "v", "<", ">"]
    markers = {ratio: marker_shapes[i % len(marker_shapes)] for i, ratio in enumerate(NE_RATIOS)}

    # color by migrants/year
    cmap = plt.cm.get_cmap("plasma")
    colors = {My: cmap(i / max(1, len(MIGRANTS_PER_YEAR)-1)) for i, My in enumerate(sorted(MIGRANTS_PER_YEAR))}


    for _, row in df.iterrows():
        x = row["Feq_migr_minGT"]        # genetic risk proxy (lower better)
        y = row["mean_pext_withI"]       # demographic risk (lower better)
        mrk = markers.get(row["Ne_ratio"], "o")
        col = colors.get(row["migrants_per_year"], "gray")
        ax.scatter(x, y, s=70, marker=mrk, edgecolor="black", linewidth=0.5, color=col)

    # helpers: marginal "good zone" lines
    ax.axvline(0.05, color="gray", linestyle="--", linewidth=1, label="F_eq = 0.05")
    ax.axhline(0.10, color="gray", linestyle=":", linewidth=1, label="P_ext = 0.10")

    ax.set_xlabel("Equilibrium inbreeding F_eq (migration, gen time = min)")
    ax.set_ylabel(f"P(extinction ≤ Q) at {TIME_HORIZON_YEARS} years (with immigration)")
    ax.set_title("Genetic (x) vs Demographic (y) risk per scenario")

    # legends
    from matplotlib.lines import Line2D
    leg1 = [Line2D([0],[0], marker=markers[r], color="w", label=f"Ne/Nc={r:.2f}",
                   markerfacecolor="lightgray", markeredgecolor="black", markersize=9)
            for r in NE_RATIOS]
    leg2 = [Line2D([0],[0], marker="o", color=c, label=f"Mig/yr={m}", markeredgecolor="black", markersize=9)
            for m, c in colors.items()]

    legend1 = ax.legend(handles=leg1, title="Ne/Nc", loc="upper right")
    ax.add_artist(legend1)
    ax.legend(handles=leg2, title="Immigration", loc="lower right")

    ax.grid(alpha=0.25)
    fig_scatter = os.path.join(out_dir, "combined_genetic_demographic_scatter.png")
    plt.tight_layout()
    plt.savefig(fig_scatter, dpi=150)
    plt.close()
    print(f"Saved scatter: {fig_scatter}")

    # FIGURE A: Submodel A (density-dependent demography, Ne-independent)
    print("Building Submodel A figure: density-dependent demography survival with CI ...")

    TS_dem = np.arange(0, TIME_HORIZON_YEARS + 1, 5)
    rng_plot = np.random.default_rng(RANDOM_SEED + 1001)

    plt.figure(figsize=(8, 5))

    for My in MIGRANTS_PER_YEAR:
        mean_surv_series = []
        lo_surv_series = []
        hi_surv_series = []

        for Tcur in TS_dem:
            if Tcur == 0:
                # At time 0 survival is trivially 1 with no uncertainty
                mean_surv_series.append(1.0)
                lo_surv_series.append(1.0)
                hi_surv_series.append(1.0)
                continue

            # collect replicate-level survival indicators across r, sigma grid
            all_survived = []
            for r, se in grid:
                extinct_flags = simulate_demography_density_independent(
                    N0=NC_POP,
                    Q=QUASI_EXT_THRESHOLD,
                    r=r,
                    sigma_e=se,
                    T_years=int(Tcur),
                    immigrants_per_year=My,
                    K=K,
                    model=DENSITY_MODEL,
                    theta=THETA,
                    replicates=REPLICATES,
                    rng=rng_plot,
                    disturb_start=DISTURBANCE_SIGMA_FACTOR_START,
                    disturb_end=DISTURBANCE_SIGMA_FACTOR_END,
                    disturb_relax=DISTURBANCE_RELAX_YEARS,
                    return_extinct_array=True,  # << NEW
                )
                # extinct_flags is length REPLICATES, 1 if extinct, 0 if alive
                survived_flags = 1.0 - extinct_flags  # 1 if alive
                all_survived.append(survived_flags)

            all_survived = np.concatenate(all_survived, axis=0)

            mean_s, lo_s, hi_s = bootstrap_ci(all_survived, n_boot=1000, rng=rng_plot)
            mean_surv_series.append(mean_s)
            lo_surv_series.append(lo_s)
            hi_surv_series.append(hi_s)

        # draw mean line
        plt.plot(
            TS_dem,
            mean_surv_series,
            label=(f"Immigration {My}/yr" if ENGLISH else f"Invandring {My}/år"),
            linewidth=2
        )
        # draw CI ribbon
        plt.fill_between(
            TS_dem,
            lo_surv_series,
            hi_surv_series,
            alpha=0.2
        )

    # 95% reference line
    plt.axhline(SURVIVAL_TARGET, color="black", linestyle="--", linewidth=1)
    plt.text(
        TIME_HORIZON_YEARS,
        SURVIVAL_TARGET + 0.003,
        f"{int(SURVIVAL_TARGET*100)}%",
        ha="right", va="bottom", fontsize=9, color="black"
    )

    plt.xlabel("Year" if ENGLISH else "År")
    plt.ylabel("Survival probability (1 − extinction risk)" if ENGLISH else "Överlevnadssannolikhet (1 − utdöenderisk)")
    plt.title(f"Sub-model A: Demography (Nc≈{NC_POP}), density regulation without Ne effects" if ENGLISH else f"Submodell A: Demografi (Nc≈{NC_POP}), täthetsreglering utan Ne-effekter")
    plt.legend(fontsize=8)
    plt.tight_layout()

    fig_A = os.path.join(out_dir, "mod_A_survival_density_demography.png")
    plt.savefig(fig_A, dpi=150)
    plt.close()
    print(f"Saved Submodel A figure: {fig_A}")

    # FIGURE B: Submodel B (genetics-only, stochastic drift + CI)
    print("Building Submodel B figure: genetic heterozygosity with drift CI ...")

    gen_time_for_plot = int(GEN_TIME_YEARS[0]) if GEN_TIME_FOR_GENETICS is None else GEN_TIME_FOR_GENETICS
    rng_gen = np.random.default_rng(RANDOM_SEED + 1500)

    plt.figure(figsize=(9, 6))

    # We'll assign each Ne_ratio a color, and each migration rate a linestyle.
    color_map = plt.cm.viridis  # stable, perceptually nice
    ratio_colors = {
        ratio: color_map(i / max(1, len(NE_RATIOS) - 1))
        for i, ratio in enumerate(sorted(NE_RATIOS))
    }

    linestyles_pool = ["-", "--", ":", "-."]  # cycle if >4 migration rates
    style_map = {
        My: linestyles_pool[i % len(linestyles_pool)]
        for i, My in enumerate(sorted(MIGRANTS_PER_YEAR))
    }

    # Loop over Ne ratio and migrants/year
    for ratio in NE_RATIOS:
        Ne = ratio * NC_METAPOP
        for My in MIGRANTS_PER_YEAR:

            # Simulate stochastic drift with immigration My
            H_traj, t_years = simulate_heterozygosity_drift_recoveringNe(
                Nc=NC_METAPOP,
                migrants_per_year=My,
                gen_time_years=gen_time_for_plot,
                years=TIME_HORIZON_YEARS,
                H0=H0,
                replicates=REPLICATES,
                rng=rng_gen,
                ne_ratio_start=NE_RATIO_START,            # <- project-level start
                ne_ratio_end=NE_RATIO_END,                # <- project-level end
                ne_ratio_relax_years=NE_RATIO_RELAX_YEARS,
                ne_gen_multiplier=_ne_gen_multiplier(gen_time_for_plot),
            )
            # Bootstrap CI across replicates at each recorded time point
            mean_curve = []
            lo_curve = []
            hi_curve = []
            for col in range(H_traj.shape[1]):
                m, lo, hi = bootstrap_ci(H_traj[:, col], n_boot=1000, rng=rng_gen)
                mean_curve.append(m)
                lo_curve.append(lo)
                hi_curve.append(hi)

            lbl = (f"Ne/Nc={ratio:.2f}, Imm={My}/yr" if ENGLISH else f"Ne/Nc={ratio:.2f}, Inv={My}/år")
            colr = ratio_colors[ratio]
            ls   = style_map[My]

            # Plot mean line
            plt.plot(
                t_years,
                mean_curve,
                linestyle=ls,
                linewidth=1.8,
                color=colr,
                label=lbl,
            )

            # Shaded CI
            plt.fill_between(
                t_years,
                lo_curve,
                hi_curve,
                color=colr,
                alpha=0.15,
            )

    # Horizontal line for viability threshold: keep this, it's biologically meaningful
    H_threshold = H_MIN_FRAC * H0
    plt.axhline(H_threshold, color="black", linestyle=":", linewidth=1)
    plt.text(
        TIME_HORIZON_YEARS,
        H_threshold + 0.01,
        (f"{int(H_MIN_FRAC*100)}% of H0" if ENGLISH else f"{int(H_MIN_FRAC*100)}% av H0"),
        ha="right", va="bottom", fontsize=9, color="black"
    )

    plt.xlabel("Year" if ENGLISH else "År")
    plt.ylabel("Heterozygosity H(t)" if ENGLISH else "Heterozygositet H(t)")
    plt.title(f"Sub-model B: Genetic variation (Nc≈{NC_METAPOP} ~ K={K}, re-established local population)" if ENGLISH else f"Submodell B: Genetisk variation (Nc≈{NC_METAPOP} ~ K={K}, återetablerad lokal population)")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    fig_B = os.path.join(out_dir, "mod_B_heterozygosity_genetics.png")
    plt.savefig(fig_B, dpi=150)
    plt.close()
    print(f"Saved Submodel B figure: {fig_B}")

    # --- FIGURE C: Submodel C (Ne-sensitive demography, CDF-based survival) ---
    print("Building Submodel C figure (CDF-based survival) ...")

    TS_plot = np.arange(0, TIME_HORIZON_YEARS + 1, 2)
    rng_plot2 = np.random.default_rng(RANDOM_SEED + 2002)

    plt.figure(figsize=(9, 6))
    rng_plot2 = np.random.default_rng(RANDOM_SEED + 2002)

    # --- Colour and style maps (match Submodel B and Blink) ---
    color_map = plt.cm.viridis
    ratio_colors = {
        ratio: color_map(i / max(1, len(NE_RATIOS) - 1))
        for i, ratio in enumerate(sorted(NE_RATIOS))
    }
    linestyles_pool = ["-", "--", ":", "-."]
    style_map = {
        My: linestyles_pool[i % len(linestyles_pool)]
        for i, My in enumerate(sorted(MIGRANTS_PER_YEAR))
    }

    for ratio in NE_RATIOS:
        Ne_demo = ratio * NC_POP
        colr = ratio_colors[ratio]

        for My in MIGRANTS_PER_YEAR:
            ls = style_map[My]
            all_times_list = []
            for r, se in grid:
                se_eff = sigma_effective(se, Ne_demo)
                times = simulate_time_to_extinction(
                    N0=NC_POP,
                    Q=QUASI_EXT_THRESHOLD,
                    r=r,
                    sigma_e_eff_base=se_eff,
                    T_years=TIME_HORIZON_YEARS,
                    immigrants_per_year=My,
                    replicates=REPLICATES,
                    rng=rng_plot2,
                    disturb_start=DISTURBANCE_SIGMA_FACTOR_START,
                    disturb_end=DISTURBANCE_SIGMA_FACTOR_END,
                    disturb_relax=DISTURBANCE_RELAX_YEARS,
                )
                all_times_list.append(times)
            all_times = np.concatenate(all_times_list, axis=0)

            mean_curve, lo_curve, hi_curve = [], [], []
            for tcur in TS_plot:
                survived_flags = (all_times > tcur).astype(float)
                m, lo, hi = bootstrap_ci(survived_flags, n_boot=1000, rng=rng_plot2)
                mean_curve.append(m)
                lo_curve.append(lo)
                hi_curve.append(hi)

            t_fail = time_to_threshold_cross(
                curve_y = mean_curve,
                curve_t = TS_plot,
                threshold = SURVIVAL_TARGET,
                direction = "down",
            )

            label = (f"Ne/Nc={ratio:.2f}, Imm={My}/yr" if ENGLISH else f"Ne/Nc={ratio:.2f}, Inv={My}/år")
            plt.plot(
                TS_plot,
                mean_curve,
                color=colr,
                linestyle=ls,
                linewidth=1.8,
                label=label,
            )
            plt.fill_between(
                TS_plot,
                lo_curve,
                hi_curve,
                color=colr,
                alpha=0.15,
            )

    plt.axhline(SURVIVAL_TARGET, color="black", linestyle="--", linewidth=1)
    plt.text(
        TIME_HORIZON_YEARS,
        SURVIVAL_TARGET + 0.02,
        f"{int(SURVIVAL_TARGET*100)}%",
        ha="right", va="bottom", fontsize=9, color="black"
    )

    plt.xlabel("Year" if ENGLISH else "År")
    plt.ylabel("Survival probability (1 − extinction time CDF)" if ENGLISH else "Överlevnadssannolikhet (1 − CDF för utdöendetid)")
    plt.title(f"Sub-model C: Demography with Ne-sensitive environmental stochasticity (Nc≈{NC_POP})" if ENGLISH else f"Submodell C: Demografi med Ne-känslig miljöstokasticitet (Nc≈{NC_POP})")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()

    fig_C = os.path.join(result_dir, "mod_C_survival_ne_sensitive.png")
    plt.savefig(fig_C, dpi=150)
    plt.close()
    print(f"Saved Submodel C figure: {fig_C}")

    # --- FIGURE Blink: eco-genetic coupling (C→B) ---
    print("Building Submodel Blink figure: eco-genetic coupling heterozygosity ...")

    plt.figure(figsize=(9, 6))
    rng_link = np.random.default_rng(RANDOM_SEED + 2500)

    # --- Colour and style maps just like Submodel B ---
    color_map = plt.cm.viridis
    ratio_colors = {
        ratio: color_map(i / max(1, len(NE_RATIOS) - 1))
        for i, ratio in enumerate(sorted(NE_RATIOS))
    }
    linestyles_pool = ["-", "--", ":", "-."]
    style_map = {
        My: linestyles_pool[i % len(linestyles_pool)]
        for i, My in enumerate(sorted(MIGRANTS_PER_YEAR))
    }

    for ratio in NE_RATIOS:
        Ne_demo = ratio * NC_POP
        colr = ratio_colors[ratio]
        for My in MIGRANTS_PER_YEAR:
            ls = style_map[My]
            H_end_all = []
            mean_curve_link = None
            t_years_link = None

            for r, se in grid:
                se_eff = sigma_effective(se, Ne_demo)
                paths, _ = simulate_demography_paths_Cstyle(
                    N0=NC_METAPOP,
                    Q=QUASI_EXT_THRESHOLD,
                    r=r,
                    sigma_e_eff_base=se_eff,
                    T_years=TIME_HORIZON_YEARS,
                    immigrants_per_year=My,
                    replicates=REPLICATES,
                    rng=rng_link,
                    disturb_start=DISTURBANCE_SIGMA_FACTOR_START,
                    disturb_end=DISTURBANCE_SIGMA_FACTOR_END,
                    disturb_relax=DISTURBANCE_RELAX_YEARS,
                )
                (
                    H_end,
                    H_mean_curve_link,
                    H_lo_curve_link,
                    H_hi_curve_link,
                    t_years_link,
                ) = simulate_genetics_on_demography_paths(
                    Nc_paths=paths,
                    gen_time_years=GEN_TIME_FOR_GENETICS,
                    H0=H0,
                    migrants_per_year=My,
                    rng=rng_link,
                    ne_ratio_start=NE_RATIO_START,
                    ne_ratio_end=NE_RATIO_END,
                    ne_ratio_relax_years=NE_RATIO_RELAX_YEARS,
                    K_scale=NC_METAPOP,
                    ne_gen_multiplier=_ne_gen_multiplier(GEN_TIME_FOR_GENETICS),
                )
                # compute recovery time for this (ratio, My, r, se) combo
                H_threshold_curve = H_MIN_FRAC * H0
                t_recover = time_to_threshold_cross(
                    curve_y = H_mean_curve_link,
                    curve_t = t_years_link,
                    threshold = H_threshold_curve,
                    direction = "up",
                )
                # store these if you want later export
                # (You can accumulate min/median across grid for reporting.)

                H_end_all.append(H_end)

            if H_mean_curve_link is None:
                continue

            label = (f"Ne/Nc={ratio:.2f}, Imm={My}/yr" if ENGLISH else f"Ne/Nc={ratio:.2f}, Inv={My}/år")

            # Mean line & CI ribbon with viridis colour
            plt.plot(
                t_years_link,
                H_mean_curve_link,
                color=colr,
                linestyle=ls,
                lw=1.8,
                label=label,
            )
            plt.fill_between(
                t_years_link,
                H_lo_curve_link,
                H_hi_curve_link,
                color=colr,
                alpha=0.15,
            )

    H_threshold = H_MIN_FRAC * H0
    plt.axhline(H_threshold, color="black", linestyle=":", linewidth=1)
    plt.text(
        TIME_HORIZON_YEARS,
        H_threshold + 0.01,
        (f"{int(H_MIN_FRAC*100)}% of H0" if ENGLISH else f"{int(H_MIN_FRAC*100)}% av H0"),
        ha="right", va="bottom", fontsize=9, color="black"
    )

    plt.xlabel("Year" if ENGLISH else "År")
    plt.ylabel("Heterozygosity H(t)" if ENGLISH else "Heterozygositet H(t)")
    plt.title("Sub-model Blink: Eco-genetic coupling (demography → genetics)" if ENGLISH else "Submodell Blink: Ekogenetisk koppling (demografi → genetik)")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    fig_Blink = os.path.join(result_dir, "mod_Blink_eco_genetic_coupling.png")
    plt.savefig(fig_Blink, dpi=150)
    plt.close()
    print(f"Saved Submodel Blink figure: {fig_Blink}")


    # --- Disturbance decay diagnostic plot (separate figure) ---
    years = np.arange(0, 30)
    factors = [disturbance_sigma_multiplier(
                y,
                start_factor=DISTURBANCE_SIGMA_FACTOR_START,
                end_factor=DISTURBANCE_SIGMA_FACTOR_END,
                relax_years=DISTURBANCE_RELAX_YEARS
            )
            for y in years]

    plt.figure(figsize=(5, 3))
    plt.plot(years, factors, lw=2)
    plt.xlabel("Years since establishment" if ENGLISH else "År sedan anläggning")
    plt.ylabel("σ_e multiplier" if ENGLISH else "σₑ-multiplikator")
    plt.title("Initial disturbance → stabilisation" if ENGLISH else "Initial störning → stabilisering")
    plt.grid(alpha=0.3)
    fig_disturb = os.path.join(result_dir, "disturbance_decay.png")
    plt.tight_layout()
    plt.savefig(fig_disturb, dpi=150)
    plt.close()
    print(f"Saved disturbance plot: {fig_disturb}")

    # Notes
    print("\nNotes:")
    print(f"- Ne-dependent volatility: sigma_eff = sigma_e * max(1, (NE_TARGET/Ne))**{ALPHA}. "
          f"With NE_TARGET={NE_TARGET}, Ne<100 amplifies risk; Ne≥100 behaves like baseline.")
    print("- Survival curves: one line per (Ne_ratio × migrants/year). Expect lower Ne to sit below higher Ne at the same immigration.")
    print("- Adjust NE_TARGET or ALPHA if you want a gentler/stronger Ne effect (e.g., ALPHA=0.5 gentler, 1.0 stronger).")

def run_survive(
    # --- “Project” tab inputs ---
    PROJECT_NAME,
    NC_POP,
    NC_METAPOP,
    TIME_HORIZON_YEARS,
    QUASI_EXT_THRESHOLD,
    DISTURBANCE_SIGMA_FACTOR_START,
    DISTURBANCE_SIGMA_FACTOR_END,
    DISTURBANCE_RELAX_YEARS,
    NE_RATIO_START,
    NE_RATIO_END,
    NE_RATIO_RELAX_YEARS,
    INIT_PROP_ADULT,
    INIT_FEMALE_FRAC,
    P_EVENT,
    SEVERITY,

    # --- “Organism” tab inputs ---
    NE_RATIOS,
    GEN_TIME_YEARS,
    MIGRANTS_PER_YEAR,
    K,
    H0,
    GEN_TIME_FOR_GENETICS,
    R_GRID,
    SIGMA_E_GRID,

    # --- “Model” tab inputs ---
    DENSITY_MODEL,
    THETA,
    GENETIC_EVAL,
    REPLICATES,
    RANDOM_SEED,
    NE_TARGET,
    ALPHA,
    SURVIVAL_TARGET,
    H_MIN_FRAC,

    # species flag and output base dir could also come from GUI
    SPECIES="B",
    base_results_dir=r"C:\Users\EricWahlsteen\OneDrive - Calluna AB\Skrivbordet\Python_test\Survive\RESULTS",
):
    """
    Run the full viability analysis with the given parameters.
    Returns the absolute path of the results folder it created.
    """

    # --- 1. assign these args into the module-level names your current code uses ---
    # WARNING: we are mutating globals in this module so the rest of your code works unchanged.
    # In long-term cleanup, you'd pass these around instead. But this is the smallest diff.
    globals().update({
        "PROJECT_NAME": PROJECT_NAME,
        "NC_POP": float(NC_POP),
        "NC_METAPOP": float(NC_METAPOP),
        "TIME_HORIZON_YEARS": int(TIME_HORIZON_YEARS),
        "QUASI_EXT_THRESHOLD": float(QUASI_EXT_THRESHOLD),

        "DISTURBANCE_SIGMA_FACTOR_START": float(DISTURBANCE_SIGMA_FACTOR_START),
        "DISTURBANCE_SIGMA_FACTOR_END":   float(DISTURBANCE_SIGMA_FACTOR_END),
        "DISTURBANCE_RELAX_YEARS":        float(DISTURBANCE_RELAX_YEARS),

        "NE_RATIO_START":        float(NE_RATIO_START),
        "NE_RATIO_END":          float(NE_RATIO_END),
        "NE_RATIO_RELAX_YEARS":  float(NE_RATIO_RELAX_YEARS),

        "INIT_PROP_ADULT":   float(INIT_PROP_ADULT),
        "INIT_FEMALE_FRAC":  float(INIT_FEMALE_FRAC),

        "P_EVENT":  float(P_EVENT),
        "SEVERITY": float(SEVERITY),

        "NE_RATIOS":           [float(x) for x in _coerce_listlike(NE_RATIOS)],
        "GEN_TIME_YEARS":      [float(x) for x in _coerce_listlike(GEN_TIME_YEARS)],
        "MIGRANTS_PER_YEAR":   [float(x) for x in _coerce_listlike(MIGRANTS_PER_YEAR)],
        "K":                   float(K),
        "H0":                  float(H0),
        "GEN_TIME_FOR_GENETICS": float(GEN_TIME_FOR_GENETICS),
        "R_GRID":              [float(x) for x in _coerce_listlike(R_GRID)],
        "SIGMA_E_GRID":        [float(x) for x in _coerce_listlike(SIGMA_E_GRID)],

        "DENSITY_MODEL":   str(DENSITY_MODEL),
        "THETA":           float(THETA),
        "GENETIC_EVAL":    str(GENETIC_EVAL),
        "REPLICATES":      int(REPLICATES),
        "RANDOM_SEED":     int(RANDOM_SEED),
        "NE_TARGET":       float(NE_TARGET),
        "ALPHA":           float(ALPHA),
        "SURVIVAL_TARGET": float(SURVIVAL_TARGET),
        "H_MIN_FRAC":      float(H_MIN_FRAC),

        "SPECIES": str(SPECIES),
    })

    # The functions catastrophe_event() and such read P_EVENT / SEVERITY
    # from closure-free arguments right now. We need them to see the updated values.
    # Easiest hack: re-bind them as globals too so they close over new P_EVENT/SEVERITY.
    def catastrophe_event_local(rng, p_event=float(P_EVENT), severity=float(SEVERITY), size=None):
        if size is None:
            if rng.random() < p_event:
                return (1.0 - severity)
            else:
                return 1.0
        hits = rng.random(size) < p_event
        return np.where(hits, 1.0 - severity, 1.0)
    globals()["catastrophe_event"] = catastrophe_event_local

    # (Optional) also update climate_modifiers if you want GUI-driven climate later

    # --- 2. run what main() currently does, but without re-setting base_results_dir inside main ---
    rng = np.random.default_rng(int(RANDOM_SEED))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_slug = re.sub(r"[^A-Za-z0-9_-]+", "_", (PROJECT_NAME or "").strip()) or "RUN"
    result_dir = os.path.join(base_results_dir, f"{project_slug}_{SPECIES}_{ts}")
    os.makedirs(result_dir, exist_ok=True)

    write_run_settings(
        result_dir=result_dir,
        species=SPECIES,
        project_name=PROJECT_NAME,
        Nc=float(NC_POP),
        NC_METAPOP=float(NC_METAPOP),
        cfg_module=cfg,
        disturbance_start=float(DISTURBANCE_SIGMA_FACTOR_START),
        disturbance_end=float(DISTURBANCE_SIGMA_FACTOR_END),
        disturbance_relax=float(DISTURBANCE_RELAX_YEARS),
        meta_weights=META_WEIGHTS,
    )

    # Now we basically inline everything your current main() does: build rows, df, plots.
    # Easiest move: instead of rewriting all that logic again,
    # call the original main() code AFTER we patched globals – but make main() able to accept result_dir override.

    _run_full_pipeline(result_dir, rng)  # we'll define this helper around your existing main() body

    return result_dir

if __name__ == "__main__":
    # fallback CLI entrypoint using whatever defaults are at top of file
    rng = np.random.default_rng(RANDOM_SEED)
    base_results_dir = r"C:\Users\EricWahlsteen\OneDrive - Calluna AB\Skrivbordet\Python_test\Survive\RESULTS"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_slug = re.sub(r"[^A-Za-z0-9_-]+", "_", (PROJECT_NAME or "").strip()) or "RUN"
    result_dir = os.path.join(base_results_dir, f"{project_slug}_{SPECIES}_{ts}")
    os.makedirs(result_dir, exist_ok=True)

    write_run_settings(
        result_dir=result_dir,
        species=SPECIES,
        project_name=PROJECT_NAME,
        Nc=NC_POP,
        NC_METAPOP=NC_METAPOP,
        cfg_module=cfg,
        disturbance_start=DISTURBANCE_SIGMA_FACTOR_START,
        disturbance_end=DISTURBANCE_SIGMA_FACTOR_END,
        disturbance_relax=DISTURBANCE_RELAX_YEARS,
        meta_weights=META_WEIGHTS,
    )

    _run_full_pipeline(result_dir, rng)
    print("Done.")

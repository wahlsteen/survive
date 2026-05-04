"""
SURVIVE v1.1 — Streamlit web app
Run locally:  streamlit run app.py
Deploy:       push to GitHub -> share.streamlit.io -> connect repo
"""
import importlib, io, shutil, tempfile, threading
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import streamlit as st

from report_builder import FIG_ORDER, FIG_CAPTIONS, build_pdf_report

st.set_page_config(page_title="SURVIVE v1.1", page_icon="🌿", layout="wide")
_run_lock = threading.Lock()

SPECIES_OPTIONS = {
    "V": ("Monocarpic biennial plant",       "config_verbascum"),
    "H": ("Polycarpic perennial plant",      "config_helichrysum"),
   # "B": ("Common toad (B. bufo)",          "config_bufo"),
    "L": ("Amphibian",                       "config_lissotriton"),
    "A": ("Bird",                            "config_bird"),
    "M": ("Mammal",                          "config_mammal"),
}

# ════════════════════════════════ SESSION STATE ═══════════════════════════════
if "run_results" not in st.session_state:
    st.session_state.run_results = None

# ════════════════════════════════ SIDEBAR ════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌿 SURVIVE v1.1")
    st.caption("Population Viability Analysis")

    st.subheader("Species")
    species_key = st.selectbox(
        "Select species", list(SPECIES_OPTIONS.keys()),
        format_func=lambda k: f"{k} — {SPECIES_OPTIONS[k][0]}",
        label_visibility="collapsed",
    )
    species_name, cfg_module_name = SPECIES_OPTIONS[species_key]

    @st.cache_resource(show_spinner=False)
    def load_cfg(name): return importlib.import_module(name)
    cfg = load_cfg(cfg_module_name)

    st.divider()
    st.subheader("Project")
    project_name = st.text_input("Project name", "Streamlit_PVA", max_chars=60)
    english = st.toggle("English figure text", value=True)

    st.divider()
    st.subheader("Population")
    nc_pop     = st.number_input("NC_POP (census, models A & C)", 1, 50000, int(cfg.K), 10)
    nc_metapop = st.number_input("NC_METAPOP (local breeding, genetics)", 1, 50000, int(cfg.K), 10)

    st.divider()
    st.subheader("Scenario grid")
    ne_ratios_str = st.text_input("Ne/Nc ratios", ", ".join(str(x) for x in cfg.NE_RATIOS))
    migrants_str  = st.text_input("Immigrants/yr", ", ".join(str(x) for x in cfg.MIGRANTS_PER_YEAR))

    st.divider()
    st.subheader("Simulation")
    replicates = st.slider("Monte Carlo replicates", 200, 5000,
                           min(int(cfg.REPLICATES), 2000), 100)

    st.divider()
    with st.expander("⚙️ Advanced parameters"):
        st.markdown("**Uncertainty grid**")
        r_grid_str     = st.text_input("r grid",  ", ".join(str(x) for x in cfg.R_GRID))
        sigma_grid_str = st.text_input("σₑ grid", ", ".join(str(x) for x in cfg.SIGMA_E_GRID))

        st.markdown("**Habitat disturbance**")
        c1, c2 = st.columns(2)
        disturb_start = c1.number_input("σ mult. yr 0",   1.0, 5.0, 1.0, 0.1)
        disturb_end   = c1.number_input("σ mult. mature", 1.0, 5.0, 1.0, 0.1)
        disturb_relax = c2.number_input("Yrs stabilise",  0,   200, 0,   5)

        st.markdown("**Founder / bottleneck**")
        c3, c4 = st.columns(2)
        ne_ratio_start = c3.number_input("Ne/Nc founding", 0.01, 1.0, float(cfg.NE_RATIOS[0]), 0.01)
        ne_ratio_end   = c3.number_input("Ne/Nc equil.",   0.01, 1.0, float(cfg.NE_RATIOS[0]), 0.01)
        ne_ratio_relax = c4.number_input("Yrs normalise",  0,    500, 0,   10)

        st.markdown("**Meta-model weights**")
        w_A = st.slider("Weight A", 0.0, 1.0, 0.34, 0.01)
        w_B = st.slider("Weight B", 0.0, 1.0, 0.33, 0.01)
        w_C = st.slider("Weight C", 0.0, 1.0, 0.33, 0.01)

        st.markdown("**Catastrophe events**")
        c5, c6 = st.columns(2)
        p_event  = c5.number_input("P(catastrophe)/yr", 0.0, 0.5, 0.01, 0.005)
        severity = c6.number_input("Severity",          0.0, 1.0, 0.9,  0.05)

    st.divider()
    run_btn = st.button("▶ Run analysis", type="primary", use_container_width=True)

# ════════════════════════════════ RUN ════════════════════════════════════════
if run_btn:
    def _floats(s): return [float(x.strip()) for x in s.split(",") if x.strip()]
    def _ints(s):   return [int(float(x.strip())) for x in s.split(",") if x.strip()]
    try:
        ne_ratios  = _floats(ne_ratios_str)
        migrants   = _ints(migrants_str)
        r_grid     = _floats(r_grid_str)
        sigma_grid = _floats(sigma_grid_str)
    except Exception as e:
        st.error(f"Could not parse input: {e}"); st.stop()

    gfg = getattr(cfg, "GEN_TIME_FOR_GENETICS", None) or cfg.GEN_TIME_YEARS[0]

    status = st.status("Running SURVIVE v1.1…", expanded=True)
    with status:
        st.write("Importing engine…")
        import survive_1_1 as eng

        st.write(f"**{species_key} — {species_name}** | Nc={nc_pop} | "
                 f"{len(ne_ratios)*len(migrants)} scenarios | {replicates:,} replicates")

        tmpdir = tempfile.mkdtemp(prefix="survive_")
        run_error = None
        try:
            with _run_lock:
                eng.cfg = cfg
                eng.ENGLISH = bool(english)
                st.write("Running simulations… (30–90 s)")
                result_dir = eng.run_survive(
                    PROJECT_NAME=project_name,
                    NC_POP=int(nc_pop), NC_METAPOP=int(nc_metapop),
                    TIME_HORIZON_YEARS=int(cfg.TIME_HORIZON_YEARS),
                    QUASI_EXT_THRESHOLD=float(cfg.QUASI_EXT_THRESHOLD),
                    DISTURBANCE_SIGMA_FACTOR_START=float(disturb_start),
                    DISTURBANCE_SIGMA_FACTOR_END=float(disturb_end),
                    DISTURBANCE_RELAX_YEARS=int(disturb_relax),
                    NE_RATIO_START=float(ne_ratio_start),
                    NE_RATIO_END=float(ne_ratio_end),
                    NE_RATIO_RELAX_YEARS=int(ne_ratio_relax),
                    INIT_PROP_ADULT=float(getattr(cfg, "INIT_PROP_ADULT", 1.0)),
                    INIT_FEMALE_FRAC=float(getattr(cfg, "INIT_FEMALE_FRAC", 0.5)),
                    P_EVENT=float(p_event), SEVERITY=float(severity),
                    NE_RATIOS=ne_ratios, GEN_TIME_YEARS=list(cfg.GEN_TIME_YEARS),
                    MIGRANTS_PER_YEAR=migrants, K=float(cfg.K), H0=float(cfg.H0),
                    GEN_TIME_FOR_GENETICS=float(gfg),
                    R_GRID=r_grid, SIGMA_E_GRID=sigma_grid,
                    DENSITY_MODEL=str(cfg.DENSITY_MODEL), THETA=float(cfg.THETA),
                    GENETIC_EVAL=str(cfg.GENETIC_EVAL), REPLICATES=int(replicates),
                    RANDOM_SEED=int(cfg.RANDOM_SEED), NE_TARGET=float(cfg.NE_TARGET),
                    ALPHA=float(cfg.ALPHA), SURVIVAL_TARGET=float(cfg.SURVIVAL_TARGET),
                    H_MIN_FRAC=float(cfg.H_MIN_FRAC), SPECIES=species_key,
                    base_results_dir=tmpdir,
                )

            st.write("Loading results…")
            rp = Path(result_dir)
            xlsx_bytes = None; df_results = None
            xf = rp / "extinction_scenarios.xlsx"
            if xf.exists():
                xlsx_bytes = xf.read_bytes()
                df_results = pd.read_excel(io.BytesIO(xlsx_bytes))
            figures = {lbl: (rp/fn).read_bytes()
                       for fn, lbl in FIG_ORDER if (rp/fn).exists()}

        except Exception:
            import traceback; run_error = traceback.format_exc()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        if run_error:
            status.update(label="Run failed ✗", state="error")
            st.error("Error during analysis:"); st.code(run_error); st.stop()
        else:
            status.update(label="Analysis complete ✓", state="complete", expanded=False)

    st.session_state.run_results = {
        "figures": figures, "df_results": df_results, "xlsx_bytes": xlsx_bytes,
        "params": dict(
            project_name=project_name, species_key=species_key, species_name=species_name,
            nc_pop=int(nc_pop), nc_metapop=int(nc_metapop),
            ne_ratios=ne_ratios, migrants=migrants, replicates=int(replicates),
            english=bool(english), r_grid=r_grid, sigma_grid=sigma_grid,
            disturb_start=float(disturb_start), disturb_end=float(disturb_end),
            disturb_relax=int(disturb_relax),
            ne_ratio_start=float(ne_ratio_start), ne_ratio_end=float(ne_ratio_end),
            ne_ratio_relax=int(ne_ratio_relax),
            w_A=float(w_A), w_B=float(w_B), w_C=float(w_C),
            p_event=float(p_event), severity=float(severity), cfg=cfg,
        ),
    }

# ════════════════════════════════ DISPLAY ════════════════════════════════════
res = st.session_state.run_results
if res is None:
    st.markdown("## SURVIVE v1.1 — Population Viability Analysis")
    st.markdown(
        "**Sub-models:** A density demography · B genetics-only · "
        "B_link eco-genetic coupling · C Ne-sensitive demography · META weighted score"
    )
    st.info("SURVIVE v1.1 is a Python-based population viability analysis (PVA) tool "
            "designed to assess the long-term viability of small or threatened populations "
            "under uncertainty. It integrates demographic and genetic sub-models within a "
            "unified framework, enabling practitioners to evaluate the relative importance "
            "of stochastic demography, genetic drift, immigration, and density dependence "
            "on population persistence over a user-defined time horizon (default 100 years)."
            "The tool is designed for ecological consulting workflows in which species-specific "
            "parameters are stored in separate configuration files, allowing rapid scenario "
            "switching without modifying the core model code. It currently supports the "
            "following life-history archetypes: short-lived monocarpic biennials, long-lived "
            "polycarpus perennials and iteroparous animals as amphibians, birds and mammals.\n\n"

            "Rather than a single monolithic model, SURVIVE decomposes the viability question "
            "into four independent or semi-coupled sub-models (A, B, Blink, C), each "
            "emphasising a different ecological mechanism. Their outputs can be combined "
            "in a configurable meta-model. This structure makes model assumptions explicit "
            "and facilitates communication with non-specialist stakeholders: each sub-model "
            "can be presented and defended independently. \n\n"

            "**Choose** a model organism in the sidebar (Species) and configure its parameters, then click **▶ Run analysis**.\n\n"
            "A typical run takes **30–90 seconds** depending on replicates and scenario count.\n\n")
            
    st.info("### Settings\n\n"
            "**Species** — select a life-history archetype. Each archetype has a set of default parameters which can be customized in the sidebar. " 
                "The archetypes are based on real species but are generalised to represent common life-history strategies.\n\n"
            "**Project** — name your project here\n\n"
            "**NC_POP** — census population size for models A & C, i.e., the number of individuals in the population\n\n"
            "**NC_METAPOP** — local breeding population size for genetics\n\n"
            "**Ne/Nc ratios** — comma-separated list of effective/census population size ratios\n\n"
            "**Immigrants/yr** — comma-separated list of immigrant numbers per year\n\n"
            "### Advanced Parameters\n\n"
            "**r grid** — list of intrinsic growth rates of the population. Can be estimated from field data and follows the equation: "
                "r = ln(Nt2/Nt1)/(t2-t1) Where N2 adn t2 are current number of individuals and N1 and t1 are original number of individuals.\n\n"
            "**sigma grid** — list of environmental stochasticity levels. Rrefers to the random, temporal variation in environmental "
                "conditions—such as climate, food supply, or habitat quality — that affects the vital rates (survival and reproduction) of all individuals in a population simultaneously\n\n"
            "**Habitat disturbance** — parameters for an initial temporary increase in environmental stochasticity, simulating habitat disturbance events for a newly created habitat\n\n"
            "**Founder / bottleneck** — parameters for an initial temporary reduction in Ne/Nc ratio, simulating founder events or bottlenecks if the populations has been translocated\n\n"
            "**Meta-model weights** — weights for combining sub-model outputs into a weighted meta-model score\n\n"
            "**Catastrophe events** — parameters for random catastrophe events causing sudden population crashes. 0.01 = 1 event per 100 years\n\n")
    st.stop()

params  = res["params"]
figures = res["figures"]
lang    = "en" if params["english"] else "sv"

st.success(f"✓ **{params['species_key']} / {params['project_name']}** | "
           f"{len(params['ne_ratios'])*len(params['migrants'])} scenarios completed")

# Tabs — Figures first
tab_figs, tab_results, tab_dl = st.tabs(["📈 Figures", "📊 Results table", "⬇️ Download"])

with tab_figs:
    if figures:
        items = list(figures.items())
        for i in range(0, len(items), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(items):
                    label, img = items[idx]
                    with col:
                        st.image(img, use_container_width=True)
                        cap = FIG_CAPTIONS.get(label, {}).get(lang, "")
                        if cap:
                            st.markdown(cap)
    else:
        st.warning("No figures were generated.")

with tab_results:
    if res["df_results"] is not None:
        st.dataframe(res["df_results"], use_container_width=True, height=500)
        st.caption("A = density demography · B = genetics · C = Ne-sensitive · META = weighted score. "
                   "Values >= SURVIVAL_TARGET = PASS.")
    else:
        st.warning("No results table available.")

with tab_dl:
    st.markdown("### Download")
    c1, c2 = st.columns(2)

    with c1:
        if res["xlsx_bytes"]:
            st.download_button(
                "⬇️ Excel results table", res["xlsx_bytes"],
                file_name=f"SURVIVE_{params['project_name']}_{params['species_key']}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    with c2:
        if st.button("📄 Generate full PDF report", type="secondary"):
            with st.spinner("Building report…"):
                try:
                    pdf_bytes = build_pdf_report(params, figures,
                                                 df_results=res["df_results"])
                    st.download_button(
                        "⬇️ Download PDF report", pdf_bytes,
                        file_name=f"SURVIVE_report_{params['project_name']}_{params['species_key']}.pdf",
                        mime="application/pdf",
                    )
                except Exception as e:
                    import traceback
                    st.error(f"PDF generation failed: {e}")
                    st.code(traceback.format_exc())

    if figures:
        st.markdown("---")
        st.markdown("**Individual figures**")
        cols = st.columns(3)
        for i, (label, img) in enumerate(figures.items()):
            fname = label.replace(" ","_").replace("—","").replace("/","_") + ".png"
            with cols[i % 3]:
                st.download_button(
                    f"⬇️ {label[:28]}", img,
                    file_name=fname, mime="image/png",
                    key=f"fig_{i}",
                )

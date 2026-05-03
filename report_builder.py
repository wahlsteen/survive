"""
report_builder.py — PDF report and figure-caption data for SURVIVE v1.1 Streamlit app.
"""
import io
from datetime import datetime

FIG_ORDER = [
    ("mod_A_survival_density_demography.png",   "Sub-model A — Density-regulated demography"),
    ("mod_B_heterozygosity_genetics.png",        "Sub-model B — Genetic variation"),
    ("mod_C_survival_ne_sensitive.png",          "Sub-model C — Ne-sensitive demography"),
    ("mod_Blink_eco_genetic_coupling.png",       "Sub-model B_link — Eco-genetic coupling"),
    ("sensitivity_sigma.png",                    "Sensitivity — sigma_e"),
    ("combined_genetic_demographic_scatter.png", "Scatter — Genetic x Demographic risk"),
    ("disturbance_decay.png",                    "Disturbance decay diagnostic"),
]

FIG_CAPTIONS = {
    "Sub-model A — Density-regulated demography": {
        "sv": ("**Submodell A – Demografi med täthetsreglering (utan Ne-effekter)**\n\n"
               "Submodell A beskriver populationens överlevnad under stokastisk demografisk "
               "tillväxt med täthetsreglering, utan att beakta effekter av liten effektiv "
               "population (Ne). Figuren visar hur den kumulativa sannolikheten för överlevnad "
               "(1 − utdöenderisk) förändras över tid för olika nivåer av immigration. Den "
               "streckade linjen markerar den konventionella tröskeln (95 % överlevnad). "
               "En flack kurva nära 1.0 indikerar demografisk stabilitet, medan brant "
               "lutning signalerar hög utdöenderisk vid givet invandringsscenario."),
        "en": ("**Sub-model A – Density-regulated demography (without Ne effects)**\n\n"
               "Sub-model A describes population survival under stochastic demographic growth "
               "with density regulation, without accounting for effects of small effective "
               "population size (Ne). The figure shows the cumulative survival probability "
               "(1 − extinction risk) over time for choosen immigration level. The dashed "
               "line marks the conventional 95% viability threshold. A flat curve near 1.0 "
               "indicates demographic stability; a steep decline signals high extinction risk."),
    },
    "Sub-model B — Genetic variation": {
        "sv": ("**Submodell B – Genetisk variation över tid**\n\n"
               "Submodell B beräknar förändringen i förväntad heterozygositet under 100 år "
               "som funktion av Ne/Nc och migration (modifierad Wright–Fisher-process). "
               "Den streckade linjen anger tröskeln för 70 % av ursprunglig heterozygositet. "
               "Brantare nedgångar visar snabbare genetisk erosion, typiskt för små eller "
               "isolerade populationer (Beebee & Griffiths, 2005; Lesbarrères et al., 2005)."),
        "en": ("**Sub-model B – Genetic variation over time**\n\n"
               "Sub-model B calculates the change in expected heterozygosity over 100 years "
               "as a function of Ne/Nc and migration (modified Wright–Fisher process). "
               "The dashed line marks the 70% threshold of initial heterozygosity. "
               "Steeper declines indicate faster genetic erosion, typical of small or "
               "isolated populations (Beebee & Griffiths, 2005; Lesbarrères et al., 2005)."),
    },
    "Sub-model C — Ne-sensitive demography": {
        "sv": ("**Submodell C – Demografi med Ne-känslig miljöstokasticitet**\n\n"
               "Submodell C integrerar effekter av liten Ne i demografisk stokasticitet. "
               "Miljövariansen ökar när Ne faller under ett tröskelvärde, vilket leder till "
               "ökad utdöenderisk. Den horisontella linjen vid 0.95 markerar kritisk tröskel."),
        "en": ("**Sub-model C – Demography with Ne-sensitive environmental stochasticity**\n\n"
               "Sub-model C integrates the effects of small Ne into demographic stochasticity. "
               "Environmental variance is amplified when Ne falls below a threshold, increasing "
               "extinction risk. The horizontal line at 0.95 marks the critical survival threshold."),
    },
    "Sub-model B_link — Eco-genetic coupling": {
        "sv": ("**Submodell B_link – Ekogenetisk koppling (demografi → genetik)**\n\n"
               "Submodell B_link beräknar simultana genetiska och demografiska förändringar. "
               "Populationsstorlekens tidsserie från demografimodellen används som indata till "
               "genetiska simuleringar, vilket gör att demografiska flaskhalsar påverkar "
               "den genetiska variationen dynamiskt."),
        "en": ("**Sub-model B_link – Eco-genetic coupling (demography → genetics)**\n\n"
               "Sub-model B_link jointly tracks genetic and demographic processes. The "
               "population size time series from the demographic model feeds into genetic "
               "simulations, so demographic bottlenecks dynamically affect genetic variation."),
    },
    "Sensitivity — sigma_e": {
        "sv": ("**Känslighetsanalys för miljöstokasticitet (σₑ)**\n\n"
               "Sambandet mellan σₑ och utdöendesannolikheten inom 100 år för olika Ne/Nc "
               "och invandringsnivåer. Varje panel = ett Ne/Nc-förhållande. Brant stigande "
               "kurvor indikerar hög känslighet för ökande miljövariation."),
        "en": ("**Sensitivity analysis – environmental stochasticity (σₑ)**\n\n"
               "Relationship between σₑ and extinction probability within 100 years for "
               "different Ne/Nc ratios and immigration levels. Each panel = one Ne/Nc ratio. "
               "Steeply rising curves indicate high sensitivity to increasing σₑ."),
    },
    "Scatter — Genetic x Demographic risk": {
        "sv": ("**Scatter: genetisk × demografisk risk**\n\n"
               "Genetisk risk (F_eq, x-axeln) mot demografisk risk (utdöendesannolikhet, "
               "y-axeln) för samtliga scenarier. Markörform = Ne/Nc, färg = immigration. "
               "Nedre vänstra hörnet = låg risk (att föredra)."),
        "en": ("**Scatter – genetic × demographic risk**\n\n"
               "Genetic risk (F_eq, x-axis) against demographic risk (extinction probability, "
               "y-axis) for all scenarios. Marker shape = Ne/Nc, colour = immigration. "
               "Lower-left corner = low risk (preferred scenarios)."),
    },
    "Disturbance decay diagnostic": {
        "sv": ("**Diagnostik: störningsrelaxation**\n\n"
               "Hur störningsmultiplikatorn på σₑ förändras över tid från anläggningsåret. "
               "Multiplikator > 1 innebär förhöjd variabilitet i tidiga år, typiskt för "
               "nyligen skapade eller restaurerade livsmiljöer."),
        "en": ("**Diagnostic – disturbance relaxation**\n\n"
               "How the σₑ multiplier changes over time from establishment. A multiplier > 1 "
               "implies elevated variability in early years, typical of newly created or "
               "restored habitats."),
    },
}


# ── PDF helpers ───────────────────────────────────────────────────────────────

def _s(text):
    """Sanitise for Latin-1 / cp1252 fpdf2 core fonts."""
    return (text
        .replace("σₑ","sigma_e").replace("σ","sigma_e")
        .replace("→","->").replace("–","-").replace("—","--")
        .replace("ₑ","e").replace("×","x").replace("≈","~")
        .replace("≥",">=").replace("≤","<=")
        .replace("−","-")
        .replace("’","’").replace("’","’")
        .replace("“",'"').replace("”",'"')
        .replace("**","").replace("*","")
    )


def _draw_table(pdf, headers, rows, col_widths):
    pdf.set_font("Helvetica","B",9)
    pdf.set_fill_color(210,225,210)
    for i,h in enumerate(headers):
        pdf.cell(col_widths[i],7,_s(h),border=1,fill=True)
    pdf.ln()
    pdf.set_font("Helvetica","",9)
    for ri,row in enumerate(rows):
        fill = ri%2==0
        if fill: pdf.set_fill_color(247,251,247)
        for i,val in enumerate(row):
            pdf.cell(col_widths[i],6,_s(str(val)),border=1,fill=fill)
        pdf.ln()
    pdf.set_fill_color(255,255,255)
    pdf.ln(3)


def _fig(pdf, img_bytes):
    if not img_bytes: return
    if pdf.h - pdf.get_y() - pdf.b_margin < 55:
        pdf.add_page()
    pdf.image(io.BytesIO(img_bytes), x=20, w=170)
    pdf.ln(4)


def build_pdf_report(params, figures):
    """Return PDF bytes for the full PVA report."""
    from fpdf import FPDF
    eng = params.get("english", True)
    cfg_obj = params["cfg"]

    class PDF(FPDF):
        def header(self):
            self.set_font("Helvetica","I",8); self.set_text_color(120)
            self.cell(0,6,
                f"SURVIVE v1.1  |  {params['project_name']}  |  {params['species_name']}",
                align="R")
            self.ln(8); self.set_text_color(0)
        def footer(self):
            self.set_y(-14); self.set_font("Helvetica","I",8); self.set_text_color(120)
            self.cell(0,6,f"{'Page' if eng else 'Sida'} {self.page_no()}",align="C")
            self.set_text_color(0)

    pdf = PDF(orientation="P",unit="mm",format="A4")
    pdf.set_margins(20,26,20); pdf.set_auto_page_break(auto=True,margin=26)
    pdf.add_page()

    W = 170  # usable page width: 210mm - 20mm left - 20mm right
    def h1(t): pdf.set_font("Helvetica","B",16); pdf.multi_cell(W,9,_s(t)); pdf.ln(2)
    def h2(t): pdf.set_font("Helvetica","B",12); pdf.multi_cell(W,7,_s(t)); pdf.ln(1)
    def body(t): pdf.set_font("Helvetica","",10); pdf.multi_cell(W,5.5,_s(t)); pdf.ln(2)

    # Title
    pdf.set_font("Helvetica","B",22); pdf.multi_cell(W,12,"SURVIVE v1.1")
    pdf.set_font("Helvetica","I",14); pdf.multi_cell(W,8,"Population Viability Analysis Report" if eng
                   else "Populationsvitalitetsanalys")
    pdf.ln(3); pdf.set_draw_color(100); pdf.line(20,pdf.get_y(),190,pdf.get_y()); pdf.ln(5)
    pdf.set_font("Helvetica","",11)
    for label, val in [
        (("Project" if eng else "Projekt"), params["project_name"]),
        (("Species" if eng else "Art"),
         f"{params['species_name']} ({params['species_key']})"),
        (("Date" if eng else "Datum"), datetime.now().strftime("%Y-%m-%d")),
    ]:
        pdf.cell(0,7,f"{label}: {val}",new_x="LMARGIN",new_y="NEXT")
    pdf.ln(8)

    # 1. Introduction
    h1("1. Introduction" if eng else "1. Inledning")
    body(
        "SURVIVE v1.1 is a Python-based population viability analysis (PVA) tool designed "
        "to assess the long-term viability of small or threatened populations under uncertainty. "
        "It integrates demographic and genetic sub-models within a unified framework, enabling "
        "practitioners to evaluate the relative importance of stochastic demography, genetic "
        "drift, immigration, and density dependence on population persistence over a user-defined "
        "time horizon."
        "The tool is designed for ecological consulting workflows in which species-specific "
        "parameters are stored in separate configuration files, allowing rapid scenario switching "
        "without modifying the core model code. It currently supports the following life-history "
        "archetypes: monocarpic perennials (e.g. Verbascum sp.), iteroparous amphibians "
        "(e.g. Bufo bufo, Lissotriton vulgaris) and long-lived perennials (e.g. Helichrysum arenarium), "
        "and custom species. "
        "Rather than a single monolithic model, SURVIVE decomposes the viability question "
        "into four independent or semi-coupled sub-models (A, B, B_link, C), each emphasising "
        "a different ecological mechanism.  This structure makes model assumptions explicit "
        "and facilitates communication with non-specialist stakeholders: each sub-model can "
        "be presented and defended independently."
        "PVA is the formal process of estimating the probability that a population will "
        "persist for a specified period. Survive implements extinction risk as the complement "
        "of the quasi-extinction probability: the probability that census size  falls to or "
        "below a quasi-extinction threshold  before the time horizon."
        if eng else
        "SURVIVE v1.1 ar ett Python-baserat PVA-verktyg for att bedömma den långsiktiga "
        "livskraften hos sma eller hotade populationer under osäkerhet. Verktyget integrerar "
        "demografiska och genetiska submodeller (A, B, B_link, C) inom ett enhetligt ramverk. "
        "PVA uppskattar sannolikheten att populationsstorleken faller under kvasi-utdöendetröskel "
        "Q fore tidshorisonten T."
    )

    # 2. Methods
    # pdf.add_page()
    h1("2. Methods" if eng else "2. Metod")

    ne_str  = ", ".join(str(x) for x in params["ne_ratios"])
    mig_str = ", ".join(str(x) for x in params["migrants"])
    r_str   = ", ".join(str(x) for x in params["r_grid"])
    s_str   = ", ".join(str(x) for x in params["sigma_grid"])
    has_d   = params["disturb_start"] != 1.0 or params["disturb_relax"] > 0
    has_b   = params["ne_ratio_start"] != params["ne_ratio_end"] or params["ne_ratio_relax"] > 0

    if eng:
        d = (f"a recently established/relocated population; sigma_e amplified "
             f"{params['disturb_start']}x at yr 0, relaxing to {params['disturb_end']}x "
             f"over {params['disturb_relax']} yr" if has_d
             else "an established population without initial disturbance")
        m = (f"PVA was performed for {params['species_name']} with a total popultaion of {params['nc_pop']} individuals. "
             f"Ne/Nc ratios where set to {ne_str} and the annual immigration {mig_str} individuals/year. "
             f" Population expansion was estimated to r = {r_str} and environmental stochasticity to sigma_e = {s_str}. The scenario represents {d}.")
        if has_b:
            m += (f" Initial Ne/Nc = {params['ne_ratio_start']}, recovering to "
                  f"{params['ne_ratio_end']} over {params['ne_ratio_relax']} yr.")
    else:
        d = (f"en nyligen etablerad/omflyttad population; sigma_e forstarktes "
             f"{params['disturb_start']}x vid ar 0, relaxerade till {params['disturb_end']}x "
             f"over {params['disturb_relax']} ar" if has_d
             else "en etablerad population utan initial storning")
        m = (f"PVA genomfordes for {params['species_name']} (Nc = {params['nc_pop']}). "
             f"Ne/Nc-kvoter: {ne_str}. Arlig immigration: {mig_str} individer/ar. "
             f"r = {r_str}; sigma_e = {s_str}. Scenariot representerar {d}.")
        if has_b:
            m += (f" Initial Ne/Nc = {params['ne_ratio_start']}, aterhämtade sig till "
                  f"{params['ne_ratio_end']} over {params['ne_ratio_relax']} ar.")
    body(m)
    pdf.ln(2)

    # Table 1 – species parameters
    h2("2.1 Species-specific parameters" if eng else "2.1 Artspecifika parametrar")
    gt = getattr(cfg_obj,"GEN_TIME_YEARS",(None,None))
    _draw_table(pdf, ["Parameter","Value","Description"], [
        ("K (carrying capacity)",   getattr(cfg_obj,"K","-"),              "Breeding adults"),
        ("Generation time",         f"{gt[0]}-{gt[1]} yr",                "Min-max range"),
        ("H0",                      getattr(cfg_obj,"H0","-"),             "Initial heterozygosity"),
        ("H_min_frac",              getattr(cfg_obj,"H_MIN_FRAC","-"),     "Min acceptable H/H0"),
        ("Ne generation model",     getattr(cfg_obj,"NE_GENERATION_MODEL","iteroparous"), "Life-history archetype"),
        ("Ne target",               getattr(cfg_obj,"NE_TARGET","-"),      "Ne threshold (model C)"),
        ("Alpha",                   getattr(cfg_obj,"ALPHA","-"),          "Ne-sigma penalty curvature"),
        ("Density model",           getattr(cfg_obj,"DENSITY_MODEL","-"),  "Ricker / theta-logistic"),
        ("Survival target",         getattr(cfg_obj,"SURVIVAL_TARGET","-"),"P(survival) PASS threshold"),
    ], [58,32,80])

    # Table 2 – run settings
    h2("2.2 Project / run settings" if eng else "2.2 Projektinstallningar")
    _draw_table(pdf, ["Parameter","Value","Note"], [
        ("Project name",        params["project_name"],                              ""),
        ("Species",             f"{params['species_key']} ({params['species_name']})",""),
        ("NC_POP",              params["nc_pop"],                                    "Census pop. (A,C)"),
        ("NC_METAPOP",          params["nc_metapop"],                                "Breeding census (B)"),
        ("Ne/Nc ratios",        ne_str,                                              "Ratios tested"),
        ("Immigrants/yr",       mig_str,                                             "Annual immigration"),
        ("r grid",              r_str,                                               "Growth rate grid"),
        ("sigma_e grid",        s_str,                                               "Env. stoch. grid"),
        ("Replicates",          params["replicates"],                                "MC replicates"),
        ("Time horizon",        str(getattr(cfg_obj,"TIME_HORIZON_YEARS",100))+" yr",""),
        ("Quasi-ext. Q",        getattr(cfg_obj,"QUASI_EXT_THRESHOLD",5),            ""),
        ("Disturb. start",      params["disturb_start"],                             "sigma_e mult. yr 0"),
        ("Disturb. end",        params["disturb_end"],                               "sigma_e mult. mature"),
        ("Disturb. relax",      str(params["disturb_relax"])+" yr",                  ""),
        ("Ne/Nc founding",      params["ne_ratio_start"],                            ""),
        ("Ne/Nc equil.",        params["ne_ratio_end"],                              ""),
        ("Ne relax",            str(params["ne_ratio_relax"])+" yr",                 ""),
        ("Meta-weight A/B/C",   f"{params['w_A']:.2f} / {params['w_B']:.2f} / {params['w_C']:.2f}", ""),
    ], [58,44,68])

    # 3. Results – one section per sub-model with caption + figure
    pdf.add_page()
    h1("3. Results" if eng else "3. Resultat")

    report_figs = [
        "Sub-model A — Density-regulated demography",
        "Sub-model B — Genetic variation",
        "Sub-model C — Ne-sensitive demography",
        "Sub-model B_link — Eco-genetic coupling",
    ]
    section_titles_en = ["3.1 Sub-model A","3.2 Sub-model B",
                         "3.3 Sub-model C","3.4 Sub-model B_link"]
    section_titles_sv = ["3.1 Submodell A","3.2 Submodell B",
                         "3.3 Submodell C","3.4 Submodell B_link"]

    lang = "en" if eng else "sv"
    for i, fig_label in enumerate(report_figs):
        if i > 0:          # 3.1 sitter kvar direkt efter rubriken "3. Results"
            pdf.add_page()
        h2(section_titles_en[i] if eng else section_titles_sv[i])
        body(FIG_CAPTIONS.get(fig_label,{}).get(lang,""))
        _fig(pdf, figures.get(fig_label))
        pdf.ln(2)

    return bytes(pdf.output())

    return bytes(pdf.output())

import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import customtkinter as ctk


###############################################################################
# Global state container
###############################################################################

class GUIState:
    def __init__(self, master):
        # --- Project tab ---
        self.PROJECT_NAME = tk.StringVar(master, value="MyProject")
        self.NC_POP = tk.StringVar(master, value="0")
        self.NC_METAPOP = tk.StringVar(master, value="0")
        self.TIME_HORIZON_YEARS = tk.StringVar(master, value="100")
        self.QUASI_EXT_THRESHOLD = tk.StringVar(master, value="5")

        self.DISTURBANCE_SIGMA_FACTOR_START = tk.StringVar(master, value="2.0")
        self.DISTURBANCE_SIGMA_FACTOR_END   = tk.StringVar(master, value="1.0")
        self.DISTURBANCE_RELAX_YEARS        = tk.StringVar(master, value="10")

        self.NE_RATIO_START         = tk.StringVar(master, value="0.1")
        self.NE_RATIO_END           = tk.StringVar(master, value="0.25")
        self.NE_RATIO_RELAX_YEARS   = tk.StringVar(master, value="10")

        self.INIT_PROP_ADULT   = tk.StringVar(master, value="0.6")
        self.INIT_FEMALE_FRAC  = tk.StringVar(master, value="0.5")

        self.P_EVENT   = tk.StringVar(master, value="0.01")
        self.SEVERITY  = tk.StringVar(master, value="0.9")

        # --- Organism tab ---
        self.SELECTED_ORGANISM = tk.StringVar(master, value="(select)")
        self.NE_RATIOS            = tk.StringVar(master, value="0 0 0")
        self.GEN_TIME_YEARS       = tk.StringVar(master, value="0 0")
        self.MIGRANTS_PER_YEAR    = tk.StringVar(master, value="0 0 0")
        self.K                    = tk.StringVar(master, value="0")
        self.H0                   = tk.StringVar(master, value="0")
        self.GEN_TIME_FOR_GENET   = tk.StringVar(master, value="0")
        self.R_GRID               = tk.StringVar(master, value="0 0 0")
        self.SIGMA_E_GRID         = tk.StringVar(master, value="0 0 0")

        # --- Model tab ---
        self.DENSITY_MODEL     = tk.StringVar(master, value="theta-logistic")
        self.THETA             = tk.StringVar(master, value="0")
        self.GENETIC_EVAL      = tk.StringVar(master, value="with_migration")

        self.REPLICATES        = tk.StringVar(master, value="999")
        self.RANDOM_SEED       = tk.StringVar(master, value="42")
        self.NE_TARGET         = tk.StringVar(master, value="100")
        self.ALPHA             = tk.StringVar(master, value="0.7")
        self.SURVIVAL_TARGET   = tk.StringVar(master, value="0.95")
        self.H_MIN_FRAC        = tk.StringVar(master, value="0.70")

        # --- New organism tab ---
        self.NEW_ORG_NAME            = tk.StringVar(master, value="")
        self.NEW_NE_RATIOS           = tk.StringVar(master, value="0 0 0")
        self.NEW_GEN_TIME_YEARS      = tk.StringVar(master, value="0 0")
        self.NEW_MIGRANTS_PER_YEAR   = tk.StringVar(master, value="0 0 0")
        self.NEW_K                   = tk.StringVar(master, value="0")
        self.NEW_H0                  = tk.StringVar(master, value="0")
        self.NEW_GEN_TIME_FOR_GENET  = tk.StringVar(master, value="0")
        self.NEW_R_GRID              = tk.StringVar(master, value="0 0 0")
        self.NEW_SIGMA_E_GRID        = tk.StringVar(master, value="0 0 0")

        # --- Results ---
        self.RESULT_DIR = tk.StringVar(master, value="(none)")


###############################################################################
# Splash Screen
###############################################################################

class Splash(ctk.CTkToplevel):
    def __init__(self, master, logo_path=None, duration_ms=1500, on_close=None):
        super().__init__(master)
        self.overrideredirect(True)
        self.configure(fg_color=("black", "black"))

        self.on_close = on_close
        self.master_root = master  # reference to real root

        # --- Window geometry (400x400 centered) ---
        w, h = 400, 400
        x = self.winfo_screenwidth() // 2 - w // 2
        y = self.winfo_screenheight() // 2 - h // 2
        self.geometry(f"{w}x{h}+{x}+{y}")

        # stay on top of hidden main window
        self.transient(master)
        self.lift()
        self.grab_set()

        # --- Image only ---
        frame = ctk.CTkFrame(self, corner_radius=0, fg_color="black")
        frame.pack(expand=True, fill="both")

        if logo_path and os.path.exists(logo_path):
            pil_img = Image.open(logo_path)
        else:
            # placeholder if logo missing
            pil_img = Image.new("RGB", (400, 400), color=(30, 60, 120))

        self.logo_img_ctk = ctk.CTkImage(
            light_image=pil_img,
            dark_image=pil_img,
            size=(400, 400),
        )

        logo_label = ctk.CTkLabel(frame, image=self.logo_img_ctk, text="")
        logo_label.pack(expand=True, fill="both")

        # auto-close after delay
        self.after(duration_ms, self._finish)

    def _finish(self):
        self.grab_release()
        self.destroy()
        self.master_root.deiconify()
        if callable(self.on_close):
            self.on_close()


###############################################################################
# Main Application Window
###############################################################################

class SurviveApp(ctk.CTkFrame):
    def __init__(self, master, state: GUIState):
        super().__init__(master)
        self.state = state

        # the master is going to be the real root window
        root = master
        root.title("Survive 1.1")
        root.geometry("1400x800")
        root.minsize(1100, 700)

        # now build the whole layout ON the root, not on self
        root.grid_rowconfigure(1, weight=1)
        root.grid_columnconfigure(0, weight=1)

        ###################################################################
        # Top navigation bar
        ###################################################################
        nav_frame = ctk.CTkFrame(root, corner_radius=0)
        nav_frame.grid(row=0, column=0, sticky="ew")
        nav_frame.grid_columnconfigure((0,1,2,3,4), weight=1)

        self.btn_project  = ctk.CTkButton(nav_frame, text="Project",  command=lambda:self.show_frame("project"))
        self.btn_organism = ctk.CTkButton(nav_frame, text="Organism", command=lambda:self.show_frame("organism"))
        self.btn_model    = ctk.CTkButton(nav_frame, text="Model",    command=lambda:self.show_frame("model"))
        self.btn_results  = ctk.CTkButton(nav_frame, text="Results",  command=lambda:self.show_frame("results"))
        self.btn_neworg   = ctk.CTkButton(nav_frame, text="New organism", command=lambda:self.show_frame("new_org"))

        self.btn_project.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.btn_organism.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.btn_model.grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        self.btn_results.grid(row=0, column=3, padx=5, pady=5, sticky="ew")
        self.btn_neworg.grid(row=0, column=4, padx=5, pady=5, sticky="ew")

        ###################################################################
        # Content area (lives on root but managed by this frame class)
        ###################################################################
        self.content = ctk.CTkFrame(root)
        self.content.grid(row=1, column=0, sticky="nsew")
        self.content.grid_rowconfigure(0, weight=1)
        self.content.grid_columnconfigure(0, weight=1)

        # create each sub-frame the same way as before
        self.frames = {}
        self.frames["project"]  = self._build_project_tab(self.content)
        self.frames["organism"] = self._build_organism_tab(self.content)
        self.frames["model"]    = self._build_model_tab(self.content)
        self.frames["results"]  = self._build_results_tab(self.content)
        self.frames["new_org"]  = self._build_new_org_tab(self.content)

        self.show_frame("project")

    ###################################################################
    # Helpers
    ###################################################################
    def _on_species_select(self, choice):
        """
        User picked a species in the dropdown.
        Load known defaults into organism fields.
        """
        preset = self._get_species_preset(choice)
        if not preset:
            return
        s = self.state
        if "NE_RATIOS" in preset:
            s.NE_RATIOS.set(preset["NE_RATIOS"])
        if "GEN_TIME_YEARS" in preset:
            s.GEN_TIME_YEARS.set(preset["GEN_TIME_YEARS"])
        if "MIGRANTS_PER_YEAR" in preset:
            s.MIGRANTS_PER_YEAR.set(preset["MIGRANTS_PER_YEAR"])
        if "K" in preset:
            s.K.set(preset["K"])
        if "H0" in preset:
            s.H0.set(preset["H0"])
        if "GEN_TIME_FOR_GENET" in preset:
            s.GEN_TIME_FOR_GENET.set(preset["GEN_TIME_FOR_GENET"])
        if "R_GRID" in preset:
            s.R_GRID.set(preset["R_GRID"])
        if "SIGMA_E_GRID" in preset:
            s.SIGMA_E_GRID.set(preset["SIGMA_E_GRID"])

    def _get_species_preset(self, species_name: str):
        """
        Return a dict of organism parameters for a given species_name.
        You will replace these dummy values with your real species library.
        """
        presets = {
            "Bufo": {
                "NE_RATIOS": "0.15 0.22 0.30",
                "GEN_TIME_YEARS": "4 5",
                "MIGRANTS_PER_YEAR": "2 1 0",
                "K": "461",
                "H0": "0.75",
                "GEN_TIME_FOR_GENET": "4",
                "R_GRID": "0.2 0.5 0.8",
                "SIGMA_E_GRID": "0.1 0.2 0.3",
            },
            "Lissotriton": {
                "NE_RATIOS": "0.10 0.18 0.25",
                "GEN_TIME_YEARS": "3 4",
                "MIGRANTS_PER_YEAR": "3 2 1",
                "K": "100",
                "H0": "0.68",
                "GEN_TIME_FOR_GENET": "3",
                "R_GRID": "0.15 0.4 0.7",
                "SIGMA_E_GRID": "0.12 0.18 0.22",
            },
        }
        return presets.get(species_name, {})

    def _load_run_settings_into_state(self, folder):
        """
        Parse run_settings.txt from a results folder and push values into GUIState.
        This understands both 'KEY: value' and 'KEY = value' lines, and ignores section headers.
        """
        settings_path = os.path.join(folder, "run_settings.txt")
        if not os.path.exists(settings_path):
            messagebox.showwarning("No settings", "run_settings.txt was not found in this folder.")
            return

        raw = {}
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # skip headers / separators
                    if line.startswith("---"):
                        continue
                    if line.startswith("===="):
                        continue
                    if line.lower().startswith("run settings"):
                        continue
                    if line.lower().startswith("timestamp"):
                        continue

                    # Try split on '=' first
                    if "=" in line:
                        left, right = line.split("=", 1)
                    elif ":" in line:
                        left, right = line.split(":", 1)
                    else:
                        continue

                    key = left.strip()
                    val = right.strip()

                    # store raw
                    raw[key] = val

        except Exception as e:
            messagebox.showerror("Error", f"Could not read run_settings.txt:\n{e}")
            return

        # Now we normalize keys from file -> our GUI variables.
        # We'll use a helper that does: if present in raw then set the StringVar.

        def set_if_present(raw_key_list, stringvar):
            for k in raw_key_list:
                if k in raw:
                    stringvar.set(raw[k])
                    return

        s = self.state

        # PROJECT_NAME
        set_if_present(["PROJECT_NAME"], s.PROJECT_NAME)

        # NC_POP (in file it's 'Nc_pop (census size)')
        set_if_present(["Nc_pop (census size)"], s.NC_POP)

        # NC_METAPOP
        # file uses 'NC_METAPOP (genetics local pond)'
        set_if_present(["NC_METAPOP (genetics local pond)", "NC_METAPOP"], s.NC_METAPOP)

        # TIME_HORIZON_YEARS
        set_if_present(["TIME_HORIZON_YEARS"], s.TIME_HORIZON_YEARS)

        # QUASI_EXT_THRESHOLD
        set_if_present(["QUASI_EXT_THRESHOLD"], s.QUASI_EXT_THRESHOLD)

        # DISTURBANCE_*
        set_if_present(["DISTURBANCE_SIGMA_FACTOR_START"], s.DISTURBANCE_SIGMA_FACTOR_START)
        set_if_present(["DISTURBANCE_SIGMA_FACTOR_END"],   s.DISTURBANCE_SIGMA_FACTOR_END)
        set_if_present(["DISTURBANCE_RELAX_YEARS"],        s.DISTURBANCE_RELAX_YEARS)

        # Species / organism fields
        set_if_present(["NE_RATIOS"],             s.NE_RATIOS)
        set_if_present(["GEN_TIME_YEARS"],        s.GEN_TIME_YEARS)
        set_if_present(["MIGRANTS_PER_YEAR"],     s.MIGRANTS_PER_YEAR)
        set_if_present(["K"],                     s.K)
        set_if_present(["H0"],                    s.H0)
        set_if_present(["GEN_TIME_FOR_GENETICS","GEN_TIME_FOR_GENET"], s.GEN_TIME_FOR_GENET)
        set_if_present(["R_GRID"],                s.R_GRID)
        set_if_present(["SIGMA_E_GRID"],          s.SIGMA_E_GRID)

        # Model tab fields
        set_if_present(["DENSITY_MODEL"],         s.DENSITY_MODEL)
        set_if_present(["THETA"],                 s.THETA)
        set_if_present(["GENETIC_EVAL"],          s.GENETIC_EVAL)
        set_if_present(["REPLICATES"],            s.REPLICATES)
        set_if_present(["RANDOM_SEED"],           s.RANDOM_SEED)
        set_if_present(["NE_TARGET"],             s.NE_TARGET)
        set_if_present(["ALPHA"],                 s.ALPHA)
        set_if_present(["SURVIVAL_TARGET"],       s.SURVIVAL_TARGET)
        set_if_present(["H_MIN_FRAC"],            s.H_MIN_FRAC)

    def show_frame(self, name: str):
        for frame_name, frame in self.frames.items():
            if frame_name == name:
                frame.grid(row=0, column=0, sticky="nsew")
            else:
                frame.grid_forget()

    def _labeled_entry(self, parent, label, textvariable, row, col=0, colspan=1, width=120):
        lbl = ctk.CTkLabel(parent, text=label)
        lbl.grid(row=row, column=col, sticky="w", padx=5, pady=(5,0))
        ent = ctk.CTkEntry(parent, textvariable=textvariable, width=width)
        ent.grid(row=row+1, column=col, columnspan=colspan, sticky="ew", padx=5, pady=(0,5))
        return ent

    def _section_label(self, parent, text, row, col=0, pady=(10,5), size=14):
        lab = ctk.CTkLabel(parent, text=text, font=ctk.CTkFont(size=size, weight="bold"))
        lab.grid(row=row, column=col, sticky="w", padx=5, pady=pady)

    ###################################################################
    # Project tab
    ###################################################################

    def _build_project_tab(self, master):
        frame = ctk.CTkFrame(master)
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure((0,1,2), weight=1)

        # Left column: basic project + buttons
        left = ctk.CTkFrame(frame)
        left.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        left.grid_columnconfigure(0, weight=1)

        # Load / new project section
        proj_top = ctk.CTkFrame(left)
        proj_top.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        load_btn = ctk.CTkButton(proj_top, text="Load previous project", command=self._load_project)
        load_btn.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        new_btn = ctk.CTkButton(proj_top, text="Start new project", command=self._new_project)
        new_btn.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # Project params
        proj_box = ctk.CTkFrame(left)
        proj_box.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        proj_box.grid_columnconfigure((0,1,2), weight=1)

        self._section_label(proj_box, "Project parameters", row=0, col=0)

        self._labeled_entry(proj_box, "PROJECT_NAME", self.state.PROJECT_NAME, row=1, col=0)
        self._labeled_entry(proj_box, "NC_POP", self.state.NC_POP, row=1, col=1)
        self._labeled_entry(proj_box, "NC_METAPOP (often ~K)", self.state.NC_METAPOP, row=1, col=2)

        self._labeled_entry(proj_box, "TIME_HORIZON_YEARS", self.state.TIME_HORIZON_YEARS, row=3, col=0)
        self._labeled_entry(proj_box, "QUASI_EXT_THRESHOLD", self.state.QUASI_EXT_THRESHOLD, row=3, col=1)

        # Disturbance block
        dist_box = ctk.CTkFrame(left)
        dist_box.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        dist_box.grid_columnconfigure((0,1,2), weight=1)

        self._section_label(dist_box, "Initial Disturbance Values", row=0, col=0)
        self._labeled_entry(dist_box, "DISTURBANCE_SIGMA_FACTOR_START", self.state.DISTURBANCE_SIGMA_FACTOR_START, row=1, col=0)
        self._labeled_entry(dist_box, "DISTURBANCE_SIGMA_FACTOR_END",   self.state.DISTURBANCE_SIGMA_FACTOR_END,   row=1, col=1)
        self._labeled_entry(dist_box, "DISTURBANCE_RELAX_YEARS",        self.state.DISTURBANCE_RELAX_YEARS,        row=1, col=2)

        # Founder values
        founder_box = ctk.CTkFrame(left)
        founder_box.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        founder_box.grid_columnconfigure((0,1,2), weight=1)

        self._section_label(founder_box, "Initial Founder Values", row=0, col=0)
        self._labeled_entry(founder_box, "NE_RATIO_START",        self.state.NE_RATIO_START,       row=1, col=0)
        self._labeled_entry(founder_box, "NE_RATIO_END",          self.state.NE_RATIO_END,         row=1, col=1)
        self._labeled_entry(founder_box, "NE_RATIO_RELAX_YEARS",  self.state.NE_RATIO_RELAX_YEARS, row=1, col=2)

        # Demography
        demo_box = ctk.CTkFrame(left)
        demo_box.grid(row=4, column=0, sticky="ew", padx=5, pady=5)
        demo_box.grid_columnconfigure((0,1), weight=1)
        self._section_label(demo_box, "Demography", row=0, col=0)
        self._labeled_entry(demo_box, "INIT_PROP_ADULT",  self.state.INIT_PROP_ADULT,  row=1, col=0)
        self._labeled_entry(demo_box, "INIT_FEMALE_FRAC", self.state.INIT_FEMALE_FRAC, row=1, col=1)

        # Catastrophe
        cat_box = ctk.CTkFrame(left)
        cat_box.grid(row=5, column=0, sticky="ew", padx=5, pady=5)
        cat_box.grid_columnconfigure((0,1), weight=1)
        self._section_label(cat_box, "Catastrophe Event", row=0, col=0)
        self._labeled_entry(cat_box, "P_EVENT",  self.state.P_EVENT,  row=1, col=0)
        self._labeled_entry(cat_box, "SEVERITY", self.state.SEVERITY, row=1, col=1)

        # Run buttons bottom
        run_box = ctk.CTkFrame(left)
        run_box.grid(row=6, column=0, sticky="ew", padx=5, pady=10)
        run_quick_btn = ctk.CTkButton(run_box, text="RUN QUICK", command=lambda: self._run_simulation(mode="quick"))
        run_quick_btn.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        run_full_btn = ctk.CTkButton(run_box, text="RUN COMPLETE", command=lambda: self._run_simulation(mode="complete"))
        run_full_btn.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # Right column could be placeholders for notes / project info
        right = ctk.CTkFrame(frame)
        right.grid(row=0, column=1, columnspan=2, sticky="nsew", padx=10, pady=10)
        right.grid_rowconfigure(2, weight=1)
        right.grid_columnconfigure(0, weight=1)

        proj_label = ctk.CTkLabel(right, text="Project info / status", font=ctk.CTkFont(size=16, weight="bold"))
        proj_label.grid(row=0, column=0, sticky="w", padx=10, pady=(10,5))

        current_dir_label = ctk.CTkLabel(right, text="Current result folder:")
        current_dir_label.grid(row=1, column=0, sticky="w", padx=10, pady=(5,0))

        current_dir_val = ctk.CTkLabel(right, textvariable=self.state.RESULT_DIR, anchor="w")
        current_dir_val.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0,10))

        return frame

    ###################################################################
    # Organism tab
    ###################################################################

    def _build_organism_tab(self, master):
        frame = ctk.CTkFrame(master)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(2, weight=1)

        # row0: organism selector + load
        head = ctk.CTkFrame(frame)
        head.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        head.grid_columnconfigure((0,1,2), weight=1)

        ctk.CTkLabel(head, text="Organism:").grid(row=0, column=0, sticky="w", padx=5)
        organism_menu = ctk.CTkOptionMenu(
            head,
            values=["(select)", "Bufo", "Lissotriton"],
            variable=self.state.SELECTED_ORGANISM,
            command=self._on_species_select  # NEW
        )
        organism_menu.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        load_org_btn = ctk.CTkButton(head, text="Load organism file", command=self._load_organism_file)
        load_org_btn.grid(row=0, column=2, sticky="ew", padx=5, pady=5)

        # params grid
        params = ctk.CTkFrame(frame)
        params.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        for i in range(4):
            params.grid_columnconfigure(i, weight=1)

        self._labeled_entry(params, "NE_RATIOS",           self.state.NE_RATIOS,           row=0, col=0)
        self._labeled_entry(params, "GEN_TIME_YEARS",      self.state.GEN_TIME_YEARS,      row=0, col=1)
        self._labeled_entry(params, "MIGRANTS_PER_YEAR",   self.state.MIGRANTS_PER_YEAR,   row=0, col=2)
        self._labeled_entry(params, "K",                   self.state.K,                   row=0, col=3)

        self._labeled_entry(params, "H0",                  self.state.H0,                  row=2, col=0)
        self._labeled_entry(params, "GEN_TIME_FOR_GENET",  self.state.GEN_TIME_FOR_GENET,  row=2, col=1)
        self._labeled_entry(params, "R_GRID",              self.state.R_GRID,              row=2, col=2)
        self._labeled_entry(params, "SIGMA_E_GRID",        self.state.SIGMA_E_GRID,        row=2, col=3)

        # run buttons
        run_box = ctk.CTkFrame(frame)
        run_box.grid(row=3, column=0, sticky="ew", padx=10, pady=10)
        run_box.grid_columnconfigure((0,1), weight=1)

        run_q = ctk.CTkButton(run_box, text="RUN QUICK", command=lambda: self._run_simulation(mode="quick"))
        run_q.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        run_c = ctk.CTkButton(run_box, text="RUN COMPLETE", command=lambda: self._run_simulation(mode="complete"))
        run_c.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        return frame

    ###################################################################
    # Model tab
    ###################################################################

    def _build_model_tab(self, master):
        frame = ctk.CTkFrame(master)
        frame.grid_columnconfigure((0,1), weight=1)
        frame.grid_rowconfigure(3, weight=1)

        # model params block
        block = ctk.CTkFrame(frame)
        block.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        block.grid_columnconfigure((0,1,2), weight=1)

        # Density model choice
        ctk.CTkLabel(block, text="DENSITY_MODEL").grid(row=0, column=0, sticky="w", padx=5)
        density_menu = ctk.CTkOptionMenu(block,
                                         values=["theta-logistic", "ricker"],
                                         variable=self.state.DENSITY_MODEL)
        density_menu.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

        self._labeled_entry(block, "THETA", self.state.THETA, row=0, col=1)

        # GENETIC_EVAL choice
        ctk.CTkLabel(block, text="GENETIC_EVAL").grid(row=0, column=2, sticky="w", padx=5)
        genetic_menu = ctk.CTkOptionMenu(block,
                                         values=["with_migration", "isolated", "best_of_both"],
                                         variable=self.state.GENETIC_EVAL)
        genetic_menu.grid(row=1, column=2, sticky="ew", padx=5, pady=5)

        # second row of model params
        block2 = ctk.CTkFrame(frame)
        block2.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
        for i in range(4):
            block2.grid_columnconfigure(i, weight=1)

        self._labeled_entry(block2, "REPLICATES",      self.state.REPLICATES,      row=0, col=0)
        self._labeled_entry(block2, "RANDOM_SEED",     self.state.RANDOM_SEED,     row=0, col=1)
        self._labeled_entry(block2, "NE_TARGET",       self.state.NE_TARGET,       row=0, col=2)
        self._labeled_entry(block2, "ALPHA",           self.state.ALPHA,           row=0, col=3)

        self._labeled_entry(block2, "SURVIVAL_TARGET", self.state.SURVIVAL_TARGET, row=2, col=0)
        self._labeled_entry(block2, "H_MIN_FRAC",      self.state.H_MIN_FRAC,      row=2, col=1)

        # run buttons
        run_box = ctk.CTkFrame(frame)
        run_box.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        run_box.grid_columnconfigure((0,1), weight=1)

        run_q = ctk.CTkButton(run_box, text="RUN QUICK", command=lambda: self._run_simulation(mode="quick"))
        run_q.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        run_c = ctk.CTkButton(run_box, text="RUN COMPLETE", command=lambda: self._run_simulation(mode="complete"))
        run_c.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        return frame

    ###################################################################
    # Results tab
    ###################################################################

    def _build_results_tab(self, master):
        # Outer frame for this tab
        frame = ctk.CTkFrame(master)
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        # 1) Paned window to allow resizing between table and plots
        paned = tk.PanedWindow(
            frame,
            orient="horizontal",
            sashrelief="raised",
            sashwidth=6,
            bg="#2b2b2b",      # dark-ish, to blend with CustomTkinter
            bd=0,
            relief="flat",
        )
        paned.grid(row=0, column=0, sticky="nsew")

        # LEFT PANE = table_frame
        table_frame = ctk.CTkFrame(paned)
        table_frame.grid_rowconfigure(1, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)

        head = ctk.CTkFrame(table_frame)
        head.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        head.grid_columnconfigure((0,1), weight=1)

        ctk.CTkLabel(
            head,
            text="Results table",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=5, pady=5)

        load_res_btn = ctk.CTkButton(
            head,
            text="Load results folder",
            command=self._load_results_folder
        )
        load_res_btn.grid(row=0, column=1, sticky="e", padx=5, pady=5)

        columns = [
            "Ne_ratio",
            "Ne",
            "A_survival_Nc",
            "B_const_H_migration_T_Nc100",
            "Bvar_H_T_Nc100",
            "Blink_H_T_Nc100",
            "C_survival_Nc461",
        ]

        self.tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            height=15
        )

        self.tree.heading("Ne_ratio", text="Ne_ratio")
        self.tree.heading("Ne", text="Ne")
        self.tree.heading("A_survival_Nc", text="A_survival_Nc461")
        self.tree.heading("B_const_H_migration_T_Nc100", text="B_const_H_migration_T_Nc100")
        self.tree.heading("Bvar_H_T_Nc100", text="Bvar_H_T_Nc100")
        self.tree.heading("Blink_H_T_Nc100", text="Blink_H_T_Nc100")
        self.tree.heading("C_survival_Nc461", text="C_survival_Nc461")

        for col in columns:
            self.tree.column(col, width=150, anchor="center")

        self.tree.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        vsb = ttk.Scrollbar(
            table_frame,
            orient="vertical",
            command=self.tree.yview
        )
        self.tree.configure(yscroll=vsb.set)
        vsb.grid(row=1, column=1, sticky="ns", padx=(0,5), pady=5)

        # RIGHT PANE = plots_frame
        plots_outer = ctk.CTkFrame(paned)
        plots_outer.grid_rowconfigure(1, weight=1)
        plots_outer.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            plots_outer,
            text="Graphs",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=5, pady=5)

        # Inner grid for plots, 2 columns like your mock
        plots_grid = ctk.CTkFrame(plots_outer)
        plots_grid.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        plots_grid.grid_columnconfigure((0,1), weight=1)

        # We'll create placeholders in a 2-col layout.
        # We'll keep references in self.plot_labels exactly like before,
        # but now it's a list of CTkLabels sitting in this grid.
        self.plot_labels = []

        # we plan for up to 6 plots: 0..5
        # arrange them row-wise: (0,1), (2,3), (4,5)
        max_plots = 6
        for idx in range(max_plots):
            r = idx // 2
            c = idx % 2
            pl = ctk.CTkLabel(
                plots_grid,
                text=f"[Plot {idx+1}]",
                anchor="center"
            )
            pl.grid(row=r, column=c, sticky="nsew", padx=5, pady=5)
            self.plot_labels.append(pl)

        # finally, add the two panes to the PanedWindow
        paned.add(table_frame)
        paned.add(plots_outer)

        return frame

    ###################################################################
    # New organism tab
    ###################################################################

    def _build_new_org_tab(self, master):
        frame = ctk.CTkFrame(master)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(3, weight=1)

        header = ctk.CTkFrame(frame)
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        header.grid_columnconfigure((0,1,2), weight=1)

        self._labeled_entry(header, "Organism name", self.state.NEW_ORG_NAME, row=0, col=0)

        load_template_btn = ctk.CTkButton(header, text="Load species", command=self._load_organism_file)
        load_template_btn.grid(row=0, column=1, padx=5, pady=(25,5), sticky="ew")

        save_species_btn = ctk.CTkButton(header, text="SAVE SPECIES", command=self._save_new_species)
        save_species_btn.grid(row=0, column=2, padx=5, pady=(25,5), sticky="ew")

        params = ctk.CTkFrame(frame)
        params.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        for i in range(4):
            params.grid_columnconfigure(i, weight=1)

        self._labeled_entry(params, "NE_RATIOS",          self.state.NEW_NE_RATIOS,          row=0, col=0)
        self._labeled_entry(params, "GEN_TIME_YEARS",     self.state.NEW_GEN_TIME_YEARS,     row=0, col=1)
        self._labeled_entry(params, "MIGRANTS_PER_YEAR",  self.state.NEW_MIGRANTS_PER_YEAR,  row=0, col=2)
        self._labeled_entry(params, "K",                  self.state.NEW_K,                  row=0, col=3)

        self._labeled_entry(params, "H0",                 self.state.NEW_H0,                 row=2, col=0)
        self._labeled_entry(params, "GEN_TIME_FOR_GENET", self.state.NEW_GEN_TIME_FOR_GENET, row=2, col=1)
        self._labeled_entry(params, "R_GRID",             self.state.NEW_R_GRID,             row=2, col=2)
        self._labeled_entry(params, "SIGMA_E_GRID",       self.state.NEW_SIGMA_E_GRID,       row=2, col=3)

        return frame

    ###################################################################
    # File operations and run stubs
    ###################################################################

    def _load_project(self):
        folder = filedialog.askdirectory(
            title="Select previous project folder (with extinction_scenarios.xlsx and run_settings.txt)"
        )
        if not folder:
            return

        # 1. remember folder
        self.state.RESULT_DIR.set(folder)

        # 2. load params from run_settings.txt into all the text fields
        self._load_run_settings_into_state(folder)

        # 3. populate table + plots in Results tab
        xlsx_path = os.path.join(folder, "extinction_scenarios.xlsx")
        self._populate_table_from_xlsx(xlsx_path)
        self._load_plots_from_folder(folder)

        # 4. jump straight to Results so user sees what they loaded
        self.show_frame("results")

    def _new_project(self):
        # reset some high-level fields
        self.state.PROJECT_NAME.set("NewProject")
        self.state.RESULT_DIR.set("(none)")
        messagebox.showinfo("New project", "Project fields cleared. Ready to edit parameters.")

    def _load_organism_file(self):
        path = filedialog.askopenfilename(
            title="Select organism .txt file",
            filetypes=[("Text files","*.txt"), ("All files","*.*")]
        )
        if not path:
            return

        data = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "\t" in line:
                        k, v = line.split("\t", 1)
                        data[k.strip()] = v.strip()
        except Exception as e:
            messagebox.showerror("Error", f"Could not read organism file:\n{e}")
            return

        # push into state
        s = self.state
        if "name" in data:
            self.state.SELECTED_ORGANISM.set(data["name"])
        if "NE_RATIOS" in data:
            s.NE_RATIOS.set(data["NE_RATIOS"])
        if "GEN_TIME_YEARS" in data:
            s.GEN_TIME_YEARS.set(data["GEN_TIME_YEARS"])
        if "MIGRANTS_PER_YEAR" in data:
            s.MIGRANTS_PER_YEAR.set(data["MIGRANTS_PER_YEAR"])
        if "K" in data:
            s.K.set(data["K"])
        if "H0" in data:
            s.H0.set(data["H0"])
        if "GEN_TIME_FOR_GENET" in data or "GEN_TIME_FOR_GENETICS" in data:
            s.GEN_TIME_FOR_GENET.set(data.get("GEN_TIME_FOR_GENET", data.get("GEN_TIME_FOR_GENETICS", s.GEN_TIME_FOR_GENET.get())))
        if "R_GRID" in data:
            s.R_GRID.set(data["R_GRID"])
        if "SIGMA_E_GRID" in data:
            s.SIGMA_E_GRID.set(data["SIGMA_E_GRID"])

        messagebox.showinfo("Organism loaded", f"Loaded organism file:\n{os.path.basename(path)}")

    def _save_new_species(self):
        path = filedialog.asksaveasfilename(
            title="Save species parameters",
            defaultextension=".txt",
            filetypes=[("Text files","*.txt"), ("All files","*.*")]
        )
        if path:
            # gather NEW_... vars and write them out
            data = {
                "name": self.state.NEW_ORG_NAME.get(),
                "NE_RATIOS": self.state.NEW_NE_RATIOS.get(),
                "GEN_TIME_YEARS": self.state.NEW_GEN_TIME_YEARS.get(),
                "MIGRANTS_PER_YEAR": self.state.NEW_MIGRANTS_PER_YEAR.get(),
                "K": self.state.NEW_K.get(),
                "H0": self.state.NEW_H0.get(),
                "GEN_TIME_FOR_GENET": self.state.NEW_GEN_TIME_FOR_GENET.get(),
                "R_GRID": self.state.NEW_R_GRID.get(),
                "SIGMA_E_GRID": self.state.NEW_SIGMA_E_GRID.get(),
            }
            with open(path, "w", encoding="utf-8") as f:
                for k,v in data.items():
                    f.write(f"{k}\t{v}\n")
            messagebox.showinfo("Saved", f"Species saved to:\n{path}")

    def _run_simulation(self, mode="quick"):
        """
        1. Collect parameters from GUIState.
        2. Call backend.run_survive(...) to actually run simulations.
        3. Point RESULT_DIR at the new folder.
        4. Refresh Results tab.
        """

        try:
            import survive_1_1 as backend  # you'll rename survive_1.1.py -> survive_1_1.py so Python can import it
        except ImportError:
            messagebox.showerror("Backend missing",
                                 "Could not import survive_1_1.py.\n"
                                 "Rename survive_1.1.py -> survive_1_1.py and make sure it's in the same folder.")
            return

        # 1. grab all fields from GUI
        s = self.state  # shorthand
        result_dir = backend.run_survive(
            PROJECT_NAME                = s.PROJECT_NAME.get(),
            NC_POP                      = s.NC_POP.get(),
            NC_METAPOP                  = s.NC_METAPOP.get(),
            TIME_HORIZON_YEARS          = s.TIME_HORIZON_YEARS.get(),
            QUASI_EXT_THRESHOLD         = s.QUASI_EXT_THRESHOLD.get(),

            DISTURBANCE_SIGMA_FACTOR_START = s.DISTURBANCE_SIGMA_FACTOR_START.get(),
            DISTURBANCE_SIGMA_FACTOR_END   = s.DISTURBANCE_SIGMA_FACTOR_END.get(),
            DISTURBANCE_RELAX_YEARS        = s.DISTURBANCE_RELAX_YEARS.get(),

            NE_RATIO_START             = s.NE_RATIO_START.get(),
            NE_RATIO_END               = s.NE_RATIO_END.get(),
            NE_RATIO_RELAX_YEARS       = s.NE_RATIO_RELAX_YEARS.get(),

            INIT_PROP_ADULT            = s.INIT_PROP_ADULT.get(),
            INIT_FEMALE_FRAC           = s.INIT_FEMALE_FRAC.get(),

            P_EVENT                    = s.P_EVENT.get(),
            SEVERITY                   = s.SEVERITY.get(),

            NE_RATIOS                  = s.NE_RATIOS.get(),
            GEN_TIME_YEARS             = s.GEN_TIME_YEARS.get(),
            MIGRANTS_PER_YEAR          = s.MIGRANTS_PER_YEAR.get(),
            K                          = s.K.get(),
            H0                         = s.H0.get(),
            GEN_TIME_FOR_GENETICS      = s.GEN_TIME_FOR_GENET.get(),
            R_GRID                     = s.R_GRID.get(),
            SIGMA_E_GRID               = s.SIGMA_E_GRID.get(),

            DENSITY_MODEL              = s.DENSITY_MODEL.get(),
            THETA                      = s.THETA.get(),
            GENETIC_EVAL               = s.GENETIC_EVAL.get(),
            REPLICATES                 = s.REPLICATES.get(),
            RANDOM_SEED                = s.RANDOM_SEED.get(),
            NE_TARGET                  = s.NE_TARGET.get(),
            ALPHA                      = s.ALPHA.get(),
            SURVIVAL_TARGET            = s.SURVIVAL_TARGET.get(),
            H_MIN_FRAC                 = s.H_MIN_FRAC.get(),

            # You can also pass SPECIES from the Organism dropdown:
            SPECIES = "B" if s.SELECTED_ORGANISM.get().lower().startswith("b") else "L",
        )

        # 2. update GUI state with the new folder path
        self.state.RESULT_DIR.set(result_dir)

        # 3. refresh Results tab from that folder
        xlsx_path = os.path.join(result_dir, "extinction_scenarios.xlsx")
        self._populate_table_from_xlsx(xlsx_path)
        self._load_plots_from_folder(result_dir)

        # 4. jump user to Results tab so they immediately see output
        self.show_frame("results")
        messagebox.showinfo("Done", f"Simulation finished.\nResults in:\n{result_dir}")

    def _load_results_folder(self):
        folder = filedialog.askdirectory(title="Select results folder (with extinction_scenarios.xlsx + pngs)")
        if not folder:
            return
        self.state.RESULT_DIR.set(folder)

        # 1. populate table from extinction_scenarios.xlsx if found
        xlsx_path = os.path.join(folder, "extinction_scenarios.xlsx")
        self._populate_table_from_xlsx(xlsx_path)

        # 2. load plots into right panel
        self._load_plots_from_folder(folder)

    def _populate_table_from_xlsx(self, xlsx_path):
        for row in self.tree.get_children():
            self.tree.delete(row)

        if not os.path.exists(xlsx_path):
            messagebox.showwarning("Missing", "No extinction_scenarios.xlsx found.")
            return

        try:
            import pandas as pd
            df = pd.read_excel(xlsx_path, sheet_name="Results")

            # We try to reconstruct the columns similar to mock:
            # Ne_ratio, Ne, A_survival..., B_const_H_migration_T..., Bvar_H_T..., Blink_H_T..., C_survival...
            # We make safe fallbacks.
            def safe_get(row, keys, default=""):
                for k in keys:
                    if k in row:
                        return row[k]
                return default

            for _, r in df.iterrows():
                self.tree.insert(
                    "",
                    "end",
                    values=(
                        safe_get(r, ["Ne_ratio"]),
                        safe_get(r, ["Ne"]),
                        safe_get(r, [c for c in df.columns if c.startswith("A_survival")]),
                        safe_get(r, [c for c in df.columns if c.startswith("B_const_H_migration_T")]),
                        safe_get(r, [c for c in df.columns if c.startswith("Bvar_H_T")]),
                        safe_get(r, [c for c in df.columns if c.startswith("Blink_H_T")]),
                        safe_get(r, [c for c in df.columns if c.startswith("C_survival")]),
                    )
                )
        except Exception as e:
            messagebox.showerror("Error reading results", str(e))

    def _load_plots_from_folder(self, folder):
        # You can reorder this list to control which plot goes where in the 2-column grid.
        plot_candidates = [
            "sensitivity_sigma.png",
            "mod_A_survival_density_demography.png",
            "mod_B_heterozygosity_genetics.png",
            "mod_C_survival_ne_sensitive.png",
            "mod_Blink_eco_genetic_coupling.png",
            "disturbance_decay.png",
        ]

        # ----- SIZE KNOB -----
        # All plots will be scaled down to fit inside this box,
        # keeping aspect ratio. Increase/decrease to taste.
        MAX_PLOT_WIDTH = 500
        MAX_PLOT_HEIGHT = 300
        # ---------------------

        loaded_imgs = []

        for fname in plot_candidates:
            path = os.path.join(folder, fname)
            if not os.path.exists(path):
                loaded_imgs.append(None)
                continue

            try:
                img = Image.open(path)

                # original size
                orig_w, orig_h = img.size

                # compute scale factors so we fit in the box without distorting aspect ratio
                scale_w = MAX_PLOT_WIDTH / orig_w
                scale_h = MAX_PLOT_HEIGHT / orig_h
                scale = min(scale_w, scale_h, 1.0)
                # min(..., 1.0) means: never upscale above original. If you DO want upscale,
                # drop the ", 1.0".

                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)

                if scale < 1.0:
                    img = img.resize((new_w, new_h), Image.LANCZOS)

                # wrap in PhotoImage for tkinter
                loaded_imgs.append(ImageTk.PhotoImage(img))

            except Exception:
                loaded_imgs.append(None)

        # push into grid labels (self.plot_labels was created in _build_results_tab)
        for i, lbl in enumerate(self.plot_labels):
            if i < len(loaded_imgs) and loaded_imgs[i] is not None:
                lbl.configure(image=loaded_imgs[i], text="")
                lbl.image_ref = loaded_imgs[i]  # prevent GC
            else:
                lbl.configure(image=None, text="[No plot]")
                lbl.image_ref = None


###############################################################################
# main start
###############################################################################

def main():
    # 1. Set dark theme etc
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    # 2. Create ONE true root window
    root = ctk.CTk()

    # 3. Hide it for now (so user only sees splash first)
    root.withdraw()

    # 4. Build state and main UI on that hidden root
    state = GUIState(root)
    app = SurviveApp(root, state)
    # app.pack_forget()  # we don't actually need to pack the frame itself

    # 5. Create splash as a Toplevel ABOVE the hidden root
    splash = Splash(
        master=root,
        logo_path=r"C:\Users\EricWahlsteen\OneDrive - Calluna AB\Skrivbordet\Python_test\Survive\GUI\logo.png",
        duration_ms=3000,
        on_close=None  # we don't strictly need a callback, deiconify happens in _finish
    )

    # 6. Now enter the single mainloop.
    # The splash will self-destruct after duration_ms, then show root.
    root.mainloop()


if __name__ == "__main__":
    main()

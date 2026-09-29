"""Comparison of SoleID with every published result on the same data sets, in one metric.

Three questions had to be settled before a table like this can be honest:

1. *Units.* Moment results are almost always reported in Nm/kg, so no conversion is needed for
   the studies that share our data sets. The shear-force literature reports normalised RMSE, and
   converting it needs the peak or the range of the reference signal on that data set. We measure
   those constants ourselves (norms.py) for the data sets we process; for studies on other data
   we can only give a bracket, and we label it as such.

2. *Window.* Our shear RMSE is computed over stance, our moment RMSE over the whole cycle. The
   shear estimate is constrained to exactly zero while the foot is off the ground, so the
   full-cycle value follows from the stance value as stance x sqrt(stance fraction); both are
   reported.

3. *Protocol.* A learned model evaluated with subjects shared between training and test is not
   comparable to one evaluated leave-subjects-out, and neither is directly comparable to a method
   with nothing to train. This cannot be fixed by arithmetic, so the protocol is a column.

Outputs: results/comparison_tables.md and the LaTeX table at config.TABLES_TEX
"""
import json
import os

import numpy as np
import pandas as pd
import config

R = config.RESULTS
TEX = config.TABLES_TEX

norm = json.load(open(os.path.join(R, "norm_constants.json")))
t = pd.read_csv(os.path.join(R, "test_all_trials.csv"))
cam = pd.read_csv(os.path.join(R, "camargo_treadmill_strides.csv"))
lg = pd.read_csv(os.path.join(R, "camargo_levelground_steps.csv"))
ramp = pd.read_csv(os.path.join(R, "camargo_ramp_steps.csv"))
w = pd.read_csv(os.path.join(R, "wang_application.csv"))
cc = pd.read_csv(os.path.join(R, "wang_crosscheck.csv"))
RA = json.load(open(os.path.join(R, "reference_agreement.json")))

SF = norm["fukuchi"]["_meta"]["stance_fraction"]


def cycle(stance_pctbw):
    """Shear RMSE over the whole cycle, given the value over stance.

    Off-contact frames are constrained to zero in every arm we report, and the reference is zero
    there too, so those frames contribute no error: RMSE_cycle = RMSE_stance * sqrt(stance frac)."""
    return stance_pctbw * np.sqrt(SF)


# --------------------------------------------------------------------------- our own numbers
OURS = {
    "fukuchi": dict(
        shear_stance=t.SoleID_stance.mean(), ankle=t.SoleID_ankle.mean(),
        knee=t.SoleID_knee.mean(), hip=t.SoleID_hip.mean(), n_subj=t.subject.nunique()),
    "camargo_tm": dict(
        shear_stance=cam.SoleID_stance.mean(), ankle=cam.SoleID_ankle.mean(),
        knee=cam.SoleID_knee.mean(), hip=cam.SoleID_hip.mean(), n_subj=cam.subject.nunique()),
    "camargo_lg": dict(
        shear_stance=lg.SoleID_stance.mean(), ankle=lg.SoleID_ankle.mean(),
        knee=lg.SoleID_knee.mean(), hip=lg.SoleID_hip.mean(), n_subj=lg.subject.nunique()),
    "camargo_ramp": dict(
        shear_stance=ramp.SoleID_stance.mean(), ankle=ramp.SoleID_ankle.mean(),
        knee=ramp.SoleID_knee.mean(), hip=ramp.SoleID_hip.mean(), n_subj=ramp.subject.nunique()),
    "wang": dict(
        shear_stance=w.SoleID_stance.mean(), ankle=w.SoleID_ankle.mean(),
        knee=w.SoleID_knee.mean(), hip=w.SoleID_hip.mean(), n_subj=w.subject.nunique()),
}


def relrmse(rmse, dataset, signal, by="range", win="cycle"):
    """Express an absolute RMSE as the literature's normalised RMSE, using the measured constant."""
    return 100 * rmse / norm[dataset][signal][win][by]


# --------------------------------------------------------------------------- the tables
def table_camargo():
    """Every result we could find that was produced on Camargo et al. 2021."""
    rows = [
        dict(study="Altai et al. 2023", inp="4 IMU", method="deep net (XCM)", split="mixed*",
             n=22, ankle=0.051, knee=0.051, hip=0.055, shear=None),
        dict(study="Altai et al. 2023", inp="4 IMU", method="deep net (XCM)",
             split="leave-subjects-out", n=22, ankle=0.124, knee=0.182, hip=0.133, shear=None),
        dict(study="Weber & Stetter 2024", inp="4 IMU", method="CNN",
             split="leave-one-subject-out", n=21, ankle=None, knee=None, hip=None, shear=None,
             pooled=(0.16, 0.18)),
        dict(study="Weber & Stetter 2024", inp="4 IMU + 11 EMG", method="CNN",
             split="leave-one-subject-out", n=21, ankle=None, knee=None, hip=None, shear=None,
             pooled=(0.16, 0.18)),
        dict(study="SoleID (treadmill)", inp="plate $F_z$+CoP, markers", method="convex QP",
             split="calibrated on another data set", n=OURS["camargo_tm"]["n_subj"],
             ankle=OURS["camargo_tm"]["ankle"], knee=OURS["camargo_tm"]["knee"],
             hip=OURS["camargo_tm"]["hip"], shear=OURS["camargo_tm"]["shear_stance"]),
        dict(study="SoleID (level ground)", inp="plate $F_z$+CoP, markers", method="convex QP",
             split="calibrated on another data set", n=OURS["camargo_lg"]["n_subj"],
             ankle=OURS["camargo_lg"]["ankle"], knee=OURS["camargo_lg"]["knee"],
             hip=OURS["camargo_lg"]["hip"], shear=OURS["camargo_lg"]["shear_stance"]),
        dict(study="SoleID (ramp)", inp="plate $F_z$+CoP, markers", method="convex QP",
             split="calibrated on another data set", n=OURS["camargo_ramp"]["n_subj"],
             ankle=OURS["camargo_ramp"]["ankle"], knee=OURS["camargo_ramp"]["knee"],
             hip=OURS["camargo_ramp"]["hip"], shear=OURS["camargo_ramp"]["shear_stance"]),
    ]
    return rows


def table_wang():
    """Camargo and Fukuchi aside, the Wang layer needs no conversion: we recomputed the
    comparator ourselves from the authors' released output files."""
    return [
        # scored like every other arm: same planar reference, whole trimmed record, one sign per
        # joint (reference_agreement.py)
        dict(arm="Wang et al., their own wearable pipeline",
             ankle=RA["wang"]["wangwear_vs_planar"]["ankle"]["rmse"],
             knee=RA["wang"]["wangwear_vs_planar"]["knee"]["rmse"],
             hip=RA["wang"]["wangwear_vs_planar"]["hip"]["rmse"],
             ar=RA["wang"]["wangwear_vs_planar"]["ankle"]["r"],
             kr=RA["wang"]["wangwear_vs_planar"]["knee"]["r"],
             hr=RA["wang"]["wangwear_vs_planar"]["hip"]["r"], shear=None),
        dict(arm="vertical-only (shear set to zero)", ankle=w.NoShear_ankle.mean(),
             knee=w.NoShear_knee.mean(), hip=w.NoShear_hip.mean(), ar=w.NoShear_ankle_r.mean(),
             kr=w.NoShear_knee_r.mean(), hr=w.NoShear_hip_r.mean(), shear=w.NoShear_stance.mean()),
        dict(arm="Newton + proportional split", ankle=w.Newton_prop_ankle.mean(),
             knee=w.Newton_prop_knee.mean(), hip=w.Newton_prop_hip.mean(),
             ar=w.Newton_prop_ankle_r.mean(), kr=w.Newton_prop_knee_r.mean(),
             hr=w.Newton_prop_hip_r.mean(), shear=w.Newton_prop_stance.mean()),
        dict(arm="SoleID (IMU kinematics)", ankle=w.SoleID_ankle.mean(), knee=w.SoleID_knee.mean(),
             hip=w.SoleID_hip.mean(), ar=w.SoleID_ankle_r.mean(), kr=w.SoleID_knee_r.mean(),
             hr=w.SoleID_hip_r.mean(), shear=w.SoleID_stance.mean()),
        dict(arm="SoleID (laboratory kinematics)", ankle=w.SoleID_lab_kin_ankle.mean(),
             knee=w.SoleID_lab_kin_knee.mean(), hip=w.SoleID_lab_kin_hip.mean(),
             ar=w.SoleID_lab_kin_ankle_r.mean(), kr=w.SoleID_lab_kin_knee_r.mean(),
             hr=w.SoleID_lab_kin_hip_r.mean(), shear=w.SoleID_lab_kin_stance.mean()),
    ]


def table_other_datasets():
    """Shear-force results obtained on data sets we do not process.

    These cannot be converted exactly: the normalising constant belongs to their data, not ours.
    The bracket below applies our two measured constants (Fukuchi and Camargo treadmill) to their
    normalised value, and therefore states the range a reader should treat as plausible rather
    than a number. Where the paper does not say whether it normalised by peak or by range, the
    bracket spans both, which is why some of them are wide.
    """
    pk = [norm["fukuchi"]["shear"]["cycle"]["peak"], norm["camargo_treadmill"]["shear"]["cycle"]["peak"]]
    rg = [norm["fukuchi"]["shear"]["cycle"]["range"], norm["camargo_treadmill"]["shear"]["cycle"]["range"]]

    def bracket(pct, by):
        base = pk if by == "peak" else rg if by == "range" else pk + rg
        v = sorted(pct / 100 * b for b in base)
        return v[0], v[-1]

    rows = [
        dict(study="Savelberg & de Lange 1999", inp="insole pressure", method="ANN", n=5,
             rep="qualitative", by=None, val=None),
        dict(study="Fong et al. 2008", inp="99 pressure sensors", method="stepwise regression",
             n=5, rep="r = 0.928 (AP)", by=None, val=None),
        dict(study="Rouhani et al. 2010", inp="insole pressure", method="PCA + neuro-fuzzy",
             n=None, rep="nRMSE 17.3 %", by="unstated", val=17.32),
        dict(study="Sim et al. 2015", inp="insole pressure", method="wavelet NN", n=None,
             rep="nRMSE 12.9 %", by="unstated", val=12.92),
        dict(study="Wei et al. 2019", inp="insole pressure", method="multistage regression",
             n=None, rep="nRMSE 10 %", by="unstated", val=10.0),
        dict(study="Wagner et al. 2024", inp="IMU kinematics only", method="VPP geometric constraint",
             n=11, rep="4.2 \\%BW (cycle)", by="absolute", val=None),
        dict(study="OpenGRF 2025", inp="kinematics (OpenSim)", method="CMC + static optimisation",
             n=7, rep="nRMSE 4.4-6.1 %", by="unstated", val=6.1),
    ]
    for r in rows:
        if r["val"] is not None:
            lo, hi = bracket(r["val"], r["by"])
            r["converted"] = f"{lo:.1f}-{hi:.1f}"
        elif r["study"].startswith("Wagner"):
            r["converted"] = "4.2"
        else:
            r["converted"] = "--"
    return rows


# --------------------------------------------------------------------------- writers
def md():
    L = []
    A = L.append
    A("# Published results on the data sets used here, in one metric\n")
    A("Generated by `comparison_tables.py`. The normalisation constants were measured by `norms.py`")
    A("(`results/norm_constants.json`).\n")

    A("## Measured normalisation constants\n")
    A("Constants needed to turn a published `nRMSE %` into absolute units, measured **per stride** on")
    A("the data sets processed here:\n")
    A("| Data set | signal | peak | range (max-min) |")
    A("|---|---|---|---|")
    for ds, lab in (("fukuchi", "Fukuchi 2018"), ("camargo_treadmill", "Camargo 2021, treadmill")):
        for sig, unit in (("shear", "%BW"), ("ankle", "N m/kg"), ("knee", "N m/kg"), ("hip", "N m/kg")):
            v = norm[ds][sig]["cycle"]
            A(f"| {lab} | {sig} ({unit}) | {v['peak']:.2f} | {v['range']:.2f} |")
    A("")

    A("## Table A: every result on the Camargo et al. (2021) data set\n")
    A("Joint moment RMSE (N m/kg) over the gait cycle. Shear in %BW (stance / whole cycle).\n")
    A("| Study | Input | Method | Protocol | Subjects | Ankle | Knee | Hip | Shear |")
    A("|---|---|---|---|---|---|---|---|---|")
    for r in table_camargo():
        f = lambda x: "-" if x is None else f"{x:.3f}"
        sh = "-" if r["shear"] is None else f"{r['shear']:.2f} / {cycle(r['shear']):.2f}"
        if r.get("pooled"):
            # the authors pool the joints; showing the value in one column would read as if it
            # were that joint's result
            a = k = h = f"{r['pooled'][0]:.2f}-{r['pooled'][1]:.2f} (pooled)"
        else:
            a, k, h = f(r["ankle"]), f(r["knee"]), f(r["hip"])
        A(f"| {r['study']} | {r['inp']} | {r['method']} | {r['split']} | {r['n']} | {a} | {k} | {h} | {sh} |")
    A("")
    A("* In the \"mixed\" protocol of Altai et al. the same subjects appear in training and test; by")
    A("  the authors' own account it gives a 59 % lower RMSE than leaving subjects out. The")
    A("  leave-subjects-out row is the comparable one.")
    A("* Weber and Stetter report one figure pooled over four moment outputs (hip flexion and")
    A("  adduction, knee, ankle) and five tasks; it is repeated in three columns because it belongs")
    A("  to none of them.")
    A("* The tasks do not match: both learned models are evaluated over tasks that include stairs,")
    A("  which are not processed here, and both are trained on subjects of this data set, which")
    A("  SoleID never saw. The rows are context, not a ranking.\n")

    A("## Table B: every result on the Wang et al. (2023) data set\n")
    A("No conversion is needed: the only comparator is the wearable pipeline released with the data,")
    A("rescored here from its own output files against the same reference, window and sign convention.\n")
    A("| Arm | Ankle | Knee | Hip | r ankle | r knee | r hip | Shear (%BW, stance) |")
    A("|---|---|---|---|---|---|---|---|")
    for r in table_wang():
        sh = "-" if r["shear"] is None else f"{r['shear']:.2f}"
        A(f"| {r['arm']} | {r['ankle']:.3f} | {r['knee']:.3f} | {r['hip']:.3f} | "
          f"{r['ar']:.2f} | {r['kr']:.2f} | {r['hr']:.2f} | {sh} |")
    A("")

    A("## Table C: the Fukuchi et al. (2018) data set\n")
    A("No published force or moment estimate on this data set was found to compare with; it serves")
    A("here as the controlled validation layer, not as a comparison.\n")
    A(f"SoleID ({OURS['fukuchi']['n_subj']} test subjects, {len(t)} trials): ankle "
      f"{OURS['fukuchi']['ankle']:.3f}, knee {OURS['fukuchi']['knee']:.3f}, hip "
      f"{OURS['fukuchi']['hip']:.3f} N m/kg; shear {OURS['fukuchi']['shear_stance']:.2f} %BW over "
      f"stance, {cycle(OURS['fukuchi']['shear_stance']):.2f} over the cycle.\n")

    A("## Table D: shear results on other data sets (approximate)\n")
    A("These cannot be converted exactly: the normalising constant belongs to their data. The range")
    A("applies the two constants measured here and, where a paper does not state how it normalised,")
    A("covers both peak and range. Its width is the reason such comparisons are not reliable.\n")
    A("| Study | Input | Method | Subjects | Reported | Approx. %BW |")
    A("|---|---|---|---|---|---|")
    for r in table_other_datasets():
        A(f"| {r['study']} | {r['inp']} | {r['method']} | {r['n'] or '-'} | {r['rep']} | {r['converted']} |")
    A("")
    A(f"For reference, SoleID: {cycle(OURS['fukuchi']['shear_stance']):.2f} %BW (Fukuchi, whole "
      f"cycle) and {cycle(OURS['camargo_tm']['shear_stance']):.2f} %BW (Camargo treadmill, whole cycle).\n")

    A("## Not convertible: correlation\n")
    A("r, R^2 and CMC cannot be turned into an RMSE; they carry different information (shape rather")
    A("than magnitude) and are reported alongside. The Wang layer shows why: the vertical-only arm")
    A("has a higher hip correlation than SoleID with IMU kinematics, but a worse RMSE.\n")
    return "\n".join(L)


def esc(x):
    """An ampersand in a study name is a column separator in LaTeX unless it is escaped."""
    return str(x).replace("&", r"\&")


def tex():
    L = []
    A = L.append
    A("% Generated by comparison_tables.py -- do not edit numbers by hand.")
    # seven columns, three of them text: too wide for one column of a two-column page
    A("\\begin{table*}[!tp]")
    A("\\caption{Published results on the Camargo \\emph{et al.}~\\cite{camargo2021} data set.")
    A("Joint moment RMSE (\\Nmkg) over the gait cycle}")
    A("\\label{tab:cmp_camargo}")
    A("\\centering")
    A("\\renewcommand{\\arraystretch}{1.2}")
    A("\\setlength{\\tabcolsep}{5pt}")
    A("\\begin{tabular}{@{}llccccc@{}}")
    A("\\toprule")
    A("\\textbf{Study} & \\textbf{Input} & \\textbf{Protocol} & \\textbf{Ank.} & \\textbf{Knee} "
      "& \\textbf{Hip} & \\textbf{Shear} \\\\")
    A(" & & & & & & (\\BW) \\\\")
    A("\\midrule")
    for r in table_camargo():
        if r.get("pooled"):
            # the authors pool the three joints, so the value belongs to none of them alone
            a = k = h = f"{r['pooled'][0]:.2f}--{r['pooled'][1]:.2f}$^\\dagger$"
        else:
            a, k, h = (f"{r[x]:.3f}" if r[x] is not None else "---" for x in ("ankle", "knee", "hip"))
        sh = "---" if r["shear"] is None else f"{r['shear']:.2f}"
        split = r["split"].replace("mixed*", "mixed$^*$").replace(
            "calibrated on another data set", "other data set")
        if r["study"].startswith(("Altai", "Weber")):
            split += "$^\\ddagger$"        # the tasks do not match: see the footnote
        A(f"{esc(r['study'])} & {esc(r['inp'])} & {esc(split)} & {a} & {k} & {h} & {sh} \\\\")
    A("\\midrule")
    _o = [r for r in table_camargo() if r["study"].startswith("SoleID")]
    if _o:
        _m = [sum(r[j] for r in _o) / len(_o) for j in ("ankle", "knee", "hip")]
        A(f"SoleID, conditions pooled & plate $F_z$+CoP, markers & other data set "
          f"& {_m[0]:.3f} & {_m[1]:.3f} & {_m[2]:.3f} & --- \\\\")
    A("\\bottomrule")
    A("\\multicolumn{7}{@{}p{0.98\\textwidth}@{}}{\\footnotesize $^*$Subjects shared between "
      "training and test (\\num{59}\\,\\% lower RMSE than leave-subjects-out, per the authors). "
      "$^\\dagger$One figure pooled over four outputs (hip flexion and adduction, knee, ankle) "
      "and five tasks, repeated because it belongs to no single column. $^\\ddagger$Tasks do not "
      "match: six modes (Altai) and five tasks (Weber), both with stairs, which we do not attempt. "
      "The last row averages our three conditions with equal weight, for orientation only.}")
    A("\\end{tabular}")
    A("\\end{table*}")
    return "\n".join(L)


# --------------------------------------------------------------------------- two further layouts
# The same results arranged differently: every setting at a glance, and every published estimator of
# joint moments or shear force on public data, with its input and protocol, in one table.
SETTINGS = [  # layer, setting, frame, unit of analysis
    ("Controlled", "Fukuchi, treadmill", t, "trials"),
    ("External", "Camargo, treadmill", cam, "strides"),
    ("", "Camargo, level ground", lg, "steps"),
    ("", "Camargo, ramps", ramp, "steps"),
    ("Application", "Wang, insoles and IMU suit", w, "trials"),
]

# Zhou et al. 2026, Table 3 (lookback of 48 frames): RMSE with sliding-window strides of 1 and 48 frames,
# averaged over thirteen tasks; ground reaction force in body weights, moments in N m/kg
ZHOU = {1: dict(ankle=0.091, knee=0.10, hip=0.11, ap=0.019),
        48: dict(ankle=0.19, knee=0.21, hip=0.21, ap=0.030)}


def _count(df, unit):
    return f"{df.subject.nunique()}, " + f"{len(df):,}".replace(",", "\\,") + f" {unit}"


def tex_summary():
    """Table: SoleID in every setting, with the Newton + proportional baseline beside it."""
    def pair(df, m, fmt):
        return f"{fmt % df[f'SoleID_{m}'].mean()} ({fmt % df[f'Newton_prop_{m}'].mean()})"

    L = []
    A = L.append
    A("% Generated by comparison_tables.py -- do not edit numbers by hand.")
    A("\\begin{table}[!ht]")
    A("\\caption{Accuracy in every setting: root-mean-square error of \\SoleID, with the Newton +")
    A("proportional baseline in parentheses. Means over trials, strides or steps}")
    A("\\label{tab:summary}")
    A("\\centering")
    A("\\footnotesize")
    A("\\renewcommand{\\arraystretch}{1.25}")
    A("\\setlength{\\tabcolsep}{3pt}")
    A("\\begin{tabular}{@{}lllcccc@{}}")
    A("\\toprule")
    A("\\textbf{Layer} & \\textbf{Setting} & \\makecell[l]{\\textbf{Subjects,}\\\\\\textbf{units}} & "
      "\\makecell{\\textbf{Shear}\\\\(\\BW)} & \\makecell{\\textbf{Ankle}\\\\(\\Nmkg)} & "
      "\\makecell{\\textbf{Knee}\\\\(\\Nmkg)} & \\makecell{\\textbf{Hip}\\\\(\\Nmkg)} \\\\")
    A("\\midrule")
    for layer, setting, df, unit in SETTINGS:
        A(f"{layer} & {setting} & {_count(df, unit)} & {pair(df, 'stance', '%.2f')} & "
          f"{pair(df, 'ankle', '%.3f')} & {pair(df, 'knee', '%.3f')} & {pair(df, 'hip', '%.3f')} \\\\")
    wp = RA["wang"]["wangwear_vs_planar"]
    A(f" & published wearable pipeline$^{{a}}$ & {_count(w, 'trials')} & --- & {wp['ankle']['rmse']:.3f} & "
      f"{wp['knee']['rmse']:.3f} & {wp['hip']['rmse']:.3f} \\\\")
    A("\\bottomrule")
    A("\\multicolumn{7}{@{}p{0.97\\textwidth}@{}}{\\footnotesize $^{a}$Wang \\emph{et al.}~\\cite{wang2023}, "
      "scored against the same reference over the same window; it sets the horizontal force to zero, so "
      "it has no shear estimate.}")
    A("\\end{tabular}")
    A("\\end{table}")
    return "\n".join(L)


def tex_published():
    """Table: every published estimator on public data next to SoleID, with input and protocol."""
    def f3(x):
        return f"{x:.3f}"

    ours = {k: OURS[k] for k in ("camargo_tm", "camargo_lg", "camargo_ramp")}
    pooled = {j: sum(o[j] for o in ours.values()) / len(ours) for j in ("ankle", "knee", "hip")}
    wp = RA["wang"]["wangwear_vs_planar"]
    plate = "force-plate $F_z$ and CoP, markers"
    L = []
    A = L.append
    A("% Generated by comparison_tables.py -- do not edit numbers by hand.")
    A("\\begin{table}[!ht]")
    A("\\caption{Published estimators of joint moments and shear force on public data sets, with their "
      "inputs and validation protocols. Root-mean-square error; moments over the gait cycle}")
    A("\\label{tab:published}")
    A("\\centering")
    A("\\footnotesize")
    A("\\renewcommand{\\arraystretch}{1.15}")
    A("\\setlength{\\tabcolsep}{3.5pt}")
    A("\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{3.2cm}"
      ">{\\raggedright\\arraybackslash}p{2.9cm}>{\\raggedright\\arraybackslash}p{2.0cm}cccc@{}}")
    A("\\toprule")
    A("\\textbf{Study} & \\textbf{Input} & \\makecell[l]{\\textbf{Training and}\\\\\\textbf{test subjects}} & "
      "\\makecell{\\textbf{Ankle}\\\\(\\Nmkg)} & \\makecell{\\textbf{Knee}\\\\(\\Nmkg)} & "
      "\\makecell{\\textbf{Hip}\\\\(\\Nmkg)} & \\makecell{\\textbf{Shear}\\\\(\\BW)} \\\\")
    A("\\midrule")
    A("\\multicolumn{7}{@{}l}{\\emph{Camargo et al.\\ data set}~\\cite{camargo2021}} \\\\")
    A("Altai \\emph{et al.}~\\cite{altai2023} & 4 IMUs & shared$^{a}$ & 0.051 & 0.051 & 0.055 & --- \\\\")
    A("Altai \\emph{et al.}~\\cite{altai2023} & 4 IMUs & held out & 0.124 & 0.182 & 0.133 & --- \\\\")
    for inp in ("4 IMUs", "4 IMUs, 11 EMG"):
        A(f"Weber and Stetter~\\cite{{weber2024}} & {inp} & held out & "
          "\\multicolumn{3}{c}{0.16--0.18$^{b}$} & --- \\\\")
    for key, lab in (("camargo_tm", "treadmill"), ("camargo_lg", "level ground"), ("camargo_ramp", "ramps")):
        o = OURS[key]
        A(f"\\SoleID, {lab} & {plate} & none$^{{c}}$ & {f3(o['ankle'])} & {f3(o['knee'])} & {f3(o['hip'])} & "
          f"{o['shear_stance']:.2f} \\\\")
    A(f"\\SoleID, three settings pooled$^{{d}}$ & {plate} & none$^{{c}}$ & {f3(pooled['ankle'])} & "
      f"{f3(pooled['knee'])} & {f3(pooled['hip'])} & --- \\\\")
    A("\\addlinespace")
    A("\\multicolumn{7}{@{}l}{\\emph{Wang et al.\\ data set}~\\cite{wang2023data}} \\\\")
    A(f"Wang \\emph{{et al.}}~\\cite{{wang2023}} & 8 IMUs, pressure insoles & none & {f3(wp['ankle']['rmse'])} & "
      f"{f3(wp['knee']['rmse'])} & {f3(wp['hip']['rmse'])} & ---$^{{e}}$ \\\\")
    o = OURS["wang"]
    A(f"\\SoleID & 8 IMUs, pressure insoles & none$^{{c}}$ & {f3(o['ankle'])} & {f3(o['knee'])} & "
      f"{f3(o['hip'])} & {o['shear_stance']:.2f} \\\\")
    A("\\addlinespace")
    A("\\multicolumn{7}{@{}l}{\\emph{Scherpereel et al.\\ data set}~\\cite{scherpereel2023data}} \\\\")
    for stride, lab in ((1, ""), (48, ", non-overlapping windows")):
        z = ZHOU[stride]
        A(f"Zhou \\emph{{et al.}}~\\cite{{zhou2026}}{lab} & 16 markers & shared$^{{f}}$ & {z['ankle']:g} & "
          f"{z['knee']:.2f} & {z['hip']:.2f} & {100 * z['ap']:.1f}$^{{g}}$ \\\\")
    A("\\bottomrule")
    A("\\multicolumn{7}{@{}p{0.97\\textwidth}@{}}{\\footnotesize CoP, center of pressure; EMG, "
      "electromyography; IMU, inertial measurement unit. The tasks differ between rows: the learned models "
      "of the Camargo data were evaluated over tasks that include stairs, which \\SoleID\\ does not attempt. "
      "$^{a}$Subjects shared between training and test; by the authors' account the error is \\num{59}\\,\\% "
      "lower than with held-out subjects. "
      "$^{b}$One value pooled over four outputs (hip flexion and adduction, knee, ankle) and five tasks. "
      "$^{c}$Four parameters selected on five subjects of another data set~\\cite{fukuchi2018} and frozen. "
      "$^{d}$Equal-weight mean of the three settings, for orientation only. "
      "$^{e}$The pipeline sets the horizontal force to zero. "
      "$^{f}$Windows drawn at random from the recordings of all twelve subjects; in the first row "
      "consecutive windows share all but one frame. Thirteen tasks, including running and jumping. "
      "$^{g}$Anteroposterior force over all frames.}")
    A("\\end{tabular}")
    A("\\end{table}")
    return "\n".join(L)


if __name__ == "__main__":
    p = os.path.join(R, "comparison_tables.md")
    open(p, "w", encoding="utf-8").write(md())
    open(TEX, "w", encoding="utf-8").write(tex())
    print("wrote", p)
    print("wrote", TEX)
    for name, fn in (("tables_summary.tex", tex_summary), ("tables_published.tex", tex_published)):
        q = os.path.join(os.path.dirname(TEX), name)
        open(q, "w", encoding="utf-8").write(fn())
        print("wrote", q)
    print()
    print("tables written; open the .md to read them")

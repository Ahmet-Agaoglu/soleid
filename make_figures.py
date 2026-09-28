"""Figures of the paper (Figs. 2-5; Fig. 1 is a drawing).

Writes PDF (vector) and PNG (for quick viewing) into config.FIGURES.
One categorical palette is used across every panel, so a reader who learns the colours in one
figure can read the others without consulting a legend again.
Sized for a two-column page: 3.5 in single column, 7.16 in double column.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import config

R = config.RESULTS
FIG = config.FIGURES

C_REF = "#2a78d6"     # measured / reference
C_SOLE = "#eb6834"    # SoleID
C_NOVPP = "#eda100"   # SoleID without the prior
C_NEWT = "#1baf7a"    # Newton + proportional split
C_KIN = "#e87ba4"     # kinematics-only / vertical-only
C_ALT = "#4a3aa7"     # secondary variant (laboratory kinematics, second dataset)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#ffffff"
SHADE = "#e9e7e1"

LAB = {
    "Kin_only": ("kinematics-only", C_KIN),
    "NoShear": ("vertical-only", C_KIN),
    "Newton_prop": ("Newton + proportional", C_NEWT),
    "SoleID_v1": ("SoleID, no prior", C_NOVPP),
    "SoleID_noVPP": ("SoleID, no prior", C_NOVPP),
    "SoleID": ("SoleID", C_SOLE),
    "SoleID_lab_kin": ("SoleID, lab kinematics", C_ALT),
}
PCT = "shear RMSE, stance (%BW)"
NMK = "hip moment RMSE (N m/kg)"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 6.8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.4, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
W1, W2 = 3.5, 7.16


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    print("  wrote", name)


def load():
    return {"test": pd.read_csv(os.path.join(R, "test_all_trials.csv")),
            "cam": pd.read_csv(os.path.join(R, "camargo_treadmill_strides.csv")),
            "ramp": pd.read_csv(os.path.join(R, "camargo_ramp_steps.csv")),
            "lg": pd.read_csv(os.path.join(R, "camargo_levelground_steps.csv")),
            "wang": pd.read_csv(os.path.join(R, "wang_application.csv")),
            "curves": np.load(os.path.join(R, "curves_fukuchi.npz"), allow_pickle=True),
            "stats": json.load(open(os.path.join(R, "statistics.json")))}


# --------------------------------------------------------------- Fig. 2: a representative test trial
def representative_trial(t):
    """The test trial shown in Fig. 2: comfortable speed (1.1-1.35 m/s), a saved time series, and
    shear and hip errors closest to the test-set medians. Calibration subjects are left out, since
    the figure stands for the test results."""
    from protocol import CAL_SUBJECTS
    c = t[t.speed.between(1.1, 1.35) & ~t.subject.isin(CAL_SUBJECTS)].copy()
    c = c[[os.path.exists(os.path.join(R, f"{s}{tr}_timeseries.csv")) for s, tr in zip(c.subject, c.trial)]]
    c["score"] = ((c.SoleID_stance - t.SoleID_stance.median()).abs() / t.SoleID_stance.std()
                  + (c.SoleID_hip - t.SoleID_hip.median()).abs() / t.SoleID_hip.std())
    r = c.sort_values("score").iloc[0]
    out = dict(subject=r.subject, trial=r.trial, speed=float(r.speed), mass=float(r.mass))
    json.dump(out, open(os.path.join(R, "fig2_trial.json"), "w"), indent=1)
    return out


def fig_timeseries(d):
    rep = representative_trial(d["test"])
    ts = pd.read_csv(os.path.join(R, f"{rep['subject']}{rep['trial']}_timeseries.csv"))
    on = ts.FyR.values > 20
    hs = np.where(np.diff(on.astype(int)) == 1)[0] + 1
    hs = hs[hs > 5 * 150]
    a, b = hs[0], hs[3]
    w = ts.iloc[a:b]
    t = w.t.values - w.t.values[0]
    BW = rep["mass"] * 9.81
    ds = (w.FyR.values > 20) & (w.FyL.values > 20)

    fig, axes = plt.subplots(4, 1, figsize=(W1, 5.4), sharex=True)
    ax = axes[0]
    style(ax)
    ax.fill_between(t, 0, 1, where=ds, transform=ax.get_xaxis_transform(), color=SHADE, lw=0)
    ax.plot(t, 100 * w.FxR_meas / BW, color=C_REF, lw=1.7, label="measured")
    ax.plot(t, 100 * w.FxR_soleid / BW, color=C_SOLE, label="SoleID")
    ax.plot(t, 100 * w.FxR_newton / BW, color=C_NEWT, ls="--", lw=1.0, label="Newton + prop.")
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_ylabel("shear force\n(%BW)")
    ax.legend(loc="upper center", ncol=3, frameon=False, columnspacing=0.9, handlelength=1.3,
              bbox_to_anchor=(0.5, 1.45))
    for ax, j, nm in zip(axes[1:], ("ank", "knee", "hip"), ("ankle", "knee", "hip")):
        style(ax)
        ax.fill_between(t, 0, 1, where=ds, transform=ax.get_xaxis_transform(), color=SHADE, lw=0)
        ax.plot(t, w[f"M{j}_ref"], color=C_REF, lw=1.7)
        ax.plot(t, w[f"M{j}_sole"], color=C_SOLE)
        ax.plot(t, w[f"M{j}_newton"], color=C_NEWT, ls="--", lw=1.0)
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_ylabel(f"{nm} moment\n(N m/kg)")
    axes[-1].set_xlabel("time (s)      shaded: double support")
    axes[-1].set_xlim(t[0], t[-1])
    fig.tight_layout()
    save(fig, "fig2_timeseries")


# --------------------------------------------------------------- Fig. 3: error against walking speed
def fig_speed(d):
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.6))
    ax = axes[0]
    style(ax)
    t = d["test"].copy()
    names = ["< 0.8", "0.8-1.2", "1.2-1.6", "> 1.6"]
    t["bin"] = pd.cut(t.speed, [0, 0.8, 1.2, 1.6, 3.0], labels=names)
    for a in ("Kin_only", "Newton_prop", "SoleID_v1", "SoleID"):
        g = t.groupby("bin", observed=True)[f"{a}_stance"].mean()
        ax.plot(range(len(g)), g.values, "o-", color=LAB[a][1], label=LAB[a][0], ms=4)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_xlabel("walking speed (m/s)")
    ax.set_ylabel(PCT)
    ax.set_ylim(0, None)
    ax.set_title("(a)  Controlled layer, by speed bin", loc="left")
    ax.legend(frameon=False, loc="upper left")

    # (b) the same dependence with real insoles and an IMU suit
    ax = axes[1]
    style(ax)
    w = d["wang"]
    for a in ("NoShear", "Newton_prop", "SoleID"):
        g = w.groupby("speed")[f"{a}_stance"].mean()
        # vertical-only shares its colour with kinematics-only in panel (a): dash it apart
        ax.plot(g.index, g.values, "o--" if a == "NoShear" else "o-", color=LAB[a][1], ms=3.8,
                label=LAB[a][0])
    ax.set_xlabel("treadmill speed (m/s)")
    ax.set_ylabel(PCT)
    ax.set_ylim(0, None)
    ax.set_title("(b)  Real insoles and IMUs, by treadmill speed", loc="left")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "fig3_speed")


# --------------------------------------------------------------- Fig. 5: ramp walking
def fig_external(d):
    """Stance shear error of each arm against the ramp inclination, ascent and descent together."""
    fig, ax = plt.subplots(1, 1, figsize=(W1, 2.7))
    style(ax)
    # inclinations as measured from the force plates by ramp_angles.py
    angles = {float(k): v for k, v in
              json.load(open(os.path.join(R, "ramp_angles.json")))["angles_deg"].items()}
    r = d["ramp"][d["ramp"].incline > 0]
    xs = [angles[i] for i in sorted(angles)]
    for arm in ("NoShear", "Newton_prop", "SoleID_noVPP", "SoleID"):
        g = r.groupby("incline")[f"{arm}_stance"].mean()
        # vertical-only dashed, as in every figure, since it shares its colour with kinematics-only
        ax.plot(xs, [g.loc[i] for i in sorted(angles)], "o--" if arm == "NoShear" else "o-",
                color=LAB[arm][1], ms=3.8, label=LAB[arm][0])
    ax.set_xlabel("ramp inclination (deg)")
    ax.set_ylabel(PCT)
    ax.set_ylim(0, 14)
    ax.legend(frameon=False, loc="lower right", ncol=2, columnspacing=0.8, handlelength=1.4,
              fontsize=5.8)
    fig.tight_layout()
    save(fig, "fig5_ramp")


# --------------------------------------------------------------- Fig. 4: curves over the cycle and where the prior helps
def fig_spm(d):
    z, st = d["curves"], d["stats"]
    subj = np.asarray(z["subjects"])
    uniq = np.unique(subj)

    def per_subject(key):
        a = np.asarray(z[key], float)
        return np.array([a[subj == s].mean(axis=0) for s in uniq])

    x = np.arange(101)
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.6))
    # vertical-only is dashed, as in the other figures, because it shares its colour with
    # kinematics-only; its shear is zero by construction, so in (a) it is drawn above the zero line
    # and labelled as such instead of disappearing under it
    spec = [((("reference_shear", C_REF, "measured"), ("SoleID_shear", C_SOLE, "SoleID"),
              ("NoShear_shear", C_KIN, "vertical-only (= 0)")), "shear force (%BW)",
             "(a)  Shear force", "SoleID_shear"),
            ((("reference_hip", C_REF, "reference"), ("SoleID_hip", C_SOLE, "SoleID"),
              ("NoShear_hip", C_KIN, "vertical-only")), "hip moment (N m/kg)",
             "(b)  Hip moment", "SoleID_hip")]
    for ax, (keys, ylab, ttl, spmkey) in zip(axes[:2], spec):
        style(ax)
        for a, b in st["spm"][spmkey]["clusters"]:
            ax.axvspan(a, b, color=C_SOLE, alpha=0.11, lw=0)
        ax.axhline(0, color=INK2, lw=0.6, zorder=1)
        for key, c, lab in keys:
            y = per_subject(key)
            m, s = y.mean(0), y.std(0)
            ax.plot(x, m, color=c, label=lab, ls="--" if key.startswith("NoShear") else "-",
                    zorder=3)
            ax.fill_between(x, m - s, m + s, color=c, alpha=0.16, lw=0)
        ax.set_xlim(0, 100)
        ax.set_xlabel("gait cycle (%)")
        ax.set_ylabel(ylab)
        ax.set_title(ttl, loc="left")
        ax.legend(frameon=False, loc="lower right")

    ax = axes[2]
    style(ax)
    t = d["test"]
    gain = (t.SoleID_v1_stance - t.SoleID_stance).values
    nc = t.newton_check.values
    ax.scatter(nc, gain, s=7, color=C_SOLE, alpha=0.5, lw=0, label="Fukuchi trials")
    for key, c, mk, lab in (("cam", C_NEWT, "s", "Camargo treadmill"),
                            ("lg", C_ALT, "^", "Camargo level ground"),
                            ("ramp", C_KIN, "v", "Camargo ramp")):
        g = d[key].groupby("subject")[["newton_check", "SoleID_noVPP_stance",
                                       "SoleID_stance"]].mean()
        ax.scatter(g.newton_check, g.SoleID_noVPP_stance - g.SoleID_stance, s=18, color=c,
                   marker=mk, lw=0, alpha=0.9, label=lab + ", subject means")
    sl, ic = np.polyfit(nc, gain, 1)
    xs = np.array([nc.min(), 13.0])
    ax.plot(xs, sl * xs + ic, color=INK2, lw=1.0, ls="--")
    s = st["gain_vs_newton_check"]
    ax.set_xlabel("whole-body dynamics residual (%BW)")
    ax.set_ylabel("benefit of the prior (%BW)")
    ax.set_title(f"(c)  Slope {s['slope']:+.2f} per %BW, r = {s['r']:.2f}", loc="left")
    ax.legend(frameon=False, loc="upper left", fontsize=6)
    fig.tight_layout()
    save(fig, "fig4_spm")


if __name__ == "__main__":
    data = load()
    os.makedirs(FIG, exist_ok=True)
    # Fig. 1 is a drawing (concept.png) and is not generated here
    for fn in (fig_timeseries, fig_speed, fig_spm, fig_external):
        fn(data)
    print("all figures written to", FIG)

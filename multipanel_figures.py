"""Multi-panel result figures: each figure answers one question and groups the panels that answer it.

    accuracy        how close the estimate is on the controlled layer
    mechanism       why the estimator works and where it matters
    robustness      how it degrades with wearable-grade inputs
    generalization  a second laboratory, overground walking, ramps, and the calibration draw
    hardware        real insoles and an IMU suit
    online          moving-horizon operation and latency

Every panel is drawn from files in results/; the numbers printed on the panels are the ones the
text reports. Writes PDF (vector) and PNG into config.FIGURES as mp_<name>.{pdf,png}.

    python multipanel_figures.py                 # all figures
    python multipanel_figures.py accuracy online # some of them
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

import config

R = config.RESULTS
FIG = config.FIGURES

# one colour per arm, the same in every figure
C = dict(ref="#2a78d6", sole="#eb6834", novpp="#eda100", newt="#1baf7a", kin="#e87ba4",
         split="#8e6bbf", lab="#4a3aa7", wang="#56606b")
TEXT_C = dict(C, novpp="#b98200")             # the amber is too light for text
INK, INK2, GRID, SHADE = "#1b1b1b", "#55534f", "#e9e7e2", "#f2efe8"
ORANGES = ["#fde3d0", "#f9c29f", "#f4a06d", "#ee7f45", "#eb6834"]
W2 = 7.09                                      # double-column width, inches

plt.rcParams.update({
    "font.family": "Arial", "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03, "legend.frameon": False,
})


def letter(ax, s, x=-0.12, y=1.04):
    """Panel label, upper case (A, B, C ...)."""
    ax.text(x, y, s.upper(), transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def soft_grid(ax, axis="y"):
    ax.grid(True, axis=axis, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"mp_{name}.{ext}"), dpi=600)
    plt.close(fig)
    print("  wrote", f"mp_{name}")


def read(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if name.endswith(".json") else pd.read_csv(p)


def per_subject_curves(z, key):
    s = np.asarray(z["subjects"])
    a = np.asarray(z[key], float)
    return np.array([a[s == u].mean(axis=0) for u in np.unique(s)])


# ================================================================ accuracy
def fig_accuracy():
    t, vs, st = read("test_all_trials.csv"), read("vpp_split_test.csv"), read("statistics.json")
    rep = read("fig2_trial.json")
    ts = pd.read_csv(os.path.join(R, f"{rep['subject']}{rep['trial']}_timeseries.csv"))
    z = np.load(os.path.join(R, "curves_fukuchi.npz"), allow_pickle=True)

    fig = plt.figure(figsize=(W2, 5.6))
    outer = GridSpec(2, 3, figure=fig, height_ratios=[1, 1.02], width_ratios=[1, 1, 1.05],
                     hspace=0.55, wspace=0.45)

    # (a) the input and the output on one trial
    sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, :2], height_ratios=[0.4, 1], hspace=0.1)
    axv, axs = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
    on = ts.FyR.values > 20
    hs = np.where(np.diff(on.astype(int)) == 1)[0] + 1
    hs = hs[hs > 5 * 150]
    w = ts.iloc[hs[0]:hs[3]]
    tt = w.t.values - w.t.values[0]
    BW = rep["mass"] * 9.81
    ds = (w.FyR.values > 20) & (w.FyL.values > 20)
    for ax in (axv, axs):
        ax.fill_between(tt, 0, 1, where=ds, transform=ax.get_xaxis_transform(), color=SHADE, lw=0)
        ax.set_xlim(tt[0], tt[-1])
    axv.fill_between(tt, 0, 100 * w.FyR / BW, color="#9fb7d9", alpha=0.55, lw=0)
    axv.plot(tt, 100 * w.FyR / BW, color="#4f76b3", lw=0.9)
    axv.set_ylabel("insole,\nvertical\n(%BW)", fontsize=6.3, linespacing=1.0)
    axv.set_ylim(0, 125)
    axv.set_yticks([0, 100])
    axv.tick_params(labelbottom=False)
    axs.axhline(0, color=INK2, lw=0.5)
    axs.plot(tt, 100 * w.FxR_meas / BW, color=C["ref"], lw=2.0, solid_capstyle="round")
    axs.plot(tt, 100 * w.FxR_newton / BW, color=C["newt"], lw=1.0, ls=(0, (4, 2)))
    axs.plot(tt, 100 * w.FxR_soleid / BW, color=C["sole"], lw=1.5)
    axs.set_ylabel("shear force (%BW)")
    axs.set_xlabel("time (s)   ·   shaded: double support")
    letter(axv, "a", x=-0.075, y=1.12)
    handles = [Line2D([], [], color=C["ref"], lw=2.0),
               Line2D([], [], color=C["newt"], lw=1.0, ls=(0, (4, 2))), Line2D([], [], color=C["sole"], lw=1.5)]
    axv.legend(handles, ["force plate (reference)", "Newton + prop.", "SoleID"], loc="lower left",
               bbox_to_anchor=(0.0, 1.0), ncol=3, fontsize=6.4, handlelength=2.2, columnspacing=1.4)

    # (b) raincloud of per-subject stance shear error
    ax = fig.add_subplot(outer[0, 2])
    subj = t.groupby("subject")
    series = [
        ("kinematics-only", subj.Kin_only_stance.mean(), t.Kin_only_stance.mean(), "kin"),
        ("Newton + prop.", subj.Newton_prop_stance.mean(), t.Newton_prop_stance.mean(), "newt"),
        ("pivot split", vs.groupby("subject").stance.mean(), vs.stance.mean(), "split"),
        ("SoleID, no prior", subj.SoleID_v1_stance.mean(), t.SoleID_v1_stance.mean(), "novpp"),
        ("SoleID", subj.SoleID_stance.mean(), t.SoleID_stance.mean(), "sole"),
    ]
    raincloud(ax, series, xmax=13.4, xticks=[2.5, 5, 7.5, 10])
    ax.set_xlabel("shear RMSE, stance (%BW)\n31 subjects; diamond: mean")
    letter(ax, "b", x=-0.52, y=1.02)

    # (c, d) gait-cycle curves with SPM tracks
    curves_with_spm(fig, outer[1, 0], z, st, "shear", "shear force (%BW)",
                    [("SoleID_shear", "sole"), ("Newton_prop_shear", "newt")], "c")
    curves_with_spm(fig, outer[1, 1], z, st, "hip", "hip moment (N m/kg)", [("SoleID_hip", "sole")], "d")

    # (e) arm x metric heatmap
    ax = fig.add_subplot(outer[1, 2])
    arms = [("SoleID", "SoleID"), ("SoleID_v1", "no prior"), ("split", "pivot split"),
            ("Newton_prop", "Newton"), ("Kin_only", "kin.-only")]
    mets = [("stance", "stance", "%.2f"), ("ds", "DS", "%.2f"), ("ss", "SS", "%.2f"),
            ("ankle", "ankle", "%.3f"), ("knee", "knee", "%.3f"), ("hip", "hip", "%.3f")]
    M = np.full((len(arms), len(mets)), np.nan)
    for i, (k, _) in enumerate(arms):
        for j, (m, _, _) in enumerate(mets):
            if k == "split":
                if m in vs:
                    M[i, j] = vs[m].mean()
            else:
                M[i, j] = t[f"{k}_{m}"].mean()
    ratio_heatmap(fig, ax, M, [a[1] for a in arms], mets, split_after=2,
                  groups=[(1, "shear (%BW)"), (4, "moment (N m/kg)")],
                  cbar_label="error relative to the best arm\nDS, SS: double, single support")
    letter(ax, "e", x=-0.3, y=1.12)
    save(fig, "accuracy")


def raincloud(ax, series, xmax, xticks, seed=3):
    """series: (label, per-subject values, mean to print, colour key), drawn top to bottom."""
    rng = np.random.default_rng(seed)
    for i, (lab, v, mean, key) in enumerate(series[::-1]):
        y, v, col = i, np.asarray(v, float), C[key]
        xs = np.linspace(v.min() - 0.3, v.max() + 0.3, 200)
        k = gaussian_kde(v)(xs)
        ax.fill_between(xs, y + 0.06, y + 0.06 + 0.36 * k / k.max(), color=col, alpha=0.75, lw=0)
        q1, q3 = np.percentile(v, [25, 75])
        ax.plot([v.min(), v.max()], [y - 0.02] * 2, color=INK, lw=0.6, alpha=0.7)
        ax.plot([q1, q3], [y - 0.02] * 2, color=INK, lw=2.4, solid_capstyle="butt", alpha=0.85)
        ax.plot(mean, y - 0.02, "D", ms=3.0, color="white", markeredgecolor=INK,
                markeredgewidth=0.7, zorder=5)
        ax.scatter(v, y - 0.2 + rng.uniform(-0.07, 0.07, len(v)), s=5, color=col, alpha=0.85, lw=0)
        ax.text(xmax, y + 0.02, f"{mean:.2f}", fontsize=6.3, va="center", ha="right",
                color=TEXT_C[key], fontweight="bold" if key == "sole" else "normal")
    ax.set_yticks(range(len(series)))
    ax.set_yticklabels([s[0] for s in series[::-1]])
    for lbl, s in zip(ax.get_yticklabels(), series[::-1]):
        lbl.set_color(TEXT_C[s[3]])
        lbl.set_fontweight("bold" if s[3] == "sole" else "normal")
    ax.set_xlim(min(float(np.min(s[1])) for s in series) - 0.9, xmax)
    ax.set_xticks(xticks)
    ax.set_ylim(-0.45, len(series) - 0.4)
    soft_grid(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def curves_with_spm(fig, spec, z, st, var, ylab, tracks, let):
    g = GridSpecFromSubplotSpec(2, 1, subplot_spec=spec, height_ratios=[1, 0.07 + 0.09 * len(tracks)],
                                hspace=0.06)
    ax, tr = fig.add_subplot(g[0]), fig.add_subplot(g[1])
    x = np.arange(101)
    for key, col, lw in ((f"reference_{var}", C["ref"], 1.8), (f"Newton_prop_{var}", C["newt"], 1.0),
                         (f"SoleID_{var}", C["sole"], 1.4)):
        y = per_subject_curves(z, key)
        m, s = y.mean(0), y.std(0)
        ax.fill_between(x, m - s, m + s, color=col, alpha=0.13, lw=0)
        ax.plot(x, m, color=col, lw=lw, ls=(0, (4, 2)) if "Newton" in key else "-")
    ax.axhline(0, color=INK2, lw=0.5)
    ax.set_xlim(0, 100)
    ax.set_ylabel(ylab)
    ax.tick_params(labelbottom=False)
    soft_grid(ax)
    letter(ax, let, x=-0.26, y=1.02)
    names = {"sole": "SoleID", "newt": "Newton"}
    for j, (name, key) in enumerate(tracks):
        yb = len(tracks) - 1 - j
        tr.plot([0, 100], [yb, yb], color=GRID, lw=4, solid_capstyle="butt")
        for c0, c1 in st["spm"][name]["clusters"]:
            tr.plot([c0, c1], [yb, yb], color=C[key], lw=4, solid_capstyle="butt")
    tr.set_xlim(0, 100)
    tr.set_ylim(-0.7, len(tracks) - 0.3)
    tr.set_yticks(range(len(tracks)))
    tr.set_yticklabels([f"{names[k]} ≠ ref." for _, k in tracks[::-1]], fontsize=5.8)
    for lbl, (_, k) in zip(tr.get_yticklabels(), tracks[::-1]):
        lbl.set_color(C[k])
    tr.tick_params(axis="y", length=0)
    tr.spines["left"].set_visible(False)
    tr.set_xlabel("gait cycle (%)")


def ratio_heatmap(fig, ax, M, rows, cols, split_after, groups, cbar_label, vmax=30):
    ratio = M / np.nanmin(M, axis=0)
    cmap = LinearSegmentedColormap.from_list("err", ["#fbf7f0", "#f6d7b5", "#e98b5a", "#b3452b", "#6b1f1a"])
    im = ax.imshow(np.log(ratio), cmap=cmap, vmin=0, vmax=np.log(vmax), aspect="auto")
    for i in range(M.shape[0]):
        for j, (_, _, fmt) in enumerate(cols):
            if np.isnan(M[i, j]):
                ax.text(j, i, "–", ha="center", va="center", fontsize=6, color=INK2)
                continue
            dark = np.log(ratio[i, j]) > np.log(vmax) * 0.55
            ax.text(j, i, fmt % M[i, j], ha="center", va="center", fontsize=5.6,
                    color="white" if dark else INK, fontweight="bold" if ratio[i, j] == 1 else "normal")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[1] for c in cols], fontsize=6.3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=6.3)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    ax.grid(which="minor", color="white", lw=1.2)
    ax.tick_params(which="minor", length=0)
    if split_after is not None:
        ax.axvline(split_after + 0.5, color="white", lw=2.5)
    for x, lab in groups:
        ax.text(x, -0.9, lab, ha="center", fontsize=6.3, color=INK2)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.06, pad=0.14, aspect=30)
    ticks = [v for v in (1, 2, 5, 10, 30) if v <= vmax]
    cb.set_ticks(np.log(ticks))
    cb.set_ticklabels([f"{v}×" for v in ticks])
    cb.ax.tick_params(labelsize=6, length=2)
    cb.set_label(cbar_label, fontsize=6, linespacing=1.1)
    cb.outline.set_visible(False)
    return im


# ================================================================ mechanism
def fig_mechanism():
    t, ab = read("test_all_trials.csv"), read("ablation_qp.json")
    st, sr, hb = read("statistics.json"), read("stats_revision.json"), read("hip_bias_breakdown.json")
    fig = plt.figure(figsize=(W2, 5.5))
    outer = GridSpec(2, 3, figure=fig, height_ratios=[1, 1], width_ratios=[1.05, 1.05, 1.0],
                     hspace=0.55, wspace=0.5)

    # (a) the ingredient ladder: what each part of the program contributes
    rows = [  # label, stance shear, hip, colour
        ("dynamics only (Newton + prop.)", t.Newton_prop_stance.mean(), t.Newton_prop_hip.mean(), C["newt"]),
        ("dynamics + smoothness + cone, no prior", t.SoleID_v1_stance.mean(), t.SoleID_v1_hip.mean(), C["novpp"]),
        ("prior alone", ab["prior"]["stance"], ab["prior"]["hip"], ORANGES[1]),
        ("prior alone, smoothed", ab["prior_smooth"]["stance"], ab["prior_smooth"]["hip"], ORANGES[1]),
        ("dynamics + prior, frame by frame", ab["blend"]["stance"], ab["blend"]["hip"], ORANGES[2]),
        ("+ temporal smoothness", ab["no_cone"]["stance"], ab["no_cone"]["hip"], ORANGES[3]),
        ("+ friction cone = SoleID", ab["full"]["stance"], ab["full"]["hip"], C["sole"]),
    ]
    sub = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0, :2], wspace=0.08, width_ratios=[1, 1])
    for k, (idx, xlab, fmt, xmax) in enumerate(((1, "shear RMSE, stance (%BW)", "%.2f", 7.4),
                                                  (2, "hip moment RMSE (N m/kg)", "%.3f", 0.47))):
        ax = fig.add_subplot(sub[k])
        y = np.arange(len(rows))[::-1]
        for yi, r in zip(y, rows):
            v = r[idx]
            ax.plot([0, v], [yi, yi], color=r[3], lw=3.2, solid_capstyle="round", alpha=0.95)
            ax.plot(v, yi, "o", ms=5.2, color=r[3], markeredgecolor="white", markeredgewidth=0.8)
            ax.text(v + xmax * 0.025, yi, fmt % v, va="center", fontsize=6.2,
                    color=INK if r[3] != C["sole"] else C["sole"],
                    fontweight="bold" if r[3] == C["sole"] else "normal")
        ax.set_xlim(0, xmax)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows] if k == 0 else [])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        soft_grid(ax, "x")
        ax.set_xlabel(xlab)
        if k == 0:
            letter(ax, "a", x=-0.98, y=1.02)
    # (b) the benefit of the prior grows with the dynamics residual
    g = GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[0, 2], width_ratios=[1, 0.22],
                                height_ratios=[0.22, 1], wspace=0.05, hspace=0.05)
    ax = fig.add_subplot(g[1, 0])
    axt, axr = fig.add_subplot(g[0, 0], sharex=ax), fig.add_subplot(g[1, 1], sharey=ax)
    x = t.newton_check.values
    yv = (t.SoleID_v1_stance - t.SoleID_stance).values
    ax.scatter(x, yv, s=7, color=C["sole"], alpha=0.45, lw=0)
    s = st["gain_vs_newton_check"]
    # the line drawn is the mixed-model slope the text reports, through the centroid of the trials
    xx = np.array([x.min(), x.max()])
    ax.plot(xx, yv.mean() + s["slope"] * (xx - x.mean()), color=INK, lw=1.0)
    for f, mk, lab in (("camargo_treadmill_strides.csv", "s", "treadmill"),
                       ("camargo_levelground_steps.csv", "^", "level ground"),
                       ("camargo_ramp_steps.csv", "D", "ramp")):
        cm = read(f).groupby("subject")[["newton_check", "SoleID_noVPP_stance", "SoleID_stance"]].mean()
        ax.scatter(cm.newton_check, cm.SoleID_noVPP_stance - cm.SoleID_stance, s=11, marker=mk,
                   color="#3b3a36", alpha=0.85, lw=0, label=f"Camargo {lab}")
    ax.text(0.04, 0.96, f"slope {s['slope']:.2f} ({s['lo']:.2f}–{s['hi']:.2f})\nr = {s['r']:.2f}, 231 trials",
            transform=ax.transAxes, va="top", fontsize=6.1)
    ax.set_ylim(-9, None)
    ax.legend(loc="lower right", fontsize=5.4, handletextpad=0.1, labelspacing=0.25, markerscale=0.9,
              borderaxespad=0.2, title="subject means", title_fontsize=5.4)
    ax.axhline(0, color=INK2, lw=0.5)
    ax.set_xlabel("whole-body dynamics residual (%BW)")
    ax.set_ylabel("benefit of the prior (%BW)")
    soft_grid(ax, "both")
    axt.hist(x, bins=28, color="#d9d6cf", lw=0)
    axr.hist(yv, bins=28, orientation="horizontal", color=ORANGES[1], lw=0)
    for a_ in (axt, axr):
        a_.axis("off")
    letter(axt, "b", x=-0.3, y=0.9)

    # (c) the advantage grows with speed
    ax = fig.add_subplot(outer[1, :2])
    names = ["< 0.8", "0.8–1.2", "1.2–1.6", "> 1.6"]
    t = t.assign(bin=pd.cut(t.speed, [0, 0.8, 1.2, 1.6, 3.0], labels=names))
    g_ = t.groupby("bin", observed=True)
    arms = [("Newton_prop", "Newton + prop.", "newt"), ("SoleID_v1", "SoleID, no prior", "novpp"),
            ("SoleID", "SoleID", "sole")]
    for i, nm in enumerate(names):
        vals = {a: g_[f"{a}_stance"].mean()[nm] for a, _, _ in arms}
        ax.plot([vals["SoleID"], vals["Newton_prop"]], [i, i], color="#cfcac0", lw=5, solid_capstyle="round",
                zorder=1)
        for a, lab, key in arms:
            ax.plot(vals[a], i, "o", ms=7 if a == "SoleID" else 6, color=C[key], markeredgecolor="white",
                    markeredgewidth=0.8, zorder=3, label=lab if i == 0 else None)
        gap = vals["Newton_prop"] - vals["SoleID"]
        ax.text(vals["Newton_prop"] + 0.25, i, f"gap {gap:.2f}", va="center", fontsize=6.1, color=INK2)
        n = int((t.bin == nm).sum())
        ax.text(0.2, i, f"n = {n}", va="center", fontsize=5.8, color=INK2)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{n} m/s" for n in names])
    ax.set_xlim(0, 11.8)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_xlabel("shear RMSE, stance (%BW)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    soft_grid(ax, "x")
    ab_ = sr["arm_by_speed"]
    ax.text(0.99, 0.02, f"Newton's excess error grows by {ab_['interaction']:.2f} %BW per m/s "
            f"({ab_['lo']:.2f}–{ab_['hi']:.2f})", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.1, color=INK2)
    ax.legend(loc="upper right", fontsize=6.2, ncol=3, bbox_to_anchor=(1.0, 1.12), handletextpad=0.2,
              columnspacing=1.0)
    letter(ax, "c", x=-0.1, y=1.04)

    # (d) the hip offset over the gait cycle, drawn as a clock
    ax = fig.add_subplot(outer[1, 2], projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    sf = 0.6                                                     # stance ~60 % of the cycle
    sectors = [("early", 0, sf / 3), ("mid", sf / 3, 2 * sf / 3), ("late", 2 * sf / 3, sf), ("swing", sf, 1.0)]
    for name, a0, a1 in sectors:
        th0, th1 = 2 * np.pi * a0, 2 * np.pi * a1
        width = (th1 - th0) / 2 * 0.86
        for j, (limb, alpha) in enumerate((("R", 1.0), ("L", 0.55))):
            v = hb["phase"][name][limb]
            ax.bar(th0 + (th1 - th0) * (0.27 + 0.46 * j), v, width=width, color=C["sole"], alpha=alpha,
                   edgecolor="white", lw=0.6)
        ax.text((th0 + th1) / 2, 0.33, {"early": "early\nstance", "mid": "mid-\nstance",
                                         "late": "late\nstance", "swing": "swing"}[name],
                ha="center", va="center", fontsize=6.0, color=INK2)
    ax.set_ylim(0, 0.3)
    ax.set_yticks([0.1, 0.2])
    ax.set_yticklabels(["0.1", "0.2"], fontsize=5.6, color=INK2)
    ax.set_rlabel_position(290)                                  # in the swing sector, where bars are ~0
    ax.set_xticks(2 * np.pi * np.array([0, sf / 3, 2 * sf / 3, sf]))
    ax.set_xticklabels([])
    ax.grid(color=GRID, lw=0.6)
    ax.spines["polar"].set_color("#cfcac0")
    ax.set_title("hip offset (N m/kg) over the gait cycle\ndark: right limb, light: left",
                 fontsize=6.3, color=INK2, pad=10)
    letter(ax, "d", x=-0.18, y=1.08)
    save(fig, "mechanism")


# ================================================================ robustness
FRAME_MS = 1000 / 150
PERTURBATIONS = [  # condition, group label, level -> row label, tolerance-side label for the spec strip
    ("marker_noise_mm", "key-point noise", lambda v: f"{v:g} mm", "mm RMS"),
    ("kin_bias_mm", "key-point bias", lambda v: f"{v:g} mm", "mm RMS"),
    ("marker_fps", "frame rate", lambda v: f"{v:g} Hz", "Hz"),
    ("fy_scale_pct", "vertical-force scale", lambda v: f"{v:+g} %", "%"),
    ("fy_noise_pctBW", "vertical-force noise", lambda v: f"{v:g} %BW", "%BW"),
    ("cop_shift_mm", "CoP shift", lambda v: f"{v:g} mm", "mm"),
    ("sync_frames", "force–kinematics offset", lambda v: f"{v * FRAME_MS:+.0f} ms", "ms"),
]


def fig_robustness():
    from matplotlib.colors import TwoSlopeNorm
    d, w = read("final_degradation_trials.csv"), read("wang_application.csv")
    mean = d.groupby(["condition", "level", "arm"])[["stance", "hip"]].mean()
    # the arms that use the measured vertical force; the kinematics-only baseline replaces every
    # force input, so it is left out of this comparison
    arms = [("SoleID", "SoleID"), ("SoleID_noVPP", "no prior"), ("Newton_prop", "Newton")]
    clean = {a: mean.loc[("clean", 0)].loc[a] for a, _ in arms} if ("clean", 0) in mean.index.droplevel(2) \
        else {a: mean.xs("clean", level=0).xs(a, level=1).iloc[0] for a, _ in arms}
    tol = {"stance": clean["SoleID"]["stance"] + 1.0, "hip": clean["SoleID"]["hip"] + 0.10}

    rows, groups = [("clean", None, "clean")], []
    for cond, glab, fmt, _ in PERTURBATIONS:
        lv = sorted(d.loc[d.condition == cond, "level"].unique(), key=float)
        if cond == "marker_fps":
            lv = sorted(lv, key=float, reverse=True)
        groups.append((glab, len(rows), len(rows) + len(lv) - 1))
        rows += [(cond, float(v), fmt(float(v))) for v in lv]
    rows.append(("wearable", None, "all at once"))

    def value(cond, lv, arm, col):
        sub = mean.xs(cond, level=0)
        sub = sub.xs(arm, level=1)
        return float(sub[col].iloc[0] if lv is None else sub.loc[lv, col])

    fig = plt.figure(figsize=(W2, 6.3))
    outer = GridSpec(2, 2, figure=fig, width_ratios=[1.35, 1], height_ratios=[1.3, 0.7],
                     wspace=0.32, hspace=0.3)
    heat = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[:, 0], wspace=0.06)
    cmap = LinearSegmentedColormap.from_list("tol", ["#2f64a8", "#9dbde3", "#f7f6f2", "#f2a38a", "#b8322a"])
    for k, (col, lab, fmt, lo, hi) in enumerate((("stance", "shear RMSE (%BW)", "%.2f", 2.5, 12.5),
                                                 ("hip", "hip moment RMSE (N m/kg)", "%.3f", 0.15, 0.82))):
        ax = fig.add_subplot(heat[k])
        M = np.array([[value(c, lv, a, col) for a, _ in arms] for c, lv, _ in rows])
        norm = TwoSlopeNorm(vcenter=tol[col], vmin=lo, vmax=hi)
        im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                strong = abs(norm(M[i, j]) - 0.5) > 0.36
                ax.text(j, i, fmt % M[i, j], ha="center", va="center", fontsize=5.2,
                        color="white" if strong else INK, fontweight="bold" if j == 0 else "normal")
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([a[1] for a in arms], fontsize=6.2, rotation=0)
        ax.xaxis.tick_top()
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[2] for r in rows] if k == 0 else [], fontsize=5.8)
        for _, a, b in groups:
            ax.axhline(a - 0.5, color="white", lw=2.2)
        ax.axhline(len(rows) - 1.5, color="white", lw=2.2)
        ax.set_xticks(np.arange(-0.5, len(arms)), minor=True)
        ax.grid(which="minor", axis="x", color="white", lw=1.0)
        ax.tick_params(which="minor", length=0)
        if k == 0:
            for glab, a, b in groups:
                ax.text(-1.95, (a + b) / 2, glab, ha="right", va="center", fontsize=6.0, color=INK,
                        fontweight="bold")
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.03, pad=0.02, aspect=22)
        cb.ax.tick_params(labelsize=5.6, length=2)
        cb.set_label(f"{lab}\nwhite = tolerance ({fmt % tol[col]})", fontsize=6, linespacing=1.1)
        cb.outline.set_visible(False)
    letter(fig.axes[0], "a", x=-1.45, y=1.03)

    # (b) the specification: which tested levels keep SoleID within tolerance on both measures
    ax = fig.add_subplot(outer[0, 1])
    y = np.arange(len(PERTURBATIONS))[::-1]
    for yi, (cond, glab, fmt, unit) in zip(y, PERTURBATIONS):
        lv = sorted(d.loc[d.condition == cond, "level"].unique(), key=float)
        if cond == "marker_fps":
            lv = sorted(lv, key=float, reverse=True)
        xs = np.linspace(0.05, 0.95, len(lv))
        ax.plot([0.02, 0.98], [yi, yi], color=GRID, lw=5, solid_capstyle="round", zorder=1)
        for x_, v in zip(xs, lv):
            ok = (value(cond, float(v), "SoleID", "stance") <= tol["stance"]
                  and value(cond, float(v), "SoleID", "hip") <= tol["hip"])
            ax.plot(x_, yi, "o", ms=6.0, zorder=3, color=C["sole"] if ok else "white",
                    markeredgecolor=C["sole"] if ok else "#b8322a", markeredgewidth=1.2)
            ax.text(x_, yi - 0.2, fmt(float(v)), ha="center", va="top", fontsize=5.2, color=INK2)
        ax.text(0.02, yi + 0.2, glab, ha="left", va="bottom", fontsize=6.0, fontweight="bold")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.8, len(PERTURBATIONS) - 0.2)
    ax.axis("off")
    ax.set_title("SoleID at each tested level: filled = within tolerance on both measures",
                 fontsize=6.1, color=INK2, loc="left", pad=6)
    letter(ax, "b", x=-0.08, y=1.02)

    # (c) all at once, synthetic, against real sensors on another data set
    sub = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1, 1], wspace=0.45)
    for k, (col, lab, fmt) in enumerate((("stance", "shear (%BW)", "%.2f"), ("hip", "hip (N m/kg)", "%.3f"))):
        ax = fig.add_subplot(sub[k])
        for arm, key in (("Newton_prop", "newt"), ("SoleID", "sole")):
            v = [value("clean", None, arm, col), value("wearable", None, arm, col), float(w[f"{arm}_{col}"].mean())]
            ax.plot([0, 1], v[:2], color=C[key], lw=1.6, zorder=2)
            ax.plot([1, 2], v[1:], color=C[key], lw=1.0, ls=(0, (2, 2)), zorder=2)
            ax.plot(range(3), v, "o", ms=5.5, color=C[key], markeredgecolor="white", markeredgewidth=0.8,
                    zorder=3)
            if key == "sole":
                for xi, vi in enumerate(v):
                    ax.annotate(fmt % vi, (xi, vi), xytext=(0, -10), textcoords="offset points",
                                ha="center", fontsize=5.6, color=C["sole"], zorder=4,
                                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))
        ax.set_xticks(range(3))
        ax.set_xticklabels(["clean", "combined", "real*"], fontsize=5.8, rotation=25, ha="right")
        ax.set_xlim(-0.6, 2.6)
        ax.set_ylabel(lab)
        ax.set_ylim(0, None)
        soft_grid(ax)
        if k == 0:
            letter(ax, "c", x=-0.55, y=1.02)
    fig.text(0.99, 0.015, "*real insoles and IMU suit; other subjects and laboratory",
             ha="right", fontsize=5.6, color=INK2)
    save(fig, "robustness")


# ================================================================ generalization
def split_violin(ax, x, left, right, width=0.42, cl=C["newt"], cr=C["sole"], seed=0):
    """Left half: one arm, right half: another; distributions of per-step errors at position x."""
    rng = np.random.default_rng(seed)
    for vals, sign, col in ((left, -1, cl), (right, 1, cr)):
        v = np.asarray(vals, float)
        v = v[np.isfinite(v)]
        if len(v) > 3000:
            v = rng.choice(v, 3000, replace=False)
        hi = np.percentile(v, 99)
        ys = np.linspace(max(v.min(), 0), hi, 160)
        k = gaussian_kde(v)(ys)
        ax.fill_betweenx(ys, x, x + sign * width * k / k.max(), color=col, alpha=0.8, lw=0)
        ax.plot([x, x + sign * width * 0.55], [np.mean(vals)] * 2, color=INK, lw=1.0)


def fig_generalization():
    tm, lg, rp = read("camargo_treadmill_strides.csv"), read("camargo_levelground_steps.csv"), read("camargo_ramp_steps.csv")
    ang = {float(k): v for k, v in read("ramp_angles.json")["angles_deg"].items()}
    fig = plt.figure(figsize=(W2, 7.0))
    outer = GridSpec(3, 2, figure=fig, height_ratios=[0.62, 1, 1], width_ratios=[1, 1.35],
                     hspace=0.62, wspace=0.3)

    # (a) treadmill: error over the whole speed range, arm by arm
    ax = fig.add_subplot(outer[0, :])
    edges = np.round(np.arange(0.5, 2.15, 0.1), 2)
    tm = tm.assign(vbin=pd.cut(tm.speed, edges, include_lowest=True))
    arms = [("SoleID", "SoleID"), ("SoleID_noVPP", "no prior"), ("Newton_prop", "Newton"),
            ("Kin_only", "kin.-only")]
    M = np.array([tm.groupby("vbin", observed=False)[f"{a}_stance"].mean().values for a, _ in arms])
    cmap = LinearSegmentedColormap.from_list("mag", ["#fbf7f0", "#f6d7b5", "#e98b5a", "#b3452b", "#6b1f1a"])
    im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=2, vmax=16)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=5.0,
                        color="white" if M[i, j] > 10 else INK)
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([a[1] for a in arms], fontsize=6.3)
    ax.set_xticks(np.arange(len(edges) - 1))
    ax.set_xticklabels([f"{(a + b) / 2:.2f}" for a, b in zip(edges[:-1], edges[1:])], fontsize=5.6)
    ax.set_xlabel("treadmill speed (m/s)")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, M.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, M.shape[0]), minor=True)
    ax.grid(which="minor", color="white", lw=1.0)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    cb.ax.tick_params(labelsize=5.6, length=2)
    cb.set_label("shear RMSE,\nstance (%BW)", fontsize=6)
    cb.outline.set_visible(False)
    ax.set_title("Camargo treadmill, 22 subjects, 19 378 strides", fontsize=6.4, color=INK2, loc="left")
    letter(ax, "a", x=-0.07, y=1.05)

    # (b) level ground by speed class, (c) ramps by measured inclination: Newton left, SoleID right
    ax = fig.add_subplot(outer[1, 0])
    for i, sc in enumerate(["slow", "normal", "fast"]):
        s = lg[lg.speed_class == sc]
        split_violin(ax, i, s.Newton_prop_stance, s.SoleID_stance, seed=i)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["slow", "normal", "fast"])
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylabel("shear RMSE, stance (%BW)")
    ax.set_xlabel("level ground, self-selected speed")
    soft_grid(ax)
    ax.legend([Line2D([], [], color=C["newt"], lw=5), Line2D([], [], color=C["sole"], lw=5)],
              ["Newton + prop.", "SoleID"], loc="upper left", fontsize=6, ncol=2, handlelength=1.2)
    letter(ax, "b", x=-0.2, y=1.03)
    ax = fig.add_subplot(outer[1, 1])
    incl = sorted(k for k in rp.incline.unique() if k > 0)
    for i, k in enumerate(incl):
        s = rp[rp.incline == k]
        split_violin(ax, i, s.Newton_prop_stance, s.SoleID_stance, seed=10 + i)
    ax.set_xticks(range(len(incl)))
    ax.set_xticklabels([f"{ang[k]:.1f}°" for k in incl])
    ax.set_xlim(-0.6, len(incl) - 0.4)
    ax.set_xlabel("ramp inclination, measured from the plates (ascent and descent pooled)")
    soft_grid(ax)
    letter(ax, "c", x=-0.12, y=1.03)

    # (d) the three settings at a glance (the numbers of the text)
    ax = fig.add_subplot(outer[2, 0])
    settings = [("treadmill", tm, "Kin_only"), ("level", lg, "NoShear"), ("ramp", rp, "NoShear")]
    cols, M = [], []
    for met, fmt in (("stance", "%.2f"), ("hip", "%.3f")):
        for lab, df, first in settings:
            cols.append((f"{lab}_{met}", lab, fmt))
            M.append([df[f"{first}_{met}"].mean(), df[f"Newton_prop_{met}"].mean(),
                      df[f"SoleID_noVPP_{met}"].mean(), df[f"SoleID_{met}"].mean()])
    M = np.array(M).T[::-1]
    ratio_heatmap(fig, ax, M, ["SoleID", "no prior", "Newton", "vert./kin.-only"], cols, split_after=2,
                  groups=[(1, "shear (%BW)"), (4, "hip (N m/kg)")],
                  cbar_label="error relative to the best arm", vmax=10)
    letter(ax, "d", x=-0.42, y=1.14)

    # (e) the calibration draw, carried to all three data sets
    sp, pw, pc = read("split_sensitivity.csv"), read("split_propagated_wang.csv"), read("split_propagated_camargo.csv")
    K = ["h_vpp", "w_prior", "w_torque", "fc_smooth"]
    wt = sp.groupby(K).size()
    wt = wt / wt.sum()
    frozen = (0.2, 1.0, 0.0, 12.0)
    g = pc.assign(stance=pc.stance * pc.n_strides, hip=pc.hip * pc.n_strides).groupby(K + ["subject"])
    cam = (g[["stance", "hip"]].sum().div(g.n_strides.sum(), axis=0)).groupby(K).mean()
    wan = pw.groupby(K + ["subject"])[["SoleID_stance", "SoleID_hip"]].mean().groupby(K).mean()
    wan.columns = ["stance", "hip"]
    sub = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[2, 1], wspace=0.25)
    for k, (met, lab, fmt, bw) in enumerate((("stance", "shear RMSE (%BW)", "%.2f", 0.25),
                                             ("hip", "hip RMSE (N m/kg)", "%.3f", 0.25))):
        ax = fig.add_subplot(sub[k])
        rows = [("Fukuchi", sp[f"test_{met}"].values, None,
                 float(sp[(sp[K] == frozen).all(axis=1)][f"test_{met}"].mean())),
                ("Camargo", cam[met].reindex(wt.index).values, wt.values, float(cam.loc[frozen, met])),
                ("Wang", wan[met].reindex(wt.index).values, wt.values, float(wan.loc[frozen, met]))]
        for i, (name, v, wts, fr) in enumerate(rows[::-1]):
            v = np.asarray(v, float)
            lo, hi = v.min(), v.max()
            pad = (hi - lo) * 0.35 + 1e-9
            xs = np.linspace(lo - pad, hi + pad, 200)
            kd = gaussian_kde(v, weights=wts, bw_method=bw)(xs)
            ax.fill_between(xs, i, i + 0.8 * kd / kd.max(), color=ORANGES[2], alpha=0.7, lw=0)
            ax.plot(xs, i + 0.8 * kd / kd.max(), color=C["sole"], lw=0.8)
            ax.plot(fr, i + 0.05, marker="*", ms=8, color=INK, markeredgecolor="white", markeredgewidth=0.5)
        ax.set_yticks(np.arange(3) + 0.3)
        ax.set_yticklabels([r[0] for r in rows[::-1]] if k == 0 else [], fontsize=6.3)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel(lab)
        soft_grid(ax, "x")
        if k == 0:
            letter(ax, "e", x=-0.45, y=1.03)
            ax.set_title("500 draws of five calibration subjects", fontsize=6.1, color=INK2, loc="left")
        else:
            ax.legend([Line2D([], [], marker="*", ms=7, color=INK, lw=0)], ["frozen choice"], loc="lower right",
                      bbox_to_anchor=(1.0, 1.0), fontsize=5.8, handletextpad=0.2, borderaxespad=0.1)
    save(fig, "generalization")


# ================================================================ hardware
def fig_hardware():
    w, wf = read("wang_application.csv"), read("wang_factorial.csv")
    wf = wf[wf.lab_ok]
    ra = read("reference_agreement.json")["wang"]["wangwear_vs_planar"]
    arms = [  # key, label, colour key
        ("wang", "Wang et al.", "wang"), ("NoShear", "vertical-only", "kin"),
        ("Newton_prop", "Newton + prop.", "newt"), ("SoleID", "SoleID (IMU)", "sole"),
        ("SoleID_lab_kin", "SoleID (lab kin.)", "lab")]
    joints = ["ankle", "knee", "hip"]

    def err(arm, j):
        return ra[j]["rmse"] if arm == "wang" else float(w[f"{arm}_{j}"].mean())

    def corr(arm, j):
        return ra[j]["r"] if arm == "wang" else float(w[f"{arm}_{j}_r"].mean())

    fig = plt.figure(figsize=(W2, 5.3))
    # reading order: (a) one trial, (b) error, (c) correlation on top; (d) the 2 x 2, (e) speed below
    outer = GridSpec(2, 3, figure=fig, height_ratios=[1, 0.8], width_ratios=[1.25, 0.85, 1.0],
                     hspace=0.42, wspace=0.95)

    # (a) one trial: knee and hip moments, reference against the two wearable pipelines
    target = w.SoleID_hip.median()
    pick = w.iloc[(w.SoleID_hip - target).abs().argsort().values[0]]
    fs = 100.0
    n0 = int(20 * fs)
    sl = slice(n0, n0 + int(3.0 * fs))
    # the three seconds shown are kept in a small file, so the figure can be redrawn without the large
    # curve file of reference_agreement.py (not committed)
    npy, ex = os.path.join(R, "reference_agreement_wang_curves.npy"), os.path.join(R, "wang_example_trial.csv")
    if os.path.exists(npy):
        curves = np.load(npy, allow_pickle=True)
        rec = next(c for c in curves if c["subject"] == pick.subject and c["trial"] == pick.trial)
        seg = pd.DataFrame({f"{j}_{s}": rec[f"{j}_{s}"][sl] for j in ("knee", "hip")
                            for s in ("planar", "wangwear", "soleid")})
        seg.to_csv(ex, index=False)
    else:
        seg = pd.read_csv(ex)
    tt = np.arange(len(seg)) / fs
    sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, 0], hspace=0.12)
    for k, j in enumerate(("knee", "hip")):
        ax = fig.add_subplot(sub[k])
        ax.plot(tt, seg[f"{j}_planar"], color=C["ref"], lw=1.9, label="reference")
        ax.plot(tt, seg[f"{j}_wangwear"], color=C["wang"], lw=1.0, ls=(0, (4, 2)), label="Wang et al.")
        ax.plot(tt, seg[f"{j}_soleid"], color=C["sole"], lw=1.4, label="SoleID")
        ax.axhline(0, color=INK2, lw=0.5)
        ax.set_ylabel(f"{j}\n(N m/kg)", fontsize=6.5)
        ax.set_xlim(0, tt[-1])
        soft_grid(ax)
        if k == 0:
            ax.tick_params(labelbottom=False)
            ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=6.1, handlelength=2.0,
                      columnspacing=1.2)
            letter(ax, "a", x=-0.2, y=1.2)
        else:
            ax.set_xlabel(f"time (s)  ·  {pick.subject}, {pick.speed:g} m/s")

    # (b) radar of joint-moment error
    ax = fig.add_subplot(outer[0, 1], projection="polar")
    ax.set_theta_offset(np.pi / 2)                               # first axis at the top
    ax.set_theta_direction(-1)
    ang = np.linspace(0, 2 * np.pi, len(joints), endpoint=False)
    top = {j: max(err(a, j) for a, _, _ in arms) * 1.08 for j in joints}
    for a, lab, key in arms:
        v = np.array([err(a, j) / top[j] for j in joints])
        vv, aa = np.append(v, v[0]), np.append(ang, ang[0])
        ls = (0, (3, 2)) if a == "SoleID_lab_kin" else "-"
        ax.plot(aa, vv, color=C[key], lw=1.6 if a == "SoleID" else 1.0, ls=ls)
        ax.fill(aa, vv, color=C[key], alpha=0.12 if a == "SoleID" else 0.04)
    ax.set_xticks(ang)
    ax.set_xticklabels([f"{j}\n{top[j] / 1.08:.2f}" for j in joints], fontsize=6.0)
    ax.tick_params(axis="x", pad=1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([])
    ax.set_ylim(0, 1.0)
    ax.grid(color=GRID, lw=0.6)
    ax.spines["polar"].set_color("#cfcac0")
    ax.set_title("moment RMSE; axis end = worst arm", fontsize=6.2, color=INK2, pad=16)
    letter(ax, "b", x=-0.25, y=1.14)

    # (e) shear error against treadmill speed
    ax = fig.add_subplot(outer[1, 2])
    nudge = {"SoleID": 0.45, "SoleID_lab_kin": -0.45}
    for a, lab, key in arms[1:]:
        g = w.groupby("speed")[f"{a}_stance"].mean()
        ax.plot(g.index, g.values, marker="o", color=C[key], lw=1.4 if a == "SoleID" else 1.0, ms=4.2,
                ls=(0, (3, 2)) if a == "SoleID_lab_kin" else "-", markeredgecolor="white", markeredgewidth=0.6)
        ax.text(g.index[-1] + 0.06, g.values[-1] + nudge.get(a, 0), f"{g.values[-1]:.1f}", fontsize=5.8,
                va="center", color=TEXT_C[key])
    ax.set_xlabel("treadmill speed (m/s)")
    ax.set_ylabel("shear RMSE, stance (%BW)")
    ax.set_xticks([0.5, 1.0, 1.5])
    ax.set_xticks(sorted(w.speed.unique()), minor=True)
    ax.set_xlim(0.35, 1.8)
    ax.set_ylim(0, None)
    soft_grid(ax)
    letter(ax, "e", x=-0.35, y=1.06)

    # (c) correlation with the reference, arm x joint
    ax = fig.add_subplot(outer[0, 2])
    Rm = np.array([[corr(a, j) for j in joints] for a, _, _ in arms])
    cmap = LinearSegmentedColormap.from_list("r", ["#f7f6f2", "#bcd3ec", "#5b8fcf", "#1f4e8c"])
    im = ax.imshow(Rm, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(Rm.shape[0]):
        for jj in range(Rm.shape[1]):
            ax.text(jj, i, f"{Rm[i, jj]:.2f}", ha="center", va="center", fontsize=6.2,
                    color="white" if Rm[i, jj] > 0.7 else INK)
    ax.set_xticks(range(len(joints)))
    ax.set_xticklabels(joints, fontsize=6.3)
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([lab for _, lab, _ in arms], fontsize=6.3)
    for lbl, (_, _, key) in zip(ax.get_yticklabels(), arms):
        lbl.set_color(TEXT_C[key])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(joints)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(arms)), minor=True)
    ax.grid(which="minor", color="white", lw=1.2)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.06, pad=0.14, aspect=25)
    cb.ax.tick_params(labelsize=5.6, length=2)
    cb.set_label("correlation r with the reference", fontsize=6)
    cb.outline.set_visible(False)
    letter(ax, "c", x=-0.78, y=1.03)

    # (d) kinematics x force source: the 2 x 2, one small grid per measure
    sub = GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[1, :2], wspace=0.3)
    for k, (m, lab, fmt) in enumerate((("hip", "hip", "%.3f"), ("knee", "knee", "%.3f"),
                                       ("ankle", "ankle", "%.3f"), ("stance", "shear", "%.2f"))):
        ax = fig.add_subplot(sub[k])
        M = np.array([[wf[f"imu_insole_{m}"].mean(), wf[f"imu_plate_{m}"].mean()],
                      [wf[f"lab_insole_{m}"].mean(), wf[f"lab_plate_{m}"].mean()]])
        cmap = LinearSegmentedColormap.from_list("sq", ["#fdf1e7", "#f4a06d", "#c2502b"])
        ax.imshow(M, cmap=cmap, vmin=M.min() * 0.6, vmax=M.max(), aspect="equal")
        for i in range(2):
            for jj in range(2):
                ax.text(jj, i, fmt % M[i, jj], ha="center", va="center", fontsize=6.0,
                        color="white" if M[i, jj] > 0.8 * M.max() else INK,
                        fontweight="bold" if (i, jj) == (0, 0) else "normal")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["insole", "plate"], fontsize=5.8)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["IMU", "lab"] if k == 0 else [], fontsize=5.8)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(lab, fontsize=6.4, pad=3)
        if k == 0:
            ax.set_ylabel("kinematics", fontsize=6.2)
            letter(ax, "d", x=-0.75, y=1.18)
        if k == 1:
            ax.text(1.18, -0.42, "force source  ·  bold: fully wearable (IMU + insole)  ·  53 trials",
                    transform=ax.transAxes, ha="center", va="top", fontsize=6.0, color=INK2)
    save(fig, "hardware")


# ================================================================ online
def fig_online():
    mf, mw, cc, st = read("mhe_fukuchi.csv"), read("mhe_wang.csv"), read("causal_chain.csv"), read("statistics.json")
    DS = {"Fukuchi (150 Hz)": (mf, C["sole"]), "Wang (100 Hz)": (mw, "#8c3a1a")}
    fig = plt.figure(figsize=(W2, 4.9))
    outer = GridSpec(2, 2, figure=fig, height_ratios=[1, 0.72], width_ratios=[1, 1], hspace=0.62, wspace=0.42)

    # (a) solve time per window against the frame budget
    ax = fig.add_subplot(outer[0, 0])
    H = sorted(mf.horizon_s.unique())
    for i, h in enumerate(H):
        s = mf[(mf.horizon_s == h) & (mf.lookahead_s == 0.05)]
        m, p95 = s.solve_ms_mean.mean(), s.solve_ms_p95.mean()
        ax.barh(i, m, height=0.52, color=ORANGES[1 + i], edgecolor="white")
        ax.plot([m, p95], [i, i], color=INK, lw=0.9)
        ax.plot(p95, i, "|", color=INK, ms=6)
        ax.text(p95 * 1.25, i, f"{m:.3f} ms (p95 {p95:.3f})", va="center", fontsize=5.9, color=INK2)
    budget = 1000 / 150
    ax.axvline(budget, color="#b8322a", lw=1.0)
    ax.text(budget * 0.9, len(H) - 0.45, "one frame at 150 Hz\n6.7 ms", ha="right", va="top", fontsize=5.9,
            color="#b8322a")
    ax.set_xscale("log")
    ax.set_xlim(0.02, 12)
    ax.set_yticks(range(len(H)))
    ax.set_yticklabels([f"{h:g} s" for h in H])
    ax.set_ylabel("horizon")
    ax.set_xlabel("solve time per window (ms, log scale)")
    ax.tick_params(axis="y", length=0)
    soft_grid(ax, "x")
    letter(ax, "a", x=-0.2, y=1.03)

    # (b) how close the moving horizon stays to the whole-record solution
    ax = fig.add_subplot(outer[0, 1])
    for i, (name, (df, col)) in enumerate(DS.items()):
        no = df[df.lookahead_s == 0.0].vs_batch_pctBW.mean()
        la = df[(df.horizon_s == 0.2) & (df.lookahead_s == 0.05)].vs_batch_pctBW.mean()
        ax.plot([la, no], [i, i], color="#d8d3c8", lw=5, solid_capstyle="round", zorder=1)
        ax.plot(no, i, "o", ms=7, color="white", markeredgecolor=col, markeredgewidth=1.4, zorder=3)
        ax.plot(la, i, "o", ms=7, color=col, markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        ax.text(no * 1.25, i, f"{no:.2f}", va="center", fontsize=6.0, color=INK2)
        ax.text(la / 1.25, i, f"{la:.3f}", va="center", ha="right", fontsize=6.0, color=col)
        ax.text(np.sqrt(la * no), i + 0.27, f"{no / la:.0f}× closer", ha="center", fontsize=5.8, color=INK2)
    ax.set_xscale("log")
    ax.set_xlim(0.006, 2.5)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks(range(len(DS)))
    ax.set_yticklabels(list(DS))
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("difference from the whole-record solution (%BW, log)")
    soft_grid(ax, "x")
    tb = max(abs(st[k][b]) for k in ("tost_mhe_fukuchi", "tost_mhe_wang") for b in ("ci_lo", "ci_hi"))
    ax.legend([Line2D([], [], marker="o", ls="", color="white", markeredgecolor=INK2, ms=6),
               Line2D([], [], marker="o", ls="", color=INK2, ms=6)],
              ["no look-ahead", "50 ms look-ahead"], loc="upper left", fontsize=6, handletextpad=0.2,
              bbox_to_anchor=(0, 1.2), ncol=2)
    ax.text(0.99, 0.02, f"equivalent to batch: |90% CI| ≤ {tb:.2f} %BW", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=5.8, color=INK2)
    letter(ax, "b", x=-0.36, y=1.03)

    # (c) the latency pilot: accuracy against the filters' look-ahead
    ax = fig.add_subplot(outer[1, 0])
    whole = cc[cc.look_s.isna()].stance.mean()
    g = cc.dropna(subset=["look_s"]).groupby("look_s")[["stance", "vs_batch"]].mean()
    x = 1000 * g.index.values
    ax.fill_between(x, 0, g.vs_batch.values, color=ORANGES[0], lw=0, step=None)
    ax.plot(x, g.vs_batch.values, color=ORANGES[3], lw=1.0, label="distance from the whole-record estimate")
    ax.plot(x, g.stance.values, "o-", color=C["sole"], lw=1.5, ms=4, markeredgecolor="white",
            markeredgewidth=0.6, label="shear RMSE against the force plate")
    ax.axhline(whole, color=INK2, lw=0.8, ls=(0, (3, 2)))
    ax.text(x[-1], whole - 0.12, f"whole record {whole:.2f}", ha="right", va="top", fontsize=5.8, color=INK2)
    for xi, yi in zip(x, g.stance.values):
        if xi in (0, 50, 150):
            ax.annotate(f"{yi:.2f}", (xi, yi), xytext=(4, 5), textcoords="offset points", fontsize=5.8,
                        color=C["sole"])
    ax.set_xlabel("filter look-ahead (ms)")
    ax.set_ylabel("%BW")
    ax.set_ylim(0, 4.6)
    ax.legend(loc="center right", fontsize=5.8, bbox_to_anchor=(1.0, 0.4))
    soft_grid(ax)
    letter(ax, "c", x=-0.2, y=1.03)

    # (d) the end-to-end latency budget
    ax = fig.add_subplot(outer[1, 1])
    parts = [("filter look-ahead", 50, 150, ORANGES[2]), ("solver look-ahead", 50, 50, ORANGES[3]),
             ("solve", 0.2, 0.2, C["sole"])]
    left = np.array([0.0, 0.0])
    for lab, lo, hi, col in parts:
        for k, v in enumerate((lo, hi)):
            ax.barh(k, v, left=left[k], height=0.5, color=col, edgecolor="white")
            if v >= 20:
                ax.text(left[k] + v / 2, k, f"{v:g}", ha="center", va="center", fontsize=5.8, color="white")
        left += np.array([lo, hi])
    for k, tot in enumerate(left):
        ax.text(tot + 4, k, f"≈ {tot:.0f} ms", va="center", fontsize=6.2, fontweight="bold", color=INK)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["50 ms filter", "150 ms filter"])
    ax.set_xlim(0, 240)
    ax.set_xlabel("end-to-end latency (ms)")
    ax.tick_params(axis="y", length=0)
    soft_grid(ax, "x")
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=p[3]) for p in parts], [p[0] for p in parts],
              loc="upper left", bbox_to_anchor=(0, 1.35), ncol=3, fontsize=5.8, handlelength=1.0,
              columnspacing=0.8)
    letter(ax, "d", x=-0.36, y=1.1)
    save(fig, "online")


# ================================================================ overview (schematic)
def fig_overview():
    """(a) the model drawing (figures/concept.png, drawn by hand), (b) the estimator, (c) the
    three validation layers with their sizes."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig = plt.figure(figsize=(W2, 4.3))
    outer = GridSpec(2, 2, figure=fig, width_ratios=[0.9, 2.0], height_ratios=[1, 1.05],
                     wspace=0.06, hspace=0.18)
    ax = fig.add_subplot(outer[:, 0])
    p = os.path.join(FIG, "concept.png")
    if os.path.exists(p):
        ax.imshow(plt.imread(p))
    ax.axis("off")
    letter(ax, "a", x=0.0, y=0.985)

    def box(ax, x, y, w, h, title, lines, face, edge, title_col=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.025",
                                    fc=face, ec=edge, lw=0.9, clip_on=False))
        ax.text(x + w / 2, y + h - 0.045, title, ha="center", va="top", fontsize=6.6,
                fontweight="bold", color=title_col)
        for i, ln in enumerate(lines):
            ax.text(x + w / 2, y + h - 0.16 - 0.092 * i, ln, ha="center", va="top", fontsize=5.4,
                    color=INK2)

    def arrow(ax, x0, x1, y, text=None):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=8, lw=0.9,
                                     color=INK2))
        if text:
            ax.text((x0 + x1) / 2, y + 0.03, text, ha="center", va="bottom", fontsize=5.4, color=INK2)

    # (b) the estimator
    ax = fig.add_subplot(outer[0, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(ax, 0.01, 0.12, 0.18, 0.78, "wearable inputs",
        ["insole:", "vertical force,", "center of pressure", "", "IMU suit or markers:", "segment motion"],
        "#e8f0fa", "#9fb7d9", title_col="#2f5e99")
    box(ax, 0.235, 0.04, 0.335, 0.94, "SoleID",
        ["one convex program per window:", "whole-body horizontal dynamics", "+ virtual-pivot-point prior", "+ temporal smoothness",
         "+ friction cone, unilateral contact", f"4 parameters, set on {len(read('calibration.json')['subjects'])} subjects,", "then frozen"],
        "#fdeee4", C["sole"], title_col=C["sole"])
    box(ax, 0.615, 0.24, 0.15, 0.54, "shear force", ["each foot,", "every frame"], "#fff7f1", ORANGES[2])
    box(ax, 0.81, 0.12, 0.185, 0.78, "joint torques",
        ["ankle, knee, hip", "(inverse dynamics)", "", "→ exoskeleton or", "prosthesis control"],
        "#f4f3ef", "#b9b4aa")
    arrow(ax, 0.19, 0.235, 0.51)
    arrow(ax, 0.57, 0.615, 0.51)
    arrow(ax, 0.765, 0.81, 0.51)
    letter(ax, "b", x=-0.035, y=0.98)

    # (c) the three validation layers
    ax = fig.add_subplot(outer[1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # every count is read from the results, after quality control
    t, cal = read("test_all_trials.csv"), read("calibration.json")
    tm, lg, rp = read("camargo_treadmill_strides.csv"), read("camargo_levelground_steps.csv"), read("camargo_ramp_steps.csv")
    wa, ang = read("wang_application.csv"), read("ramp_angles.json")["angles_deg"].values()

    def n(x):
        return f"{x:,}".replace(",", " ")
    cards = [
        ("controlled", "Fukuchi et al. (2018)", ORANGES[1],
         ["split-belt treadmill, 8 speeds", "force plates stand in for insoles", "",
          f"calibration: {len(cal['subjects'])} subjects, {cal['n_trials']} trials",
          f"test: {t.subject.nunique()} subjects, {len(t)} trials"]),
        ("external", "Camargo et al. (2021)", ORANGES[3],
         ["a second laboratory, no retuning", f"{tm.subject.nunique()} subjects", f"treadmill: {n(len(tm))} strides",
          f"level ground: {n(len(lg))} steps", f"ramps {min(ang):.1f}–{max(ang):.1f}°: {n(len(rp))} steps"]),
        ("application", "Wang et al. (2023)", "#b3452b",
         ["real pressure insoles", "and an IMU suit", "", "reference: instrumented", "treadmill and markers",
          f"{wa.subject.nunique()} subjects, {len(wa)} trials"]),
    ]
    wcard, gap = 0.3, 0.045
    for k, (layer, src, col, lines) in enumerate(cards):
        x = 0.01 + k * (wcard + gap)
        ax.add_patch(FancyBboxPatch((x, 0.04), wcard, 0.86, boxstyle="round,pad=0.008,rounding_size=0.025",
                                    fc="#faf9f6", ec="#d9d5cc", lw=0.8, clip_on=False))
        ax.add_patch(FancyBboxPatch((x, 0.78), wcard, 0.12, boxstyle="round,pad=0.008,rounding_size=0.025",
                                    fc=col, ec=col, lw=0.8, clip_on=False))
        ax.text(x + wcard / 2, 0.84, f"{layer} layer", ha="center", va="center", fontsize=6.6,
                fontweight="bold", color="white")
        ax.text(x + wcard / 2, 0.715, src, ha="center", va="center", fontsize=6.0, color=INK,
                style="italic")
        for i, ln in enumerate(lines):
            ax.text(x + wcard / 2, 0.61 - 0.098 * i, ln, ha="center", va="center", fontsize=5.6, color=INK2)
        if k < 2:
            arrow(ax, x + wcard + 0.004, x + wcard + gap - 0.004, 0.45)
    ax.text(0.5, -0.05, "the parameters frozen on the controlled layer are used unchanged in the other two",
            ha="center", va="center", fontsize=5.8, color=INK2)
    letter(ax, "c", x=-0.035, y=0.96)
    save(fig, "overview")


FIGS = {"overview": fig_overview, "accuracy": fig_accuracy, "mechanism": fig_mechanism,
        "robustness": fig_robustness, "generalization": fig_generalization, "hardware": fig_hardware,
        "online": fig_online}

if __name__ == "__main__":
    for name in (sys.argv[1:] or list(FIGS)):
        FIGS[name]()

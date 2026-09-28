"""Statistical analysis for the SoleID paper.

Three families, each answering a different question:

  1. Linear mixed models -- how much better, with a confidence interval, accounting for the fact
     that trials are nested within subjects.
  2. Equivalence tests (TOST) -- is the difference small enough to call two things the same
     (moving-horizon vs batch; external validation vs the controlled layer).
  3. Statistical parametric mapping over the gait cycle -- where in the stride the estimate
     departs from the reference, with the multiple comparisons across the curve controlled.

Output: results/statistics.json (machine readable) and results/statistics.txt (for the paper).
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats

import soleid_planar as sp_

warnings.filterwarnings("ignore")
OUT = sp_.OUT
ARMS = ["Kin_only", "Newton_prop", "SoleID_v1", "SoleID"]     # v1 == SoleID without the VPP prior
PRETTY = {"Kin_only": "kinematics-only", "Newton_prop": "Newton + proportional split",
          "SoleID_v1": "SoleID without the prior", "SoleID_noVPP": "SoleID without the prior",
          "SoleID": "SoleID", "NoShear": "vertical-only (shear set to zero)"}
REPORT = []
JSON = {}


def say(line=""):
    REPORT.append(line)
    print(line)


# --------------------------------------------------------------------------- 1. mixed models
def long_form(df, arms, key, extra=()):
    """Stack the per-arm error columns into one long table. The overground modes carry `incline`
    or `speed_class` instead of a numeric speed, so `speed` is optional."""
    base = ["subject"] + (["speed"] if "speed" in df.columns else []) + \
           [c for c in extra if c in df.columns]
    rows = []
    for a in arms:
        col = f"{a}_{key}"
        if col not in df:
            continue
        sub = df[base].copy()
        sub["arm"] = a
        sub["y"] = df[col].values
        rows.append(sub)
    out = pd.concat(rows, ignore_index=True)
    return out.dropna(subset=["y"])


def mixed_model(df, arms, key, label, reference="SoleID", speed_term=True, extra=()):
    """y ~ arm (+ speed) with a random intercept per subject; contrasts against `reference`."""
    from statsmodels.regression.mixed_linear_model import MixedLM   # what statsmodels.formula.api.mixedlm is
    d = long_form(df, arms, key, extra)
    d["arm"] = pd.Categorical(d["arm"], categories=[reference] + [a for a in arms if a != reference])
    has_speed = "speed" in d.columns and d["speed"].nunique() > 2
    formula = "y ~ C(arm)" + (" + speed" if speed_term and has_speed else "")
    m = MixedLM.from_formula(formula, d, groups=d["subject"]).fit(reml=True, method="lbfgs")
    out = {"n_obs": int(len(d)), "n_subjects": int(d.subject.nunique()), "reference": reference,
           "contrasts": {}, "speed_slope": None}
    for name in m.params.index:
        if name.startswith("C(arm)"):
            arm = name.split("[T.")[1].rstrip("]")
            ci = m.conf_int().loc[name]
            out["contrasts"][arm] = dict(estimate=float(m.params[name]), lo=float(ci[0]),
                                         hi=float(ci[1]), p=float(m.pvalues[name]))
        elif name == "speed":
            ci = m.conf_int().loc[name]
            out["speed_slope"] = dict(estimate=float(m.params[name]), lo=float(ci[0]), hi=float(ci[1]),
                                      p=float(m.pvalues[name]))
    say(f"\n{label}  (mixed model, subject as random intercept; {out['n_obs']} observations, "
        f"{out['n_subjects']} subjects)")
    for arm, c in out["contrasts"].items():
        say(f"    {PRETTY.get(arm, arm):32s} minus {reference}: "
            f"{c['estimate']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]  p = {c['p']:.2e}")
    if out["speed_slope"]:
        s = out["speed_slope"]
        say(f"    speed slope: {s['estimate']:+.3f} per m/s [{s['lo']:+.3f}, {s['hi']:+.3f}]  p = {s['p']:.2e}")
    return out


def bootstrap_ci(x, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    b = rng.choice(np.asarray(x, float), (n, len(x)), replace=True).mean(axis=1)
    return float(np.mean(x)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


# --------------------------------------------------------------------------- 2. equivalence
def tost_paired(a, b, bound, label):
    """Two one-sided tests on paired differences; equivalence if the 90 % CI of the mean
    difference lies inside +-bound."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    n = len(d)
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    t_lo = (m + bound) / se
    t_hi = (m - bound) / se
    p_lo = 1 - stats.t.cdf(t_lo, n - 1)
    p_hi = stats.t.cdf(t_hi, n - 1)
    p = max(p_lo, p_hi)
    crit = stats.t.ppf(0.95, n - 1)
    ci = (m - crit * se, m + crit * se)
    ok = ci[0] > -bound and ci[1] < bound
    say(f"    {label}: mean difference {m:+.4f}, 90 % CI [{ci[0]:+.4f}, {ci[1]:+.4f}], "
        f"bound +-{bound}, equivalent: {'yes' if ok else 'no'} (p = {p:.1e}, n = {n})")
    return dict(mean=float(m), ci_lo=float(ci[0]), ci_hi=float(ci[1]), bound=float(bound),
                equivalent=bool(ok), p=float(p), n=int(n))


def tost_independent(a, b, bound, label):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    m = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    dof = len(a) + len(b) - 2
    crit = stats.t.ppf(0.95, dof)
    ci = (m - crit * se, m + crit * se)
    ok = ci[0] > -bound and ci[1] < bound
    say(f"    {label}: difference of means {m:+.3f}, 90 % CI [{ci[0]:+.3f}, {ci[1]:+.3f}], "
        f"bound +-{bound}, equivalent: {'yes' if ok else 'no'}")
    return dict(mean=float(m), ci_lo=float(ci[0]), ci_hi=float(ci[1]), bound=float(bound),
                equivalent=bool(ok), n_a=int(len(a)), n_b=int(len(b)))


# --------------------------------------------------------------------------- 3. SPM
def spm_paired(y1, y0, label, alpha=0.05):
    """Paired SPM{t} over the gait cycle; falls back to a cluster permutation test if spm1d is
    unavailable. Returns the significant clusters as percentages of the cycle."""
    y1, y0 = np.asarray(y1, float), np.asarray(y0, float)
    keep = np.isfinite(y1).all(1) & np.isfinite(y0).all(1)
    y1, y0 = y1[keep], y0[keep]
    try:
        import spm1d
        t = spm1d.stats.ttest_paired(y1, y0)
        ti = t.inference(alpha, two_tailed=True, interp=True)
        clusters = [(float(c.endpoints[0]), float(c.endpoints[1])) for c in ti.clusters]
        zstar = float(ti.zstar)
        method = "spm1d"
    except Exception as e:                        # cluster-based permutation fallback
        clusters, zstar, method = _cluster_perm(y1, y0, alpha), np.nan, f"permutation ({e.__class__.__name__})"
    frac = sum(b - a for a, b in clusters) / (y1.shape[1] - 1) * 100
    say(f"    {label}: {len(clusters)} significant cluster(s), {frac:.0f} % of the cycle"
        + (f", t* = {zstar:.2f}" if np.isfinite(zstar) else "") + f"  [{method}]")
    for a, b in clusters:
        say(f"        {a:.0f}-{b:.0f} % of the cycle")
    return dict(n_clusters=len(clusters), clusters=clusters, zstar=zstar, method=method,
                percent_of_cycle=float(frac), n=int(len(y1)))


def _cluster_perm(y1, y0, alpha=0.05, n_perm=1000, seed=0):
    d = y1 - y0
    n, p = d.shape
    tmap = d.mean(0) / (d.std(0, ddof=1) / np.sqrt(n) + 1e-12)
    thr = stats.t.ppf(1 - alpha / 2, n - 1)

    def clusters_of(t):
        out, start = [], None
        for i, v in enumerate(np.abs(t) > thr):
            if v and start is None:
                start = i
            elif not v and start is not None:
                out.append((start, i - 1)); start = None
        if start is not None:
            out.append((start, p - 1))
        return out

    obs = clusters_of(tmap)
    mass = [np.abs(tmap[a:b + 1]).sum() for a, b in obs]
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        sign = rng.choice([-1.0, 1.0], size=(n, 1))
        dd = d * sign
        tt = dd.mean(0) / (dd.std(0, ddof=1) / np.sqrt(n) + 1e-12)
        cl = clusters_of(tt)
        null.append(max([np.abs(tt[a:b + 1]).sum() for a, b in cl], default=0.0))
    crit = np.percentile(null, 95)
    return [(float(a), float(b)) for (a, b), m in zip(obs, mass) if m > crit]


# --------------------------------------------------------------------------- driver
def main():
    say("=" * 78)
    say("SoleID -- statistical analysis")
    say("=" * 78)

    test = pd.read_csv(os.path.join(OUT, "test_all_trials.csv"))
    say(f"\nControlled layer: {test.subject.nunique()} subjects, {len(test)} trials")

    say("\n--- 1. mixed models (Fukuchi test set) ---")
    JSON["mixed_stance"] = mixed_model(test, ARMS, "stance", "Shear RMSE during stance (%BW)", extra=("age",))
    JSON["mixed_ds"] = mixed_model(test, ARMS, "ds", "Shear RMSE during double support (%BW)")
    JSON["mixed_hip"] = mixed_model(test, ARMS, "hip", "Hip moment RMSE (Nm/kg)")
    JSON["mixed_knee"] = mixed_model(test, ARMS, "knee", "Knee moment RMSE (Nm/kg)")

    say("\n    age group (SoleID stance): " +
        ", ".join(f"{k} {v:.2f} %BW" for k, v in test.groupby("age").SoleID_stance.mean().items()))
    m, lo, hi = bootstrap_ci(test.SoleID_stance)
    say(f"    SoleID stance mean {m:.2f} %BW [{lo:.2f}, {hi:.2f}] (bootstrap)")
    m, lo, hi = bootstrap_ci(test.SoleID_hip)
    say(f"    SoleID hip mean {m:.3f} Nm/kg [{lo:.3f}, {hi:.3f}] (bootstrap)")
    JSON["paired_better"] = dict(
        vs_newton=float((test.SoleID_stance < test.Newton_prop_stance).mean()),
        vs_noVPP=float((test.SoleID_stance < test.SoleID_v1_stance).mean()))
    say(f"    SoleID beats the Newton baseline in {100 * JSON['paired_better']['vs_newton']:.1f} % of trials, "
        f"the no-prior arm in {100 * JSON['paired_better']['vs_noVPP']:.1f} %")

    # the prior-gain relationship, with a mixed model over trials
    test = test.assign(gain=test.SoleID_v1_stance - test.SoleID_stance)
    from statsmodels.regression.mixed_linear_model import MixedLM   # what statsmodels.formula.api.mixedlm is
    mg = MixedLM.from_formula("gain ~ newton_check", test, groups=test["subject"]).fit(reml=True, method="lbfgs")
    ci = mg.conf_int().loc["newton_check"]
    JSON["gain_vs_newton_check"] = dict(slope=float(mg.params["newton_check"]), lo=float(ci[0]),
                                        hi=float(ci[1]), p=float(mg.pvalues["newton_check"]),
                                        r=float(np.corrcoef(test.newton_check, test.gain)[0, 1]))
    say(f"\n    prior gain vs whole-body dynamics residual: slope "
        f"{JSON['gain_vs_newton_check']['slope']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] per %BW, "
        f"p = {JSON['gain_vs_newton_check']['p']:.1e}, r = {JSON['gain_vs_newton_check']['r']:.2f}")

    # ---------------------------------------------------------------- external layers
    say("\n--- 2. external validation and application ---")
    cam = pd.read_csv(os.path.join(OUT, "camargo_treadmill_strides.csv"))
    cam = cam.rename(columns={c: c.replace("SoleID_noVPP", "SoleID_v1") for c in cam.columns})
    JSON["camargo_treadmill_stance"] = mixed_model(cam, ARMS, "stance",
                                                   "Camargo treadmill, shear RMSE stance (%BW)")
    for name, f in (("ramp", "camargo_ramp_steps.csv"), ("levelground", "camargo_levelground_steps.csv")):
        p = os.path.join(OUT, f)
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        df = df.rename(columns={c: c.replace("SoleID_noVPP", "SoleID_v1") for c in df.columns})
        JSON[f"camargo_{name}_stance"] = mixed_model(df, ["NoShear", "Newton_prop", "SoleID_v1", "SoleID"],
                                                     "stance", f"Camargo {name}, shear RMSE stance (%BW)")

    wang = pd.read_csv(os.path.join(OUT, "wang_application.csv"))
    JSON["wang_hip"] = mixed_model(wang, ["NoShear", "Newton_prop", "SoleID_lab_kin", "SoleID"], "hip",
                                   "Wang, hip moment RMSE (Nm/kg)")
    JSON["wang_stance"] = mixed_model(wang, ["NoShear", "Newton_prop", "SoleID_lab_kin", "SoleID"], "stance",
                                      "Wang, shear RMSE stance (%BW)")

    # ---------------------------------------------------------------- equivalence
    say("\n--- 3. equivalence tests (TOST, 90 % CI) ---")
    mhe = pd.read_csv(os.path.join(OUT, "mhe_fukuchi.csv"))
    sel = mhe[(mhe.horizon_s == 0.2) & (mhe.lookahead_s == 0.05)]
    JSON["tost_mhe_fukuchi"] = tost_paired(sel.mhe_stance_pctBW, sel.batch_stance_pctBW, 0.5,
                                           "moving horizon (0.2 s, 50 ms look-ahead) vs batch, Fukuchi")
    mw = pd.read_csv(os.path.join(OUT, "mhe_wang.csv"))
    selw = mw[(mw.horizon_s == 0.2) & (mw.lookahead_s == 0.05)]
    JSON["tost_mhe_wang"] = tost_paired(selw.mhe_stance_pctBW, selw.batch_stance_pctBW, 0.5,
                                        "moving horizon vs batch, Wang")
    JSON["tost_generalisation"] = tost_independent(
        cam.groupby("subject").SoleID_stance.mean(), test.groupby("subject").SoleID_stance.mean(), 1.0,
        "Camargo treadmill vs Fukuchi (subject means), generalisation")

    # ---------------------------------------------------------------- SPM
    say("\n--- 4. statistical parametric mapping over the gait cycle ---")
    cp = os.path.join(OUT, "curves_fukuchi.npz")
    if os.path.exists(cp):
        z = np.load(cp, allow_pickle=True)
        subj = np.asarray(z["subjects"])
        uniq = np.unique(subj)

        def per_subject(key):
            """One curve per subject: trials within a subject are not independent, so the subject
            is the unit of analysis."""
            a = np.asarray(z[key], float)
            return np.array([a[subj == s].mean(axis=0) for s in uniq])

        say(f"    curves: {len(subj)} trials averaged to {len(uniq)} subjects (the unit of analysis)")
        spm = {}
        for arm in ("SoleID", "SoleID_noVPP", "Newton_prop", "NoShear"):
            k = f"{arm}_shear"
            if k in z:
                spm[f"{arm}_shear"] = spm_paired(per_subject(k), per_subject("reference_shear"),
                                                 f"{PRETTY.get(arm, arm)}: shear")
        for j in ("ankle", "knee", "hip"):
            spm[f"SoleID_{j}"] = spm_paired(per_subject(f"SoleID_{j}"), per_subject(f"reference_{j}"),
                                            f"SoleID: {j} moment")
            spm[f"NoShear_{j}"] = spm_paired(per_subject(f"NoShear_{j}"), per_subject(f"reference_{j}"),
                                             f"vertical-only: {j} moment")
        # how large are the differences the test flags?
        d = per_subject("SoleID_shear") - per_subject("reference_shear")
        say(f"    SoleID shear: mean absolute difference over the cycle {np.abs(d.mean(0)).mean():.2f} %BW, "
            f"largest {np.abs(d.mean(0)).max():.2f} %BW at {int(np.argmax(np.abs(d.mean(0))))} % of the cycle")
        dh = per_subject("SoleID_hip") - per_subject("reference_hip")
        say(f"    SoleID hip moment: mean absolute difference {np.abs(dh.mean(0)).mean():.3f} Nm/kg, "
            f"largest {np.abs(dh.mean(0)).max():.3f} Nm/kg at {int(np.argmax(np.abs(dh.mean(0))))} %")
        JSON["spm"] = spm
        JSON["spm_effect_size"] = dict(
            shear_mean_abs=float(np.abs(d.mean(0)).mean()), shear_max_abs=float(np.abs(d.mean(0)).max()),
            hip_mean_abs=float(np.abs(dh.mean(0)).mean()), hip_max_abs=float(np.abs(dh.mean(0)).max()))
    else:
        say("    curves file missing; run curves.py first")

    with open(os.path.join(OUT, "statistics.json"), "w") as f:
        json.dump(JSON, f, indent=2)
    with open(os.path.join(OUT, "statistics.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(REPORT))
    say(f"\nwritten: {os.path.join(OUT, 'statistics.json')} and statistics.txt")


if __name__ == "__main__":
    main()

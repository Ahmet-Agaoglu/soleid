"""Statistical analyses that go beyond statistics.py, each computed rather than argued.

  1. Each trial contributes one row per arm, so trial is a second level of clustering under
     subject, and the arm effect may differ between subjects. Both are handled by modelling the
     within-trial difference between two arms with a random intercept per subject: differencing
     removes everything the arms share in a trial, and the subject intercept on the difference is
     a subject-specific arm effect. (A variance component for trial inside the stacked model is
     the textbook alternative; with 231 components it is numerically singular here.)
  2. A common speed slope does not show that the advantage grows with speed. The speed slope of
     the within-trial difference is exactly the arm x speed interaction, so test that.
  3. Multiplicity: Holm adjustment across the family of primary contrasts.
  4. Age: a difference with an interval and an equivalence test, instead of "no practical effect".
  5. The bootstrap: resample subjects, not trials, since trials within a subject are not
     independent.
  6. SPM: repeat the family of curve tests at a Bonferroni-adjusted alpha and report whether any
     conclusion changes.

Mixed models are fitted on a standardized response and rescaled, with a fallback through several
optimizers: moment differences are small numbers and L-BFGS alone sometimes stops at the start.

Output: results/stats_revision.json
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
from statsmodels.regression.mixed_linear_model import MixedLM   # what statsmodels.formula.api.mixedlm is

import soleid_planar as sp_
from statistics import ARMS, spm_paired

warnings.filterwarnings("ignore")
OUT = sp_.OUT
RES = {}


def robust_fit(formula, d, resp):
    """Fit on resp / sd(resp); return {term: (estimate, lo, hi, p)} on the original scale."""
    sc = float(d[resp].std()) or 1.0
    d2 = d.copy()
    d2[resp] = d2[resp] / sc
    last = None
    for method in ("lbfgs", "powell", "nm", "bfgs"):
        try:
            m = MixedLM.from_formula(formula, d2, groups=d2["subject"]).fit(reml=True, method=method)
            ci = m.conf_int().loc[m.fe_params.index]
            if np.isfinite(ci.values).all() and float((ci[1] - ci[0]).max()) < 50:
                return {t: (float(m.fe_params[t]) * sc, float(ci.loc[t, 0]) * sc,
                            float(ci.loc[t, 1]) * sc, float(m.pvalues[t])) for t in m.fe_params.index}
        except Exception as e:           # singular Hessian on one optimizer: try the next
            last = e
    raise RuntimeError(f"no optimizer converged for {formula}: {last!r}")


def paired(test, arm, key, ref="SoleID"):
    """arm minus SoleID within each trial; speed centered so the intercept is the contrast at the
    mean speed and the slope is the arm x speed interaction."""
    d = pd.DataFrame(dict(subject=test.subject.values,
                          diff=(test[f"{arm}_{key}"] - test[f"{ref}_{key}"]).values,
                          speed_c=(test.speed - test.speed.mean()).values))
    return robust_fit("diff ~ speed_c", d, "diff")


def main():
    test = pd.read_csv(os.path.join(OUT, "test_all_trials.csv"))
    pub = json.load(open(os.path.join(OUT, "statistics.json")))

    # ---- 1 + 3. within-trial paired models, and Holm
    fam = []
    RES["paired"] = {}
    for key in ("stance", "ds", "hip", "knee"):
        out = {}
        for arm in [a for a in ARMS if a != "SoleID"]:
            r = paired(test, arm, key)
            e, lo, hi, p = r["Intercept"]
            p0 = pub[f"mixed_{key}"]["contrasts"][arm]
            out[arm] = dict(estimate=e, lo=lo, hi=hi, p=p, speed_slope=r["speed_c"][0],
                            speed_lo=r["speed_c"][1], speed_hi=r["speed_c"][2], speed_p=r["speed_c"][3],
                            published=float(p0["estimate"]), published_lo=float(p0["lo"]),
                            published_hi=float(p0["hi"]))
            fam.append((f"{key}:{arm}", p))
        RES["paired"][key] = out
        print(f"\n{key}: within-trial paired model vs the published stacked model")
        for arm, c in out.items():
            print(f"   {arm:12s} {c['estimate']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]   "
                  f"published {c['published']:+.3f} [{c['published_lo']:+.3f}, {c['published_hi']:+.3f}]"
                  f"   speed slope {c['speed_slope']:+.3f} (p {c['speed_p']:.1e})")
    ps = sorted(fam, key=lambda x: x[1])
    k = len(ps)
    adj, running = {}, 0.0
    for i, (name, p) in enumerate(ps):
        running = max(running, min(1.0, (k - i) * p))
        adj[name] = running
    RES["holm"] = dict(family_size=k, max_adjusted_p=float(max(adj.values())), adjusted=adj)
    print(f"\nHolm over {k} primary contrasts: largest adjusted p = {max(adj.values()):.1e}")

    # ---- 2. does the advantage grow with speed?
    c = RES["paired"]["stance"]["Newton_prop"]
    RES["arm_by_speed"] = dict(interaction=c["speed_slope"], lo=c["speed_lo"], hi=c["speed_hi"],
                               p=c["speed_p"])
    d = pd.DataFrame(dict(subject=test.subject.values,
                          ratio=(test.Newton_prop_stance / test.SoleID_stance).values,
                          speed=test.speed.values))
    r = robust_fit("ratio ~ speed", d, "ratio")["speed"]
    RES["ratio_by_speed"] = dict(slope=r[0], lo=r[1], hi=r[2], p=r[3])
    print(f"\nNewton minus SoleID, change per m/s: {c['speed_slope']:+.3f} "
          f"[{c['speed_lo']:+.3f}, {c['speed_hi']:+.3f}] %BW, p = {c['speed_p']:.1e}")
    print(f"Newton/SoleID ratio, change per m/s: {r[0]:+.3f} [{r[1]:+.3f}, {r[2]:+.3f}], p = {r[3]:.1e}")

    # ---- 4. age (between subjects: 11 older, 20 young)
    d = pd.DataFrame(dict(subject=test.subject.values, y=test.SoleID_stance.values,
                          older=(test.age == "Older").astype(float).values,
                          speed_c=(test.speed - test.speed.mean()).values))
    e, lo95, hi95, p = robust_fit("y ~ older + speed_c", d, "y")["older"]
    se = (hi95 - lo95) / (2 * 1.96)
    lo90, hi90 = e - 1.645 * se, e + 1.645 * se
    bound = 0.5
    RES["age"] = dict(estimate=e, lo95=lo95, hi95=hi95, lo90=lo90, hi90=hi90, p=p, bound=bound,
                      equivalent=bool(lo90 > -bound and hi90 < bound),
                      means=test.groupby("age").SoleID_stance.mean().to_dict(),
                      n_subjects=test.groupby("age").subject.nunique().to_dict())
    print(f"\nolder minus young, speed-adjusted: {e:+.3f} %BW, 95 % CI [{lo95:+.3f}, {hi95:+.3f}]; "
          f"90 % CI [{lo90:+.3f}, {hi90:+.3f}] inside +-{bound}: {RES['age']['equivalent']}")

    # ---- 5. subject-level bootstrap
    rng = np.random.default_rng(0)
    groups = {s: g for s, g in test.groupby("subject")}
    subs = np.array(list(groups))
    cols = ("SoleID_stance", "SoleID_hip")
    bs = {c: [] for c in cols}
    for _ in range(5000):
        sample = pd.concat([groups[s] for s in rng.choice(subs, len(subs), replace=True)])
        for c in cols:
            bs[c].append(sample[c].mean())
    RES["bootstrap_subjects"] = {c: dict(mean=float(test[c].mean()), lo=float(np.percentile(v, 2.5)),
                                         hi=float(np.percentile(v, 97.5))) for c, v in bs.items()}
    print("subject-level bootstrap:",
          {c: f"{v['mean']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]" for c, v in RES["bootstrap_subjects"].items()})

    # ---- 6. SPM with the family controlled
    cp = os.path.join(OUT, "curves_fukuchi.npz")
    if os.path.exists(cp):
        z = np.load(cp, allow_pickle=True)
        subj = np.asarray(z["subjects"])
        uniq = np.unique(subj)

        def per(key):
            a = np.asarray(z[key], float)
            return np.array([a[subj == s].mean(axis=0) for s in uniq])

        tests = [(f"{a}_shear", "reference_shear")
                 for a in ("SoleID", "SoleID_noVPP", "Newton_prop", "NoShear") if f"{a}_shear" in z]
        tests += [(f"{a}_{j}", f"reference_{j}") for a in ("SoleID", "NoShear") for j in ("ankle", "knee", "hip")]
        alpha = 0.05 / len(tests)
        RES["spm_bonferroni"] = dict(n_tests=len(tests), alpha=alpha, results={})
        pub_spm = pub.get("spm", {})
        print(f"\nSPM at Bonferroni alpha {alpha:.4f} over {len(tests)} tests (published: 0.05 each)")
        for a, b in tests:
            r = spm_paired(per(a), per(b), a, alpha=alpha)
            RES["spm_bonferroni"]["results"][a] = dict(
                percent_of_cycle=r["percent_of_cycle"], n_clusters=r["n_clusters"],
                published_percent=pub_spm.get(a, {}).get("percent_of_cycle"))
    json.dump(RES, open(os.path.join(OUT, "stats_revision.json"), "w"), indent=1, default=float)
    return RES


if __name__ == "__main__":
    main()

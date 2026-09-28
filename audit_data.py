"""Audit part 2: protocol integrity, result files, per-data-set physical sanity, and the two
numbers that looked odd (the correlation drop at the hip in the Wang layer, and the centre of
pressure shift that *improved* the hip error in the degradation study).

Run:  python audit_data.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

import experiments as ex
import soleid_planar as sp_
import config

OUT = sp_.OUT
CAL = json.load(open(os.path.join(OUT, "calibration.json")))
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))


def note(msg):
    print(f"  [note] {msg}")


# ------------------------------------------------------------------ 1. protocol integrity
def audit_protocol():
    print("== protocol integrity ==")
    from protocol import CAL_SUBJECTS
    test = pd.read_csv(os.path.join(OUT, "test_all_trials.csv"))
    leak = sorted(set(test.subject) & set(CAL_SUBJECTS))
    check("calibration subjects absent from the test set", not leak, f"leaked: {leak}" if leak else
          f"{test.subject.nunique()} test subjects, {len(CAL_SUBJECTS)} calibration subjects")

    chosen = CAL["chosen"]
    note(f"frozen parameters: {chosen}")
    # every runner must read the same frozen values
    import camargo_run, wang_run, camargo_overground, moving_horizon
    for mod in (camargo_run, wang_run, camargo_overground, moving_horizon):
        same = all(abs(mod.CAL[k] - chosen[k]) < 1e-12 for k in chosen)
        check(f"{mod.__name__} uses the frozen parameters", same, str(mod.CAL))
    check("soleid_planar.V2_PARAMS defaults are not silently different",
          all(abs(sp_.V2_PARAMS[k] - chosen[k]) < 1e-12 for k in ("w_prior", "h_vpp", "fc_smooth")),
          str(sp_.V2_PARAMS))


# ------------------------------------------------------------------ 2. result files
def audit_files():
    print("\n== result files ==")
    files = {
        "test_all_trials.csv": ["SoleID_stance", "SoleID_hip", "Newton_prop_stance", "Kin_only_stance"],
        "camargo_treadmill_strides.csv": ["SoleID_stance", "SoleID_hip"],
        "camargo_ramp_steps.csv": ["SoleID_stance", "SoleID_hip"],
        "wang_application.csv": ["SoleID_stance", "SoleID_hip"],
        "final_degradation_trials.csv": ["stance", "hip"],
        "mhe_fukuchi.csv": ["vs_batch_pctBW", "solve_ms_mean"],
        "mhe_wang.csv": ["vs_batch_pctBW", "solve_ms_mean"],
        "wang_crosscheck.csv": ["ours_vs_theirs_ankle_r"],
    }
    lg = os.path.join(OUT, "camargo_levelground_steps.csv")
    if os.path.exists(lg):
        files["camargo_levelground_steps.csv"] = ["SoleID_stance", "SoleID_hip"]
    for f, cols in files.items():
        p = os.path.join(OUT, f)
        if not os.path.exists(p):
            check(f"{f} exists", False, "missing")
            continue
        df = pd.read_csv(p)
        miss = [c for c in cols if c not in df.columns]
        nan = {c: int(df[c].isna().sum()) for c in cols if c in df.columns and df[c].isna().any()}
        check(f"{f}: columns present and finite", not miss and not nan,
              f"rows={len(df)}" + (f", missing={miss}" if miss else "") + (f", NaN={nan}" if nan else ""))
        num = df.select_dtypes(include=[np.number])
        inf = int(np.isinf(num.to_numpy()).sum())
        check(f"{f}: no infinities", inf == 0, f"{inf} infinite values")


# ------------------------------------------------------------------ 3. physical sanity per set
def sanity_fukuchi():
    print("\n== Fukuchi: physical sanity (5 random test trials) ==")
    from protocol import CAL_SUBJECTS, available_trials
    rng = np.random.default_rng(0)
    trials = [t for t in available_trials() if t[0] not in CAL_SUBJECTS]
    pick = [trials[i] for i in rng.choice(len(trials), 5, replace=False)]
    for subj, trial, speed, age in pick:
        try:
            d = ex.prepare(subj, trial)
        except sp_.MarkerGapError as e:
            note(f"{subj} {trial}: skipped ({e})")
            continue
        sp_.configure(subj, trial)
        BW = sp_.MASS * 9.81
        okv, oks, okf, okc = [], [], [], []
        for s in ("R", "L"):
            Fy, cop = d["grf"][s]["Fy"], d["grf"][s]["copx"]
            on = Fy > sp_.F_CONTACT
            okv.append(0.9 <= Fy.max() / BW <= 1.4)
            oks.append(0.5 <= on.mean() <= 0.8)
            heel, toe = d["P"][s + ".Heel"][:d["n"], 0], d["P"][s + ".MT5"][:d["n"], 0]
            L = np.mean(np.abs(toe - heel)[on])
            okf.append(0.10 <= L <= 0.30)
            lo, hi = np.minimum(heel, toe) - 0.06, np.maximum(heel, toe) + 0.06
            okc.append(np.mean(((cop >= lo) & (cop <= hi))[on]) > 0.75)
        check(f"{subj} {trial} ({speed} m/s): peak Fy/BW, stance %, foot length, COP inside foot",
              all(okv) and all(oks) and all(okf) and all(okc),
              f"Fy/BW={d['grf']['R']['Fy'].max() / BW:.2f}, stance={100 * (d['grf']['R']['Fy'] > 20).mean():.0f}%, "
              f"footlen={L:.3f} m")


def sanity_camargo():
    print("\n== Camargo: subject data and covariates ==")
    info = pd.read_csv(config.CAMARGO_INFO).set_index("Subject")
    tm = pd.read_csv(os.path.join(OUT, "camargo_treadmill_strides.csv"))
    bad = []
    for s, g in tm.groupby("subject"):
        if abs(g.mass.iloc[0] - float(info.loc[s, "Weight"])) > 1e-6:
            bad.append(s)
    check("subject masses match SubjectInfo.csv", not bad, f"mismatched: {bad}" if bad else f"{tm.subject.nunique()} subjects")
    check("treadmill speeds inside the protocol range", tm.speed.between(0.4, 2.1).all(),
          f"{tm.speed.min():.2f}..{tm.speed.max():.2f} m/s")
    check("stride durations plausible", tm.stride_s.between(0.6, 2.0).mean() > 0.98,
          f"{tm.stride_s.min():.2f}..{tm.stride_s.max():.2f} s")
    rp = pd.read_csv(os.path.join(OUT, "camargo_ramp_steps.csv"))
    check("ramp: both feet represented about equally", abs(rp.side.value_counts(normalize=True).iloc[0] - 0.5) < 0.06,
          rp.side.value_counts(normalize=True).round(3).to_dict())
    check("ramp: stance durations plausible", rp.stance_s.between(0.3, 1.5).mean() > 0.95,
          f"{rp.stance_s.min():.2f}..{rp.stance_s.max():.2f} s")


def sanity_wang():
    print("\n== Wang: alignment and insole handling ==")
    w = pd.read_csv(os.path.join(OUT, "wang_application.csv"))
    check("time alignment correlation high on every trial", w.r_align.min() > 0.70,
          f"min {w.r_align.min():.3f}, mean {w.r_align.mean():.3f}")
    check("time offsets consistent across trials", w.tau.std() < 0.2,
          f"{w.tau.min():.2f}..{w.tau.max():.2f} s (sd {w.tau.std():.3f})")
    check("insole vertical force error in the expected band", 5 < w.fy_rmse_pctBW.mean() < 15,
          f"{w.fy_rmse_pctBW.mean():.1f} %BW")
    check("the wearable arm does not depend on a rigid frame alignment", (w.dx_align == 0).all(),
          "dx_align is zero for every trial")


# ------------------------------------------------------------------ 4. the two odd numbers
def investigate_hip_correlation():
    print("\n== anomaly 1: hip correlation lower for SoleID than for the vertical-only arm (Wang) ==")
    w = pd.read_csv(os.path.join(OUT, "wang_application.csv"))
    for arm in ("NoShear", "Newton_prop", "SoleID", "SoleID_lab_kin"):
        note(f"{arm:15s} hip RMSE {w[f'{arm}_hip'].mean():.3f} Nm/kg, r {w[f'{arm}_hip_r'].mean():.3f}")
    note("RMSE and r answer different questions: r ignores a constant offset and a scale factor.")
    # decompose one trial into bias and variance
    import wang_adapter as wa, wang_run as wr
    info = wa.subject_info()
    d = wa.load_trial("Subj04", "walk_36")
    mass, height = info["Subj04"]["mass"], info["Subj04"]["height"]
    n = len(d["t"])
    sp_.MASS, sp_.HEIGHT, sp_.FS = mass, height, wr.FS
    trc = wa.read_trc(d["paths"]["trc"])
    n = min(n, len(trc[0]))
    Plab, pel, ok = wr.planar_from_markers(trc, n)
    Plab = {k: (sp_.lowpass(v, 6.0, wr.FS) if isinstance(v, np.ndarray) else v) for k, v in Plab.items()}
    Plab["R.Hip"] = Plab["L.Hip"] = wr.hip_from_pelvis_planar(sp_.lowpass(pel, 6.0, wr.FS), height)
    segs = wr.build(Plab, mass, height, n)
    grf = {s: {k: v[:n] for k, v in d["ref"][s].items()} for s in ("R", "L")}
    ref = sp_.inverse_dynamics(segs, "R", grf["R"]["Fx"], grf["R"]["Fy"], grf["R"]["copx"])["hip"] / mass
    zero = sp_.inverse_dynamics(segs, "R", np.zeros(n), grf["R"]["Fy"], grf["R"]["copx"])["hip"] / mass
    on = grf["R"]["Fy"] > sp_.F_CONTACT
    note(f"vertical-only hip: bias {np.mean(zero[on] - ref[on]):+.3f}, sd of error "
         f"{np.std(zero[on] - ref[on]):.3f}, r {np.corrcoef(zero[on], ref[on])[0, 1]:.3f} Nm/kg")
    note("a large, smooth systematic offset keeps r high while RMSE is poor; adding an imperfect "
         "shear estimate removes most of the offset but adds scatter, which lowers r.")
    check("SoleID reduces the hip RMSE despite the lower correlation",
          w.SoleID_hip.mean() < w.NoShear_hip.mean(),
          f"{w.SoleID_hip.mean():.3f} vs {w.NoShear_hip.mean():.3f} Nm/kg")


def investigate_cop_shift():
    print("\n== anomaly 2: a COP shift that improves the hip error (degradation study) ==")
    dg = pd.read_csv(os.path.join(OUT, "final_degradation_trials.csv"))
    s = dg[(dg.arm == "SoleID") & (dg.condition.isin(["clean", "cop_shift_mm"]))]
    g = s.groupby("level")[["stance", "hip"]].mean().round(3)
    print(g.to_string())
    # is there a systematic bias in the clean hip moment that the shift cancels?
    tr = pd.read_csv(os.path.join(OUT, "test_all_trials.csv"))
    note("if the clean hip moment carries a systematic offset, a forward COP shift can cancel part "
         "of it; that would show up as a signed bias rather than a random error.")
    from protocol import CAL_SUBJECTS, available_trials
    trials = [t for t in available_trials() if t[0] not in CAL_SUBJECTS][:6]
    bias = []
    for subj, trial, speed, age in trials:
        try:
            d = ex.prepare(subj, trial)
        except sp_.MarkerGapError:
            continue
        sp_.configure(subj, trial)
        S = ex.newton_sum(d)
        pr = ex.vpp_prior(d, CAL["chosen"]["h_vpp"])
        fR, _, _ = ex.solve(d, S, fc_smooth=CAL["chosen"]["fc_smooth"], prior=pr,
                            w_prior=CAL["chosen"]["w_prior"], w_torque=0.0)
        ref = sp_.inverse_dynamics(d["segs"], "R", d["grf"]["R"]["Fx"], d["grf"]["R"]["Fy"], d["grf"]["R"]["copx"])["hip"] / sp_.MASS
        est = sp_.inverse_dynamics(d["segs"], "R", fR, d["grf"]["R"]["Fy"], d["grf"]["R"]["copx"])["hip"] / sp_.MASS
        m = d["trim"] & (d["grf"]["R"]["Fy"] > sp_.F_CONTACT)
        bias.append(float(np.mean(est[m] - ref[m])))
    note(f"signed hip-moment bias over {len(bias)} trials: mean {np.mean(bias):+.3f} Nm/kg "
         f"(sd {np.std(bias):.3f})")
    check("the hip bias is small compared with the reported RMSE", abs(np.mean(bias)) < 0.15,
          f"bias {np.mean(bias):+.3f} vs RMSE {tr.SoleID_hip.mean():.3f} Nm/kg")


if __name__ == "__main__":
    audit_protocol()
    audit_files()
    sanity_fukuchi()
    sanity_camargo()
    sanity_wang()
    investigate_hip_correlation()
    investigate_cop_shift()
    bad = [n for n, ok in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} checks passed")
    if bad:
        print("FAILED:")
        for n in bad:
            print("  -", n)
        sys.exit(1)

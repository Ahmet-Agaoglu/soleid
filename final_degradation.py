"""Final degradation analysis on the TEST subjects with the frozen SoleID v1.0 parameters.

Each perturbation is applied to the inputs of the estimators; the reference (clean measured GRF +
clean kinematics -> inverse dynamics) never changes. One 'wearable' scenario combines realistic
levels of all perturbations.

Output: <OUT>/final_degradation_trials.csv (per trial x condition) and a printed summary.
"""
import json
import os
import sys
import time
import zlib

import numpy as np
import pandas as pd

import experiments as ex
import soleid_planar as sp_
from protocol import CAL_SUBJECTS, available_trials

OUT = os.environ.get("SOLEID_OUT", sp_.OUT)
os.makedirs(OUT, exist_ok=True)
CAL = json.load(open(os.path.join(sp_.OUT, "calibration.json")))["chosen"]

CONDITIONS = [("clean", None, 0)]
CONDITIONS += [("marker_noise_mm", "marker_noise_mm", lv) for lv in (5, 10, 20)]
CONDITIONS += [("kin_bias_mm", "kin_bias_mm", lv) for lv in (10, 20, 30)]
CONDITIONS += [("marker_fps", "marker_fps", lv) for lv in (60, 30)]
CONDITIONS += [("fy_scale_pct", "fy_scale_pct", lv) for lv in (5, 8, 12, -8)]
CONDITIONS += [("fy_noise_pctBW", "fy_noise_pctBW", lv) for lv in (2, 5)]
CONDITIONS += [("cop_shift_mm", "cop_shift_mm", lv) for lv in (5, 10, 15)]
CONDITIONS += [("sync_frames", "sync_frames", lv) for lv in (1, 2, 3, -2)]
CONDITIONS += [("wearable", "wearable", 1)]


def degrade(d, kind, level, rng):
    e = dict(d)
    e["grf"] = {s: dict(d["grf"][s]) for s in ("R", "L")}
    n = d["n"]

    def with_markers(P):
        e["P"] = P
        e["segs"] = sp_.build_segments(P, n)
        e["pelvis"] = sp_.pelvis_centroid(P, n)

    if kind == "marker_noise_mm":
        P = {k: (sp_.lowpass(v + rng.normal(0, level / 1000.0, v.shape), 6.0, sp_.FS) if isinstance(v, np.ndarray) else v)
             for k, v in d["P"].items()}
        with_markers(P)
    elif kind == "kin_bias_mm":     # slowly varying (0.5 Hz) systematic position error, per marker
        P = {}
        for k, v in d["P"].items():
            if isinstance(v, np.ndarray):
                w = rng.normal(0, 1.0, v.shape)
                slow = sp_.lowpass(w, 0.5, sp_.FS)
                slow *= (level / 1000.0) / (slow.std(axis=0, keepdims=True) + 1e-12)
                P[k] = v + slow
            else:
                P[k] = v
        with_markers(P)
    elif kind == "marker_fps":
        P = {}
        t = np.arange(n) / sp_.FS
        tl = np.arange(0, t[-1], 1.0 / level)
        for k, v in d["P"].items():
            if isinstance(v, np.ndarray):
                low = np.column_stack([np.interp(tl, t, v[:, i]) for i in range(v.shape[1])])
                back = np.column_stack([np.interp(t, tl, low[:, i]) for i in range(v.shape[1])])
                P[k] = sp_.lowpass(back, min(6.0, level / 2.5), sp_.FS)
            else:
                P[k] = v
        with_markers(P)
    elif kind == "fy_scale_pct":
        g = 1 + level / 100.0
        for s in ("R", "L"):
            Fy = d["grf"][s]["Fy"]
            e["grf"][s]["Fy"] = np.where(Fy > sp_.F_CONTACT, Fy * g, 0.0)
    elif kind == "fy_noise_pctBW":
        for s in ("R", "L"):
            Fy = d["grf"][s]["Fy"]
            noisy = Fy + rng.normal(0, level / 100.0 * sp_.MASS * 9.81, n)
            noisy = sp_.lowpass(noisy, 20.0, sp_.FS)
            e["grf"][s]["Fy"] = np.where(Fy > sp_.F_CONTACT, np.maximum(noisy, sp_.F_CONTACT + 1), 0.0)
    elif kind == "cop_shift_mm":
        for s in ("R", "L"):
            e["grf"][s]["copx"] = d["grf"][s]["copx"] + level / 1000.0
    elif kind == "sync_frames":
        k = int(level)
        for s in ("R", "L"):
            for key in ("Fy", "copx"):
                e["grf"][s][key] = np.roll(d["grf"][s][key], k)
    elif kind == "wearable":        # realistic cheap system: all at once
        e = degrade(d, "marker_noise_mm", 10, rng)
        e = degrade(e, "kin_bias_mm", 20, rng)
        e = degrade(e, "marker_fps", 60, rng)
        e = degrade(e, "fy_scale_pct", 8, rng)
        e = degrade(e, "fy_noise_pctBW", 2, rng)
        e = degrade(e, "cop_shift_mm", 10, rng)
        e = degrade(e, "sync_frames", 1, rng)
    else:
        raise ValueError(kind)
    return e


def evaluate(d_in, d_clean):
    """Run all arms on d_in; metrics against the clean reference."""
    n = d_clean["n"]
    grf = d_in["grf"]
    S = ex.newton_sum(d_in)
    pr = ex.vpp_prior(d_in, CAL["h_vpp"])
    fR, _, _ = ex.solve(d_in, S, fc_smooth=CAL["fc_smooth"], prior=pr, w_prior=CAL["w_prior"], w_torque=0.0)
    vR, _, _ = ex.solve(d_in, S, fc_smooth=CAL["fc_smooth"], prior=None, w_prior=0.0, w_torque=0.0)
    bR, _ = sp_.newton_baseline(S, grf["R"]["Fy"], grf["L"]["Fy"])
    kin, _ = sp_.kinematics_only_baseline(d_in["P"], d_in["segs"], n)

    FxR, FyR_c, FyL_c, cR_c = d_clean["grf"]["R"]["Fx"], d_clean["grf"]["R"]["Fy"], d_clean["grf"]["L"]["Fy"], d_clean["grf"]["R"]["copx"]
    onR, onL, tr = FyR_c > sp_.F_CONTACT, FyL_c > sp_.F_CONTACT, d_clean["trim"]
    BW = sp_.MASS * 9.81
    ref = sp_.inverse_dynamics(d_clean["segs"], "R", FxR, FyR_c, cR_c)
    hs = sp_.heel_strikes(FyR_c)
    hs = hs[(hs > 150) & (hs < n - 150)]
    cm = sp_.stride_normalize(FxR, hs)
    out = {}
    arms = (("SoleID", fR, grf["R"]["Fy"], grf["R"]["copx"]),
            ("SoleID_noVPP", vR, grf["R"]["Fy"], grf["R"]["copx"]),
            ("Newton_prop", bR, grf["R"]["Fy"], grf["R"]["copx"]),
            ("Kin_only", kin["R"]["Fx"], kin["R"]["Fy"], kin["R"]["copx"]))
    for name, est, Fy_used, cop_used in arms:
        m = sp_.inverse_dynamics(d_in["segs"], "R", est, Fy_used, cop_used)
        ce = sp_.stride_normalize(est, hs)
        out[name] = dict(
            stance=100 * np.sqrt(np.mean((est[onR & tr] - FxR[onR & tr]) ** 2)) / BW,
            ds=100 * np.sqrt(np.mean((est[onR & onL & tr] - FxR[onR & onL & tr]) ** 2)) / BW,
            hip=float(np.sqrt(np.mean((m["hip"][tr] - ref["hip"][tr]) ** 2))) / sp_.MASS,
            knee=float(np.sqrt(np.mean((m["knee"][tr] - ref["knee"][tr]) ** 2))) / sp_.MASS,
            ankle=float(np.sqrt(np.mean((m["ankle"][tr] - ref["ankle"][tr]) ** 2))) / sp_.MASS,
            prop_peak=float(np.mean(ce.max(1) - cm.max(1))) if len(ce) else np.nan)
    return out


def main():
    trials = [t for t in available_trials() if t[0] not in CAL_SUBJECTS]
    rows = []
    t0 = time.time()
    for i, (subj, trial, speed, age) in enumerate(trials):
        try:
            d = ex.prepare(subj, trial)
        except sp_.MarkerGapError:
            continue
        # the same seed in every run; Python's hash() of a string changes from one process to the next
        rng = np.random.default_rng(zlib.crc32(f"{subj}/{trial}".encode()))
        sp_.configure(subj, trial)
        for cname, kind, lv in CONDITIONS:
            dd = d if kind is None else degrade(d, kind, lv, rng)
            res = evaluate(dd, d)
            for arm, m in res.items():
                rows.append(dict(subject=subj, trial=trial, speed=speed, age=age, condition=cname, level=lv, arm=arm, **m))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(trials)}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(OUT, "final_degradation_trials.partial.csv"), index=False)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "final_degradation_trials.csv"), index=False)
    summarize(df)


def summarize(df):
    pd.set_option("display.width", 220)
    g = df.groupby(["condition", "level", "arm"])[["stance", "ds", "hip", "knee", "prop_peak"]].mean()
    for key in ("stance", "hip"):
        piv = g[key].unstack("arm")[["Kin_only", "Newton_prop", "SoleID_noVPP", "SoleID"]]
        print(f"\n== {key} (mean over test trials) ==")
        print(piv.round(3).to_string())
    n_tr = df.drop_duplicates(["subject", "trial"]).shape[0]
    print(f"\ntrials: {n_tr}, subjects: {df.subject.nunique()}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        summarize(pd.read_csv(os.path.join(OUT, "final_degradation_trials.csv")))
    else:
        main()

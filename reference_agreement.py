"""How far is our reference from each laboratory's own, and how does the estimate score against
theirs?

Every joint-moment error in the paper is measured against a planar reference: the measured force
run through the same planar inverse dynamics as the estimate. That makes the comparison between
arms clean, because modeling error common to both cancels, but it also means the errors describe
the effect of substituting the estimated shear, not the accuracy of the moments themselves. The
remedy is to bring in the reference each laboratory computed itself, with its own model:

  Fukuchi   Visual3D sagittal moments, one stride-averaged curve per trial   (N m / kg)
  Camargo   OpenSim inverse dynamics, per frame                              (N m)
  Wang      OpenSim inverse dynamics, per frame, from the laboratory markers (N m); and the
            authors' own wearable result from IMU kinematics and insoles     (N m)

For each data set and joint we report, with windows matched:
  1. our planar reference against the laboratory reference         (how far apart the references are)
  2. SoleID against the laboratory reference                        (the estimate on their scale)
  3. SoleID against our planar reference                            (the number already in the paper)
and for Wang the full square: {SoleID, the published wearable pipeline} x {planar, laboratory}.

Sign. The laboratory moments are generalized forces about anatomical coordinates, ours are about
the lab z axis, so the conventions differ by joint. The sign is fixed once per data set and joint,
as the sign of the median per-trial correlation, and applied to every trial. Choosing it per trial
to maximize each trial's correlation -- which the earlier cross-check did -- flatters the
agreement, and the share of trials whose own correlation disagrees with the fixed sign is reported
instead.

Output: results/reference_agreement_<dataset>.csv, results/reference_agreement.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

import experiments as ex
import soleid_planar as sp_

CAL = json.load(open(os.path.join(sp_.OUT, "calibration.json")))["chosen"]
JOINTS = ("ankle", "knee", "hip")


def soleid_R(d):
    S = ex.newton_sum(d, 1.0)
    pr = ex.vpp_prior(d, CAL["h_vpp"])
    fR, _, _ = ex.solve(d, S, fc_smooth=CAL["fc_smooth"], prior=pr, w_prior=CAL["w_prior"],
                        w_torque=0.0)
    return fR


# --------------------------------------------------------------------------------- Fukuchi
KNT = dict(ankle="RAnkleMomentZ", knee="RKneeMomentZ", hip="RHipMomentZ")


def fukuchi_trial(subj, trial):
    d = ex.prepare(subj, trial)
    sp_.configure(subj, trial)
    _, _, knt = sp_.load()
    g = d["grf"]["R"]
    fR = soleid_R(d)
    ref = sp_.inverse_dynamics(d["segs"], "R", g["Fx"], g["Fy"], g["copx"])
    est = sp_.inverse_dynamics(d["segs"], "R", fR, g["Fy"], g["copx"])
    n = d["n"]
    hs = sp_.heel_strikes(g["Fy"])
    hs = hs[(hs > int(sp_.FS)) & (hs < n - int(sp_.FS))]
    row = dict(subject=subj, trial=trial)
    for j in JOINTS:
        # the laboratory curve is one stride-averaged waveform, so ours is averaged the same way
        row[f"{j}_planar"] = sp_.stride_normalize(ref[j] / sp_.MASS, hs).mean(axis=0)
        row[f"{j}_soleid"] = sp_.stride_normalize(est[j] / sp_.MASS, hs).mean(axis=0)
        row[f"{j}_lab"] = knt[KNT[j]].values.astype(float)
    return row


# --------------------------------------------------------------------------------- Camargo
def camargo_trial(subj, trial):
    import camargo_adapter as ca
    import camargo_run as cr
    d = ca.load_trial(subj, trial, "treadmill")
    n = d["n"]
    mass, height = float(cr.INFO.loc[subj, "Weight"]), float(cr.INFO.loc[subj, "Height"])
    sp_.MASS, sp_.HEIGHT, sp_.FS = mass, height, ca.FS
    segs, pel = ca.build_segments(d["P"], n, mass, height)
    grf = d["grf"]
    lab = pd.read_csv(os.path.join(ca.CSV, subj, "treadmill", "id", trial + ".csv"))
    n = min(n, len(lab))
    trim = np.zeros(d["n"], bool)
    trim[int(ca.FS):d["n"] - int(ca.FS)] = True
    trim &= d["speed"] > cr.MIN_SPEED
    dd = dict(n=d["n"], segs=segs, pelvis=pel, grf=grf, trim=trim)
    fR = soleid_R(dd)
    ref = sp_.inverse_dynamics(segs, "R", grf["R"]["Fx"], grf["R"]["Fy"], grf["R"]["copx"])
    est = sp_.inverse_dynamics(segs, "R", fR, grf["R"]["Fy"], grf["R"]["copx"])
    cols = dict(ankle="ankle_angle_r_moment", knee="knee_angle_r_moment", hip="hip_flexion_r_moment")
    w = trim[:n]
    row = dict(subject=subj, trial=trial)
    for j in JOINTS:
        row[f"{j}_planar"] = ref[j][:n][w] / mass
        row[f"{j}_soleid"] = est[j][:n][w] / mass
        row[f"{j}_lab"] = lab[cols[j]].values[:n][w] / mass
    return row


# --------------------------------------------------------------------------------- Wang
def wang_trial(subj, trial, info):
    import wang_adapter as wa
    import wang_run as wr
    d = wa.load_trial(subj, trial)
    mass, height = info[subj]["mass"], info[subj]["height"]
    n = len(d["t"])
    sp_.MASS, sp_.HEIGHT, sp_.FS = mass, height, wr.FS
    trc = wa.read_trc(d["paths"]["trc"])
    n = min(n, len(trc[0]))
    Plab, pelvis_lab, lab_ok = wr.planar_from_markers(trc, n)
    if not lab_ok:
        raise ValueError("marker gaps")
    Plab = {k: (sp_.lowpass(v, 6.0, wr.FS) if isinstance(v, np.ndarray) else v)
            for k, v in Plab.items()}
    pelvis_lab = sp_.lowpass(pelvis_lab, 6.0, wr.FS)
    Plab["R.Hip"] = Plab["L.Hip"] = wr.hip_from_pelvis_planar(pelvis_lab, height)
    segs_lab = wr.build(Plab, mass, height, n)

    onR0 = d["ref"]["R"]["Fy"][:n] > sp_.F_CONTACT
    onL0 = d["ref"]["L"]["Fy"][:n] > sp_.F_CONTACT
    Fy_tot = np.where(onR0, d["ref"]["R"]["Fy"][:n], 0) + np.where(onL0, d["ref"]["L"]["Fy"][:n], 0)
    k = wa.xsens_planar(subj, trial, n, Fy_total=Fy_tot, mass=mass)
    n = min(n, k["n"])
    Pw = {"R.Hip": k["pos"]["RightUpperLeg"], "R.Knee": k["pos"]["RightLowerLeg"],
          "R.Ankle": k["pos"]["RightFoot"], "R.MT5": k["pos"]["RightToe"],
          "L.Hip": k["pos"]["LeftUpperLeg"], "L.Knee": k["pos"]["LeftLowerLeg"],
          "L.Ankle": k["pos"]["LeftFoot"], "L.MT5": k["pos"]["LeftToe"],
          "Pelvis": k["pos"]["Pelvis"]}
    Pw = {key: sp_.lowpass(v[:n], 6.0, wr.FS) for key, v in Pw.items()}
    segs_w = wr.build(Pw, mass, height, n)
    grf_ref = {s: {kk: vv[:n] for kk, vv in d["ref"][s].items()} for s in ("R", "L")}
    ins = {s: dict(Fy=d["insole"][s]["Fy"][:n],
                   copx=wr.insole_cop_to_lab(Pw, s, d["insole"][s]["copx"], n)) for s in ("R", "L")}
    trim = np.zeros(n, bool)
    trim[int(wr.FS):n - int(wr.FS)] = True
    dd = dict(n=n, segs=segs_w, pelvis=Pw["Pelvis"], grf=ins, trim=trim)
    S = mass * sp_.lowpass(k["com_acc"][:n, 0], 10.0, wr.FS)
    pr = ex.vpp_prior(dd, CAL["h_vpp"])
    fR, _, _ = ex.solve(dd, S, fc_smooth=CAL["fc_smooth"], prior=pr, w_prior=CAL["w_prior"],
                        w_torque=0.0)
    est = sp_.inverse_dynamics(segs_w, "R", fR, ins["R"]["Fy"], ins["R"]["copx"])
    ref = sp_.inverse_dynamics(segs_lab, "R", grf_ref["R"]["Fx"], grf_ref["R"]["Fy"],
                               grf_ref["R"]["copx"])
    m = min(n, len(d["id_lab"]), len(d["id_wear"]))
    w = trim[:m]
    cols = dict(ankle="ankle_angle_r_moment", knee="knee_angle_r_moment", hip="hip_flexion_r_moment")
    row = dict(subject=subj, trial=trial)
    for j in JOINTS:
        row[f"{j}_planar"] = ref[j][:m][w] / mass
        row[f"{j}_soleid"] = est[j][:m][w] / mass
        row[f"{j}_lab"] = d["id_lab"][cols[j]].values[:m][w] / mass
        row[f"{j}_wangwear"] = d["id_wear"][cols[j]].values[:m][w] / mass
    return row


# --------------------------------------------------------------------------------- scoring
def fixed_signs(rows, a="planar", b="lab"):
    """One sign per joint: the sign of the median per-trial correlation between our reference and
    the laboratory's. Pooling the samples first is tempting and wrong -- between-trial differences
    in level dominate the pooled correlation and can reverse it even when every trial agrees
    (it did on the controlled layer, at the knee and the hip)."""
    out = {}
    for j in JOINTS:
        rs = []
        for r in rows:
            x, y = r[f"{j}_{a}"], r[f"{j}_{b}"]
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() > 10:
                rs.append(np.corrcoef(x[ok], y[ok])[0, 1])
        out[j] = 1.0 if np.median(rs) >= 0 else -1.0
    return out


def score(rows, a, b, sign):
    """Per-trial RMSE, bias (a - b after the sign is applied) and r; then means."""
    res = {}
    for j in JOINTS:
        rm, bi, rr, disagree = [], [], [], 0
        for r in rows:
            x, y = sign[j] * r[f"{j}_{a}"], r[f"{j}_{b}"]
            ok = np.isfinite(x) & np.isfinite(y)
            x, y = x[ok], y[ok]
            if len(x) < 10:
                continue
            c = np.corrcoef(x, y)[0, 1]
            rm.append(np.sqrt(np.mean((x - y) ** 2)))
            bi.append(np.mean(x - y))
            rr.append(c)
            disagree += c < 0
        res[j] = dict(rmse=float(np.mean(rm)), bias=float(np.mean(bi)), r=float(np.mean(rr)),
                      n=len(rm), sign=sign[j], trials_against_sign=int(disagree))
    return res


def trials_for(ds):
    if ds == "fukuchi":
        from protocol import CAL_SUBJECTS, available_trials
        return [(s, t) for s, t, *_ in available_trials() if s not in CAL_SUBJECTS]
    if ds == "camargo":
        import camargo_adapter as ca
        return [(s, t) for s in ca.subjects() for t in ca.trials(s)]
    if ds == "wang":
        done = pd.read_csv(os.path.join(sp_.OUT, "wang_application.csv"))
        return list(zip(done.subject, done.trial))


def run(ds, limit=None):
    trials = trials_for(ds)[:limit] if limit else trials_for(ds)
    info = None
    if ds == "wang":
        import wang_adapter as wa
        info = wa.subject_info()
    rows = []
    for i, (s, t) in enumerate(trials):
        try:
            if ds == "fukuchi":
                rows.append(fukuchi_trial(s, t))
            elif ds == "camargo":
                rows.append(camargo_trial(s, t))
            else:
                rows.append(wang_trial(s, t, info))
        except Exception as e:
            print(f"  skip {s} {t}: {e!r}", flush=True)
        if (i + 1) % 25 == 0:
            print(f"  {ds} {i+1}/{len(trials)}", flush=True)
    np.save(os.path.join(sp_.OUT, f"reference_agreement_{ds}_curves.npy"),
            np.array(rows, dtype=object), allow_pickle=True)      # re-scoring needs no re-run
    sign = fixed_signs(rows)
    out = dict(n_trials=len(rows), n_subjects=len({r["subject"] for r in rows}), sign=sign,
               planar_vs_lab=score(rows, "planar", "lab", sign),
               soleid_vs_lab=score(rows, "soleid", "lab", sign),
               soleid_vs_planar=score(rows, "soleid", "planar", {j: 1.0 for j in JOINTS}))
    if ds == "wang":
        one = {j: 1.0 for j in JOINTS}
        # the published wearable result is an OpenSim output like the lab reference: same convention
        out["wangwear_vs_lab"] = score(rows, "wangwear", "lab", one)
        # against our planar reference it is mapped into the planar convention with the same
        # fixed sign, so its bias reads like SoleID's (pipeline minus reference)
        out["wangwear_vs_planar"] = score(rows, "wangwear", "planar", sign)
    path = os.path.join(sp_.OUT, "reference_agreement.json")
    allres = json.load(open(path)) if os.path.exists(path) else {}
    allres[ds] = out
    json.dump(allres, open(path, "w"), indent=1)
    report(ds, out)
    return out


def report(ds, out):
    print(f"\n{ds}: {out['n_trials']} trials, {out['n_subjects']} subjects; "
          f"fixed signs {out['sign']}")
    keys = [k for k in out if k.endswith(("_lab", "_planar"))]
    for key in keys:
        print(f"  {key}")
        for j in JOINTS:
            r = out[key][j]
            print(f"    {j:6s} RMSE {r['rmse']:.3f}  bias {r['bias']:+.3f}  r {r['r']:.3f}  "
                  f"(n {r['n']}, against the fixed sign: {r['trials_against_sign']})")


if __name__ == "__main__":
    ds = sys.argv[1]
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run(ds, lim)

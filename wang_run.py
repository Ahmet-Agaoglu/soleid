"""Run SoleID (frozen v1.0) on the Wang et al. 2023 wearable dataset — application layer.

Arms
  REF        lab kinematics (.trc markers) + treadmill 3D GRF -> planar inverse dynamics (reference)
  SoleID     Xsens kinematics + insole vertical force & COP   -> full wearable chain
  SoleID_lab lab kinematics + insole vertical force & COP     -> isolates the kinematics source
  Newton     Xsens kinematics + insole vertical force, proportional split in double support
  NoShear    Xsens kinematics + insole vertical force, shear = 0 (what Wang et al. did)

Frozen parameters come from results/calibration.json (Fukuchi calibration subjects); nothing is
tuned on this dataset.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

import soleid_planar as sp_
import experiments as ex
import wang_adapter as wa

CAL = json.load(open(os.path.join(sp_.OUT, "calibration.json")))["chosen"]
FS = 100.0
MARK = dict(hip=None, knee=("RLFE", "LLFE"), ankle=("RLM", "LLM"), toe=("R5MT", "L5MT"),
            heel=("RCAL", "LCAL"), asis=("RASI", "LASI"), sacr="SACR")


def planar_from_markers(trc, n, max_gap=0.05):
    """Planar positions from the lab markers, in the OpenSim/XLD frame (x forward, y up).
    Short marker gaps are interpolated; the flag reports whether every required marker stayed
    below max_gap missing samples."""
    t, M = trc
    state = {"ok": True}

    def xy(k):
        a = np.column_stack([M[k][:n, 0], M[k][:n, 1]])
        bad = np.isnan(a[:, 0])
        if bad.any():
            if bad.mean() > max_gap:
                state["ok"] = False
            a = pd.DataFrame(a).interpolate(limit_direction="both").values
        return a
    P = {}
    for side, (kn, an, to, he) in (("R", ("RLFE", "RLM", "R5MT", "RCAL")),
                                   ("L", ("LLFE", "LLM", "L5MT", "LCAL"))):
        P[side + ".Knee"], P[side + ".Ankle"] = xy(kn), xy(an)
        P[side + ".MT5"], P[side + ".Heel"] = xy(to), xy(he)
    # pelvis landmarks: ASIS pair + sacrum (no PSIS markers in this set)
    P["R.ASIS"], P["L.ASIS"] = xy("RASI"), xy("LASI")
    P["R.PSIS"] = P["L.PSIS"] = xy("SACR")
    pelvis = 0.25 * (P["R.ASIS"] + P["L.ASIS"] + P["R.PSIS"] + P["L.PSIS"])
    # hip joint centres: Harrington-like offset below/behind the pelvis centroid is not available
    # in 2-D from these markers; use the ASIS/sacrum centroid shifted down by 9 % of height later.
    P["pelvis_pts"] = ["R.ASIS", "L.ASIS", "R.PSIS", "L.PSIS"]
    return P, pelvis, state["ok"]


def hip_from_pelvis_planar(pelvis, height):
    """Planar hip joint centre below the pelvis landmark centroid. Harrington (2007):
    vertical offset = -(0.30 PW + 10.9 mm); with PW ~ 0.138 height this is ~ 0.048 height."""
    out = pelvis.copy()
    out[:, 1] -= 0.048 * height
    return out


def build(P_joints, mass, height, n):
    """Build the 7-segment planar model dict used by soleid_planar.inverse_dynamics."""
    sp_.MASS, sp_.HEIGHT, sp_.FS = mass, height, FS
    segs = {}
    for s in ("R", "L"):
        hip, knee, ank, toe = (P_joints[s + k][:n] for k in (".Hip", ".Knee", ".Ankle", ".MT5"))
        for name, prox, dist in (("foot", ank, toe), ("shank", knee, ank), ("thigh", hip, knee)):
            par = sp_.SEG[name]
            L = np.linalg.norm(dist - prox, axis=1).mean()
            m = par["m"] * mass
            com = prox + par["c"] * (dist - prox)
            _, acom = sp_.deriv(com, FS)
            th = np.unwrap(np.arctan2((dist - prox)[:, 1], (dist - prox)[:, 0]))
            _, alpha = sp_.deriv(th, FS)
            segs[(s, name)] = dict(m=m, I=m * (par["rg"] * L) ** 2, com=com, acom=acom,
                                   alpha=alpha, prox=prox, dist=dist, L=L)
    midhip = 0.5 * (P_joints["R.Hip"][:n] + P_joints["L.Hip"][:n])
    L_hat = 0.288 * height
    com = midhip + np.array([0.0, sp_.SEG["hat"]["c"] * L_hat])
    _, acom = sp_.deriv(com, FS)
    m = sp_.SEG["hat"]["m"] * mass
    segs[("B", "hat")] = dict(m=m, I=m * (sp_.SEG["hat"]["rg"] * L_hat) ** 2, com=com, acom=acom,
                              alpha=np.zeros(n), prox=midhip, dist=com, L=L_hat)
    return segs


def insole_cop_to_lab(P_joints, side, cop_local, n, heel_key=None):
    """The insole COP is expressed in the foot (calcaneus) frame: a distance from the heel end
    along the foot. Place it in the lab frame on the heel->toe line.
    heel_key: a measured heel marker, else the heel is estimated from ankle and toe."""
    ank, toe = P_joints[side + ".Ankle"][:n], P_joints[side + ".MT5"][:n]
    if heel_key is not None and heel_key in P_joints:
        heel = P_joints[heel_key][:n]
    else:                                   # ankle sits ~25 % of foot length ahead of the heel
        v = toe - ank
        L = np.linalg.norm(v, axis=1, keepdims=True)
        u = v / np.maximum(L, 1e-9)
        heel = ank - 0.25 * (L / 0.75) * u
    v = toe - heel
    u = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
    return heel[:, 0] + cop_local[:n] * u[:, 0]


def align_x(P_joints, grf, n, side_pref=("R", "L")):
    """Rigid horizontal offset between the Xsens frame and the insole/OpenSim COP frame:
    match the mean 'under-foot' position during stance to the mean measured COP."""
    num, den = 0.0, 0
    for s in side_pref:
        on = grf[s]["Fy"][:n] > sp_.F_CONTACT
        if on.sum() < 50:
            continue
        under = 0.5 * (P_joints[s + ".Ankle"][:n, 0] + P_joints[s + ".MT5"][:n, 0])
        num += np.mean(grf[s]["copx"][:n][on] - under[on]) * on.sum()
        den += on.sum()
    dx = num / max(den, 1)
    for k, v in P_joints.items():
        if isinstance(v, np.ndarray):
            v[:, 0] += dx
    return dx


def run_trial(subj, trial, info):
    d = wa.load_trial(subj, trial)
    mass, height = info[subj]["mass"], info[subj]["height"]
    n = len(d["t"])
    sp_.MASS, sp_.HEIGHT, sp_.FS = mass, height, FS
    BW = mass * 9.81

    # ---------- lab kinematics
    trc = wa.read_trc(d["paths"]["trc"])
    n = min(n, len(trc[0]))
    Plab, pelvis_lab, lab_ok = planar_from_markers(trc, n)
    Plab = {k: (sp_.lowpass(v, 6.0, FS) if isinstance(v, np.ndarray) else v) for k, v in Plab.items()}
    pelvis_lab = sp_.lowpass(pelvis_lab, 6.0, FS)
    hip_lab = hip_from_pelvis_planar(pelvis_lab, height)
    Plab["R.Hip"] = Plab["L.Hip"] = hip_lab
    segs_lab = build(Plab, mass, height, n)

    # ---------- wearable kinematics (Xsens), resampled and time-aligned to the force files
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
    Pw = {key: sp_.lowpass(v[:n], 6.0, FS) for key, v in Pw.items()}
    grf_ins = {s: {kk: vv[:n] for kk, vv in d["insole"][s].items()} for s in ("R", "L")}
    grf_ref = {s: {kk: vv[:n] for kk, vv in d["ref"][s].items()} for s in ("R", "L")}
    segs_w = build(Pw, mass, height, n)
    pelvis_w = Pw["Pelvis"]
    # The insole COP is expressed in the foot frame; it is placed on the heel->toe line of the
    # SAME kinematics source the arm uses, so each arm is self-consistent and no rigid alignment
    # between the Xsens and lab frames is needed (the VPP prior only sees pelvis - COP).
    ins_w = {s: dict(Fy=grf_ins[s]["Fy"],
                     copx=insole_cop_to_lab(Pw, s, grf_ins[s]["copx"], n)) for s in ("R", "L")}
    ins_lab = {s: dict(Fy=grf_ins[s]["Fy"],
                       copx=insole_cop_to_lab(Plab, s, grf_ins[s]["copx"], n, heel_key=s + ".Heel"))
               for s in ("R", "L")}
    dx = 0.0

    # trim filter edges and the first/last second
    trim = np.zeros(n, bool)
    trim[int(FS):n - int(FS)] = True
    onR = grf_ref["R"]["Fy"][:n] > sp_.F_CONTACT
    onL = grf_ref["L"]["Fy"][:n] > sp_.F_CONTACT
    ds = onR & onL

    # ---------- reference: lab kinematics + treadmill 3D GRF
    for key in ("R", "L"):
        for kk in ("Fx", "Fy", "copx"):
            grf_ref[key][kk] = grf_ref[key][kk][:n]
    ref = {s: sp_.inverse_dynamics(segs_lab, s, grf_ref[s]["Fx"], grf_ref[s]["Fy"], grf_ref[s]["copx"])
           for s in ("R", "L")}

    # ---------- arms
    def qp(segs, pelvis, grf, S=None):
        dd = dict(n=n, segs=segs, pelvis=pelvis, grf=grf, trim=trim)
        S = ex.newton_sum(dd) if S is None else S
        pr = ex.vpp_prior(dd, CAL["h_vpp"])
        fR, fL, _ = ex.solve(dd, S, fc_smooth=CAL["fc_smooth"], prior=pr, w_prior=CAL["w_prior"], w_torque=0.0)
        return S, fR, fL

    # wearable whole-body horizontal force: Xsens COM acceleration (sensor-derived, no drift)
    S_xsens = mass * sp_.lowpass(k["com_acc"][:n, 0], 10.0, FS)
    S_w, sR, sL = qp(segs_w, pelvis_w, ins_w, S=S_xsens)
    S_l, lR, lL = qp(segs_lab, pelvis_lab, ins_lab)
    bR, bL = sp_.newton_baseline(S_w, ins_w["R"]["Fy"], ins_w["L"]["Fy"])
    zR = np.zeros(n)

    arms = {
        "SoleID": (segs_w, sR, ins_w["R"]),
        "SoleID_lab_kin": (segs_lab, lR, ins_lab["R"]),
        "Newton_prop": (segs_w, bR, ins_w["R"]),
        "NoShear": (segs_w, zR, ins_w["R"]),
    }
    row = dict(subject=subj, trial=trial, speed=d["speed"], mass=mass, n=n, dx_align=dx,
               tau=k["tau"], r_align=k["r_align"], lab_ok=bool(lab_ok),
               fy_rmse_pctBW=100 * np.sqrt(np.mean((grf_ins["R"]["Fy"][trim] - grf_ref["R"]["Fy"][trim]) ** 2)) / BW,
               newton_check_pctBW=100 * np.sqrt(np.mean((S_w[trim] - (np.where(onR, grf_ref["R"]["Fx"], 0) +
                                                                     np.where(onL, grf_ref["L"]["Fx"], 0))[trim]) ** 2)) / BW,
               newton_check_lab_pctBW=100 * np.sqrt(np.mean((S_l[trim] - (np.where(onR, grf_ref["R"]["Fx"], 0) +
                                                                         np.where(onL, grf_ref["L"]["Fx"], 0))[trim]) ** 2)) / BW)
    for name, (segs, est, g) in arms.items():
        row[f"{name}_stance"] = 100 * np.sqrt(np.mean((est[onR & trim] - grf_ref["R"]["Fx"][onR & trim]) ** 2)) / BW
        row[f"{name}_ds"] = 100 * np.sqrt(np.mean((est[ds & trim] - grf_ref["R"]["Fx"][ds & trim]) ** 2)) / BW
        m = sp_.inverse_dynamics(segs, "R", est, g["Fy"], g["copx"])
        for j in ("ankle", "knee", "hip"):
            a, b = m[j][trim] / mass, ref["R"][j][trim] / mass
            good = lab_ok and np.isfinite(a).all() and np.isfinite(b).all()
            row[f"{name}_{j}"] = float(np.sqrt(np.mean((a - b) ** 2))) if good else np.nan
            row[f"{name}_{j}_r"] = float(np.corrcoef(a, b)[0, 1]) if good else np.nan
    return row


def prepare_for_mhe(subj, trial, info):
    """Assemble the wearable-arm inputs of one Wang trial for the moving-horizon study:
    Xsens kinematics + insole vertical force/COP, plus the measured shear for scoring."""
    d = wa.load_trial(subj, trial)
    mass, height = info[subj]["mass"], info[subj]["height"]
    n = len(d["t"])
    sp_.MASS, sp_.HEIGHT, sp_.FS = mass, height, FS
    trc = wa.read_trc(d["paths"]["trc"])
    n = min(n, len(trc[0]))
    Plab, pelvis_lab, lab_ok = planar_from_markers(trc, n)
    Plab = {k: (sp_.lowpass(v, 6.0, FS) if isinstance(v, np.ndarray) else v) for k, v in Plab.items()}
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
    Pw = {key: sp_.lowpass(v[:n], 6.0, FS) for key, v in Pw.items()}
    ins_w = {s: dict(Fy=d["insole"][s]["Fy"][:n],
                     copx=insole_cop_to_lab(Pw, s, d["insole"][s]["copx"], n)) for s in ("R", "L")}
    ins_w["R"]["Fx"] = d["ref"]["R"]["Fx"][:n]        # measured shear, for scoring only
    ins_w["L"]["Fx"] = d["ref"]["L"]["Fx"][:n]
    segs = build(Pw, mass, height, n)
    trim = np.zeros(n, bool)
    trim[int(FS):n - int(FS)] = True
    S = mass * sp_.lowpass(k["com_acc"][:n, 0], 10.0, FS)
    dd = dict(n=n, segs=segs, pelvis=Pw["Pelvis"], grf=ins_w, trim=trim)
    return dd, S, f"wang:{subj}:{trial}"


def main(subjects=None, trials=None):
    info = wa.subject_info()
    subs = subjects or sorted([d for d in os.listdir(wa.ROOT) if d.startswith("Subj")])
    trs = trials or wa.WALKS
    rows = []
    for s in subs:
        for t in trs:
            try:
                rows.append(run_trial(s, t, info))
                print(f"  {s} {t}: SoleID stance {rows[-1]['SoleID_stance']:.2f} %BW, hip {rows[-1]['SoleID_hip']:.3f}", flush=True)
            except Exception as e:
                print(f"  {s} {t}: ERROR {e!r}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(sp_.OUT, "wang_application.csv"), index=False)
    summarize(df)
    return df


def summarize(df):
    pd.set_option("display.width", 220)
    arms = ["NoShear", "Newton_prop", "SoleID_lab_kin", "SoleID"]
    n_lab = int(df.lab_ok.sum()) if "lab_ok" in df else len(df)
    print(f"\nWANG APPLICATION: {df.subject.nunique()} subjects, {len(df)} trials, "
          f"{n_lab} with a complete marker set for the joint-moment reference")
    print("insole vertical force error vs treadmill: %.1f %%BW" % df.fy_rmse_pctBW.mean())
    print("whole-body Newton check (Xsens / lab kin): %.2f / %.2f %%BW" % (df.newton_check_pctBW.mean(), df.newton_check_lab_pctBW.mean()))
    for key in ("stance", "ds", "ankle", "knee", "hip"):
        cols = [f"{a}_{key}" for a in arms if f"{a}_{key}" in df]
        print(f"\n{key}: " + "  ".join(f"{a}={df[f'{a}_{key}'].mean():.3f}" for a in arms if f"{a}_{key}" in df))
        if key in ("ankle", "knee", "hip"):
            print("   r: " + "  ".join(f"{a}={df[f'{a}_{key}_r'].mean():.3f}" for a in arms))
        g = df.groupby("speed")[cols].mean().round(3)
        print(g.to_string())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        summarize(pd.read_csv(os.path.join(sp_.OUT, "wang_application.csv")))
    elif len(sys.argv) > 1 and sys.argv[1] == "one":
        info = wa.subject_info()
        r = run_trial("Subj04", "walk_36", info)
        for k, v in r.items():
            print(f"  {k}: {v}")
    else:
        main()

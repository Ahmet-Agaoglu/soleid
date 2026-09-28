"""Adapter for the Camargo et al. 2021 dataset (Mendeley) -> SoleID planar pipeline.

External validation set: different lab, different marker protocol, instrumented treadmill with a
continuously varying belt speed, plus ramp trials.

Conventions found in the data (verified, not assumed):
  markers   millimetres; x medio-lateral, y vertical, z fore-aft; walking towards -z
  forces    newtons at 1000 Hz; Treadmill_R_* / Treadmill_L_*; same lab frame, metres for the COP
  ->        planar frame used here: x = -z_marker (forward), y = y_marker (up)
Marker sampling is 200 Hz; forces are low-pass filtered at 20 Hz and decimated by 5.
"""
import os

import numpy as np
import pandas as pd

import soleid_planar as sp_
import config

CSV = config.CAMARGO_CSV
FS = 200.0

MARKERS = dict(knee=("R_Knee_Lat", "L_Knee_Lat"), ankle=("R_Ankle_Lat", "L_Ankle_Lat"),
               heel=("R_Heel", "L_Heel"), toe=("R_Toe_Tip", "L_Toe_Tip"),
               asis=("R_ASIS", "L_ASIS"), psis=("R_PSIS", "L_PSIS"))


def subjects():
    return sorted(d for d in os.listdir(CSV) if d.startswith("AB"))


def trials(subj, mode="treadmill"):
    p = os.path.join(CSV, subj, mode, "fp")
    if not os.path.isdir(p):
        return []
    out = []
    for f in sorted(os.listdir(p)):
        if not f.endswith(".csv"):
            continue
        if all(os.path.exists(os.path.join(CSV, subj, mode, s, f)) for s in ("markers", "conditions")):
            out.append(f[:-4])
    return out


def _xy(df, name):
    """Marker (N,2) in the planar frame: x forward (= -z_marker), y up, metres."""
    return np.column_stack([-df[name + "_z"].values, df[name + "_y"].values]) / 1000.0


def _lat(df, name):
    """Mediolateral marker coordinate (metres). Not part of the planar model; used only to decide
    which foot a force plate belongs to."""
    return df[name + "_x"].values / 1000.0


def load_trial(subj, trial, mode="treadmill", fc_marker=6.0, fc_force=20.0):
    base = os.path.join(CSV, subj, mode)
    mk = pd.read_csv(os.path.join(base, "markers", trial + ".csv"))
    fp = pd.read_csv(os.path.join(base, "fp", trial + ".csv"))
    cond = pd.read_csv(os.path.join(base, "conditions", trial + ".csv"))
    # forces: 1000 Hz -> 200 Hz
    step = int(round(np.median(np.diff(mk.Header.values)) / np.median(np.diff(fp.Header.values))))
    grf = {}
    for side, tag in (("R", "Treadmill_R"), ("L", "Treadmill_L")):
        Fx = sp_.lowpass(-fp[tag + "_vz"].values, fc_force, 1000.0)[::step]
        Fy = sp_.lowpass(fp[tag + "_vy"].values, fc_force, 1000.0)[::step]
        cop = -fp[tag + "_pz"].values[::step]
        grf[side] = dict(Fx=Fx, Fy=Fy, copx=cop)
    n = min(len(mk), min(len(v["Fx"]) for v in grf.values()))
    for s in grf:
        for k in grf[s]:
            grf[s][k] = grf[s][k][:n]
        grf[s]["copx"] = sp_.hold_cop(grf[s]["copx"], grf[s]["Fy"])
    P = {}
    for side in ("R", "L"):
        for key, (rn, ln) in MARKERS.items():
            nm = rn if side == "R" else ln
            if key in ("asis", "psis"):
                continue
            P[f"{side}.{ {'knee':'Knee','ankle':'Ankle','heel':'Heel','toe':'MT5'}[key] }"] = _xy(mk, nm)[:n]
    for side, (rn, ln) in (("R", MARKERS["asis"]), ("L", MARKERS["asis"])):
        pass
    P["R.ASIS"], P["L.ASIS"] = _xy(mk, "R_ASIS")[:n], _xy(mk, "L_ASIS")[:n]
    P["R.PSIS"], P["L.PSIS"] = _xy(mk, "R_PSIS")[:n], _xy(mk, "L_PSIS")[:n]
    gaps = {k: float(np.isnan(v[:, 0]).mean()) for k, v in P.items()}
    P = {k: sp_.lowpass(pd.DataFrame(v).interpolate(limit_direction="both").values, fc_marker, FS)
         for k, v in P.items()}
    P["pelvis_pts"] = ["R.ASIS", "L.ASIS", "R.PSIS", "L.PSIS"]
    speed = np.interp(np.arange(n) / FS, cond.Header.values - cond.Header.values[0], cond.Speed.values)
    return dict(subj=subj, trial=trial, mode=mode, n=n, P=P, grf=grf, speed=speed, gaps=gaps)


def hip_centres(P, n, height_m):
    """Planar hip joint centre: pelvis landmark centroid shifted down by 0.048 * height
    (Harrington 2007 vertical offset, as in the Wang adapter)."""
    pel = 0.25 * (P["R.ASIS"][:n] + P["L.ASIS"][:n] + P["R.PSIS"][:n] + P["L.PSIS"][:n])
    hip = pel.copy()
    hip[:, 1] -= 0.048 * height_m
    return pel, hip


def build_segments(P, n, mass, height):
    sp_.MASS, sp_.HEIGHT, sp_.FS = mass, height, FS
    pel, hip = hip_centres(P, n, height)
    P = dict(P)
    P["R.Hip"] = P["L.Hip"] = hip
    segs = {}
    for s in ("R", "L"):
        chain = (P[s + ".Hip"][:n], P[s + ".Knee"][:n], P[s + ".Ankle"][:n], P[s + ".MT5"][:n])
        for name, prox, dist in (("foot", chain[2], chain[3]), ("shank", chain[1], chain[2]),
                                 ("thigh", chain[0], chain[1])):
            par = sp_.SEG[name]
            L = np.linalg.norm(dist - prox, axis=1).mean()
            m = par["m"] * mass
            com = prox + par["c"] * (dist - prox)
            _, acom = sp_.deriv(com, FS)
            th = np.unwrap(np.arctan2((dist - prox)[:, 1], (dist - prox)[:, 0]))
            _, alpha = sp_.deriv(th, FS)
            segs[(s, name)] = dict(m=m, I=m * (par["rg"] * L) ** 2, com=com, acom=acom, alpha=alpha,
                                   prox=prox, dist=dist, L=L)
    L_hat = 0.288 * height
    com = pel + np.array([0.0, sp_.SEG["hat"]["c"] * L_hat])
    _, acom = sp_.deriv(com, FS)
    m = sp_.SEG["hat"]["m"] * mass
    segs[("B", "hat")] = dict(m=m, I=m * (sp_.SEG["hat"]["rg"] * L_hat) ** 2, com=com, acom=acom,
                              alpha=np.zeros(n), prox=pel, dist=com, L=L_hat)
    return segs, pel


def sanity(d, mass=None):
    """Frame checks that must hold if the conventions above are right."""
    n, grf, P = d["n"], d["grf"], d["P"]
    out = {}
    onR = grf["R"]["Fy"] > sp_.F_CONTACT
    heel, toe = P["R.Heel"][:n, 0], P["R.MT5"][:n, 0]
    cop = grf["R"]["copx"]
    lo, hi = np.minimum(heel, toe) - 0.05, np.maximum(heel, toe) + 0.05
    inside = ((cop >= lo) & (cop <= hi))[onR]
    out["cop_inside_foot_pct"] = 100 * inside.mean()
    out["toe_ahead_of_heel_pct"] = 100 * (toe > heel)[onR].mean()
    out["foot_len_m"] = float(np.mean(np.abs(toe - heel)[onR]))
    out["stance_pct"] = 100 * onR.mean()
    if mass:
        out["fy_peak_BW"] = float(grf["R"]["Fy"].max() / (mass * 9.81))
    return out


if __name__ == "__main__":
    subs = subjects()
    print("subjects converted so far:", len(subs), subs[:6])
    s = subs[0]
    tr = trials(s)
    print(f"{s}: {len(tr)} treadmill trials -> {tr[:4]}")
    d = load_trial(s, tr[0])
    print("n =", d["n"], " speed range %.2f..%.2f m/s" % (d["speed"].min(), d["speed"].max()))
    print("marker gaps:", {k: round(v, 3) for k, v in d["gaps"].items() if v > 0} or "none")
    print("sanity:", {k: round(v, 2) for k, v in sanity(d).items()})

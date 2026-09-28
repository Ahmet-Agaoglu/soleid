"""
SoleID — planar (sagittal) feasibility prototype on the Fukuchi 2018 WBDS dataset.

Question: given per-foot VERTICAL ground force + centre of pressure + marker kinematics,
how well can the horizontal (anterior-posterior) ground force and the sagittal joint
moments be recovered?

Arms
  A  reference : full measured GRF (Fx, Fy, COP)      -> planar inverse dynamics
  B  Newton    : Fx_total = sum(m a_x);  double support split proportional to Fy
  C  Kin-only  : no force measurement at all. Contact timing from foot-marker height, total
                 Fx and Fy from whole-body Newton, double-support split by a smooth-transition
                 assumption (Ren et al. 2008 style, cubic decay of the trailing foot), COP from
                 heel-to-toe progression. Emulates kinematics-only methods (OpenGRF, IMU+Moco).
  D1 SoleID v1 : Fx per foot = unknown input, QP with whole-body dynamics (soft),
                 temporal smoothness, friction cone, zero when foot in air
  D2 SoleID v2 : v1 + virtual-pivot-point (VPP) direction prior (force vector points from the
                 COP to a point h_vpp above the pelvis centroid) + joint-torque smoothness

Frame: X anterior, Y up (lab). Units: m, N, kg, s.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.signal import butter, filtfilt

import osqp
import config

DATA = config.WBDS
OUT = config.RESULTS
os.makedirs(OUT, exist_ok=True)

SUBJ = "WBDS01"
TRIAL = "walkT05"
MASS = 74.3      # kg   (WBDSinfo.csv) -- overwritten by configure()
HEIGHT = 1.725   # m
PLATE = {"L": "1", "R": "2"}   # plate -> foot mapping, set by configure()


def configure(subj, trial):
    """Set subject/trial globals from WBDSinfo.csv and detect plate-foot mapping."""
    global SUBJ, TRIAL, MASS, HEIGHT, PLATE
    SUBJ, TRIAL = subj, trial
    info = pd.read_csv(os.path.join(DATA, "WBDSinfo.csv"))
    row = info[info.FileName == f"{subj}{trial}grf.txt"].iloc[0]
    MASS, HEIGHT = float(row.Mass), float(row.Height) / 100.0
    mk = pd.read_csv(os.path.join(DATA, f"{subj}{trial}mkr.txt"), sep="	")
    gr = pd.read_csv(os.path.join(DATA, f"{subj}{trial}grf.txt"), sep="	")
    zR = mk["R.AnkleZ"].mean()
    z1 = gr.COPz1[gr.Fy1 > 20].mean()
    z2 = gr.COPz2[gr.Fy2 > 20].mean()
    PLATE = {"R": "1", "L": "2"} if abs(z1 - zR) < abs(z2 - zR) else {"R": "2", "L": "1"}
    return dict(mass=MASS, height=HEIGHT, plate=PLATE, speed=float(row["GaitSpeed(m/s)"]))
G = 9.81
FS = 150.0       # marker rate; GRF (300 Hz) is decimated to this
MU = 0.8         # friction cone
F_CONTACT = 20.0 # N, foot-on-ground threshold
F_COP = 50.0     # N, below this the plate COP is unreliable -> held from nearest reliable sample
HIP_FROM_PELVIS = True   # hip joint centre by Harrington 2007 regression (pelvis markers) instead of GTR
MAX_GAP_FRAC = 0.05      # trials with more than this fraction of NaN in a required marker are rejected
# SoleID (full) weights: set by the calibration protocol (protocol.py); defaults = prototype values
# defaults equal the frozen calibration (results/calibration.json); runners may override
V2_PARAMS = dict(w_prior=1.0, h_vpp=0.2, w_torque=0.0, fc_smooth=12.0)

# Winter (2009) anthropometric table: mass fraction, COM fraction from proximal, rg/L about COM
SEG = {
    "foot":  dict(m=0.0145, c=0.50, rg=0.475),
    "shank": dict(m=0.0465, c=0.433, rg=0.302),
    "thigh": dict(m=0.100, c=0.433, rg=0.323),
    "hat":   dict(m=0.678, c=0.626, rg=0.496),
}


# Acceptance band for the mean vertical force / body weight ratio. The population sits at
# 0.93 (5-95 %: 0.87-0.96), so the band is wide enough to keep it and narrow enough to
# reject a missing belt (~0.5) or a force/mass scale error (~1.85).
BW_RATIO_LO, BW_RATIO_HI = 0.70, 1.30


def bw_ratio(grf, mass, mask=None):
    """Mean measured vertical force over both feet, divided by body weight.

    Over complete strides the body does not accelerate vertically on average, so this ratio must
    be 1. Values near 0.5 mean only one foot was recorded, values far above 1 mean the mass or the
    force scale is wrong, and values a little below 1 mean part of the weight was carried
    elsewhere (a handrail)."""
    tot = None
    for s in ("R", "L"):
        Fy = grf[s]["Fy"]
        tot = np.where(Fy > F_CONTACT, Fy, 0.0) if tot is None else tot + np.where(Fy > F_CONTACT, Fy, 0.0)
    if mask is not None:
        tot = tot[mask]
    return float(np.mean(tot) / (mass * 9.81))


def lowpass(x, fc, fs, order=4):
    b, a = butter(order, fc / (fs / 2), btype="low")
    return filtfilt(b, a, x, axis=0)


def load():
    mk = pd.read_csv(os.path.join(DATA, f"{SUBJ}{TRIAL}mkr.txt"), sep="\t")
    gr = pd.read_csv(os.path.join(DATA, f"{SUBJ}{TRIAL}grf.txt"), sep="\t")
    knt = pd.read_csv(os.path.join(DATA, f"{SUBJ}{TRIAL}knt.txt"), sep="\t")
    return mk, gr, knt


class MarkerGapError(ValueError):
    pass


def _marker3d(mk, n):
    xyz = np.column_stack([mk[n + "X"].values, mk[n + "Y"].values, mk[n + "Z"].values]) / 1000.0
    if np.isnan(xyz).any():   # short gaps: linear interpolation; long gaps: reject trial
        frac = float(np.isnan(xyz[:, 0]).mean())
        if frac > MAX_GAP_FRAC:
            raise MarkerGapError(f"{n}: {100 * frac:.1f}% missing")
        xyz = pd.DataFrame(xyz).interpolate(limit_direction="both").values
    return lowpass(xyz, 6.0, FS)


def hip_joint_centres(mk):
    """Harrington et al. (2007) regression: HJC in the pelvis frame (origin mid-ASIS; x anterior,
    y superior, z right). PW = inter-ASIS distance, PD = mid-ASIS to mid-PSIS distance (in mm)."""
    ra, la, rp, lp = (_marker3d(mk, k) for k in ("R.ASIS", "L.ASIS", "R.PSIS", "L.PSIS"))
    o = 0.5 * (ra + la)
    mp = 0.5 * (rp + lp)
    z = ra - la
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    xt = o - mp
    y = np.cross(z, xt)
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    x = np.cross(y, z)
    PW = np.linalg.norm(ra - la, axis=1).mean() * 1000.0
    PD = np.linalg.norm(o - mp, axis=1).mean() * 1000.0
    xh = (-0.24 * PD - 9.9) / 1000.0
    yh = (-0.30 * PW - 10.9) / 1000.0
    zh = (0.33 * PW + 7.3) / 1000.0
    out = {}
    for side, sign in (("R", 1.0), ("L", -1.0)):
        out[side] = o + xh * x + yh * y + sign * zh * z
    return out


PELVIS_MARKERS = ("R.ASIS", "L.ASIS", "R.PSIS", "L.PSIS")
LEG_MARKERS = ("R.Knee", "R.Ankle", "R.Heel", "R.MT5", "L.Knee", "L.Ankle", "L.Heel", "L.MT5")


def planar_markers(mk):
    """Return dict of (N,2) arrays [x, y] in metres, low-pass filtered at 6 Hz.
    Hip joint centres: Harrington regression when all four pelvis markers are usable
    (P['hip_source'] = 'harrington'); otherwise fall back to the GTR markers ('gtr_fallback').
    P['pelvis_pts'] lists the usable pelvis markers (for the trunk / VPP reference point)."""
    P, missing = {}, {}
    for n in LEG_MARKERS + PELVIS_MARKERS + ("R.GTR", "L.GTR"):
        try:
            P[n] = _marker3d(mk, n)[:, :2]
        except MarkerGapError as e:
            missing[n] = str(e)
    for r in LEG_MARKERS:
        if r not in P:
            raise MarkerGapError(missing[r])
    pelvis_ok = all(k in P for k in PELVIS_MARKERS)
    if HIP_FROM_PELVIS and pelvis_ok:
        hjc = hip_joint_centres(mk)
        P["R.Hip"], P["L.Hip"] = hjc["R"][:, :2], hjc["L"][:, :2]
        P["hip_source"] = "harrington"
    else:
        for sd in ("R", "L"):
            if sd + ".GTR" not in P:
                raise MarkerGapError(f"no hip reference: {missing.get(sd + '.GTR', 'pelvis: ' + str(missing))}")
        P["R.Hip"], P["L.Hip"] = P["R.GTR"], P["L.GTR"]
        P["hip_source"] = "gtr_fallback"
    P["pelvis_pts"] = [k for k in PELVIS_MARKERS if k in P]
    return P


def pelvis_centroid(P, n):
    """Trunk / VPP reference point: centroid of usable pelvis markers, else mid-GTR."""
    pts = P["pelvis_pts"]
    if pts:
        return np.mean([P[k][:n] for k in pts], axis=0)
    return 0.5 * (P["R.GTR"][:n] + P["L.GTR"][:n])


def grf_to_marker_rate(gr):
    """Filter GRF at 20 Hz (300 Hz), then take every 2nd sample -> 150 Hz aligned with markers.
    Plate-foot mapping from configure() (checked against ankle marker Z)."""
    fs_g = 300.0
    out = {}
    for side, p in PLATE.items():
        Fx = lowpass(gr[f"Fx{p}"].values, 20.0, fs_g)
        Fy = lowpass(gr[f"Fy{p}"].values, 20.0, fs_g)
        copx = gr[f"COPx{p}"].values / 1000.0  # not filtered (only used when Fy>thr)
        copx = hold_cop(copx, Fy)
        out[side] = dict(Fx=Fx[::2], Fy=Fy[::2], copx=copx[::2])
    n = min(len(out["L"]["Fx"]), len(out["R"]["Fx"]))
    for s in out:
        for k in out[s]:
            out[s][k] = out[s][k][:n]
    return out, n


def hold_cop(copx, Fy):
    """Plate COP is unreliable at low vertical force (COP = M/F). Within each contact episode
    (Fy > F_CONTACT) replace samples with Fy < F_COP by the nearest reliable value."""
    copx = copx.copy()
    on = Fy > F_CONTACT
    good = Fy > F_COP
    d = np.diff(on.astype(int))
    starts, ends = list(np.where(d == 1)[0] + 1), list(np.where(d == -1)[0] + 1)
    if on[0]:
        starts = [0] + starts
    if on[-1]:
        ends = ends + [len(on)]
    for a, b in zip(starts, ends):
        g = np.where(good[a:b])[0]
        if len(g) == 0:
            continue
        seg = copx[a:b]
        first, last = g[0], g[-1]
        seg[:first] = seg[first]
        seg[last + 1:] = seg[last]
        copx[a:b] = seg
    return copx


def deriv(x, fs):
    v = np.gradient(x, 1.0 / fs, axis=0)
    a = np.gradient(v, 1.0 / fs, axis=0)
    return v, a


def cross2(r, F):
    """z-component of r x F for (N,2) arrays."""
    return r[:, 0] * F[:, 1] - r[:, 1] * F[:, 0]


def build_segments(P, n):
    """Segment COM positions/accelerations, angles, inertias for both legs + HAT."""
    segs = {}
    for s in ("R", "L"):
        hip, knee, ank, toe = P[s + ".Hip"][:n], P[s + ".Knee"][:n], P[s + ".Ankle"][:n], P[s + ".MT5"][:n]
        for name, prox, dist in (("foot", ank, toe), ("shank", knee, ank), ("thigh", hip, knee)):
            par = SEG[name]
            L = np.linalg.norm(dist - prox, axis=1).mean()
            m = par["m"] * MASS
            I = m * (par["rg"] * L) ** 2
            com = prox + par["c"] * (dist - prox)
            _, acom = deriv(com, FS)
            th = np.unwrap(np.arctan2((dist - prox)[:, 1], (dist - prox)[:, 0]))
            _, alpha = deriv(th, FS)
            segs[(s, name)] = dict(m=m, I=I, com=com, acom=acom, alpha=alpha,
                                   prox=prox, dist=dist, L=L)
    # HAT: rigidly above the pelvis centroid (ASIS+PSIS; bony landmarks, far less soft-tissue
    # artefact than the GTR markers). No trunk markers in this data set.
    midhip = pelvis_centroid(P, n)
    L_hat = 0.288 * HEIGHT
    com = midhip + np.array([0.0, SEG["hat"]["c"] * L_hat])
    _, acom = deriv(com, FS)
    m = SEG["hat"]["m"] * MASS
    segs[("B", "hat")] = dict(m=m, I=m * (SEG["hat"]["rg"] * L_hat) ** 2, com=com, acom=acom,
                              alpha=np.zeros(n), prox=midhip, dist=com, L=L_hat)
    return segs


def inverse_dynamics(segs, side, Fx, Fy, copx, copy=None):
    """Bottom-up planar Newton-Euler for one leg. Returns moments (N m) at ankle, knee, hip:
    moment applied by the proximal segment on the distal segment, +z (counter-clockwise, X->Y).
    copy: height of the centre of pressure; None (level ground) means y = 0."""
    gvec = np.array([0.0, -G])
    foot, shank, thigh = segs[(side, "foot")], segs[(side, "shank")], segs[(side, "thigh")]
    n = len(Fx)
    F_grf = np.column_stack([Fx, Fy])
    on = Fy > F_CONTACT
    cy = np.zeros(n) if copy is None else np.where(on, copy, 0.0)
    r_cop = np.column_stack([np.where(on, copx, foot["com"][:, 0]), cy])
    # foot
    F_ank = foot["m"] * foot["acom"] - foot["m"] * gvec - F_grf
    M_ank = (foot["I"] * foot["alpha"]
             - cross2(r_cop - foot["com"], F_grf)
             - cross2(foot["prox"] - foot["com"], F_ank))
    # shank (distal joint = ankle carries -F_ank, -M_ank)
    F_knee = shank["m"] * shank["acom"] - shank["m"] * gvec + F_ank
    M_knee = (shank["I"] * shank["alpha"] + M_ank
              - cross2(shank["prox"] - shank["com"], F_knee)
              - cross2(shank["dist"] - shank["com"], -F_ank))
    # thigh
    F_hip = thigh["m"] * thigh["acom"] - thigh["m"] * gvec + F_knee
    M_hip = (thigh["I"] * thigh["alpha"] + M_knee
             - cross2(thigh["prox"] - thigh["com"], F_hip)
             - cross2(thigh["dist"] - thigh["com"], -F_knee))
    return dict(ankle=M_ank, knee=M_knee, hip=M_hip)


def whole_body_ax_sum(segs):
    """S(t) = sum_s m_s * a_x,s  (= total horizontal ground force by Newton)."""
    return sum(v["m"] * v["acom"][:, 0] for v in segs.values())


def soleid_qp(S, FyR, FyL, w_dyn=1.0, fc_smooth=8.0, w_mag=1e-4, mu=MU):
    """Estimate per-foot horizontal force as unknown inputs.
    min w_dyn|fR+fL-S|^2 + w_s|D2 fR|^2 + w_s|D2 fL|^2 + w_mag(|fR|^2+|fL|^2)
    s.t. foot in air -> f = 0 ; on ground -> |f| <= mu*Fy."""
    n = len(S)
    dt = 1.0 / FS
    w_s = w_dyn / (2 * np.pi * fc_smooth * dt) ** 4
    I = sp.identity(n, format="csc")
    D2 = sp.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n), format="csc")
    A_dyn = sp.hstack([I, I], format="csc")                     # fR + fL
    D2b = sp.block_diag([D2, D2], format="csc")
    P = 2 * (w_dyn * A_dyn.T @ A_dyn + w_s * D2b.T @ D2b + w_mag * sp.identity(2 * n)).tocsc()
    q = -2 * w_dyn * (A_dyn.T @ S)
    # bounds
    onR, onL = FyR > F_CONTACT, FyL > F_CONTACT
    lb = np.concatenate([np.where(onR, -mu * FyR, 0.0), np.where(onL, -mu * FyL, 0.0)])
    ub = np.concatenate([np.where(onR, mu * FyR, 0.0), np.where(onL, mu * FyL, 0.0)])
    prob = osqp.OSQP()
    prob.setup(P=P, q=q, A=sp.identity(2 * n, format="csc"), l=lb, u=ub,
               verbose=False, eps_abs=1e-6, eps_rel=1e-6, max_iter=20000, polish=True)
    res = prob.solve()
    f = res.x
    return f[:n], f[n:], res.info.status, res.info.run_time


def vpp_prior(pelvis, grf, h_vpp=0.3):
    """Per-foot shear prior from the virtual-pivot-point regularity of walking (Maus et al. 2010):
    the ground force vector points from the centre of pressure towards a point h_vpp above the
    pelvis centroid. The lever is taken in the gravity direction, so an inclined surface only
    enters through the height of the centre of pressure (zero on level ground)."""
    out = {}
    px, py = pelvis[:, 0], pelvis[:, 1] + h_vpp
    for s in ("R", "L"):
        Fy, cop = grf[s]["Fy"], grf[s]["copx"]
        cy = grf[s].get("copy")
        cy = np.zeros_like(Fy) if cy is None else cy
        on = Fy > F_CONTACT
        lever = np.maximum(py - np.where(on, cy, 0.0), 0.2)
        out[s] = np.where(on, Fy * (px - cop) / lever, 0.0)
    return out


def soleid_qp_v2(S, grf, segs, pelvis, w_dyn=1.0, fc_smooth=8.0, w_mag=1e-4, mu=MU,
                 w_prior=1.0, h_vpp=0.3, w_torque=1.0):
    """SoleID v2: v1 + VPP direction prior + joint-torque smoothness (both linear in Fx)."""
    FyR, FyL = grf["R"]["Fy"], grf["L"]["Fy"]
    n = len(S)
    dt = 1.0 / FS
    w_s = w_dyn / (2 * np.pi * fc_smooth * dt) ** 4
    I = sp.identity(n, format="csc")
    D2 = sp.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n), format="csc")
    A_dyn = sp.hstack([I, I], format="csc")
    D2b = sp.block_diag([D2, D2], format="csc")
    P = w_dyn * A_dyn.T @ A_dyn + w_s * D2b.T @ D2b + w_mag * sp.identity(2 * n)
    q = -w_dyn * (A_dyn.T @ S)
    onR, onL = FyR > F_CONTACT, FyL > F_CONTACT
    if w_prior > 0:
        pr = vpp_prior(pelvis, grf, h_vpp)
        Wp = sp.diags(np.concatenate([onR, onL]).astype(float))
        prv = np.concatenate([pr["R"], pr["L"]])
        P = P + w_prior * Wp
        q = q - w_prior * (Wp @ prv)
    if w_torque > 0:
        # M_j = M_j|Fx=0 - y_j Fx  (planar, force at ground level) -> smoothness of M_j is linear in Fx
        w_t = w_torque * w_s / (0.9 ** 2)
        Z = sp.csc_matrix((n - 2, n))
        for side, blk in (("R", 0), ("L", 1)):
            M0 = inverse_dynamics(segs, side, np.zeros(n), grf[side]["Fy"], grf[side]["copx"])
            lev = dict(ankle=segs[(side, "foot")]["prox"][:, 1], knee=segs[(side, "shank")]["prox"][:, 1],
                       hip=segs[(side, "thigh")]["prox"][:, 1])
            for j in ("ankle", "knee", "hip"):
                Bj = D2 @ sp.diags(lev[j])
                Bfull = sp.hstack([Bj, Z] if blk == 0 else [Z, Bj], format="csc")
                P = P + w_t * Bfull.T @ Bfull
                q = q - w_t * (Bfull.T @ (D2 @ M0[j]))
    lb = np.concatenate([np.where(onR, -mu * FyR, 0.0), np.where(onL, -mu * FyL, 0.0)])
    ub = np.concatenate([np.where(onR, mu * FyR, 0.0), np.where(onL, mu * FyL, 0.0)])
    prob = osqp.OSQP()
    prob.setup(P=sp.csc_matrix(2 * P), q=2 * q, A=sp.identity(2 * n, format="csc"), l=lb, u=ub,
               verbose=False, eps_abs=1e-6, eps_rel=1e-6, max_iter=50000, polish=True)
    res = prob.solve()
    return res.x[:n], res.x[n:], res.info.status, res.info.run_time


def kinematics_only_baseline(P, segs, n, thr=0.015):
    """Arm C: estimate per-foot GRF (Fx, Fy) and COP without any force measurement."""
    gvec_y = G
    S_x = sum(v["m"] * v["acom"][:, 0] for v in segs.values())
    S_y = sum(v["m"] * (v["acom"][:, 1] + gvec_y) for v in segs.values())
    out = {}
    contact = {}
    for s in ("R", "L"):
        heel, toe = P[s + ".Heel"][:n], P[s + ".MT5"][:n]
        h = np.minimum(heel[:, 1], toe[:, 1])
        on = h < (h.min() + thr)
        # remove very short blips
        d = np.diff(on.astype(int))
        starts, ends = list(np.where(d == 1)[0] + 1), list(np.where(d == -1)[0] + 1)
        if on[0]:
            starts = [0] + starts
        if on[-1]:
            ends = ends + [n]
        for a, b in zip(starts, ends):
            if b - a < 0.2 * FS:
                on[a:b] = False
        contact[s] = on
        # COP: heel x at heel strike -> toe x at toe-off, linear progression in time
        cop = np.full(n, np.nan)
        d = np.diff(on.astype(int))
        starts, ends = list(np.where(d == 1)[0] + 1), list(np.where(d == -1)[0] + 1)
        if on[0]:
            starts = [0] + starts
        if on[-1]:
            ends = ends + [n]
        for a, b in zip(starts, ends):
            tau = np.linspace(0, 1, b - a)
            cop[a:b] = heel[a, 0] + tau * (toe[b - 1, 0] - heel[a, 0])
        out[s] = dict(copx=cop)
    onR, onL = contact["R"], contact["L"]
    FxR, FyR, FxL, FyL = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    # single support: everything on the stance foot; double support: smooth transition of trailing foot
    ds = onR & onL
    # identify double-support episodes and which foot is trailing (the one whose contact started earlier)
    d = np.diff(ds.astype(int))
    starts, ends = list(np.where(d == 1)[0] + 1), list(np.where(d == -1)[0] + 1)
    if ds[0]:
        starts = [0] + starts
    if ds[-1]:
        ends = ends + [n]
    trail_w = np.zeros(n)     # weight of the trailing foot (1 -> 0), leading foot gets 1 - w
    trail_is_R = np.zeros(n, dtype=bool)
    for a, b in zip(starts, ends):
        # trailing foot = the one that was already on the ground just before a
        i = max(a - 1, 0)
        trailR = onR[i] and not onL[i]
        tau = np.linspace(0, 1, b - a)
        trail_w[a:b] = 1 - 3 * tau ** 2 + 2 * tau ** 3      # smooth cubic decay
        trail_is_R[a:b] = trailR
    wR = np.where(ds, np.where(trail_is_R, trail_w, 1 - trail_w), onR.astype(float))
    wL = np.where(ds, np.where(trail_is_R, 1 - trail_w, trail_w), onL.astype(float))
    FxR, FyR = wR * S_x, wR * S_y
    FxL, FyL = wL * S_x, wL * S_y
    out["R"].update(Fx=FxR, Fy=np.maximum(FyR, 0.0))
    out["L"].update(Fx=FxL, Fy=np.maximum(FyL, 0.0))
    for s in ("R", "L"):
        out[s]["copx"] = np.where(np.isnan(out[s]["copx"]), segs[(s, "foot")]["com"][:, 0], out[s]["copx"])
    return out, contact


def newton_baseline(S, FyR, FyL):
    onR, onL = FyR > F_CONTACT, FyL > F_CONTACT
    tot = np.where(onR, FyR, 0.0) + np.where(onL, FyL, 0.0)
    tot[tot < 1e-6] = np.nan
    fR = np.where(onR, S * np.where(onR, FyR, 0.0) / tot, 0.0)
    fL = np.where(onL, S * np.where(onL, FyL, 0.0) / tot, 0.0)
    return np.nan_to_num(fR), np.nan_to_num(fL)


def heel_strikes(Fy):
    on = Fy > F_CONTACT
    idx = np.where(np.diff(on.astype(int)) == 1)[0] + 1
    return idx


def stride_normalize(x, hs, npts=101):
    curves = []
    for a, b in zip(hs[:-1], hs[1:]):
        if b - a < 0.6 * FS or b - a > 2.0 * FS:
            continue
        tt = np.linspace(a, b, npts)
        curves.append(np.interp(tt, np.arange(len(x)), x))
    return np.array(curves)


def rmse(a, b, mask=None):
    if mask is not None:
        a, b = a[mask], b[mask]
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main(subj="WBDS01", trial="walkT05"):
    cfg = configure(subj, trial)
    mk, gr, knt = load()
    P = planar_markers(mk)
    grf, n = grf_to_marker_rate(gr)
    n = min(n, len(P["R.Knee"]))
    for s in grf:
        for k in grf[s]:
            grf[s][k] = grf[s][k][:n]
    segs = build_segments(P, n)
    BW = MASS * G

    FxR, FyR, cR = grf["R"]["Fx"], grf["R"]["Fy"], grf["R"]["copx"]
    FxL, FyL, cL = grf["L"]["Fx"], grf["L"]["Fy"], grf["L"]["copx"]
    onR, onL = FyR > F_CONTACT, FyL > F_CONTACT
    ds = onR & onL                     # double support
    ssR = onR & ~onL                   # right single support

    # ---- whole-body Newton check: does sum(m a_x) reproduce measured Fx_total?
    S = whole_body_ax_sum(segs)
    Fx_tot_meas = np.where(onR, FxR, 0.0) + np.where(onL, FxL, 0.0)
    trim = np.zeros(n, dtype=bool)                    # drop filter edges (1 s each end)
    trim[int(1.0 * FS):n - int(1.0 * FS)] = True
    newton_r = float(np.corrcoef(S[trim], Fx_tot_meas[trim])[0, 1])
    newton_rmse = rmse(S[trim], Fx_tot_meas[trim])

    # ---- arm A: reference inverse dynamics with full measured GRF
    ref = {s: inverse_dynamics(segs, s, grf[s]["Fx"], grf[s]["Fy"], grf[s]["copx"]) for s in ("R", "L")}

    # ---- arm B: Newton + proportional split
    bR, bL = newton_baseline(S, FyR, FyL)
    base = {"R": inverse_dynamics(segs, "R", bR, FyR, cR), "L": inverse_dynamics(segs, "L", bL, FyL, cL)}

    # ---- arm C: kinematics-only (no force measurement)
    kin, kin_contact = kinematics_only_baseline(P, segs, n)
    kinR, kinL = kin["R"]["Fx"], kin["L"]["Fx"]
    kino = {"R": inverse_dynamics(segs, "R", kin["R"]["Fx"], kin["R"]["Fy"], kin["R"]["copx"]),
            "L": inverse_dynamics(segs, "L", kin["L"]["Fx"], kin["L"]["Fy"], kin["L"]["copx"])}
    # contact-timing agreement with force-based contact (right foot)
    kin_timing = dict(agreement_pct=float(100 * np.mean(kin_contact["R"] == onR)),
                      vertical_rmse_pctBW=float(100 * np.sqrt(np.mean((kin["R"]["Fy"] - FyR) ** 2)) / BW))

    # ---- arm D1: SoleID v1 (dynamics + smoothness)
    vR, vL, status1, rt1 = soleid_qp(S, FyR, FyL)
    sole1 = {"R": inverse_dynamics(segs, "R", vR, FyR, cR), "L": inverse_dynamics(segs, "L", vL, FyL, cL)}
    # ---- arm D2: SoleID v2 (+ VPP prior + torque smoothness)
    pelvis = pelvis_centroid(P, n)
    dR, dL, status, rt = soleid_qp_v2(S, grf, segs, pelvis, **V2_PARAMS)
    sole = {"R": inverse_dynamics(segs, "R", dR, FyR, cR), "L": inverse_dynamics(segs, "L", dL, FyL, cL)}

    # ---- metrics: horizontal force
    res = dict(subject=SUBJ, trial=TRIAL, mass=MASS, speed=cfg["speed"], n_samples=int(n), fs=FS, qp_status=status,
               qp_time_s=rt, newton_check=dict(r=newton_r, rmse_N=newton_rmse, rmse_pctBW=100 * newton_rmse / BW))
    m = {}
    for name, (eR, eL) in (("SoleID", (dR, dL)), ("SoleID_v1", (vR, vL)), ("Newton_prop", (bR, bL)), ("Kin_only", (kinR, kinL))):
        d = {}
        for lab, mask in (("all", trim), ("stance_R", onR & trim),
                          ("double_support", ds & trim), ("single_support_R", ssR & trim)):
            e = rmse(eR, FxR, mask)
            d[lab] = dict(rmse_N=e, rmse_pctBW=100 * e / BW,
                          r=float(np.corrcoef(eR[mask], FxR[mask])[0, 1]) if np.sum(mask) > 10 else None)
        m[name] = d
    res["Fx_right_foot"] = m

    # peak braking / propulsion per right stride
    hs = heel_strikes(FyR)
    hs = hs[(hs > int(FS)) & (hs < n - int(FS))]
    def peaks(x):
        cur = stride_normalize(x, hs)
        return cur.min(axis=1), cur.max(axis=1), cur
    pm, pM, curFx = peaks(FxR)
    peak = {}
    for name, e in (("SoleID", dR), ("SoleID_v1", vR), ("Newton_prop", bR), ("Kin_only", kinR)):
        em, eM, _ = peaks(e)
        peak[name] = dict(braking_peak_err_N_mean=float(np.mean(em - pm)), braking_peak_err_N_sd=float(np.std(em - pm)),
                          propulsion_peak_err_N_mean=float(np.mean(eM - pM)), propulsion_peak_err_N_sd=float(np.std(eM - pM)),
                          n_strides=int(len(em)))
    res["Fx_right_peaks"] = peak
    res["measured_peaks_N"] = dict(braking_mean=float(pm.mean()), propulsion_mean=float(pM.mean()))

    # ---- metrics: joint moments (right leg), Nm/kg, stance-phase and full stride
    jm = {}
    for name, arm in (("SoleID", sole), ("SoleID_v1", sole1), ("Newton_prop", base), ("Kin_only", kino)):
        d = {}
        for j in ("ankle", "knee", "hip"):
            a, b = arm["R"][j] / MASS, ref["R"][j] / MASS
            d[j] = dict(rmse_all=rmse(a, b, trim), rmse_stance=rmse(a, b, onR & trim),
                        rmse_double_support=rmse(a, b, ds & trim),
                        ref_peak_abs=float(np.max(np.abs(b[trim]))))
        jm[name] = d
    res["joint_moments_right_Nm_per_kg"] = jm
    res["kin_only_contact"] = kin_timing
    res["hip_source"] = P["hip_source"]
    res["pelvis_markers_used"] = len(P["pelvis_pts"])

    # ---- sanity: our reference planar ID vs data set's own (Visual3D) sagittal moments, stride-normalized
    san = {}
    for j, col in (("ankle", "RAnkleMomentZ"), ("knee", "RKneeMomentZ"), ("hip", "RHipMomentZ")):
        ours = stride_normalize(ref["R"][j] / MASS, hs).mean(axis=0)
        theirs = knt[col].values
        r = float(np.corrcoef(ours, theirs)[0, 1])
        sign = 1.0 if r >= 0 else -1.0
        san[j] = dict(r=r, sign_flip=sign < 0, rmse_after_sign=rmse(sign * ours, theirs),
                      our_peak=float(np.max(np.abs(ours))), their_peak=float(np.max(np.abs(theirs))))
    res["reference_vs_dataset_moments"] = san

    with open(os.path.join(OUT, f"{SUBJ}{TRIAL}_metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    # time series for later plotting
    pd.DataFrame(dict(t=np.arange(n) / FS, FxR_meas=FxR, FxR_soleid=dR, FxR_soleid_v1=vR, FxR_newton=bR, FxR_kin=kinR, FyR_kin=kin["R"]["Fy"], FyR=FyR, FyL=FyL,
                      FxL_meas=FxL, FxL_soleid=dL, FxL_soleid_v1=vL, FxL_newton=bL, S_newton=S,
                      Mank_ref=ref["R"]["ankle"] / MASS, Mank_sole=sole["R"]["ankle"] / MASS, Mank_newton=base["R"]["ankle"] / MASS,
                      Mknee_ref=ref["R"]["knee"] / MASS, Mknee_sole=sole["R"]["knee"] / MASS, Mknee_newton=base["R"]["knee"] / MASS,
                      Mhip_ref=ref["R"]["hip"] / MASS, Mhip_sole=sole["R"]["hip"] / MASS, Mhip_newton=base["R"]["hip"] / MASS,
                      Mank_kin=kino["R"]["ankle"] / MASS, Mknee_kin=kino["R"]["knee"] / MASS, Mhip_kin=kino["R"]["hip"] / MASS)
                 ).to_csv(os.path.join(OUT, f"{SUBJ}{TRIAL}_timeseries.csv"), index=False)
    return res


if __name__ == "__main__":
    a = sys.argv[1:]
    print(json.dumps(main(*a) if a else main(), indent=2))

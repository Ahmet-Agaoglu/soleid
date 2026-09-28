"""Camargo 2021 overground modes: ramp (six inclinations) and levelground (three speeds).

Unlike the treadmill mode, the force data come from five embedded plates (FP1..FP5) with no
left/right labelling, and each trial contains only a handful of instrumented steps. Two things
are therefore needed here and nowhere else:

  1. plate-to-foot assignment: every contact interval on a plate is attributed to the foot whose
     heel/toe markers are closest to the measured centre of pressure during that interval;
  2. a non-flat ground: on the ramp the plates sit at different heights, so the centre of pressure
     has a height of its own, which now enters the inverse dynamics and the VPP prior.

Evaluation is per instrumented step, not per stride: only steps that land fully on a plate are
scored, and the shear component of that plate is the ground truth.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

import camargo_adapter as ca
import experiments as ex
import soleid_planar as sp_
import config

CAL = json.load(open(os.path.join(sp_.OUT, "calibration.json")))["chosen"]
INFO = pd.read_csv(config.CAMARGO_INFO).set_index("Subject")
FS = ca.FS
MIN_CONTACT_S = 0.25          # a plausible stance duration
MAX_CONTACT_S = 1.60          # longer contacts are standing on the plate, not a walking step
PEAK_BW_LO, PEAK_BW_HI = 0.70, 1.80   # plausible peak vertical force for one foot, in body weights
PLATE_MARGIN = 0.25           # m, how far the COP may sit from the foot markers to still match


def _plates(fp):
    """Force sources in an overground trial. Ramp trials expose FP1..FP5 only; levelground trials
    also cross the instrumented treadmill, whose two belts are already side-labelled. The labelled
    belts are treated like any other plate, which gives an independent check of the plate-to-foot
    assignment (reported as `label_check`)."""
    out = [c.rsplit("_vy", 1)[0] for c in fp.columns if c.endswith("_vy")]
    return [g for g in sorted(set(out)) if g != "Combined"]


def _known_side(plate):
    if plate.endswith("_R"):
        return "R"
    if plate.endswith("_L"):
        return "L"
    return None


def load_overground(subj, trial, mode, fc_marker=6.0, fc_force=20.0, mass=None):
    """Per-foot Fx, Fy, cop_x, cop_y in the planar frame (x forward = -z_marker, y up)."""
    base = os.path.join(ca.CSV, subj, mode)
    mk = pd.read_csv(os.path.join(base, "markers", trial + ".csv"))
    fp = pd.read_csv(os.path.join(base, "fp", trial + ".csv"))
    cond = pd.read_csv(os.path.join(base, "conditions", trial + ".csv"))
    step = int(round(np.median(np.diff(mk.Header.values)) / np.median(np.diff(fp.Header.values))))
    n = min(len(mk), len(fp) // step)

    P = {}
    for side, (kn, an, he, to) in (("R", ("R_Knee_Lat", "R_Ankle_Lat", "R_Heel", "R_Toe_Tip")),
                                   ("L", ("L_Knee_Lat", "L_Ankle_Lat", "L_Heel", "L_Toe_Tip"))):
        P[side + ".Knee"], P[side + ".Ankle"] = ca._xy(mk, kn)[:n], ca._xy(mk, an)[:n]
        P[side + ".Heel"], P[side + ".MT5"] = ca._xy(mk, he)[:n], ca._xy(mk, to)[:n]
    for nm in ("R_ASIS", "L_ASIS", "R_PSIS", "L_PSIS"):
        P[nm.replace("_", ".")] = ca._xy(mk, nm)[:n]
    lat = {s: 0.5 * (ca._lat(mk, s + "_Heel")[:n] + ca._lat(mk, s + "_Toe_Tip")[:n]) for s in ("R", "L")}
    lat = {s: pd.Series(v).interpolate(limit_direction="both").values for s, v in lat.items()}
    gaps = {k: float(np.isnan(v[:, 0]).mean()) for k, v in P.items()}
    if max(gaps.values()) > 0.05:
        raise ValueError(f"marker gaps: {max(gaps, key=gaps.get)} {max(gaps.values()):.2f}")
    P = {k: sp_.lowpass(pd.DataFrame(v).interpolate(limit_direction="both").values, fc_marker, FS)
         for k, v in P.items()}
    P["pelvis_pts"] = ["R.ASIS", "L.ASIS", "R.PSIS", "L.PSIS"]

    grf = {s: dict(Fx=np.zeros(n), Fy=np.zeros(n), copx=np.zeros(n), copy=np.zeros(n)) for s in ("R", "L")}
    steps, label_check = [], []
    for pl in _plates(fp):
        Fy = sp_.lowpass(fp[f"{pl}_vy"].values, fc_force, 1000.0)[::step][:n]
        Fx = sp_.lowpass(-fp[f"{pl}_vz"].values, fc_force, 1000.0)[::step][:n]
        cx = -fp[f"{pl}_pz"].values[::step][:n]
        cy = fp[f"{pl}_py"].values[::step][:n]
        clat = fp[f"{pl}_px"].values[::step][:n]
        on = Fy > sp_.F_CONTACT
        d = np.diff(on.astype(int))
        starts, ends = list(np.where(d == 1)[0] + 1), list(np.where(d == -1)[0] + 1)
        if on[0]:
            starts = [0] + starts
        if on[-1]:
            ends = ends + [n]
        for a, b in zip(starts, ends):
            if not (MIN_CONTACT_S * FS <= (b - a) <= MAX_CONTACT_S * FS):
                continue
            core = slice(a + (b - a) // 4, b - (b - a) // 4)      # middle half of the contact
            copm, coplat = np.median(cx[core]), np.median(clat[core])
            dist = {}
            for s in ("R", "L"):
                mid = 0.5 * (P[s + ".Heel"][core, 0] + P[s + ".MT5"][core, 0])
                d_fore = np.median(np.abs(mid - copm))
                d_lat = np.median(np.abs(lat[s][core] - coplat))
                dist[s] = float(np.hypot(d_fore, d_lat))   # the feet separate mediolaterally
            side = min(dist, key=dist.get)
            if dist[side] > PLATE_MARGIN:
                continue
            known = _known_side(pl)
            if known is not None:
                # Do NOT override the geometry: the belt label is fixed in the laboratory, while
                # the subject traverses the treadmill in both directions during the loops.
                label_check.append(side == known)
            if mass is not None:
                peak = float(np.max(Fy[a:b])) / (mass * 9.81)
                if not (PEAK_BW_LO <= peak <= PEAK_BW_HI):
                    continue
            sl = slice(a, b)
            grf[side]["Fx"][sl] += Fx[sl]
            grf[side]["Fy"][sl] += Fy[sl]
            grf[side]["copx"][sl] = cx[sl]
            grf[side]["copy"][sl] = cy[sl]
            steps.append(dict(plate=pl, side=side, start=a, end=b, dist=dist[side],
                              cop_height=float(np.median(cy[core]))))
    for s in ("R", "L"):
        grf[s]["copx"] = sp_.hold_cop(grf[s]["copx"], grf[s]["Fy"])
    label = cond.Label.astype(str).values if "Label" in cond else np.array(["?"] * len(cond))
    return dict(subj=subj, trial=trial, mode=mode, n=n, P=P, grf=grf, steps=steps,
                labels=label, gaps=gaps, label_check=label_check)


def levelground_meta(trial):
    """levelground_<ccw|cw>_<slow|normal|fast>_<trial>_<rep>"""
    p = trial.split("_")
    return dict(direction=p[1] if len(p) > 1 else "?", speed_class=p[2] if len(p) > 2 else "?")


def incline_of(trial):
    """ramp_<k>_<l|r>_... : k indexes the inclination, l/r the direction of travel."""
    parts = trial.split("_")
    if parts[0] != "ramp":
        return 0.0
    try:
        return float(parts[1])
    except ValueError:
        return np.nan


def run_trial(subj, trial, mode, return_check=False):
    mass, height = float(INFO.loc[subj, "Weight"]), float(INFO.loc[subj, "Height"])
    d = load_overground(subj, trial, mode, mass=mass)
    n = d["n"]
    if not d["steps"]:
        raise ValueError("no instrumented steps")
    sp_.MASS, sp_.HEIGHT, sp_.FS = mass, height, FS
    segs, pel = ca.build_segments(d["P"], n, mass, height)
    grf = d["grf"]
    BW = mass * 9.81

    trim = np.zeros(n, bool)
    trim[int(0.3 * FS):n - int(0.3 * FS)] = True
    dd = dict(n=n, segs=segs, pelvis=pel, grf=grf, trim=trim)
    S = ex.newton_sum(dd)
    prior = ex.vpp_prior(dd, CAL["h_vpp"])
    sR, sL, _ = ex.solve(dd, S, fc_smooth=CAL["fc_smooth"], prior=prior, w_prior=CAL["w_prior"], w_torque=0.0)
    vR, vL, _ = ex.solve(dd, S, fc_smooth=CAL["fc_smooth"], prior=None, w_prior=0.0, w_torque=0.0)
    bR, bL = sp_.newton_baseline(S, grf["R"]["Fy"], grf["L"]["Fy"])

    est = dict(SoleID=dict(R=sR, L=sL), SoleID_noVPP=dict(R=vR, L=vL), Newton_prop=dict(R=bR, L=bL),
               NoShear=dict(R=np.zeros(n), L=np.zeros(n)))
    ref = {s: sp_.inverse_dynamics(segs, s, grf[s]["Fx"], grf[s]["Fy"], grf[s]["copx"], grf[s]["copy"])
           for s in ("R", "L")}

    rows = []
    for st in d["steps"]:
        a, b = st["start"], st["end"]
        if not trim[a:b].all():
            continue
        side, sl = st["side"], slice(a, b)
        meas = grf[side]["Fx"][sl]
        meta = levelground_meta(trial) if mode == "levelground" else dict(direction="-", speed_class="-")
        row = dict(subject=subj, trial=trial, mode=mode, side=side, plate=st["plate"],
                   incline=incline_of(trial), cop_height=st["cop_height"], mass=mass,
                   direction=meta["direction"], speed_class=meta["speed_class"],
                   stance_s=(b - a) / FS,
                   newton_check=100 * np.sqrt(np.mean((S[sl] - (grf["R"]["Fx"] + grf["L"]["Fx"])[sl]) ** 2)) / BW)
        for name, e in est.items():
            ee = e[side][sl]
            row[f"{name}_stance"] = 100 * np.sqrt(np.mean((ee - meas) ** 2)) / BW
            M = sp_.inverse_dynamics(segs, side, e[side], grf[side]["Fy"], grf[side]["copx"], grf[side]["copy"])
            for j in ("ankle", "knee", "hip"):
                row[f"{name}_{j}"] = float(np.sqrt(np.mean((M[j][sl] / mass - ref[side][j][sl] / mass) ** 2)))
        rows.append(row)
    return (rows, d["label_check"]) if return_check else rows


def main(mode="ramp", limit_subjects=None):
    subs = ca.subjects()
    if limit_subjects:
        subs = subs[:limit_subjects]
    rows, skipped, checks = [], 0, []
    for s in subs:
        got = 0
        for t in ca.trials(s, mode):
            try:
                r, ck = run_trial(s, t, mode, return_check=True)
                rows += r
                checks += ck
                got += len(r)
            except Exception as e:
                skipped += 1
        print(f"  {s}: {got} instrumented steps", flush=True)
    if checks:
        print("plate-to-foot assignment agreed with the (lab-fixed) belt label in %.1f%% of %d "
              "labelled contacts; see the direction split below" % (100 * np.mean(checks), len(checks)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(sp_.OUT, f"camargo_{mode}_steps.csv"), index=False)
    print(f"skipped trials: {skipped}")
    summarize(df, mode)
    return df


def summarize(df, mode):
    pd.set_option("display.width", 220)
    arms = ["NoShear", "Newton_prop", "SoleID_noVPP", "SoleID"]
    print(f"\nCAMARGO {mode.upper()}: {df.subject.nunique()} subjects, "
          f"{df.groupby(['subject', 'trial']).ngroups} trials, {len(df)} instrumented steps")
    for key in ("stance", "knee", "hip"):
        print(f"\n{key}: " + "  ".join(f"{a}={df[f'{a}_{key}'].mean():.3f}" for a in arms))
        by = "incline" if mode == "ramp" else "speed_class"
        if by in df:
            print(df.groupby(by)[[f"{a}_{key}" for a in arms]].mean().round(3).to_string())
    print("\nwhole-body Newton check: %.2f %%BW" % df.newton_check.mean())
    print("SoleID better than NoShear in %.0f%% of steps (stance)" % (100 * (df.SoleID_stance < df.NoShear_stance).mean()))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "ramp"
    if mode == "summary":
        for m in ("ramp", "levelground"):
            p = os.path.join(sp_.OUT, f"camargo_{m}_steps.csv")
            if os.path.exists(p):
                summarize(pd.read_csv(p), m)
    else:
        main(mode, limit_subjects=int(sys.argv[2]) if len(sys.argv) > 2 else None)

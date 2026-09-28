"""Design loop on the CALIBRATION subjects only (protocol step 3):
  A. phase-dependent VPP prior weight (propulsion-peak bias)
  B. pilot degradation test (kinematic noise / frame rate, vertical-force error, COP shift, sync)
     -> decides whether kinematic re-tracking (or stronger smoothing) is needed.
Nothing here touches the test subjects."""
import json
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

import osqp
import experiments as ex
import soleid_planar as sp_
from protocol import CAL_SUBJECTS, available_trials

CAL = json.load(open(os.path.join(sp_.OUT, "calibration.json")))["chosen"]


# ----------------------------------------------------------------------------- solver with w_prior(t)
def solve_tv(d, S, prior, w_prior_t, fc_smooth, w_dyn=1.0, w_mag=1e-4, mu=sp_.MU):
    """SoleID QP with a time-varying prior weight vector (per foot)."""
    n = d["n"]
    dt = 1.0 / sp_.FS
    FyR, FyL = d["grf"]["R"]["Fy"], d["grf"]["L"]["Fy"]
    w_s = w_dyn / (2 * np.pi * fc_smooth * dt) ** 4
    I = sp.identity(n, format="csc")
    D2 = sp.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n), format="csc")
    A_dyn = sp.hstack([I, I], format="csc")
    D2b = sp.block_diag([D2, D2], format="csc")
    P = w_dyn * A_dyn.T @ A_dyn + w_s * D2b.T @ D2b + w_mag * sp.identity(2 * n)
    q = -w_dyn * (A_dyn.T @ S)
    onR, onL = FyR > sp_.F_CONTACT, FyL > sp_.F_CONTACT
    Wp = sp.diags(np.concatenate([w_prior_t["R"] * onR, w_prior_t["L"] * onL]))
    pr = np.concatenate([prior["R"], prior["L"]])
    P = P + Wp
    q = q - Wp @ pr
    lb = np.concatenate([np.where(onR, -mu * FyR, 0.0), np.where(onL, -mu * FyL, 0.0)])
    ub = np.concatenate([np.where(onR, mu * FyR, 0.0), np.where(onL, mu * FyL, 0.0)])
    prob = osqp.OSQP()
    prob.setup(P=sp.csc_matrix(2 * P), q=2 * q, A=sp.identity(2 * n, format="csc"), l=lb, u=ub,
               verbose=False, eps_abs=1e-6, eps_rel=1e-6, max_iter=50000, polish=True)
    r = prob.solve()
    return r.x[:n], r.x[n:]


def stance_phase(Fy):
    """tau in [0,1] within each contact episode (from the measured vertical force), 0 elsewhere."""
    on = Fy > sp_.F_CONTACT
    tau = np.zeros(len(Fy))
    d = np.diff(on.astype(int))
    starts, ends = list(np.where(d == 1)[0] + 1), list(np.where(d == -1)[0] + 1)
    if on[0]:
        starts = [0] + starts
    if on[-1]:
        ends = ends + [len(on)]
    for a, b in zip(starts, ends):
        tau[a:b] = np.linspace(0, 1, b - a)
    return tau


def weight_shape(tau, shape, w0):
    if shape == "const":
        return np.full_like(tau, w0)
    if shape.startswith("late"):          # late<tau0>_<wmin>: linear decay from tau0 to 1
        _, t0, wmin = shape.split("_")
        t0, wmin = float(t0), float(wmin)
        w = np.where(tau > t0, w0 * (1 - (1 - wmin) * (tau - t0) / (1 - t0)), w0)
        return w
    if shape.startswith("window"):        # window_<wmin>: low weight in first and last 20 % of stance
        wmin = float(shape.split("_")[1])
        w = np.full_like(tau, w0)
        w[(tau < 0.2) | (tau > 0.8)] = w0 * wmin
        return w
    raise ValueError(shape)


def peaks(d, fR):
    FyR, FxR = d["grf"]["R"]["Fy"], d["grf"]["R"]["Fx"]
    hs = sp_.heel_strikes(FyR)
    hs = hs[(hs > 150) & (hs < d["n"] - 150)]
    cm = sp_.stride_normalize(FxR, hs)
    ce = sp_.stride_normalize(fR, hs)
    return float(np.mean(ce.max(1) - cm.max(1))), float(np.mean(ce.min(1) - cm.min(1)))


def load_cal():
    trials = [t for t in available_trials() if t[0] in CAL_SUBJECTS]
    data = {}
    for subj, trial, speed, age in trials:
        try:
            data[(subj, trial)] = ex.prepare(subj, trial)
        except sp_.MarkerGapError:
            pass
    return data


# ----------------------------------------------------------------------------- A. phase weight
def phase_weight_experiment(data):
    shapes = ["const", "late_0.5_0.3", "late_0.6_0.2", "late_0.6_0.0", "late_0.7_0.0", "window_0.3", "window_0.0"]
    rows = []
    for shape in shapes:
        st, ds, hip, pk, bk = [], [], [], [], []
        for (subj, trial), d in data.items():
            sp_.configure(subj, trial)
            S = ex.newton_sum(d)
            pr = ex.vpp_prior(d, CAL["h_vpp"])
            wt = {s: weight_shape(stance_phase(d["grf"][s]["Fy"]), shape, CAL["w_prior"]) for s in ("R", "L")}
            fR, fL = solve_tv(d, S, pr, wt, CAL["fc_smooth"])
            m = ex.metrics(d, fR)
            p, b = peaks(d, fR)
            st.append(m["stance"]); ds.append(m["ds"]); hip.append(m["hip"]); pk.append(p); bk.append(b)
        rows.append(dict(shape=shape, stance=np.mean(st), ds=np.mean(ds), hip=np.mean(hip),
                         prop_peak_err_N=np.mean(pk), brake_peak_err_N=np.mean(bk)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(sp_.OUT, "design_phase_weight.csv"), index=False)
    print("\nA. phase-dependent VPP weight (calibration subjects):")
    print(df.round(3).to_string(index=False))
    return df


# ----------------------------------------------------------------------------- B. degradation pilot
def degrade(d, kind, level, rng):
    """Return a shallow copy of d with degraded inputs. Reference (metrics) stays clean."""
    e = dict(d)
    e["grf"] = {s: dict(d["grf"][s]) for s in ("R", "L")}
    n = d["n"]
    if kind == "marker_noise_mm":       # white noise on marker positions before the 6 Hz filter
        P = {}
        for k, v in d["P"].items():
            if isinstance(v, np.ndarray):
                noisy = v + rng.normal(0, level / 1000.0, v.shape)
                P[k] = sp_.lowpass(noisy, 6.0, sp_.FS)
            else:
                P[k] = v
        e["P"] = P
        e["segs"] = sp_.build_segments(P, n)
        e["pelvis"] = sp_.pelvis_centroid(P, n)
    elif kind == "marker_fps":          # resample markers to `level` fps and back (linear), refilter
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
        e["P"] = P
        e["segs"] = sp_.build_segments(P, n)
        e["pelvis"] = sp_.pelvis_centroid(P, n)
    elif kind == "fy_scale_pct":        # per-trial gain error on the vertical force (+ 2 %BW white noise)
        g = 1 + level / 100.0
        for s in ("R", "L"):
            Fy = d["grf"][s]["Fy"] * g + rng.normal(0, 0.02 * sp_.MASS * 9.81, n)
            Fy = sp_.lowpass(Fy, 20.0, sp_.FS)
            e["grf"][s]["Fy"] = np.where(d["grf"][s]["Fy"] > sp_.F_CONTACT, np.maximum(Fy, sp_.F_CONTACT + 1), 0.0)
    elif kind == "cop_shift_mm":        # constant anterior shift of the COP
        for s in ("R", "L"):
            e["grf"][s]["copx"] = d["grf"][s]["copx"] + level / 1000.0
    elif kind == "sync_frames":         # force channels shifted in time relative to kinematics
        k = int(level)
        for s in ("R", "L"):
            for key in ("Fy", "copx"):
                e["grf"][s][key] = np.roll(d["grf"][s][key], k)
    else:
        raise ValueError(kind)
    return e


def run_arms(d_in, d_clean):
    """SoleID (full), SoleID - VPP, Newton+prop on degraded inputs; metrics vs clean reference."""
    n = d_clean["n"]
    S = ex.newton_sum(d_in)
    grf = d_in["grf"]
    pr = ex.vpp_prior(d_in, CAL["h_vpp"])
    # full
    fR, fL, _ = ex.solve(d_in, S, fc_smooth=CAL["fc_smooth"], prior=pr, w_prior=CAL["w_prior"], w_torque=0.0)
    # no prior
    vR, vL, _ = ex.solve(d_in, S, fc_smooth=CAL["fc_smooth"], prior=None, w_prior=0.0, w_torque=0.0)
    # newton proportional
    bR, bL = sp_.newton_baseline(S, grf["R"]["Fy"], grf["L"]["Fy"])
    out = {}
    FxR, FyR_c, FyL_c = d_clean["grf"]["R"]["Fx"], d_clean["grf"]["R"]["Fy"], d_clean["grf"]["L"]["Fy"]
    onR, onL, tr = FyR_c > sp_.F_CONTACT, FyL_c > sp_.F_CONTACT, d_clean["trim"]
    BW = sp_.MASS * 9.81
    ref = sp_.inverse_dynamics(d_clean["segs"], "R", FxR, FyR_c, d_clean["grf"]["R"]["copx"])["hip"] / sp_.MASS
    for name, est in (("SoleID", fR), ("SoleID_noVPP", vR), ("Newton_prop", bR)):
        est_m = sp_.inverse_dynamics(d_in["segs"], "R", est, grf["R"]["Fy"], grf["R"]["copx"])["hip"] / sp_.MASS
        out[name] = dict(stance=100 * np.sqrt(np.mean((est[onR & tr] - FxR[onR & tr]) ** 2)) / BW,
                         hip=float(np.sqrt(np.mean((est_m[tr] - ref[tr]) ** 2))))
    return out


def degradation_pilot(data):
    rng = np.random.default_rng(0)
    plan = [("marker_noise_mm", [0, 5, 10, 20]), ("marker_fps", [150, 60, 30]),
            ("fy_scale_pct", [0, 5, 8, 12]), ("cop_shift_mm", [0, 5, 10, 15]), ("sync_frames", [0, 1, 2, 3])]
    rows = []
    for kind, levels in plan:
        for lv in levels:
            acc = {a: dict(stance=[], hip=[]) for a in ("SoleID", "SoleID_noVPP", "Newton_prop")}
            for (subj, trial), d in data.items():
                sp_.configure(subj, trial)
                dd = degrade(d, kind, lv, rng) if lv not in (0, 150) else d
                res = run_arms(dd, d)
                for a in acc:
                    acc[a]["stance"].append(res[a]["stance"]); acc[a]["hip"].append(res[a]["hip"])
            row = dict(kind=kind, level=lv)
            for a in acc:
                row[f"{a}_stance"] = np.mean(acc[a]["stance"]); row[f"{a}_hip"] = np.mean(acc[a]["hip"])
            rows.append(row)
            print(f"  {kind:16s} {lv:>5}: SoleID {row['SoleID_stance']:.2f}%BW / hip {row['SoleID_hip']:.3f} | noVPP {row['SoleID_noVPP_stance']:.2f} / {row['SoleID_noVPP_hip']:.3f} | Newton {row['Newton_prop_stance']:.2f} / {row['Newton_prop_hip']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(sp_.OUT, "design_degradation_pilot.csv"), index=False)
    return df


if __name__ == "__main__":
    data = load_cal()
    print(f"calibration trials loaded: {len(data)}")
    what = sys.argv[1] if len(sys.argv) > 1 else "both"
    if what in ("phase", "both"):
        phase_weight_experiment(data)
    if what in ("degrade", "both"):
        print("\nB. degradation pilot (calibration subjects); stance Fx RMSE %BW / hip moment RMSE Nm/kg")
        degradation_pilot(data)

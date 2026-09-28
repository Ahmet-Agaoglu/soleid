"""Formulation experiments for SoleID (planar): trunk gain, VPP direction prior, torque smoothness.
Calibration (trunk gain, VPP height, weights) is done on WBDS01 T05 only and then applied
unchanged to the other cases (leave-subject-out spirit)."""
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp

import osqp
import soleid_planar as sp_

BWf = lambda: sp_.MASS * 9.81


def prepare(subj, trial):
    sp_.configure(subj, trial)
    mk, gr, knt = sp_.load()
    P = sp_.planar_markers(mk)
    grf, n = sp_.grf_to_marker_rate(gr)
    n = min(n, len(P["R.Knee"]))
    for s in grf:
        for k in grf[s]:
            grf[s][k] = grf[s][k][:n]
    segs = sp_.build_segments(P, n)
    pelvis = sp_.pelvis_centroid(P, n)
    trim = np.zeros(n, dtype=bool)
    trim[150:n - 150] = True
    return dict(P=P, grf=grf, n=n, segs=segs, pelvis=pelvis, trim=trim)


def newton_sum(d, hat_gain=1.0):
    segs = d["segs"]
    S = sum(v["m"] * v["acom"][:, 0] for k, v in segs.items() if k[1] != "hat")
    return S + hat_gain * segs[("B", "hat")]["m"] * segs[("B", "hat")]["acom"][:, 0]


def vpp_prior(d, h_vpp):
    """Per-foot shear prior: the force vector points from the centre of pressure towards a point
    h_vpp above the pelvis centroid. The lever is vertical (gravity-referenced), so an inclined
    surface enters only through the height of the centre of pressure (zero on level ground)."""
    out = {}
    px, py = d["pelvis"][:, 0], d["pelvis"][:, 1] + h_vpp
    for s in ("R", "L"):
        Fy, cop = d["grf"][s]["Fy"], d["grf"][s]["copx"]
        cy = d["grf"][s].get("copy")
        cy = np.zeros_like(Fy) if cy is None else cy
        on = Fy > sp_.F_CONTACT
        lever = np.maximum(py - np.where(on, cy, 0.0), 0.2)
        out[s] = np.where(on, Fy * (px - cop) / lever, 0.0)
    return out


def torque_lever(d, side):
    """Joint heights y_j(t): M_j = M_j|Fx=0 - y_j * Fx (planar, force at ground level)."""
    segs = d["segs"]
    return dict(ankle=segs[(side, "foot")]["prox"][:, 1], knee=segs[(side, "shank")]["prox"][:, 1],
                hip=segs[(side, "thigh")]["prox"][:, 1])


def solve(d, S, fc_smooth=8.0, w_dyn=1.0, w_mag=1e-4, prior=None, w_prior=0.0, w_torque=0.0, mu=sp_.MU):
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
    if prior is not None and w_prior > 0:
        onR, onL = FyR > sp_.F_CONTACT, FyL > sp_.F_CONTACT
        Wp = sp.diags(np.concatenate([onR, onL]).astype(float))
        pr = np.concatenate([prior["R"], prior["L"]])
        P = P + w_prior * Wp
        q = q - w_prior * (Wp @ pr)
    if w_torque > 0:
        # smoothness of joint moments of both legs: || D2 (M0_j - y_j f) ||^2  ->  || D2 Y_j f - D2 M0_j ||^2
        # M0 (moment with Fx = 0) computed from the reference ID with Fx set to zero
        w_t = w_torque * w_s / (0.9 ** 2)   # scale like force smoothness at hip lever arm ~0.9 m
        for side, blk in (("R", 0), ("L", 1)):
            M0 = sp_.inverse_dynamics(d["segs"], side, np.zeros(n), d["grf"][side]["Fy"], d["grf"][side]["copx"])
            lev = torque_lever(d, side)
            for j in ("ankle", "knee", "hip"):
                Yj = sp.diags(lev[j])
                Bj = D2 @ Yj                                   # (n-2, n)
                Bfull = sp.hstack([Bj if blk == 0 else sp.csc_matrix((n - 2, n)),
                                   Bj if blk == 1 else sp.csc_matrix((n - 2, n))], format="csc")
                target = D2 @ M0[j]
                P = P + w_t * Bfull.T @ Bfull
                q = q - w_t * (Bfull.T @ target)
    onR, onL = FyR > sp_.F_CONTACT, FyL > sp_.F_CONTACT
    lb = np.concatenate([np.where(onR, -mu * FyR, 0.0), np.where(onL, -mu * FyL, 0.0)])
    ub = np.concatenate([np.where(onR, mu * FyR, 0.0), np.where(onL, mu * FyL, 0.0)])
    prob = osqp.OSQP()
    prob.setup(P=sp.csc_matrix(2 * P), q=q * 2, A=sp.identity(2 * n, format="csc"), l=lb, u=ub,
               verbose=False, eps_abs=1e-6, eps_rel=1e-6, max_iter=50000, polish=True)
    r = prob.solve()
    return r.x[:n], r.x[n:], r.info.status


def metrics(d, fR):
    FxR, FyR, FyL = d["grf"]["R"]["Fx"], d["grf"]["R"]["Fy"], d["grf"]["L"]["Fy"]
    onR, onL = FyR > sp_.F_CONTACT, FyL > sp_.F_CONTACT
    tr = d["trim"]
    BW = BWf()
    f = lambda m: 100 * np.sqrt(np.mean((fR[m] - FxR[m]) ** 2)) / BW
    # hip moment error (right) vs reference ID
    ref = sp_.inverse_dynamics(d["segs"], "R", FxR, FyR, d["grf"]["R"]["copx"])["hip"] / sp_.MASS
    est = sp_.inverse_dynamics(d["segs"], "R", fR, FyR, d["grf"]["R"]["copx"])["hip"] / sp_.MASS
    hip = float(np.sqrt(np.mean((ref[tr] - est[tr]) ** 2)))
    return dict(stance=f(onR & tr), ds=f(onR & onL & tr), hip=hip)


if __name__ == "__main__":
    cases = [("WBDS01", "walkT05"), ("WBDS01", "walkT01"), ("WBDS01", "walkT08"),
             ("WBDS02", "walkT05"), ("WBDS05", "walkT05"), ("WBDS10", "walkT05"), ("WBDS20", "walkT05")]
    data = {c: prepare(*c) for c in cases}

    # ---------- calibration on WBDS01 T05 only
    cal = data[("WBDS01", "walkT05")]
    FxR, FyR, FxL, FyL = cal["grf"]["R"]["Fx"], cal["grf"]["R"]["Fy"], cal["grf"]["L"]["Fx"], cal["grf"]["L"]["Fy"]
    tot = np.where(FyR > 20, FxR, 0) + np.where(FyL > 20, FxL, 0)
    tr = cal["trim"]
    gains = np.arange(0.5, 1.21, 0.05)
    errs = [np.sqrt(np.mean((newton_sum(cal, g)[tr] - tot[tr]) ** 2)) for g in gains]
    g_best = float(gains[int(np.argmin(errs))])
    hs = np.arange(0.0, 1.01, 0.1)
    perr = []
    for h in hs:
        pr = vpp_prior(cal, h)
        m = (FyR > 20) & tr
        perr.append(np.sqrt(np.mean((pr["R"][m] - FxR[m]) ** 2)))
    h_best = float(hs[int(np.argmin(perr))])
    print(f"calibrated on WBDS01 T05: trunk gain = {g_best:.2f}  (Newton-sum rmse {100*min(errs)/BWf():.2f} %BW), "
          f"VPP height = {h_best:.1f} m (prior-alone stance rmse {100*min(perr)/BWf():.2f} %BW)")

    variants = {
        "base":                 dict(gain=1.0, w_prior=0.0, w_torque=0.0),
        "trunk gain":           dict(gain=g_best, w_prior=0.0, w_torque=0.0),
        "VPP prior w=0.05":     dict(gain=1.0, w_prior=0.05, w_torque=0.0),
        "VPP prior w=0.2":      dict(gain=1.0, w_prior=0.2, w_torque=0.0),
        "torque smooth w=1":    dict(gain=1.0, w_prior=0.0, w_torque=1.0),
        "gain+VPP0.05":         dict(gain=g_best, w_prior=0.05, w_torque=0.0),
        "gain+VPP0.05+torque":  dict(gain=g_best, w_prior=0.05, w_torque=1.0),
    }
    rows = []
    for c, d in data.items():
        sp_.configure(*c)
        for name, v in variants.items():
            S = newton_sum(d, v["gain"])
            pr = vpp_prior(d, h_best) if v["w_prior"] > 0 else None
            fR, fL, st = solve(d, S, prior=pr, w_prior=v["w_prior"], w_torque=v["w_torque"])
            m = metrics(d, fR)
            rows.append(dict(case=f"{c[0][-2:]}-{c[1][-3:]}", variant=name, **m, status=st))
        # prior alone (no QP) as an extra reference
        pr = vpp_prior(d, h_best)
        m = metrics(d, pr["R"])
        rows.append(dict(case=f"{c[0][-2:]}-{c[1][-3:]}", variant="VPP prior alone", **m, status="-"))
    df = pd.DataFrame(rows)
    piv = df.pivot(index="variant", columns="case", values="stance").loc[list(variants) + ["VPP prior alone"]]
    pd.set_option("display.width", 200)
    print("\nStance-phase Fx RMSE (%BW), right foot:")
    print(piv.round(2).assign(mean=piv.mean(axis=1).round(2)).to_string())
    piv = df.pivot(index="variant", columns="case", values="ds").loc[list(variants) + ["VPP prior alone"]]
    print("\nDouble-support Fx RMSE (%BW):")
    print(piv.round(2).assign(mean=piv.mean(axis=1).round(2)).to_string())
    piv = df.pivot(index="variant", columns="case", values="hip").loc[list(variants) + ["VPP prior alone"]]
    print("\nHip moment RMSE (Nm/kg), all samples:")
    print(piv.round(3).assign(mean=piv.mean(axis=1).round(3)).to_string())
    df.to_csv(os.path.join(sp_.OUT, "experiments_summary.csv"), index=False)

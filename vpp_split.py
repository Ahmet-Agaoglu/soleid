"""The arm the ablation was missing: a plain virtual-pivot-point split, in the manner of
Castellaz et al. (IEEE TNSRE 2025).

Their method obtains the total ground reaction force from a pelvis inertial unit, constrains each
foot's force to lie along the line from its center of pressure to the pivot point, and recovers
the two magnitudes by least squares at each time step, with the vertical component held
non-negative. The insole supplies the center of pressure; the per-foot vertical force is part of
what is solved for.

Reimplementing it here isolates the one thing this paper adds to that idea: the per-foot vertical
force as a measurement. This arm is therefore given the same contact timing and the same measured
center of pressure as SoleID, and is denied only the per-foot vertical force magnitude. Anything
else would confound the comparison.

Planar reduction: two balance equations (horizontal and vertical) and two unknown magnitudes, so
the double-support system is square rather than overdetermined; in single support the balance is
assigned to the loaded foot, as in the original.

Output: results/vpp_split_test.csv, vpp_split_log.txt
"""
import json
import os
import time

import numpy as np
import pandas as pd

import experiments as ex
import soleid_planar as sp_
from protocol import CAL_SUBJECTS, available_trials

CAL = json.load(open(os.path.join(sp_.OUT, "calibration.json")))["chosen"]
LOG = open(os.path.join(sp_.OUT, "vpp_split_log.txt"), "w", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


def nnls2(U, b):
    """Non-negative least squares for a 2x2 system, solved in closed form.

    With only two variables the active set is small enough to enumerate: solve unconstrained, and
    if either magnitude comes out negative, clamp it to zero and refit the other one alone. This
    is exact, and it vectorizes over frames, which per-frame scipy.optimize.nnls would not.
    U: (n,2,2) columns are the two direction vectors; b: (n,2).
    """
    n = U.shape[0]
    det = U[:, 0, 0] * U[:, 1, 1] - U[:, 0, 1] * U[:, 1, 0]
    safe = np.abs(det) > 1e-9
    lam = np.zeros((n, 2))
    d = np.where(safe, det, 1.0)
    lam[:, 0] = (b[:, 0] * U[:, 1, 1] - b[:, 1] * U[:, 0, 1]) / d
    lam[:, 1] = (U[:, 0, 0] * b[:, 1] - U[:, 1, 0] * b[:, 0]) / d
    lam[~safe] = 0.0

    for j in (0, 1):                      # clamp a negative magnitude and refit the other
        bad = lam[:, j] < 0
        if bad.any():
            k = 1 - j
            u = U[bad][:, :, k]
            denom = np.sum(u * u, axis=1)
            denom[denom < 1e-12] = 1.0
            lam[bad, j] = 0.0
            lam[bad, k] = np.maximum(np.sum(u * b[bad], axis=1) / denom, 0.0)
    return lam


def vpp_split(d, h):
    """Per-foot force from the whole-body balance and the pivot-point directions alone."""
    n = d["n"]
    grf = d["grf"]
    segs = d["segs"]
    onR = grf["R"]["Fy"] > sp_.F_CONTACT
    onL = grf["L"]["Fy"] > sp_.F_CONTACT

    Sx = sum(v["m"] * v["acom"][:, 0] for v in segs.values())
    Sz = sum(v["m"] * (v["acom"][:, 1] + sp_.G) for v in segs.values())
    vpp = d["pelvis"] + np.array([0.0, h])

    u = {}
    for s in ("R", "L"):
        v = vpp - np.column_stack([grf[s]["copx"], np.zeros(n)])
        u[s] = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)

    out = {s: np.zeros((n, 2)) for s in ("R", "L")}
    ds = onR & onL
    if ds.any():
        U = np.stack([u["R"][ds], u["L"][ds]], axis=2)       # (m,2,2)
        lam = nnls2(U, np.column_stack([Sx[ds], Sz[ds]]))
        out["R"][ds] = lam[:, [0]] * u["R"][ds]
        out["L"][ds] = lam[:, [1]] * u["L"][ds]
    for s, on_s, other in (("R", onR, onL), ("L", onL, onR)):
        ss = on_s & ~other                                    # single support: balance to that foot
        out[s][ss, 0] = Sx[ss]
        out[s][ss, 1] = np.maximum(Sz[ss], 0.0)
    return out


def metrics(d, est):
    """Same windows and units as every other arm, plus the vertical force this arm must estimate."""
    grf = d["grf"]
    FxR, FzR = grf["R"]["Fx"], grf["R"]["Fy"]
    onR = FzR > sp_.F_CONTACT
    onL = grf["L"]["Fy"] > sp_.F_CONTACT
    tr = d["trim"]
    BW = sp_.MASS * 9.81
    st, ds = onR & tr, onR & onL & tr
    fx, fz = est["R"][:, 0], est["R"][:, 1]

    def r(a, b, m):
        return 100 * np.sqrt(np.mean((a[m] - b[m]) ** 2)) / BW

    ref = sp_.inverse_dynamics(d["segs"], "R", FxR, FzR, grf["R"]["copx"])
    # this arm supplies its own vertical force, so the moments use it -- that is the method
    got = sp_.inverse_dynamics(d["segs"], "R", fx, fz, grf["R"]["copx"])
    out = dict(stance=r(fx, FxR, st), ds=r(fx, FxR, ds), vert_stance=r(fz, FzR, st))
    for j in ("ankle", "knee", "hip"):
        out[j] = float(np.sqrt(np.mean((got[j][tr] - ref[j][tr]) ** 2)) / sp_.MASS)
    return out


def main():
    trials = [t for t in available_trials() if t[0] not in CAL_SUBJECTS]
    say(f"VPP-split arm on {len(trials)} test trials, h = {CAL['h_vpp']} m")
    rows, t0 = [], time.time()
    for i, (subj, trial, speed, age) in enumerate(trials):
        try:
            d = ex.prepare(subj, trial)
        except Exception:
            continue
        sp_.configure(subj, trial)
        m = metrics(d, vpp_split(d, CAL["h_vpp"]))
        m.update(subject=subj, trial=trial, speed=speed)
        rows.append(m)
        if (i + 1) % 50 == 0:
            say(f"  {i+1}/{len(trials)} [{(time.time()-t0)/60:.1f} min]")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(sp_.OUT, "vpp_split_test.csv"), index=False)

    t = pd.read_csv(os.path.join(sp_.OUT, "test_all_trials.csv"))
    say(f"\n{len(df)} trials, {df.subject.nunique()} subjects\n")
    say(f"{'arm':28s} {'shear st':>9s} {'shear ds':>9s} {'vert st':>9s} "
        f"{'ankle':>7s} {'knee':>7s} {'hip':>7s}")
    say(f"{'VPP split (no measured Fz)':28s} {df.stance.mean():9.2f} {df.ds.mean():9.2f} "
        f"{df.vert_stance.mean():9.2f} {df.ankle.mean():7.3f} {df.knee.mean():7.3f} "
        f"{df.hip.mean():7.3f}")
    for arm, lab in (("Newton_prop", "Newton + proportional"), ("SoleID_v1", "SoleID, no prior"),
                     ("SoleID", "SoleID")):
        say(f"{lab:28s} {t[arm+'_stance'].mean():9.2f} {t[arm+'_ds'].mean():9.2f} "
            f"{'0.00':>9s} {t[arm+'_ankle'].mean():7.3f} {t[arm+'_knee'].mean():7.3f} "
            f"{t[arm+'_hip'].mean():7.3f}")
    say("\n(the other arms are given the measured vertical force, so their vertical error is zero "
        "by construction; that is the difference being measured)")

    j = df.merge(t[["subject", "trial", "SoleID_stance", "SoleID_hip"]], on=["subject", "trial"])
    say(f"\nSoleID better than the VPP split in {100*(j.SoleID_stance < j.stance).mean():.1f} % of "
        f"trials (shear), {100*(j.SoleID_hip < j.hip).mean():.1f} % (hip)")
    LOG.close()
    return df


if __name__ == "__main__":
    main()

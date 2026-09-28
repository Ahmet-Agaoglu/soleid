"""Audit part 1: the physics and the estimator, tested against cases whose answer is known
independently of the pipeline.

Every check prints PASS/FAIL with the numbers, so a failure is actionable rather than a verdict.
Run:  python audit_core.py
"""
import json
import os
import sys

import numpy as np

import experiments as ex
import soleid_planar as sp_

CAL = json.load(open(os.path.join(sp_.OUT, "calibration.json")))["chosen"]
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))


# ----------------------------------------------------------------------------- synthetic model
def synth_segments(n, mass=70.0, height=1.75, pose=None, fs=100.0):
    """A still, upright leg: hip (0,0.9), knee (0,0.5), ankle (0,0.1), toe (0.15,0.05).
    All velocities and accelerations are zero, so the inverse dynamics reduces to statics."""
    sp_.MASS, sp_.HEIGHT, sp_.FS = mass, height, fs
    pts = pose or dict(hip=(0.0, 0.90), knee=(0.0, 0.50), ankle=(0.0, 0.10), toe=(0.15, 0.05))
    P = {k: np.tile(np.array(v, float), (n, 1)) for k, v in pts.items()}
    segs = {}
    for name, prox, dist in (("foot", "ankle", "toe"), ("shank", "knee", "ankle"), ("thigh", "hip", "knee")):
        par = sp_.SEG[name]
        a, b = P[prox], P[dist]
        L = np.linalg.norm(b - a, axis=1).mean()
        m = par["m"] * mass
        com = a + par["c"] * (b - a)
        segs[("R", name)] = dict(m=m, I=m * (par["rg"] * L) ** 2, com=com,
                                 acom=np.zeros((n, 2)), alpha=np.zeros(n), prox=a, dist=b, L=L)
    segs[("B", "hat")] = dict(m=sp_.SEG["hat"]["m"] * mass, I=1.0, com=np.tile([0.0, 1.2], (n, 1)),
                              acom=np.zeros((n, 2)), alpha=np.zeros(n),
                              prox=np.tile([0.0, 0.9], (n, 1)), dist=np.tile([0.0, 1.2], (n, 1)), L=0.5)
    return segs, P


def test_statics():
    """Standing on one leg: the ankle moment must equal the weight of everything above the ankle
    times the horizontal distance from the ankle to the centre of pressure (plus the foot's own
    weight term). Computed here by hand, independently of inverse_dynamics()."""
    n, mass = 50, 70.0
    segs, P = synth_segments(n, mass)
    g = 9.81
    m_foot = sp_.SEG["foot"]["m"] * mass
    # vertical force: the whole body weight minus... in this synthetic leg only the leg segments
    # exist, so use a GRF that equals the weight of (shank + thigh + hat) + foot, applied at cop_x
    m_above = (sp_.SEG["shank"]["m"] + sp_.SEG["thigh"]["m"] + sp_.SEG["hat"]["m"]) * mass
    Fy = (m_above + m_foot) * g
    for cop_x in (-0.05, 0.0, 0.05, 0.10):
        Fx = np.zeros(n)
        M = sp_.inverse_dynamics(segs, "R", Fx, np.full(n, Fy), np.full(n, cop_x))
        # hand calculation for the ankle: sum of moments about the ankle joint of the foot segment
        ankle = P["ankle"][0]
        com_f = segs[("R", "foot")]["com"][0]
        m_ankle_hand = -((cop_x - ankle[0]) * Fy - (0.0 - ankle[1]) * 0.0) \
                       - ((com_f[0] - ankle[0]) * (-m_foot * g))
        err = abs(M["ankle"][10] - m_ankle_hand)
        check(f"statics: ankle moment at cop_x={cop_x:+.2f} m", err < 1e-6,
              f"pipeline {M['ankle'][10]:+.3f} Nm, hand {m_ankle_hand:+.3f} Nm")


def test_swing():
    """Foot in the air (no ground force): the ankle moment must equal the foot's own inertia term.
    With zero acceleration and a horizontal force of zero, that moment is the foot weight term."""
    n, mass = 50, 70.0
    segs, P = synth_segments(n, mass)
    M = sp_.inverse_dynamics(segs, "R", np.zeros(n), np.zeros(n), np.zeros(n))
    m_foot = sp_.SEG["foot"]["m"] * mass
    ankle, com_f = P["ankle"][0], segs[("R", "foot")]["com"][0]
    hand = -((com_f[0] - ankle[0]) * (-m_foot * 9.81))
    check("swing: ankle moment equals the foot weight term", abs(M["ankle"][10] - hand) < 1e-6,
          f"pipeline {M['ankle'][10]:+.4f} Nm, hand {hand:+.4f} Nm")


def test_linearity_in_shear():
    """The pipeline assumes M_j = M_j|Fx=0 - y_j Fx. Verify that on the synthetic model."""
    n, mass = 50, 70.0
    segs, P = synth_segments(n, mass)
    Fy = np.full(n, 600.0)
    M0 = sp_.inverse_dynamics(segs, "R", np.zeros(n), Fy, np.zeros(n))
    Fx = np.full(n, 120.0)
    M1 = sp_.inverse_dynamics(segs, "R", Fx, Fy, np.zeros(n))
    for j, prox in (("ankle", "ankle"), ("knee", "knee"), ("hip", "hip")):
        y = P[prox][0, 1]
        pred = M0[j][10] - y * 120.0
        check(f"linearity in shear at the {j}", abs(M1[j][10] - pred) < 1e-6,
              f"actual {M1[j][10]:+.2f}, predicted {pred:+.2f} (lever {y:.2f} m)")


def test_cop_height():
    """With a raised centre of pressure the lever arm must shrink accordingly."""
    n, mass = 50, 70.0
    segs, P = synth_segments(n, mass)
    Fx, Fy = np.full(n, 100.0), np.full(n, 600.0)
    M0 = sp_.inverse_dynamics(segs, "R", Fx, Fy, np.zeros(n))
    h = 0.2
    M1 = sp_.inverse_dynamics(segs, "R", Fx, Fy, np.zeros(n), copy=np.full(n, h))
    d = M1["hip"][10] - M0["hip"][10]
    check("cop height enters the moment arm", abs(d - h * 100.0) < 1e-6,
          f"difference {d:+.3f} Nm, expected {h * 100.0:+.3f}")


# ----------------------------------------------------------------------------- estimator checks
def load_real(subj="WBDS02", trial="walkT05"):
    d = ex.prepare(subj, trial)
    sp_.configure(subj, trial)
    return d


def test_constraints(d):
    S = ex.newton_sum(d)
    prior = ex.vpp_prior(d, CAL["h_vpp"])
    fR, fL, st = ex.solve(d, S, fc_smooth=CAL["fc_smooth"], prior=prior, w_prior=CAL["w_prior"], w_torque=0.0)
    check("solver reports success", st == "solved", st)
    for s, f in (("R", fR), ("L", fL)):
        Fy = d["grf"][s]["Fy"]
        air = Fy <= sp_.F_CONTACT
        check(f"{s}: shear is zero while the foot is in the air", np.max(np.abs(f[air])) < 1e-6,
              f"max |Fx| in air = {np.max(np.abs(f[air])):.2e} N")
        on = ~air
        viol = np.max(np.abs(f[on]) - sp_.MU * Fy[on])
        check(f"{s}: friction cone respected", viol < 1e-3, f"worst violation {viol:.2e} N")
    return fR, fL, S


def test_recovery(d):
    """Plumbing test: if the true shear is supplied as the prior with a large weight, the solution
    must reproduce it (otherwise the prior is not entering the cost as intended)."""
    S = ex.newton_sum(d)
    truth = {s: d["grf"][s]["Fx"].copy() for s in ("R", "L")}
    fR, fL, st = ex.solve(d, S, fc_smooth=CAL["fc_smooth"], prior=truth, w_prior=1e4, w_torque=0.0)
    on = d["grf"]["R"]["Fy"] > sp_.F_CONTACT
    err = np.sqrt(np.mean((fR[on] - truth["R"][on]) ** 2))
    check("prior plumbing: truth as prior is recovered", err < 2.0, f"RMSE {err:.3f} N")


def test_dynamics_term(d, fR, fL, S):
    """The dynamics residual of the solution should be small relative to the measured total."""
    tot = fR + fL
    r = np.sqrt(np.mean((tot[d["trim"]] - S[d["trim"]]) ** 2))
    meas = (np.where(d["grf"]["R"]["Fy"] > sp_.F_CONTACT, d["grf"]["R"]["Fx"], 0) +
            np.where(d["grf"]["L"]["Fy"] > sp_.F_CONTACT, d["grf"]["L"]["Fx"], 0))
    rm = np.sqrt(np.mean((S[d["trim"]] - meas[d["trim"]]) ** 2))
    check("solution follows the whole-body dynamics term", r < max(5.0, 0.5 * rm),
          f"|fR+fL-S| = {r:.2f} N ; |S-measured| = {rm:.2f} N")


def test_metric_independence(d, fR):
    """Recompute the headline metric by a second, independent path."""
    BW = sp_.MASS * 9.81
    on = d["grf"]["R"]["Fy"] > sp_.F_CONTACT
    m = on & d["trim"]
    a = 100 * np.sqrt(np.mean((fR[m] - d["grf"]["R"]["Fx"][m]) ** 2)) / BW
    err = fR - d["grf"]["R"]["Fx"]
    b = 100 * float(np.sqrt(np.sum(err[m] ** 2) / m.sum())) / BW
    check("stance metric reproducible by an independent path", abs(a - b) < 1e-9, f"{a:.6f} vs {b:.6f}")


def test_reference_uses_measured_force(d):
    """The reference must use the measured shear; a silent fallback to the estimate would make
    every comparison meaningless."""
    ref1 = sp_.inverse_dynamics(d["segs"], "R", d["grf"]["R"]["Fx"], d["grf"]["R"]["Fy"], d["grf"]["R"]["copx"])
    ref2 = sp_.inverse_dynamics(d["segs"], "R", np.zeros(d["n"]), d["grf"]["R"]["Fy"], d["grf"]["R"]["copx"])
    diff = np.max(np.abs(ref1["hip"] - ref2["hip"]))
    check("reference depends on the measured shear", diff > 1.0, f"max difference {diff:.1f} Nm")


if __name__ == "__main__":
    print("== analytic physics ==")
    test_statics()
    test_swing()
    test_linearity_in_shear()
    test_cop_height()
    print("\n== estimator on real data (WBDS02 walkT05) ==")
    d = load_real()
    fR, fL, S = test_constraints(d)
    test_recovery(d)
    test_dynamics_term(d, fR, fL, S)
    test_metric_independence(d, fR)
    test_reference_uses_measured_force(d)
    bad = [n for n, ok in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} checks passed")
    if bad:
        print("FAILED:", "; ".join(bad))
        sys.exit(1)

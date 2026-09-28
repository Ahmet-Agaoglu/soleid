"""Which of the two wearable inputs costs more: the kinematics or the insole?

The application layer already has two arms that differ only in the kinematics source, and the
paper reads them as separating the error into its two causes. They do not: both use the real
insole, so the kinematics contribution is isolated but the insole's never is. This script adds
the missing cells and closes the square.

              insole force        plate force
  IMU kin     SoleID              (new)
  lab kin     SoleID_lab_kin      (new: the floor)

The fourth cell is worth as much as the third. With laboratory kinematics and the treadmill's own
vertical force and center of pressure, the only thing still estimated is the shear, so that arm is
the error the method cannot go below on this data set -- everything else is measured.

Center of pressure. The insole reports its center of pressure in the foot frame, and each arm
places it on the heel-toe line of the kinematics that arm uses, which is why no rigid alignment
between the Xsens and laboratory frames is needed. The plate reports a point on the ground
instead (its vertical coordinate is exactly zero in these files), so to keep the two forces
interchangeable we project that point onto the laboratory heel-toe line, obtaining the same kind
of along-the-foot distance the insole gives, and then place it exactly as the insole's is placed.
The four arms then differ only in which sensor supplied the vertical force and the center of
pressure, and in which kinematics drove the model.

Output: results/wang_factorial.csv, results/wang_factorial_log.txt
"""
import os

import numpy as np
import pandas as pd

import experiments as ex
import soleid_planar as sp_
import wang_adapter as wa
from wang_run import (CAL, FS, build, hip_from_pelvis_planar, insole_cop_to_lab,
                      planar_from_markers)

LOG = open(os.path.join(sp_.OUT, "wang_factorial_log.txt"), "w", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


def plate_cop_local(P, side, copx, n, heel_key=None):
    """The plate's center of pressure as a distance from the heel end along the foot, which is how
    the insole reports its own. The plate point lies on the ground, so it is projected onto the
    heel-toe line rather than differenced along x only."""
    ank, toe = P[side + ".Ankle"][:n], P[side + ".MT5"][:n]
    if heel_key is not None and heel_key in P:
        heel = P[heel_key][:n]
    else:
        v = toe - ank
        L = np.linalg.norm(v, axis=1, keepdims=True)
        heel = ank - 0.25 * (L / 0.75) * (v / np.maximum(L, 1e-9))
    v = toe - heel
    u = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
    p = np.column_stack([copx[:n], np.zeros(n)])
    return np.einsum("ij,ij->i", p - heel, u)


def run_trial(subj, trial, info):
    d = wa.load_trial(subj, trial)
    mass, height = info[subj]["mass"], info[subj]["height"]
    n = len(d["t"])
    sp_.MASS, sp_.HEIGHT, sp_.FS = mass, height, FS
    BW = mass * 9.81

    trc = wa.read_trc(d["paths"]["trc"])
    n = min(n, len(trc[0]))
    Plab, pelvis_lab, lab_ok = planar_from_markers(trc, n)
    Plab = {k: (sp_.lowpass(v, 6.0, FS) if isinstance(v, np.ndarray) else v)
            for k, v in Plab.items()}
    pelvis_lab = sp_.lowpass(pelvis_lab, 6.0, FS)
    Plab["R.Hip"] = Plab["L.Hip"] = hip_from_pelvis_planar(pelvis_lab, height)
    segs_lab = build(Plab, mass, height, n)

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
    segs_w = build(Pw, mass, height, n)
    grf_ins = {s: {kk: vv[:n] for kk, vv in d["insole"][s].items()} for s in ("R", "L")}
    grf_ref = {s: {kk: vv[:n] for kk, vv in d["ref"][s].items()} for s in ("R", "L")}

    # the two forces, each expressed as an along-the-foot distance so they are interchangeable
    cop_local = {
        "insole": {s: grf_ins[s]["copx"] for s in ("R", "L")},
        "plate": {s: plate_cop_local(Plab, s, grf_ref[s]["copx"], n, heel_key=s + ".Heel")
                  for s in ("R", "L")},
    }
    Fz = {"insole": {s: grf_ins[s]["Fy"] for s in ("R", "L")},
          "plate": {s: grf_ref[s]["Fy"] for s in ("R", "L")}}

    trim = np.zeros(n, bool)
    trim[int(FS):n - int(FS)] = True
    onR = grf_ref["R"]["Fy"][:n] > sp_.F_CONTACT
    ds = onR & (grf_ref["L"]["Fy"][:n] > sp_.F_CONTACT)

    ref = {s: sp_.inverse_dynamics(segs_lab, s, grf_ref[s]["Fx"], grf_ref[s]["Fy"],
                                   grf_ref[s]["copx"]) for s in ("R", "L")}

    S_xsens = mass * sp_.lowpass(k["com_acc"][:n, 0], 10.0, FS)
    KIN = {"imu": (segs_w, Pw, Pw["Pelvis"], None, S_xsens),
           "lab": (segs_lab, Plab, pelvis_lab, "Heel", None)}

    row = dict(subject=subj, trial=trial, speed=d["speed"], mass=mass, n=n, lab_ok=bool(lab_ok),
               fy_rmse_pctBW=100 * np.sqrt(np.mean((grf_ins["R"]["Fy"][trim]
                                                    - grf_ref["R"]["Fy"][trim]) ** 2)) / BW)
    for kin, (segs, P, pelvis, hk, S_fixed) in KIN.items():
        for src in ("insole", "plate"):
            g = {s: dict(Fy=Fz[src][s],
                         copx=insole_cop_to_lab(P, s, cop_local[src][s], n,
                                                heel_key=(s + "." + hk) if hk else None))
                 for s in ("R", "L")}
            dd = dict(n=n, segs=segs, pelvis=pelvis, grf=g, trim=trim)
            S = ex.newton_sum(dd) if S_fixed is None else S_fixed
            pr = ex.vpp_prior(dd, CAL["h_vpp"])
            fR, _, _ = ex.solve(dd, S, fc_smooth=CAL["fc_smooth"], prior=pr,
                                w_prior=CAL["w_prior"], w_torque=0.0)
            name = f"{kin}_{src}"
            row[f"{name}_stance"] = 100 * np.sqrt(np.mean(
                (fR[onR & trim] - grf_ref["R"]["Fx"][onR & trim]) ** 2)) / BW
            row[f"{name}_ds"] = 100 * np.sqrt(np.mean(
                (fR[ds & trim] - grf_ref["R"]["Fx"][ds & trim]) ** 2)) / BW
            m = sp_.inverse_dynamics(segs, "R", fR, g["R"]["Fy"], g["R"]["copx"])
            for j in ("ankle", "knee", "hip"):
                a, b = m[j][trim] / mass, ref["R"][j][trim] / mass
                ok = lab_ok and np.isfinite(a).all() and np.isfinite(b).all()
                row[f"{name}_{j}"] = float(np.sqrt(np.mean((a - b) ** 2))) if ok else np.nan
    return row


def main():
    info = wa.subject_info()
    done = pd.read_csv(os.path.join(sp_.OUT, "wang_application.csv"))
    trials = list(zip(done.subject, done.trial))
    say(f"{len(trials)} trials, the same ones as the application layer")
    rows = []
    for i, (s, t) in enumerate(trials):
        try:
            rows.append(run_trial(s, t, info))
        except Exception as e:
            say(f"  skip {s} {t}: {e!r}")
        if (i + 1) % 10 == 0:
            say(f"  {i+1}/{len(trials)}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(sp_.OUT, "wang_factorial.csv"), index=False)

    ok = df[df.lab_ok] if "lab_ok" in df else df
    say(f"\n{len(ok)} trials with usable laboratory markers; insole vs treadmill vertical force "
        f"{df.fy_rmse_pctBW.mean():.1f} %BW\n")
    for metric in ("hip", "knee", "ankle", "stance"):
        unit = "%BW" if metric == "stance" else "Nm/kg"
        say(f"  {metric} ({unit})")
        say(f"{'':>12s} {'insole':>10s} {'plate':>10s} {'insole cost':>12s}")
        for kin in ("imu", "lab"):
            a = ok[f"{kin}_insole_{metric}"].mean()
            b = ok[f"{kin}_plate_{metric}"].mean()
            say(f"{kin:>12s} {a:10.3f} {b:10.3f} {a-b:12.3f}")
        ki = ok[f"imu_insole_{metric}"].mean() - ok[f"lab_insole_{metric}"].mean()
        kp = ok[f"imu_plate_{metric}"].mean() - ok[f"lab_plate_{metric}"].mean()
        say(f"{'kin cost':>12s} {ki:10.3f} {kp:10.3f}")
        say("")
    LOG.close()
    return df


if __name__ == "__main__":
    main()

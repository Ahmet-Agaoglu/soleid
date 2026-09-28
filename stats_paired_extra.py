"""The remaining contrasts of the paper, in the within-trial paired form, and two further numbers.

  1. Application layer: laboratory minus IMU kinematics with the real insole (the contrast quoted
     in IV-E), paired within trial, random intercept per subject.
  2. The 2x2 on the Wang data: the four conditional replacement effects (each input replaced with
     the other held at either level), the direct comparison of the two costs, and the interaction
     (whether the cost of one input depends on the level of the other). Note that "the ordering is
     the same from either corner" is an identity -- both corners reduce to imu_plate - lab_insole --
     so it is not reported as a check.
  3. The kinematics-only arm's own vertical-force error over stance, for Table IV.

Output: results/stats_paired_extra.json
"""
import json
import os
import warnings

import numpy as np
import pandas as pd

import experiments as ex
import soleid_planar as sp_
from protocol import CAL_SUBJECTS, available_trials
from stats_revision import robust_fit

warnings.filterwarnings("ignore")
OUT = sp_.OUT


def paired_diff(subject, diff, speed):
    d = pd.DataFrame(dict(subject=np.asarray(subject), diff=np.asarray(diff, float),
                          speed_c=np.asarray(speed, float) - np.mean(speed)))
    d = d[np.isfinite(d["diff"])]
    e, lo, hi, p = robust_fit("diff ~ speed_c", d, "diff")["Intercept"]
    return dict(estimate=e, lo=lo, hi=hi, p=p, n=int(len(d)))


def main():
    res = {}
    # ---- 1. application layer, kinematics contrast with the real insole
    w = pd.read_csv(os.path.join(OUT, "wang_application.csv"))
    res["wang_kin_contrast"] = {k: paired_diff(w.subject, w[f"SoleID_lab_kin_{k}"] - w[f"SoleID_{k}"],
                                               w.speed) for k in ("hip", "stance")}
    # ---- 2. the square
    f = pd.read_csv(os.path.join(OUT, "wang_factorial.csv"))
    f = f[f.lab_ok]
    sq = {}
    for k in ("hip", "knee", "ankle", "stance"):
        c = {a: f[f"{a}_{k}"] for a in ("imu_insole", "imu_plate", "lab_insole", "lab_plate")}
        sq[k] = dict(
            kin_with_insole=paired_diff(f.subject, c["imu_insole"] - c["lab_insole"], f.speed),
            kin_with_plate=paired_diff(f.subject, c["imu_plate"] - c["lab_plate"], f.speed),
            insole_with_imu=paired_diff(f.subject, c["imu_insole"] - c["imu_plate"], f.speed),
            insole_with_lab=paired_diff(f.subject, c["lab_insole"] - c["lab_plate"], f.speed),
            kin_minus_insole=paired_diff(f.subject, c["imu_plate"] - c["lab_insole"], f.speed),
            interaction=paired_diff(f.subject, (c["imu_insole"] - c["lab_insole"])
                                    - (c["imu_plate"] - c["lab_plate"]), f.speed),
            n=int(len(f)))
    res["square"] = sq
    for k, v in sq.items():
        print(f"\n{k}")
        for name in ("kin_with_insole", "kin_with_plate", "insole_with_imu", "insole_with_lab",
                     "kin_minus_insole", "interaction"):
            r = v[name]
            print(f"   {name:17s} {r['estimate']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]  p {r['p']:.1e}")
    r = res["wang_kin_contrast"]
    print(f"\nWang, lab minus IMU kinematics (insole): hip {r['hip']['estimate']:+.3f} "
          f"[{r['hip']['lo']:+.3f}, {r['hip']['hi']:+.3f}], stance {r['stance']['estimate']:+.3f} "
          f"[{r['stance']['lo']:+.3f}, {r['stance']['hi']:+.3f}] p {r['stance']['p']:.2f}")

    # ---- 3. the kinematics-only arm's vertical force
    CAL = json.load(open(os.path.join(OUT, "calibration.json")))["chosen"]
    errs = []
    for s, t, *_ in [x for x in available_trials() if x[0] not in CAL_SUBJECTS]:
        try:
            d = ex.prepare(s, t)
        except Exception:
            continue
        sp_.configure(s, t)
        kin, _ = sp_.kinematics_only_baseline(d["P"], d["segs"], d["n"])
        g = d["grf"]["R"]
        on = (g["Fy"] > sp_.F_CONTACT) & d["trim"]
        errs.append(100 * float(np.sqrt(np.mean((kin["R"]["Fy"][on] - g["Fy"][on]) ** 2)))
                    / (sp_.MASS * 9.81))
    res["kin_only_vertical_stance"] = dict(mean=float(np.mean(errs)), n=len(errs))
    print(f"\nkinematics-only vertical force RMSE over stance: {np.mean(errs):.2f} %BW (n {len(errs)})")
    json.dump(res, open(os.path.join(OUT, "stats_paired_extra.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    main()

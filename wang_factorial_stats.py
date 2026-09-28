"""Confidence intervals for the two input costs, on the same footing.

The kinematics contrast in the paper already carries a mixed-model interval; the insole contrast
from the factorial would otherwise be a bare difference of means, which is not a fair pairing. Both
are estimated here in one model per outcome, with the wearable configuration (IMU kinematics,
insole force) as the reference, so each cost is read from the corner a user of the wearable
actually stands in. The other corner is reported too, because a statement that "the insole costs
more" is only safe if the ordering does not depend on which corner it is measured from.

Output: results/wang_factorial_stats.json
"""
import json
import os

import pandas as pd

import soleid_planar as sp_
from statistics import mixed_model

ARMS = ["imu_insole", "imu_plate", "lab_insole", "lab_plate"]


def main():
    df = pd.read_csv(os.path.join(sp_.OUT, "wang_factorial.csv"))
    df = df[df.lab_ok]
    out = {}
    for key in ("hip", "knee", "ankle", "stance"):
        m = mixed_model(df, ARMS, key, f"Wang factorial, {key}", reference="imu_insole")
        c = m["contrasts"]
        # cost = how much worse the wearable input is than the laboratory one, so flip the sign
        out[key] = dict(
            insole_cost=dict(estimate=-c["imu_plate"]["estimate"], lo=-c["imu_plate"]["hi"],
                             hi=-c["imu_plate"]["lo"], p=c["imu_plate"]["p"]),
            kin_cost=dict(estimate=-c["lab_insole"]["estimate"], lo=-c["lab_insole"]["hi"],
                          hi=-c["lab_insole"]["lo"], p=c["lab_insole"]["p"]),
            floor=float(df["lab_plate_" + key].mean()),
            wearable=float(df["imu_insole_" + key].mean()),
        )
        # the same two costs measured from the other corner
        out[key]["insole_cost_lab_kin"] = float(df[f"lab_insole_{key}"].mean()
                                                - df[f"lab_plate_{key}"].mean())
        out[key]["kin_cost_plate"] = float(df[f"imu_plate_{key}"].mean()
                                           - df[f"lab_plate_{key}"].mean())
    print()
    print(f"{'':>8s} {'kinematics cost [95% CI]':>30s} {'insole cost [95% CI]':>30s} "
          f"{'same order from the other corner?':>34s}")
    for key, r in out.items():
        k, i = r["kin_cost"], r["insole_cost"]
        same = (k["estimate"] > i["estimate"]) == (r["kin_cost_plate"] > r["insole_cost_lab_kin"])
        print(f"{key:>8s} {k['estimate']:8.3f} [{k['lo']:6.3f}, {k['hi']:6.3f}]  p={k['p']:.1e}"
              f"   {i['estimate']:8.3f} [{i['lo']:6.3f}, {i['hi']:6.3f}]  p={i['p']:.1e}"
              f"   {'yes' if same else 'NO'}")
        r["order_robust"] = bool(same)
    with open(os.path.join(sp_.OUT, "wang_factorial_stats.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


if __name__ == "__main__":
    main()

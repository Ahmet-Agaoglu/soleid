"""How often does the friction cone actually bind?

The constraint |Fx| <= mu*Fz guarantees that the estimate stays physically possible, but a
guarantee that is never reached does no work in the solution. If it never binds, then the choice
of mu does not influence any reported number, which is worth stating explicitly.

Output: results/friction_cone.json
"""
import json
import os

import numpy as np

import experiments as ex
import soleid_planar as sp_
from protocol import CAL_SUBJECTS, available_trials

CAL = json.load(open(os.path.join(sp_.OUT, "calibration.json")))["chosen"]


def main():
    trials = [t for t in available_trials() if t[0] not in CAL_SUBJECTS]
    frac_active, peak_ratio, n_ok = [], [], 0
    for i, (subj, trial, *_ ) in enumerate(trials):
        try:
            d = ex.prepare(subj, trial)
        except Exception:
            continue
        sp_.configure(subj, trial)
        S = ex.newton_sum(d, 1.0)
        pr = ex.vpp_prior(d, CAL["h_vpp"])
        fR, fL, _ = ex.solve(d, S, fc_smooth=CAL["fc_smooth"], prior=pr,
                             w_prior=CAL["w_prior"], w_torque=0.0)
        for f, s in ((fR, "R"), (fL, "L")):
            Fy = d["grf"][s]["Fy"]
            on = (Fy > sp_.F_CONTACT) & d["trim"]
            if on.sum() < sp_.FS:
                continue
            ratio = np.abs(f[on]) / Fy[on]
            # "binding" means the solution sits on the bound to within a thousandth
            frac_active.append(float(np.mean(ratio > 0.999 * sp_.MU)))
            peak_ratio.append(float(ratio.max()))
        n_ok += 1
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(trials)}", flush=True)

    out = dict(n_trials=n_ok, n_limbs=len(frac_active), mu=sp_.MU,
               frac_frames_active=float(np.mean(frac_active)),
               max_frac_in_any_limb=float(np.max(frac_active)),
               peak_ratio_mean=float(np.mean(peak_ratio)),
               peak_ratio_max=float(np.max(peak_ratio)))
    with open(os.path.join(sp_.OUT, "friction_cone.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\n{out['n_trials']} trials, {out['n_limbs']} limb-trials, mu = {out['mu']}")
    print(f"  frames on the bound: {100*out['frac_frames_active']:.4f} % "
          f"(worst limb {100*out['max_frac_in_any_limb']:.4f} %)")
    print(f"  |Fx|/Fz reached: mean peak {out['peak_ratio_mean']:.2f}, "
          f"largest {out['peak_ratio_max']:.2f}")


if __name__ == "__main__":
    main()

"""How much does the sole normal differ from the vertical, and what does treating one as the
other cost?

A pressure-sensing insole responds to load normal to its own surface, which is the sole of the
shoe, not to the global vertical. The two coincide only while the foot is flat. Equation (1)
treats the measured force as vertical, so the size of that approximation should be a number rather
than a caveat.

For each loaded frame we take the sole line from heel to fifth metatarsal, form its normal in the
sagittal plane, and compute what an ideal insole would report, F.n, from the measured force
vector. Comparing that with the measured vertical component gives the error the approximation
introduces. Everything is weighted the way it matters: a large pitch angle at 20 N is irrelevant,
a small one at body weight is not.

Output: results/sole_normal.json
"""
import json
import os

import numpy as np

import experiments as ex
import soleid_planar as sp_
from protocol import CAL_SUBJECTS, available_trials


def main(limit=None):
    trials = [t for t in available_trials() if t[0] not in CAL_SUBJECTS]
    if limit:
        trials = trials[:limit]
    ang_w, err, rel, frames, n_ok = [], [], [], 0, 0
    big = []                       # pitch while the foot carries more than half body weight
    for subj, trial, *_ in trials:
        try:
            d = ex.prepare(subj, trial)
        except Exception:
            continue
        sp_.configure(subj, trial)
        BW = sp_.MASS * 9.81
        for s in ("R", "L"):
            g = d["grf"][s]
            Fz, Fx = g["Fy"], g["Fx"]
            on = (Fz > sp_.F_CONTACT) & d["trim"]
            if on.sum() < sp_.FS:
                continue
            heel, toe = d["P"][f"{s}.Heel"], d["P"][f"{s}.MT5"]
            v = toe - heel                                   # sole line, heel to toe
            th = np.arctan2(v[:, 1], v[:, 0])                # pitch, positive = toe above heel
            nx, nz = -np.sin(th), np.cos(th)                 # unit normal to the sole
            reading = Fx * nx + Fz * nz                      # what an ideal insole would report
            w = Fz[on]
            ang_w.append(float(np.sum(np.abs(np.degrees(th[on])) * w) / np.sum(w)))
            err.append(100 * float(np.sqrt(np.mean((reading[on] - Fz[on]) ** 2))) / BW)
            rel.append(100 * float(np.mean(np.abs(reading[on] - Fz[on]) / np.maximum(Fz[on], 1))))
            hi = on & (Fz > 0.5 * BW)
            if hi.any():
                big.append(float(np.mean(np.abs(np.degrees(th[hi])))))
            frames += int(on.sum())
        n_ok += 1

    out = dict(n_trials=n_ok, n_frames=frames,
               pitch_deg_force_weighted=float(np.mean(ang_w)),
               pitch_deg_above_half_BW=float(np.mean(big)),
               vertical_error_pctBW=float(np.mean(err)),
               vertical_error_pct_of_local=float(np.mean(rel)))
    with open(os.path.join(sp_.OUT, "sole_normal.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"{out['n_trials']} trials, {out['n_frames']} loaded frames")
    print(f"  sole pitch, weighted by vertical force : {out['pitch_deg_force_weighted']:.1f} deg")
    print(f"  sole pitch while carrying > 0.5 BW     : {out['pitch_deg_above_half_BW']:.1f} deg")
    print(f"  treating the sole-normal reading as the vertical force:")
    print(f"     RMSE {out['vertical_error_pctBW']:.2f} %BW   "
          f"(mean {out['vertical_error_pct_of_local']:.1f} % of the instantaneous force)")


if __name__ == "__main__":
    main()

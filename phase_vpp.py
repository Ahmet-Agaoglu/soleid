"""Does a phase-dependent virtual pivot point remove the systematic hip bias?

Motivation. Vielemeyer et al. (2021) report that the ground reaction forces intersect
significantly *above* the centre of mass during single support, but only *around* it during
double support -- that is, the pivot point effectively collapses towards the centre of mass when
both feet are loaded. Our frozen estimator applies one height h = 0.20 m in every phase, and
carries a systematic prior bias of about -19.6 N that reaches the hip moment as a positive
offset (hip_bias_decomp.py). A global change of h does not remove it, but a global change cannot: the correction the literature points to is phase-dependent, and only a
phase-dependent *weight* was tried before, never a phase-dependent *height*.

Protocol. The search runs on the five calibration subjects only, with the same selection rule as
the original calibration (lowest stance shear RMSE). The test set is touched once, at the end,
and only to report -- never to choose. The frozen estimator is left untouched in
experiments.py; this is a separate, clearly labelled variant.

Output: results/phase_vpp_grid_narrow.csv (first grid, no options); results/phase_vpp_grid.csv and
results/phase_vpp_test.csv (extended grid and the test run: --wide --test); phase_vpp_log.txt
"""
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import experiments as ex
import soleid_planar as sp_
from protocol import CAL_SUBJECTS, available_trials

CAL = json.load(open(os.path.join(sp_.OUT, "calibration.json")))["chosen"]
LOG = open(os.path.join(sp_.OUT, "phase_vpp_log.txt"), "w", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


def vpp_prior_phase(d, h_single, h_double, blend=0):
    """The direction prior with a pivot height that depends on the support phase.

    h_single applies while one foot is loaded, h_double while both are. With blend > 0 the
    height is moved between the two over that many frames on each side of a transition, so the
    prior does not step discontinuously into the smoothness term.
    """
    FyR, FyL = d["grf"]["R"]["Fy"], d["grf"]["L"]["Fy"]
    onR, onL = FyR > sp_.F_CONTACT, FyL > sp_.F_CONTACT
    ds = onR & onL
    h = np.where(ds, h_double, h_single).astype(float)
    if blend > 0:
        k = np.ones(2 * blend + 1) / (2 * blend + 1)
        h = np.convolve(np.pad(h, blend, mode="edge"), k, mode="valid")

    out = {}
    px, py = d["pelvis"][:, 0], d["pelvis"][:, 1] + h
    for s in ("R", "L"):
        Fy, cop = d["grf"][s]["Fy"], d["grf"][s]["copx"]
        cy = d["grf"][s].get("copy")
        cy = np.zeros_like(Fy) if cy is None else cy
        on = Fy > sp_.F_CONTACT
        lever = np.maximum(py - np.where(on, cy, 0.0), 0.2)
        out[s] = np.where(on, Fy * (px - cop) / lever, 0.0)
    return out


def full_metrics(d, fR, prior):
    """RMSE as in the frozen protocol, plus the signed biases the experiment is aimed at."""
    FxR, FyR, FyL = d["grf"]["R"]["Fx"], d["grf"]["R"]["Fy"], d["grf"]["L"]["Fy"]
    onR, onL = FyR > sp_.F_CONTACT, FyL > sp_.F_CONTACT
    tr = d["trim"]
    BW = sp_.MASS * 9.81
    st, ds = onR & tr, onR & onL & tr

    def rmse(m):
        return 100 * np.sqrt(np.mean((fR[m] - FxR[m]) ** 2)) / BW

    ref = sp_.inverse_dynamics(d["segs"], "R", FxR, FyR, d["grf"]["R"]["copx"])["hip"] / sp_.MASS
    est = sp_.inverse_dynamics(d["segs"], "R", fR, FyR, d["grf"]["R"]["copx"])["hip"] / sp_.MASS
    return dict(
        stance=rmse(st), ds=rmse(ds),
        hip=float(np.sqrt(np.mean((ref[tr] - est[tr]) ** 2))),
        # over the whole record the swing phase, where the two agree, dilutes the offset;
        # the stance-window value is the one that matches how the bias was first reported
        hip_bias=float(np.mean(est[tr] - ref[tr])),
        hip_bias_stance=float(np.mean(est[st] - ref[st])),
        sol_bias_N=float(np.mean(fR[st] - FxR[st])),
        prior_bias_N=float(np.mean(prior["R"][st] - FxR[st])),
    )


def run_set(trials, h_s, h_d, blend, cache):
    rows = []
    for key in trials:
        d = cache[key]
        sp_.configure(*key)
        S = ex.newton_sum(d, 1.0)
        pr = vpp_prior_phase(d, h_s, h_d, blend)
        fR, _, _ = ex.solve(d, S, fc_smooth=CAL["fc_smooth"], prior=pr,
                            w_prior=CAL["w_prior"], w_torque=0.0)
        rows.append(full_metrics(d, fR, pr))
    return pd.DataFrame(rows).mean().to_dict()


def load(trials):
    cache = {}
    for subj, trial, *_ in trials:
        try:
            cache[(subj, trial)] = ex.prepare(subj, trial)
        except Exception as e:
            say(f"  skip {subj} {trial}: {e!r}")
    return cache


def main():
    t0 = time.time()
    cal_trials = [t for t in available_trials() if t[0] in CAL_SUBJECTS]
    say(f"calibration trials: {len(cal_trials)} from {len(CAL_SUBJECTS)} subjects")
    cache = load(cal_trials)
    keys = list(cache)
    say(f"loaded {len(keys)}\n")

    if "--wide" in sys.argv:
        # the first grid put its optimum at the top edge of h_single, so extend it far enough
        # to find the turning point. Raising h lengthens the lever and therefore shrinks the
        # prior towards zero, so a monotone improvement here would mean "a weaker prior is
        # better", not "a higher pivot is better" -- the two must be told apart.
        H_S = [0.20, 0.40, 0.50, 0.60, 0.80, 1.20, 2.00, 4.00]
        H_D = [0.10, 0.20, 0.40, 0.80]
        BLEND = [0]
    else:
        H_S = [0.15, 0.20, 0.25, 0.30, 0.40]
        H_D = [0.0, 0.05, 0.10, 0.20, 0.30]
        BLEND = [0, 5]
    rows = []
    for h_s, h_d, b in itertools.product(H_S, H_D, BLEND):
        m = run_set(keys, h_s, h_d, b, cache)
        m.update(h_single=h_s, h_double=h_d, blend=b)
        rows.append(m)
        tag = "  <- frozen method" if (h_s == 0.20 and h_d == 0.20 and b == 0) else ""
        say(f"  h_s={h_s:.2f} h_d={h_d:.2f} blend={b}: stance {m['stance']:.3f}  ds {m['ds']:.3f}  "
            f"hip {m['hip']:.4f}  hip_bias {m['hip_bias']:+.4f}  prior_bias {m['prior_bias_N']:+.1f} N{tag}")
    df = pd.DataFrame(rows)
    name = "phase_vpp_grid.csv" if "--wide" in sys.argv else "phase_vpp_grid_narrow.csv"
    df.to_csv(os.path.join(sp_.OUT, name), index=False)

    base = df[(df.h_single == 0.20) & (df.h_double == 0.20) & (df.blend == 0)].iloc[0]
    say(f"\nfrozen method on the calibration set: stance {base.stance:.3f}, ds {base.ds:.3f}, "
        f"hip {base.hip:.4f}, hip bias {base.hip_bias:+.4f}, prior bias {base.prior_bias_N:+.1f} N")

    best = df.sort_values("stance").iloc[0]
    say(f"best by stance RMSE (the original selection rule): h_single={best.h_single:.2f} "
        f"h_double={best.h_double:.2f} blend={int(best.blend)}")
    say(f"   stance {best.stance:.3f} ({100*(best.stance/base.stance-1):+.1f} %), "
        f"ds {best.ds:.3f} ({100*(best.ds/base.ds-1):+.1f} %), "
        f"hip {best.hip:.4f} ({100*(best.hip/base.hip-1):+.1f} %), "
        f"hip bias {best.hip_bias:+.4f} (was {base.hip_bias:+.4f})")

    bh = df.sort_values("hip").iloc[0]
    say(f"best by hip RMSE:            h_single={bh.h_single:.2f} h_double={bh.h_double:.2f} "
        f"blend={int(bh.blend)} -> hip {bh.hip:.4f}, stance {bh.stance:.3f}, "
        f"hip bias {bh.hip_bias:+.4f}")
    ba = df.reindex(df.hip_bias.abs().sort_values().index).iloc[0]
    say(f"smallest |hip bias|:         h_single={ba.h_single:.2f} h_double={ba.h_double:.2f} "
        f"blend={int(ba.blend)} -> hip bias {ba.hip_bias:+.4f}, hip {ba.hip:.4f}, "
        f"stance {ba.stance:.3f}")
    say(f"\n[{time.time() - t0:.0f} s] grid done, {len(df)} configurations")
    return df, best, base


if __name__ == "__main__":
    df, best, base = main()
    if "--test" in sys.argv:
        say("\n" + "=" * 70)
        say("applying the selected configuration to the test set, once, to report")
        test_trials = [t for t in available_trials() if t[0] not in CAL_SUBJECTS]
        cache = load(test_trials)
        keys = list(cache)
        say(f"test trials loaded: {len(keys)}")
        out = []
        for name, (h_s, h_d, b) in (("frozen", (0.20, 0.20, 0)),
                                    ("phase", (best.h_single, best.h_double, int(best.blend)))):
            rows = []
            for key in keys:
                d = cache[key]
                sp_.configure(*key)
                S = ex.newton_sum(d, 1.0)
                pr = vpp_prior_phase(d, h_s, h_d, b)
                fR, _, _ = ex.solve(d, S, fc_smooth=CAL["fc_smooth"], prior=pr,
                                    w_prior=CAL["w_prior"], w_torque=0.0)
                m = full_metrics(d, fR, pr)
                m.update(subject=key[0], trial=key[1], variant=name)
                rows.append(m)
            r = pd.DataFrame(rows)
            out.append(r)
            say(f"  {name:6s}: stance {r.stance.mean():.3f}  ds {r.ds.mean():.3f}  "
                f"hip {r.hip.mean():.4f}  hip_bias {r.hip_bias.mean():+.4f}  "
                f"prior_bias {r.prior_bias_N.mean():+.1f} N")
        pd.concat(out).to_csv(os.path.join(sp_.OUT, "phase_vpp_test.csv"), index=False)
        say("written: results/phase_vpp_test.csv")
    LOG.close()

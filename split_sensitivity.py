"""How much of the reported accuracy depends on which five subjects were used for calibration?

The evaluation protocol fixes every parameter on five calibration subjects and then freezes them.
A fair question is whether that particular set of five was lucky or unlucky: a different draw
might have chosen different parameters and produced a different test result.

The question can be answered exactly rather than argued about, because the solution for a given
parameter configuration on a given trial does not depend on the split at all. So we solve every
configuration on every trial once (stage 1), and then any split is a re-aggregation of that table
(stage 2): choose the configuration that minimises mean stance RMSE over the calibration
subjects, and report its mean over the remaining subjects.

Stage 1 output: results/grid_all_trials.csv (about 17 min)
Stage 2 output: results/split_sensitivity.csv, split_sensitivity_log.txt
"""
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

import experiments as ex
import soleid_planar as sp_
from protocol import CAL_SUBJECTS, available_trials

GRID = list(itertools.product([0.1, 0.2, 0.3, 0.4],      # h_vpp
                              [0.5, 1.0, 2.0],           # w_prior
                              [0.0, 1.0, 2.0],           # w_torque
                              [5.0, 8.0, 12.0]))         # fc_smooth
ALL = os.path.join(sp_.OUT, "grid_all_trials.csv")
LOG = os.path.join(sp_.OUT, "split_sensitivity_log.txt")


def say(f, *a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    f.write(s + "\n")
    f.flush()


# --------------------------------------------------------------------------- stage 1
def sweep():
    trials = available_trials()          # every trial that passes the quality gate
    print(f"{len(trials)} trials x {len(GRID)} configurations")
    rows, t0 = [], time.time()
    for i, (subj, trial, speed, age) in enumerate(trials):
        try:
            d = ex.prepare(subj, trial)
        except Exception as e:
            print(f"  skip {subj} {trial}: {e!r}")
            continue
        sp_.configure(subj, trial)
        S = ex.newton_sum(d, 1.0)
        for h, wp, wt, fc in GRID:
            pr = ex.vpp_prior(d, h)
            fR, _, _ = ex.solve(d, S, fc_smooth=fc, prior=pr, w_prior=wp, w_torque=wt)
            m = ex.metrics(d, fR)
            rows.append(dict(subject=subj, trial=trial, speed=speed, age=age,
                             h_vpp=h, w_prior=wp, w_torque=wt, fc_smooth=fc,
                             stance=m["stance"], ds=m["ds"], hip=m["hip"]))
        if (i + 1) % 20 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(trials)}  [{el/60:.1f} min, eta {el/(i+1)*(len(trials)-i-1)/60:.1f} min]")
    df = pd.DataFrame(rows)
    df.to_csv(ALL, index=False)
    print(f"wrote {ALL}: {len(df)} rows, {df.subject.nunique()} subjects, "
          f"{time.time()-t0:.0f} s")
    return df


# --------------------------------------------------------------------------- stage 2
KEYS = ["h_vpp", "w_prior", "w_torque", "fc_smooth"]


def evaluate_split(per_subj_cfg, cal_subjects, all_subjects):
    """Choose a configuration on the calibration subjects, score it on the rest.

    per_subj_cfg: mean metric per (subject, configuration), so a split is a groupby.
    The selection rule is the one the protocol actually used: lowest mean stance RMSE.
    """
    test_subjects = [s for s in all_subjects if s not in cal_subjects]
    cal = per_subj_cfg[per_subj_cfg.subject.isin(cal_subjects)]
    chosen = cal.groupby(KEYS, as_index=False).stance.mean().sort_values("stance").iloc[0]
    sel = {k: chosen[k] for k in KEYS}
    m = per_subj_cfg[per_subj_cfg.subject.isin(test_subjects)]
    for k, v in sel.items():
        m = m[m[k] == v]
    return dict(**sel, cal_stance=float(chosen.stance), test_stance=float(m.stance.mean()),
                test_ds=float(m.ds.mean()), test_hip=float(m.hip.mean()),
                n_test_subj=len(test_subjects))


def analyse(n_draws=500, seed=0):
    f = open(LOG, "w", encoding="utf-8")
    df = pd.read_csv(ALL)
    # weight every subject equally, so a subject with more trials does not dominate a draw
    per_subj = df.groupby(["subject"] + KEYS, as_index=False)[["stance", "ds", "hip"]].mean()
    subjects = sorted(per_subj.subject.unique())
    say(f, f"{len(subjects)} subjects, {len(GRID)} configurations, "
           f"{df.trial.nunique()} distinct trial names, {len(df)} solves")

    actual = evaluate_split(per_subj, CAL_SUBJECTS, subjects)
    say(f, f"\nthe split that was actually used ({', '.join(CAL_SUBJECTS)}):")
    say(f, f"   chosen h={actual['h_vpp']}, w_prior={actual['w_prior']}, "
           f"w_torque={actual['w_torque']}, fc={actual['fc_smooth']}")
    say(f, f"   test: stance {actual['test_stance']:.3f}, ds {actual['test_ds']:.3f}, "
           f"hip {actual['test_hip']:.4f}  (n = {actual['n_test_subj']} subjects)")

    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_draws):
        cal = list(rng.choice(subjects, size=len(CAL_SUBJECTS), replace=False))
        r = evaluate_split(per_subj, cal, subjects)
        r["cal_subjects"] = " ".join(sorted(cal))
        rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(sp_.OUT, "split_sensitivity.csv"), index=False)

    say(f, f"\n{n_draws} random draws of {len(CAL_SUBJECTS)} calibration subjects:")
    for k, lab, fmt in (("test_stance", "shear stance (%BW)", "{:.3f}"),
                        ("test_ds", "shear double support", "{:.3f}"),
                        ("test_hip", "hip moment (Nm/kg)", "{:.4f}")):
        v = res[k]
        pct = 100 * (v < actual[k]).mean()
        say(f, f"   {lab:24s} mean {fmt.format(v.mean())}  sd {fmt.format(v.std())}  "
               f"min {fmt.format(v.min())}  max {fmt.format(v.max())}  "
               f"| ours {fmt.format(actual[k])} -> better than {100-pct:.0f} % of draws")

    say(f, "\nwhich configuration each draw selected:")
    for k in KEYS:
        vc = res[k].value_counts(normalize=True).sort_index()
        say(f, f"   {k:10s} " + "  ".join(f"{i}: {100*v:.0f} %" for i, v in vc.items())
               + f"   | ours {actual[k]}")

    say(f, "\nthe spread of the *chosen parameters* matters less than it looks: the spread of the")
    say(f, "test result is what a reader cares about, and it is reported above.")

    # how much of the variation comes from the choice of parameters at all?
    best_cfg = (per_subj.groupby(KEYS, as_index=False).stance.mean()
                .sort_values("stance").iloc[0])
    say(f, f"\noracle (the configuration that is best on *all* subjects): "
           f"h={best_cfg.h_vpp}, w_prior={best_cfg.w_prior}, w_torque={best_cfg.w_torque}, "
           f"fc={best_cfg.fc_smooth}, stance {best_cfg.stance:.3f}")
    worst = per_subj.groupby(KEYS, as_index=False).stance.mean().sort_values("stance").iloc[-1]
    say(f, f"worst configuration in the grid: stance {worst.stance:.3f} "
           f"(h={worst.h_vpp}, w_prior={worst.w_prior}, w_torque={worst.w_torque}, fc={worst.fc_smooth})")
    f.close()
    return res, actual


if __name__ == "__main__":
    if "--analyse" in sys.argv:
        analyse(n_draws=int(sys.argv[sys.argv.index("--draws") + 1]) if "--draws" in sys.argv else 500)
    else:
        sweep()
        analyse()

"""Before calling the designed chain a failure, try the variants that a reader would ask about.

Four ways to spend the same look-ahead budget, differing in where the smoother is put:

  truncated     everything is a zero-phase filter clipped to the window (the baseline)
  force         only the force channel is smoothed; markers stay truncated + gradient^2
  state         markers raw, velocity and acceleration read from the smoother state
  pos+state     markers smoothed first, then the state derivatives (two low passes in series)
  pos+grad      markers smoothed first, then plain gradient^2

The force channel is smoothed in every arm but `truncated`, since its penalty is pure delay.
"""
import os
import time

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

import causal_design as cd
import experiments as ex
import soleid_planar as sp_
from protocol import CAL_SUBJECTS, available_trials

MODES = {                    # marker filter, derivative
    "truncated": ("trunc", "grad"),
    "force":     ("trunc", "grad"),
    "state":     ("raw", "state"),
    "pos+state": ("smooth", "state"),
    "pos+grad":  ("smooth", "grad"),
}


def prepare(subj, trial, look_s, mode):
    mk_mode, dv_mode = MODES[mode]
    real_lp, real_hold, real_dv = sp_.lowpass, sp_.hold_cop, sp_.deriv

    def lp(x, fc, fs, order=4):
        tag = "marker" if fc < 10 else "force"
        if tag == "force":
            if mode == "truncated":
                return cd.bounded_lowpass(x, fc, fs, look_s, order)
            return cd.smooth_fixed_lag(x, fc, fs, look_s)
        if mk_mode == "raw":
            return np.asarray(x, float)
        if mk_mode == "smooth":
            return cd.smooth_fixed_lag(x, fc, fs, look_s)
        return cd.bounded_lowpass(x, fc, fs, look_s, order)

    def dv(x, fs):
        return real_dv(x, fs) if dv_mode == "grad" else cd.deriv_fixed_lag(x, fs, look_s)

    def hold(copx, Fy):
        L = int(round(look_s * sp_.FS))
        out = real_hold(copx, Fy)
        good = Fy > sp_.F_COP
        idx = np.where(good, np.arange(len(copx)), -1)
        np.maximum.accumulate(idx, out=idx)
        ahead = np.minimum(np.arange(len(copx)) + L, len(copx) - 1)
        use = np.where(good, np.arange(len(copx)), np.where(good[ahead], ahead, idx))
        return np.where(Fy > sp_.F_CONTACT, copx[np.where(use < 0, 0, use)], out)

    sp_.lowpass, sp_.hold_cop, sp_.deriv = lp, hold, dv
    try:
        return ex.prepare(subj, trial)
    finally:
        sp_.lowpass, sp_.hold_cop, sp_.deriv = real_lp, real_hold, real_dv


def batch(subj, trial):
    real_lp = sp_.lowpass
    sp_.lowpass = lambda x, fc, fs, order=4: filtfilt(
        *butter(order, fc / (fs / 2), btype="low"), x, axis=0)
    try:
        return ex.prepare(subj, trial)
    finally:
        sp_.lowpass = real_lp


def main(n_trials=6, looks=(0.0, 0.02, 0.05, 0.10)):
    W = cd.n_past_for(sp_.FS)
    trials = [t for t in available_trials() if t[0] not in CAL_SUBJECTS][:n_trials]
    rows, t0 = [], time.time()
    for subj, trial, *_ in trials:
        d0 = batch(subj, trial)
        sp_.configure(subj, trial)
        b, ref = cd.solve_and_score(d0, warmup=W)
        rows.append(dict(subject=subj, trial=trial, mode="batch", look_s=np.nan,
                         **b, vs_batch=0.0))
        for mode in MODES:
            for L in looks:
                d = prepare(subj, trial, L, mode)
                sp_.configure(subj, trial)
                m, _ = cd.solve_and_score(d, ref, warmup=W)
                rows.append(dict(subject=subj, trial=trial, mode=mode, look_s=L, **m))
        print(f"  {subj} {trial} [{(time.time()-t0)/60:.1f} min]", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(sp_.OUT, "causal_variants.csv"), index=False)
    print(f"\n{len(trials)} trials; whole-record pipeline "
          f"{df[df['mode']=='batch'].stance.mean():.3f} %BW\n")
    print(f"{'look-ahead':>12s}" + "".join(f"{m:>12s}" for m in MODES))
    for L in looks:
        cells = "".join(f"{df[(df['mode']==m)&(df.look_s==L)].stance.mean():12.3f}" for m in MODES)
        print(f"{1000*L:9.0f} ms" + cells)
    return df


if __name__ == "__main__":
    main()

"""What is the latency of the whole chain, not just of the solver?

The estimator is solved over a sliding window, but its inputs were produced by zero-phase
filtering of the complete recording, which uses the entire future. A latency quoted for the solver
alone is therefore not the latency of the method. This script replaces every zero-phase filter
with a bounded-look-ahead version and measures what the bound costs.

The construction: to produce the filtered value at frame k with look-ahead L, apply the same
forward-backward filter to the segment [k-W, k+L] and keep the sample at k. Every output then
depends on the past and on L frames of future, which is exactly what a real-time system with L of
latency has. W is fixed at 0.5 s, long enough that the past side is settled.

Implemented with a sliding-window view so the whole signal is filtered in one call rather than
frame by frame.

Output: results/causal_chain.csv, causal_chain_log.txt
"""
import json
import os
import time

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, filtfilt

import experiments as ex
import soleid_planar as sp_
from protocol import CAL_SUBJECTS, available_trials

CAL = json.load(open(os.path.join(sp_.OUT, "calibration.json")))["chosen"]
W_PAST_S = 0.5
LOG = open(os.path.join(sp_.OUT, "causal_chain_log.txt"), "w", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


def bounded_lowpass(x, fc, fs, look_s, order=4):
    """Zero-phase low pass restricted to a window that ends look_s after the output sample."""
    b, a = butter(order, fc / (fs / 2), btype="low")
    if look_s is None:                       # the published pipeline: the whole record
        return filtfilt(b, a, x, axis=0)     # not sp_.lowpass: that name is patched while we run
    W, L = int(round(W_PAST_S * fs)), int(round(look_s * fs))
    x = np.asarray(x, float)
    one_d = x.ndim == 1
    X = x[:, None] if one_d else x
    n, m = X.shape
    pad = np.vstack([np.repeat(X[:1], W, axis=0), X, np.repeat(X[-1:], L + 1, axis=0)])
    out = np.empty_like(X)
    for j in range(m):
        win = sliding_window_view(pad[:, j], W + L + 1)[:n]      # (n, W+L+1)
        out[:, j] = filtfilt(b, a, win, axis=1)[:, W]
    return out[:, 0] if one_d else out


def prepare_causal(subj, trial, look_s):
    """ex.prepare with every zero-phase filter given a finite look-ahead."""
    real_lowpass = sp_.lowpass
    real_hold = sp_.hold_cop

    def patched_lowpass(x, fc, fs, order=4):
        return bounded_lowpass(x, fc, fs, look_s, order)

    def patched_hold(copx, Fy):
        """hold_cop replaces unreliable samples from the nearest reliable one in either
        direction; with a finite look-ahead only the past side and L frames ahead are visible."""
        if look_s is None:
            return real_hold(copx, Fy)
        L = int(round(look_s * sp_.FS))
        out = real_hold(copx, Fy)
        # forward fill from the last reliable sample, allowing L frames of look-ahead
        good = Fy > sp_.F_COP
        idx = np.where(good, np.arange(len(copx)), -1)
        np.maximum.accumulate(idx, out=idx)
        ahead = np.minimum(np.arange(len(copx)) + L, len(copx) - 1)
        use = np.where(good, np.arange(len(copx)), np.where(good[ahead], ahead, idx))
        use = np.where(use < 0, 0, use)
        return np.where(Fy > sp_.F_CONTACT, copx[use], out)

    sp_.lowpass, sp_.hold_cop = patched_lowpass, patched_hold
    try:
        return ex.prepare(subj, trial)
    finally:
        sp_.lowpass, sp_.hold_cop = real_lowpass, real_hold


def solve_and_score(d, ref_fR=None):
    sp_.FS = sp_.FS
    S = ex.newton_sum(d, 1.0)
    pr = ex.vpp_prior(d, CAL["h_vpp"])
    fR, _, _ = ex.solve(d, S, fc_smooth=CAL["fc_smooth"], prior=pr,
                        w_prior=CAL["w_prior"], w_torque=0.0)
    grf = d["grf"]
    on = (grf["R"]["Fy"] > sp_.F_CONTACT) & d["trim"]
    BW = sp_.MASS * 9.81
    out = dict(stance=100 * np.sqrt(np.mean((fR[on] - grf["R"]["Fx"][on]) ** 2)) / BW)
    if ref_fR is not None:
        m = min(len(fR), len(ref_fR))
        o = on[:m]
        out["vs_batch"] = 100 * np.sqrt(np.mean((fR[:m][o] - ref_fR[:m][o]) ** 2)) / BW
    return out, fR


def main(n_trials=12, looks=(0.0, 0.05, 0.10, 0.15, 0.20, 0.30)):
    trials = [t for t in available_trials() if t[0] not in CAL_SUBJECTS][:n_trials]
    say(f"causal preprocessing chain: {len(trials)} trials, past window {W_PAST_S} s")
    rows, t0 = [], time.time()
    for subj, trial, speed, age in trials:
        try:
            d0 = prepare_causal(subj, trial, None)          # the published pipeline
        except Exception as e:
            say(f"  skip {subj} {trial}: {e!r}")
            continue
        sp_.configure(subj, trial)
        base, ref = solve_and_score(d0)
        rows.append(dict(subject=subj, trial=trial, look_s=np.nan, **base, vs_batch=0.0))
        for L in looks:
            try:
                d = prepare_causal(subj, trial, L)
            except Exception as e:
                say(f"  skip {subj} {trial} L={L}: {e!r}")
                continue
            sp_.configure(subj, trial)
            m, _ = solve_and_score(d, ref)
            rows.append(dict(subject=subj, trial=trial, look_s=L, **m))
        say(f"  {subj} {trial} done [{(time.time()-t0)/60:.1f} min]")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(sp_.OUT, "causal_chain.csv"), index=False)

    b = df[df.look_s.isna()]
    say(f"\npublished pipeline (whole-record filtering): stance {b.stance.mean():.3f} %BW\n")
    say(f"{'filter look-ahead':>18s} {'stance':>9s} {'vs batch':>10s}")
    for L in looks:
        g = df[df.look_s == L]
        if len(g):
            say(f"{1000*L:15.0f} ms {g.stance.mean():9.3f} {g.vs_batch.mean():10.3f}")
    say("\nThe estimator's own look-ahead (50 ms) adds to whichever row is chosen, because the "
        "solver needs filtered samples beyond the read-out frame.")
    LOG.close()
    return df


if __name__ == "__main__":
    main()

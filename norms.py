"""Normalisation constants, so that results reported in different metrics can be compared.

The literature reports the same quantity in at least four ways: RMSE in absolute units, RMSE
normalised by the peak of the reference signal, RMSE normalised by its range (max - min), and
correlation. Converting between the first three needs the peak and the range of the reference
signal *on the same data set* — which we can compute, because we process the same data.

This script measures those constants for each data set and signal, over two evaluation windows
(stance only, and the full gait cycle), and writes results/norm_constants.json.

Correlation cannot be converted into an RMSE and is not attempted here.
"""
import json
import os

import numpy as np
import pandas as pd

import soleid_planar as sp_
import config

OUT = sp_.OUT
N_SAMPLE = 40        # trials per data set; the reference statistics converge long before this


def stats(sig, on):
    """Peak, range and RMS of a reference signal, over stance and over the whole record."""
    d = {}
    for win, m in (("stance", on), ("cycle", np.ones_like(on, bool))):
        x = sig[m]
        if x.size < 10:
            continue
        d[win] = dict(peak=float(np.max(np.abs(x))), range=float(np.ptp(x)),
                      rms=float(np.sqrt(np.mean(x ** 2))))
    return d


def merge(acc, key, d):
    for win, v in d.items():
        for k, val in v.items():
            acc.setdefault(key, {}).setdefault(win, {}).setdefault(k, []).append(val)


def finish(acc):
    return {sig: {win: {k: float(np.mean(v)) for k, v in d.items()} for win, d in wins.items()}
            for sig, wins in acc.items()}


# ------------------------------------------------------------------ Fukuchi, from stored curves
def fukuchi():
    z = np.load(os.path.join(OUT, "curves_fukuchi.npz"), allow_pickle=True)
    sf = np.asarray(z["stance_frac"], float).mean()
    acc = {}
    # curves are one gait cycle resampled to 101 points; stance is the leading fraction
    ns = int(round(sf * 101))
    on = np.zeros(101, bool)
    on[:ns] = True
    for sig, key in (("reference_shear", "shear"), ("reference_ankle", "ankle"),
                     ("reference_knee", "knee"), ("reference_hip", "hip")):
        a = np.asarray(z[sig], float)
        for row in a:
            merge(acc, key, stats(row, on))
    out = finish(acc)
    out["_meta"] = dict(n_trials=int(len(z["subjects"])), stance_fraction=float(sf),
                        source="curves_fukuchi.npz", units="shear %BW, moments Nm/kg")
    return out


# ------------------------------------------------------------- Camargo and Wang, reference only
def _reference_stats(loader, items, label):
    """Reference signals without solving the QP: only the measured GRF is needed.

    Statistics are taken **per stride** and then averaged, to match the Fukuchi curves, where one
    curve is one gait cycle. Taking the maximum over a whole trial instead would report the single
    largest instantaneous value of the fastest stride, which is not the same quantity and is not
    what the literature normalises by.
    """
    acc, used, strides = {}, 0, 0
    for it in items:
        try:
            segs, grf, n, fs = loader(it)
        except Exception:
            continue
        on = grf["R"]["Fy"] > sp_.F_CONTACT
        if on.sum() < fs:
            continue
        BW = sp_.MASS * 9.81
        ref = sp_.inverse_dynamics(segs, "R", grf["R"]["Fx"], grf["R"]["Fy"], grf["R"]["copx"])
        sig = dict(shear=100 * grf["R"]["Fx"] / BW,
                   **{j: ref[j] / sp_.MASS for j in ("ankle", "knee", "hip")})
        hs = sp_.heel_strikes(grf["R"]["Fy"])
        hs = hs[(hs > fs) & (hs < n - fs)]
        for a, b in zip(hs[:-1], hs[1:]):
            if not (0.5 * fs < b - a < 2.5 * fs):
                continue
            sl = slice(a, b)
            for key, x in sig.items():
                merge(acc, key, stats(x[sl], on[sl]))
            strides += 1
        used += 1
        if used >= N_SAMPLE:
            break
    out = finish(acc)
    out["_meta"] = dict(n_trials=used, n_strides=strides, source=label,
                        units="shear %BW, moments Nm/kg", statistic="per stride, then averaged")
    return out


def camargo():
    import camargo_adapter as ca
    info = pd.read_csv(config.CAMARGO_INFO).set_index("Subject")

    def loader(it):
        subj, trial = it
        d = ca.load_trial(subj, trial, "treadmill")
        mass, height = float(info.loc[subj, "Weight"]), float(info.loc[subj, "Height"])
        sp_.MASS, sp_.HEIGHT, sp_.FS = mass, height, ca.FS
        segs, _ = ca.build_segments(d["P"], d["n"], mass, height)
        return segs, d["grf"], d["n"], ca.FS

    items = [(s, t) for s in ca.subjects() for t in ca.trials(s, "treadmill")[:2]]
    return _reference_stats(loader, items, "Camargo treadmill")


# The Wang layer needs no conversion: the only comparator on that data set is the wearable
# pipeline published with it, and we recomputed its output in our own metric directly
# (results/wang_crosscheck.csv). Nothing there is reported in a normalised form.


if __name__ == "__main__":
    out = {"fukuchi": fukuchi()}
    for name, fn in (("camargo_treadmill", camargo),):
        try:
            out[name] = fn()
            print(f"{name}: {out[name]['_meta']['n_trials']} trials")
        except Exception as e:
            print(f"{name}: skipped ({e!r})")
    with open(os.path.join(OUT, "norm_constants.json"), "w") as f:
        json.dump(out, f, indent=1)
    for ds, d in out.items():
        print(f"\n== {ds}")
        for sig in ("shear", "ankle", "knee", "hip"):
            if sig not in d:
                continue
            for win in ("stance", "cycle"):
                if win in d[sig]:
                    v = d[sig][win]
                    print(f"   {sig:6s} {win:6s} peak {v['peak']:7.3f}  range {v['range']:7.3f}"
                          f"  rms {v['rms']:7.3f}")

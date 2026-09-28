"""Carry the calibration-split uncertainty through to the external and application layers.

split_sensitivity.py answers the question on the controlled layer: if a different five subjects
had been used for calibration, what would the test result have been? The same question applies
downstream, because a different calibration set would have frozen different parameters and those
parameters are what Camargo and Wang are then processed with.

Propagating it is cheap because the 500 draws select only 13 distinct configurations, five of
which account for 87 % of draws. So we run those 13 on the external and application layers once
and weight each by how often it was selected. The result is the distribution of the reported
number over the choice of calibration subjects -- not a single frozen value.

Output: results/split_propagated.csv, split_propagated_log.txt
"""
import os
import sys
import time

import numpy as np
import pandas as pd

import experiments as ex
import soleid_planar as sp_

OUT = sp_.OUT
KEYS = ["h_vpp", "w_prior", "w_torque", "fc_smooth"]
LOG = open(os.path.join(OUT, "split_propagated_log.txt"), "w", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


def configurations():
    """The distinct configurations the 500 draws chose, with how often each was chosen."""
    sp = pd.read_csv(os.path.join(OUT, "split_sensitivity.csv"))
    w = sp.groupby(KEYS).size().rename("draws").reset_index()
    w["weight"] = w.draws / w.draws.sum()
    return w.sort_values("draws", ascending=False).reset_index(drop=True)


def metrics_for(est, meas, on, ds, segs, Fy, cop, ref, mass, sl=None):
    BW = mass * 9.81
    sl = slice(None) if sl is None else sl
    out = {}
    m_on, m_ds = on[sl], ds[sl]
    e, y = est[sl], meas[sl]
    out["stance"] = 100 * np.sqrt(np.mean((e[m_on] - y[m_on]) ** 2)) / BW
    out["ds"] = (100 * np.sqrt(np.mean((e[m_ds] - y[m_ds]) ** 2)) / BW) if m_ds.sum() > 5 else np.nan
    M = sp_.inverse_dynamics(segs, "R", est, Fy, cop)
    for j in ("ankle", "knee", "hip"):
        out[j] = float(np.sqrt(np.mean((M[j][sl] / mass - ref[j][sl] / mass) ** 2)))
    return out


# ------------------------------------------------------------------ Camargo treadmill
def camargo(cfgs, limit=None):
    import camargo_adapter as ca
    import camargo_run as cr
    info = cr.INFO
    rows, t0 = [], time.time()
    items = [(s, t) for s in ca.subjects() for t in ca.trials(s, "treadmill")]
    if limit:
        items = items[:limit]
    say(f"Camargo treadmill: {len(items)} subject-trials x {len(cfgs)} configurations")
    for i, (subj, trial) in enumerate(items):
        try:
            d = ca.load_trial(subj, trial, "treadmill")
        except Exception as e:
            say(f"  skip {subj} {trial}: {e!r}")
            continue
        n = d["n"]
        mass, height = float(info.loc[subj, "Weight"]), float(info.loc[subj, "Height"])
        sp_.MASS, sp_.HEIGHT, sp_.FS = mass, height, ca.FS
        segs, pel = ca.build_segments(d["P"], n, mass, height)
        grf = d["grf"]
        onR = grf["R"]["Fy"] > sp_.F_CONTACT
        ds = onR & (grf["L"]["Fy"] > sp_.F_CONTACT)
        trim = np.zeros(n, bool)
        trim[int(ca.FS):n - int(ca.FS)] = True
        trim &= d["speed"] > cr.MIN_SPEED
        dd = dict(n=n, segs=segs, pelvis=pel, grf=grf, trim=trim)
        S = ex.newton_sum(dd)
        ref = sp_.inverse_dynamics(segs, "R", grf["R"]["Fx"], grf["R"]["Fy"], grf["R"]["copx"])
        hs = sp_.heel_strikes(grf["R"]["Fy"])
        strides = [(a, b) for a, b in zip(hs[:-1], hs[1:])
                   if 0.5 * ca.FS < b - a < 2.5 * ca.FS and trim[a:b].all() and onR[a:b].sum() >= 20]
        if not strides:
            continue
        for _, c in cfgs.iterrows():
            pr = ex.vpp_prior(dd, c.h_vpp)
            fR, _, _ = ex.solve(dd, S, fc_smooth=c.fc_smooth, prior=pr,
                                w_prior=c.w_prior, w_torque=c.w_torque)
            # average over strides here rather than keeping every stride: 157 trials x 13
            # configurations x ~120 strides is enough rows to exhaust memory, and the analysis
            # only ever uses per-subject means
            per = [metrics_for(fR, grf["R"]["Fx"], onR, ds, segs, grf["R"]["Fy"],
                               grf["R"]["copx"], ref, mass, slice(a, b)) for a, b in strides]
            m = {k: float(np.nanmean([p[k] for p in per])) for k in per[0]}
            m.update(subject=subj, trial=trial, layer="camargo_treadmill",
                     n_strides=len(per), **{k: c[k] for k in KEYS})
            rows.append(m)
        if (i + 1) % 20 == 0:
            el = time.time() - t0
            say(f"  {i+1}/{len(items)} [{el/60:.1f} min, eta {el/(i+1)*(len(items)-i-1)/60:.1f} min]")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ Wang
def wang(cfgs):
    """Re-run the published Wang pipeline once per configuration.

    Rather than reimplement the trial processing, we swap the frozen parameters that
    wang_run reads and call its own run_trial, so the arm being varied is exactly the one the
    paper reports. Rows are written as they are produced: the run takes about an hour, and a
    partial table of whole trials is still usable.
    """
    import wang_adapter as wa
    import wang_run as wr
    info = wa.subject_info()
    subs = sorted(d for d in os.listdir(wa.ROOT) if d.startswith("Subj"))
    frozen = dict(wr.CAL)
    path = os.path.join(OUT, "split_propagated_wang.csv")
    rows, t0, written = [], time.time(), False
    say(f"\nWang: {len(subs)} subjects x {len(wa.WALKS)} trials x {len(cfgs)} configurations")
    try:
        for i, subj in enumerate(subs):
            for trial in wa.WALKS:
                for _, c in cfgs.iterrows():
                    wr.CAL = {k: c[k] for k in KEYS}
                    try:
                        r = wr.run_trial(subj, trial, info)
                    except Exception as e:
                        say(f"  skip {subj} {trial} {dict(wr.CAL)}: {e!r}")
                        continue
                    r.update(layer="wang", **{k: c[k] for k in KEYS})
                    rows.append(r)
                if rows:
                    pd.DataFrame(rows).to_csv(path, index=False)
                    written = True
            say(f"  {i+1}/{len(subs)} subjects [{(time.time()-t0)/60:.1f} min, "
                f"eta {(time.time()-t0)/(i+1)*(len(subs)-i-1)/60:.1f} min]")
    finally:
        wr.CAL = frozen
    say(f"  wrote {path}" if written else "  nothing written")
    return pd.DataFrame(rows)


def report(df, cfgs, layer):
    """Weight each configuration by how often the 500 draws selected it."""
    w = cfgs.set_index(KEYS).weight
    cols = ["stance", "ds", "ankle", "knee", "hip"]
    if "n_strides" in df:          # one row per trial: weight each trial by its strides
        g = df.assign(**{c: df[c] * df.n_strides for c in cols}).groupby(KEYS + ["subject"])
        per = g[cols].sum().div(g.n_strides.sum(), axis=0)
    else:
        per = df.groupby(KEYS + ["subject"])[cols].mean()
    per = per.groupby(KEYS).mean()          # subject-weighted, as in split_sensitivity.py
    per["weight"] = w
    say(f"\n=== {layer}: distribution over the choice of calibration subjects")
    say(f"{'configuration':34s} {'draws':>6s} {'stance':>8s} {'hip':>8s}")
    for k, r in per.sort_values("weight", ascending=False).iterrows():
        say(f"  h={k[0]:.1f} wp={k[1]:.1f} wt={k[2]:.1f} fc={k[3]:<5.1f}"
            f" {100*r.weight:5.1f}% {r.stance:8.3f} {r.hip:8.4f}")
    for col, lab, fmt in (("stance", "shear stance (%BW)", "{:.3f}"),
                          ("hip", "hip moment (Nm/kg)", "{:.4f}")):
        v, wt = per[col].values, per.weight.values
        mean = float(np.sum(v * wt))
        sd = float(np.sqrt(np.sum(wt * (v - mean) ** 2)))
        say(f"   {lab:22s} weighted mean {fmt.format(mean)}  sd {fmt.format(sd)}  "
            f"range {fmt.format(v.min())}-{fmt.format(v.max())}")
    return per


if __name__ == "__main__":
    cfgs = configurations()
    say(f"{len(cfgs)} distinct configurations selected by the 500 draws; "
        f"the top five account for {100*cfgs.weight.head(5).sum():.0f} %\n")
    say(cfgs.to_string(index=False))
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None

    if "--wang" in sys.argv:
        d = wang(cfgs)
        if len(d):
            per = (d.groupby(KEYS + ["subject"])[["SoleID_stance", "SoleID_hip"]].mean()
                   .groupby(KEYS).mean()
                   .rename(columns={"SoleID_stance": "stance", "SoleID_hip": "hip"}))
            say(f"\n=== Wang: {d.subject.nunique()} subjects, "
                f"{d.groupby(['subject','trial']).ngroups} trials")
            w = cfgs.set_index(KEYS).weight
            per["weight"] = w
            for k, r in per.sort_values("weight", ascending=False).iterrows():
                say(f"  h={k[0]:.1f} wp={k[1]:.1f} wt={k[2]:.1f} fc={k[3]:<5.1f}"
                    f" {100*r.weight:5.1f}% stance {r.stance:7.3f}  hip {r.hip:7.4f}")
            for col, lab, fmt in (("stance", "shear stance (%BW)", "{:.3f}"),
                                  ("hip", "hip moment (Nm/kg)", "{:.4f}")):
                v, wt = per[col].values, per.weight.values
                m = float(np.sum(v * wt))
                sd = float(np.sqrt(np.sum(wt * (v - m) ** 2)))
                fr = float(per.loc[(0.2, 1.0, 0.0, 12.0), col])
                say(f"   {lab:22s} weighted mean {fmt.format(m)}  sd {fmt.format(sd)}  "
                    f"range {fmt.format(v.min())}-{fmt.format(v.max())}  | frozen "
                    f"{fmt.format(fr)}, better than {100*np.sum(wt[v > fr]):.0f} % by weight")
        LOG.close()
        sys.exit(0)

    d = camargo(cfgs, limit)
    if len(d):
        d.to_csv(os.path.join(OUT, "split_propagated_camargo.csv"), index=False)
        say(f"\nwrote split_propagated_camargo.csv: {len(d)} rows, "
            f"{d.subject.nunique()} subjects, {d.groupby(KEYS).ngroups} configurations")
        try:
            report(d, cfgs, "Camargo treadmill")
        except Exception as e:      # the table is written; do not lose it to a formatting bug
            say(f"report failed: {e!r}")
    LOG.close()

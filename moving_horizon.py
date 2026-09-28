"""Moving-horizon (real-time) variant of SoleID.

The batch QP over a whole recording is replaced by a short sliding window that is re-solved at
every sample: the window covers [t - H + 1 + LA, t + LA], the problem is the batch one restricted
to that window, and only the sample at t is kept. LA is a look-ahead, i.e. the latency the
estimator needs beyond the sensor delay.

The QP structure is fixed for a given window length, so the solver is set up ONCE and each step
only updates the cost vector, the diagonal of P (the prior weight follows foot contact) and the
friction-cone bounds, with warm starting. The reported time per window therefore reflects what a
real-time implementation would spend.
"""
import json
import os
import sys
import time

import numpy as np
import osqp
import pandas as pd
import scipy.sparse as sp

import experiments as ex
import soleid_planar as sp_

CAL = json.load(open(os.path.join(sp_.OUT, "calibration.json")))["chosen"]


class WindowSolver:
    """Fixed-size sliding-window QP for the stacked variable [fR; fL] (2m values)."""

    def __init__(self, m, fs, w_prior, fc_smooth, w_dyn=1.0, w_mag=1e-4, mu=sp_.MU):
        self.m, self.fs, self.w_prior, self.mu = m, fs, w_prior, mu
        dt = 1.0 / fs
        w_s = w_dyn / (2 * np.pi * fc_smooth * dt) ** 4
        I = sp.identity(m, format="csc")
        D2 = sp.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(m - 2, m), format="csc")
        A_dyn = sp.hstack([I, I], format="csc")
        D2b = sp.block_diag([D2, D2], format="csc")
        P0 = 2 * (w_dyn * A_dyn.T @ A_dyn + w_s * D2b.T @ D2b + w_mag * sp.identity(2 * m))
        # add an explicit (zero) diagonal so the prior can be updated without changing sparsity
        P0 = (P0 + 0.0 * sp.identity(2 * m)).tocsc()
        P0.sort_indices()
        self.P = sp.triu(P0, format="csc")          # OSQP takes the upper triangle
        self.P.sort_indices()
        self.w_dyn = w_dyn
        self.A_dyn = A_dyn
        # index of each diagonal entry inside P.data
        diag_idx = np.empty(2 * m, dtype=int)
        for j in range(2 * m):
            s0, s1 = self.P.indptr[j], self.P.indptr[j + 1]
            rows = self.P.indices[s0:s1]
            diag_idx[j] = s0 + int(np.where(rows == j)[0][0])
        self.diag_idx = diag_idx
        self.P_base = self.P.data.copy()
        self.prob = osqp.OSQP()
        self.prob.setup(P=self.P, q=np.zeros(2 * m), A=sp.identity(2 * m, format="csc"),
                        l=-np.ones(2 * m), u=np.ones(2 * m), verbose=False,
                        eps_abs=1e-5, eps_rel=1e-5, max_iter=4000, polish=False, warm_starting=True)

    def solve(self, S, FyR, FyL, prR, prL):
        m = self.m
        onR, onL = FyR > sp_.F_CONTACT, FyL > sp_.F_CONTACT
        mask = np.concatenate([onR, onL]).astype(float)
        Pdata = self.P_base.copy()
        Pdata[self.diag_idx] += 2 * self.w_prior * mask
        q = -2 * (self.w_dyn * (self.A_dyn.T @ S) + self.w_prior * mask * np.concatenate([prR, prL]))
        lb = np.concatenate([np.where(onR, -self.mu * FyR, 0.0), np.where(onL, -self.mu * FyL, 0.0)])
        ub = np.concatenate([np.where(onR, self.mu * FyR, 0.0), np.where(onL, self.mu * FyL, 0.0)])
        self.prob.update(Px=Pdata, q=q, l=lb, u=ub)
        t0 = time.perf_counter()
        r = self.prob.solve()
        dt = time.perf_counter() - t0
        return r.x, dt


def run_mhe(S, grf, prior, fs, horizon_s=0.4, lookahead_s=0.05, w_prior=None, fc_smooth=None):
    """Sample-by-sample sliding-window solution; returns (fR, fL, solve times)."""
    w_prior = CAL["w_prior"] if w_prior is None else w_prior
    fc_smooth = CAL["fc_smooth"] if fc_smooth is None else fc_smooth
    n = len(S)
    m = int(round(horizon_s * fs))
    LA = int(round(lookahead_s * fs))
    solver = WindowSolver(m, fs, w_prior, fc_smooth)
    FyR, FyL = grf["R"]["Fy"], grf["L"]["Fy"]
    fR, fL = np.full(n, np.nan), np.full(n, np.nan)
    times = np.empty(n)
    k = 0
    for t_out in range(n):
        b = min(n, t_out + LA + 1)
        a = b - m
        if a < 0:
            continue
        sl = slice(a, b)
        x, dt = solver.solve(S[sl], FyR[sl], FyL[sl], prior["R"][sl], prior["L"][sl])
        if x is None or np.any(np.isnan(x)):
            continue
        j = t_out - a
        fR[t_out], fL[t_out] = x[j], x[m + j]
        times[k] = dt
        k += 1
    return fR, fL, times[:k]


def evaluate_trial(d, label, horizons=(0.2, 0.3, 0.4, 0.6), lookaheads=(0.0, 0.05, 0.1),
                   fs=None, S=None, extra=None):
    fs = fs or sp_.FS
    n = d["n"]
    S = ex.newton_sum(d) if S is None else S
    prior = ex.vpp_prior(d, CAL["h_vpp"])
    t0 = time.perf_counter()
    bR, bL, _ = ex.solve(d, S, fc_smooth=CAL["fc_smooth"], prior=prior, w_prior=CAL["w_prior"], w_torque=0.0)
    t_batch = time.perf_counter() - t0
    BW = sp_.MASS * 9.81
    meas = d["grf"]["R"].get("Fx")
    onR = d["grf"]["R"]["Fy"] > sp_.F_CONTACT
    rows = []
    for H in horizons:
        for LA in lookaheads:
            fR, fL, times = run_mhe(S, d["grf"], prior, fs, horizon_s=H, lookahead_s=LA)
            ok = ~np.isnan(fR)
            tr = d["trim"] & onR & ok
            row = dict(label=label, horizon_s=H, lookahead_s=LA, n=n, fs=fs,
                       vs_batch_pctBW=100 * np.sqrt(np.mean((fR[tr] - bR[tr]) ** 2)) / BW,
                       solve_ms_mean=1000 * times.mean(), solve_ms_p95=1000 * np.percentile(times, 95),
                       solve_ms_max=1000 * times.max(), batch_s=t_batch, windows=len(times),
                       budget_ms=1000.0 / fs)
            if meas is not None:
                row["mhe_stance_pctBW"] = 100 * np.sqrt(np.mean((fR[tr] - meas[tr]) ** 2)) / BW
                row["batch_stance_pctBW"] = 100 * np.sqrt(np.mean((bR[tr] - meas[tr]) ** 2)) / BW
            if extra:
                row.update(extra)
            rows.append(row)
            print("  H=%.2f LA=%.2f: vs batch %.2f %%BW | stance %.2f (batch %.2f) | %.2f ms/window (p95 %.2f, budget %.1f)"
                  % (H, LA, row["vs_batch_pctBW"], row.get("mhe_stance_pctBW", np.nan),
                     row.get("batch_stance_pctBW", np.nan), row["solve_ms_mean"], row["solve_ms_p95"],
                     row["budget_ms"]), flush=True)
    return rows


def summarize(df, name):
    g = df.groupby(["horizon_s", "lookahead_s"])[
        ["vs_batch_pctBW", "mhe_stance_pctBW", "batch_stance_pctBW", "solve_ms_mean", "solve_ms_p95"]].mean()
    print(f"\n== {name}: {df.label.nunique()} trials ==")
    print(g.round(3).to_string())


def fukuchi(n_subjects=8, trials_per_subject=("walkT03", "walkT05", "walkT07")):
    from protocol import CAL_SUBJECTS, available_trials
    avail = [t for t in available_trials() if t[0] not in CAL_SUBJECTS]
    subs, picked = [], []
    for subj, trial, speed, age in avail:
        if trial not in trials_per_subject:
            continue
        if subj not in subs:
            if len(subs) >= n_subjects:
                continue
            subs.append(subj)
        picked.append((subj, trial, speed, age))
    rows = []
    for subj, trial, speed, age in picked:
        try:
            d = ex.prepare(subj, trial)
        except sp_.MarkerGapError:
            continue
        sp_.configure(subj, trial)
        print(f"{subj} {trial} ({speed} m/s)", flush=True)
        rows += evaluate_trial(d, f"fukuchi:{subj}:{trial}", extra=dict(dataset="fukuchi", subject=subj, speed=speed))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(sp_.OUT, "mhe_fukuchi.csv"), index=False)
    summarize(df, "Fukuchi (150 Hz)")
    return df


def wang(subjects=None, trials=("walk_18", "walk_36", "walk_54")):
    import wang_adapter as wa
    import wang_run as wr
    info = wa.subject_info()
    subs = subjects or sorted([x for x in os.listdir(wa.ROOT) if x.startswith("Subj")])[:6]
    rows = []
    for s in subs:
        for t in trials:
            try:
                d, S, label = wr.prepare_for_mhe(s, t, info)
            except Exception as e:
                print(f"  {s} {t}: ERROR {e!r}", flush=True)
                continue
            print(f"{s} {t}", flush=True)
            rows += evaluate_trial(d, label, fs=wr.FS, S=S,
                                   extra=dict(dataset="wang", subject=s, speed=wa.SPEED[t]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(sp_.OUT, "mhe_wang.csv"), index=False)
    summarize(df, "Wang (100 Hz, real sensors)")
    return df


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "fukuchi"
    if what == "fukuchi":
        fukuchi()
    elif what == "wang":
        wang()
    elif what == "summary":
        for f, nm in (("mhe_fukuchi.csv", "Fukuchi"), ("mhe_wang.csv", "Wang")):
            p = os.path.join(sp_.OUT, f)
            if os.path.exists(p):
                summarize(pd.read_csv(p), nm)

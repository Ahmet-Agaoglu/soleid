"""A causal estimator designed for the purpose, rather than a zero-phase filter truncated to a
window.

`causal_chain.py` answered "what does a bounded look-ahead cost?" by clipping `filtfilt` to a
window. That is the honest way to pose the question but a poor way to answer it: at short
look-ahead the backward pass has almost no data to settle into, so the truncated filter is not
merely delayed, it is wrong. This script asks what a chain designed for the constraint can do.

Three things were needed, and the first two only became visible by measuring.

1. *Where the cost is.* Made causal one at a time, the 20 Hz force filter costs more than the 6 Hz
   marker filter, and `hold_cop` costs nothing at all. The force penalty is pure delay --- force is
   never differentiated --- so it is the part a better filter can simply remove.

2. *Where the estimate lives.* Filtering the marker positions and then differencing twice throws
   away the estimator's own acceleration state, which is the quantity the dynamics actually needs.
   The smoother is therefore placed at the derivative, not at the filter.

3. *What sets the accuracy.* Acceleration weights noise by omega^2, so what governs it is the
   rolloff above the cutoff, which is the model order: m states gives -20m dB/decade. A constant-
   acceleration model (m = 3) rolls off far too gently. We use m = 8 for the plain reason that the
   filter being replaced is 8th order --- a 4th-order Butterworth applied forwards and backwards
   --- so the two differ in causality and in nothing else.

The design is fixed by one criterion with no reference to the data: q/r is chosen so that the
infinite-lag smoother turns over where the zero-phase filter does, measured from that filter's own
impulse response rather than from its nominal cutoff (forward-backward 4th-order Butterworth at
6 Hz actually turns over near 5.4 Hz, because the response is squared).

Once the covariance recursion has converged the smoother is linear and time-invariant, so it is
realised as a kernel: the impulse responses of the position, velocity and acceleration states are
read off once per (cutoff, look-ahead) and applied by a sliding dot product. The kernel spans L
frames of future and W_PAST_S of past, which is exactly what a real-time system with L of latency
has. Past data is free --- it costs memory, not latency --- and the window has to be long enough
for the acceleration kernel to settle, its amplitude being orders of magnitude above the position
kernel's.

The recursion is run in units of samples rather than seconds and the derivative kernels scaled
back afterwards: in physical units q/r reaches 1e22 at m = 8 and the Riccati equation is badly
conditioned.

Output: results/causal_design_decomp.csv, results/causal_design_sweep.csv, causal_design_log.txt
"""
import json
import os
import time
from math import factorial as fct

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.linalg import solve_discrete_are
from scipy.signal import butter, fftconvolve, filtfilt

import experiments as ex
import soleid_planar as sp_
from protocol import CAL_SUBJECTS, available_trials

CAL = json.load(open(os.path.join(sp_.OUT, "calibration.json")))["chosen"]
W_PAST_S = 0.5        # floor on the past window; the real length is derived from the poles
TAIL = 1e-13          # how far the kernel must have decayed by the end of that window
ORDER = 8             # the order of the filter being replaced: butter(4) applied twice
LOG = open(os.path.join(sp_.OUT, "causal_design_log.txt"), "w", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


# --------------------------------------------------------------------------------------------
# the truncated zero-phase filter: the baseline, as in causal_chain.py
# --------------------------------------------------------------------------------------------
def bounded_lowpass(x, fc, fs, look_s, order=4):
    b, a = butter(order, fc / (fs / 2), btype="low")
    if look_s is None:
        return filtfilt(b, a, x, axis=0)      # not sp_.lowpass: that name is patched while we run
    W, L = int(round(W_PAST_S * fs)), int(round(look_s * fs))
    X = np.asarray(x, float)
    one_d = X.ndim == 1
    X = X[:, None] if one_d else X
    n, m = X.shape
    pad = np.vstack([np.repeat(X[:1], W, axis=0), X, np.repeat(X[-1:], L + 1, axis=0)])
    out = np.empty_like(X)
    for j in range(m):
        win = sliding_window_view(pad[:, j], W + L + 1)[:n]
        out[:, j] = filtfilt(b, a, win, axis=1)[:, W]
    return out[:, 0] if one_d else out


# --------------------------------------------------------------------------------------------
# the fixed-lag smoother
# --------------------------------------------------------------------------------------------
def _poly_model(m, q):
    """m-fold integrated white noise, in units of samples: the top derivative is the disturbance.

    A is the Taylor propagator and Q its integral over one step; both are written out rather than
    obtained from a matrix exponential, so the entries stay exact.
    """
    A = np.zeros((m, m))
    for i in range(m):
        for j in range(i, m):
            A[i, j] = 1.0 / fct(j - i)
    Q = np.empty((m, m))
    for i in range(m):
        for j in range(m):
            p = 2 * m - i - j - 1
            Q[i, j] = q / (fct(m - i - 1) * fct(m - j - 1) * p)
    H = np.zeros((1, m))
    H[0, 0] = 1.0
    return A, Q, H


def _steady_state(m, q, r=1.0):
    """Kalman and RTS gains once the covariance recursion has converged. The covariances do not
    depend on the data, so this is solved once and reused for every signal."""
    A, Q, H = _poly_model(m, q)
    try:
        Pp = solve_discrete_are(A.T, H.T, Q, np.array([[r]]))
    except Exception:                                   # fall back to iterating the recursion
        P = np.eye(m) * 1e6
        for _ in range(200000):
            Pp = A @ P @ A.T + Q
            K = Pp @ H.T / (H @ Pp @ H.T + r).item()
            Pn = (np.eye(m) - K @ H) @ Pp
            if np.max(np.abs(Pn - P)) < 1e-18 * max(1.0, np.max(np.abs(P))):
                P = Pn
                break
            P = Pn
        Pp = A @ P @ A.T + Q
    K = Pp @ H.T / (H @ Pp @ H.T + r).item()
    P = (np.eye(m) - K @ H) @ Pp
    return A, K[:, 0], P @ A.T @ np.linalg.inv(Pp)


def settling(fs, q, m=ORDER):
    """Frames for the impulse response to decay to TAIL, from the slowest closed-loop pole.

    Guessing this is not safe. The pole radius rises with the order (0.878 at m = 3, 0.955 at
    m = 8), and the acceleration kernel is scaled by fs^2 on top, so a tail that is negligible for
    position is not negligible for acceleration: at m = 8 the window has to be near 4 s. Past data
    costs memory, not latency, so a generous window is free.
    """
    A, K, C = _steady_state(m, q)
    rho = np.abs(np.linalg.eigvals(A - np.outer(K, np.eye(m)[0]) @ A)).max()
    return int(max(round(W_PAST_S * fs), np.ceil(np.log(TAIL) / np.log(rho))))


def smoother_kernels(fs, q, L_frames, n_past, m=ORDER):
    """Impulse responses of the position, velocity and acceleration states, as taps over
    [-L, n_past], in physical units.

    The smoother is LTI once the gains have settled, so its impulse responses define it; reading
    them off numerically avoids unrolling the recursion by hand. The position kernel sums to one,
    the derivative kernels to zero, so only the first is normalised.
    """
    A, K, C = _steady_state(m, q)
    n, i0 = 4 * (n_past + L_frames) + 512, 2 * (n_past + L_frames) + 256
    z = np.zeros(n)
    z[i0] = 1.0
    xf = np.empty((n, m))
    x = np.zeros(m)
    for k in range(n):                                  # forward pass
        xp = A @ x
        x = xp + K * (z[k] - xp[0])
        xf[k] = x
    Axf = xf @ A.T                                      # x_{k+1|k}
    s = xf.copy()
    for _ in range(L_frames):        # x_{k|k+j} = x_{k|k} + C (x_{k+1|k+j-1} - x_{k+1|k})
        s = xf + (np.vstack([s[1:], s[-1:]]) - Axf) @ C.T
    d = np.arange(n) - i0                               # output at k responds to the input at k-d
    sel = (d >= -L_frames) & (d <= n_past)
    kp, kv, ka = s[sel, 0], s[sel, 1], s[sel, 2]
    return kp / kp.sum(), kv * fs, ka * fs * fs         # samples -> seconds


def apply_kernel(x, ker, L, W):
    """y[k] = sum_d ker[d] x[k-d] over d in [-L, W]: L frames of future, W of past.

    `ker` runs from d = -L to d = W, which is exactly the order a 'valid' convolution wants, so
    the taps go in as they are stored.
    """
    X = np.asarray(x, float)
    one_d = X.ndim == 1
    X = X[:, None] if one_d else X
    pad = np.vstack([np.repeat(X[:1], W, axis=0), X, np.repeat(X[-1:], L, axis=0)])
    out = fftconvolve(pad, ker[:, None], mode="valid", axes=0)
    return out[:, 0] if one_d else out


def _f3db(ker, fs, n=16384):
    """-3 dB point of a kernel. A pure delay does not change the magnitude response, so the taps
    can be laid down from zero."""
    pad = np.zeros(n)
    pad[:len(ker)] = ker
    mag = np.abs(np.fft.rfft(pad))
    mag /= mag[0]
    f = np.fft.rfftfreq(n, 1.0 / fs)
    i = int(np.argmax(mag < 1 / np.sqrt(2)))
    return float(np.interp(1 / np.sqrt(2), [mag[i], mag[i - 1]], [f[i], f[i - 1]]))


def _zero_phase_f3db(fc, fs, order=4, n=16384):
    """The -3 dB point of filtfilt itself, which is not fc: the response is squared."""
    b, a = butter(order, fc / (fs / 2), btype="low")
    imp = np.zeros(n)
    imp[n // 2] = 1.0
    return _f3db(np.fft.ifftshift(filtfilt(b, a, imp)), fs, n)


def design_q(fc, fs, m=ORDER):
    """Pick q so the infinite-lag smoother turns over where the zero-phase filter does.

    The design and the window depend on each other --- q sets the poles, the poles set how long
    the kernel must be, and the kernel is what the -3 dB point is read from --- so the window is
    recomputed inside the bisection rather than fixed in advance.
    """
    target = _zero_phase_f3db(fc, fs)
    lo, hi = -30.0, 10.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        q = 10.0 ** mid
        w = settling(fs, q, m)
        kp, _, _ = smoother_kernels(fs, q, w, w, m)
        if _f3db(kp, fs) < target:
            lo = mid
        else:
            hi = mid
    q = 10.0 ** (0.5 * (lo + hi))
    return q, target, settling(fs, q, m)


_QC, _KER = {}, {}


def q_for(fc, fs, m=ORDER):
    if (fc, fs, m) not in _QC:
        _QC[(fc, fs, m)] = design_q(fc, fs, m)
    return _QC[(fc, fs, m)]


def n_past_for(fs, m=ORDER):
    """One window for the whole chain: the longest any of its filters needs."""
    return max(q_for(fc, fs, m)[2] for fc in (6.0, 20.0))


def kernel_for(fc, fs, look_s, m=ORDER):
    key = (round(fc, 6), round(fs, 3), round(look_s, 6), m)
    if key not in _KER:
        n_past, L = n_past_for(fs, m), int(round(look_s * fs))
        q = q_for(fc, fs, m)[0]
        _KER[key] = (smoother_kernels(fs, q, L, n_past, m), L, n_past)
    return _KER[key]


def smooth_fixed_lag(x, fc, fs, look_s):
    (kp, _, _), L, W = kernel_for(fc, fs, look_s)
    return apply_kernel(x, kp, L, W)


def deriv_fixed_lag(x, fs, look_s, fc=6.0):
    """Velocity and acceleration read off the smoother's own state, rather than obtained by
    differencing a filtered position twice."""
    (_, kv, ka), L, W = kernel_for(fc, fs, look_s)
    return apply_kernel(x, kv, L, W), apply_kernel(x, ka, L, W)


# --------------------------------------------------------------------------------------------
# patching
# --------------------------------------------------------------------------------------------
def prepare_with(subj, trial, look_s, which=("marker", "force", "cop"), design="designed"):
    """ex.prepare with the named elements of the chain restricted to a look-ahead of look_s.

    The two low-pass calls are told apart by their cutoff: 6 Hz is the marker filter, 20 Hz the
    force filter.
    """
    real_lowpass, real_hold, real_deriv = sp_.lowpass, sp_.hold_cop, sp_.deriv

    def patched_lowpass(x, fc, fs, order=4):
        tag = "marker" if fc < 10 else "force"
        if look_s is None or tag not in which:
            b, a = butter(order, fc / (fs / 2), btype="low")
            return filtfilt(b, a, x, axis=0)
        if design == "designed":
            # markers are left raw: the smoother in patched_deriv is matched to raw noise, and
            # filtering first would put two low passes in series. Raw marker noise is ~1 mm, which
            # is negligible in the segment geometry that still uses these positions.
            return np.asarray(x, float) if tag == "marker" else smooth_fixed_lag(x, fc, fs, look_s)
        return bounded_lowpass(x, fc, fs, look_s, order)

    def patched_deriv(x, fs):
        if look_s is None or design != "designed" or "marker" not in which:
            return real_deriv(x, fs)
        return deriv_fixed_lag(x, fs, look_s)

    def patched_hold(copx, Fy):
        if look_s is None or "cop" not in which:
            return real_hold(copx, Fy)
        L = int(round(look_s * sp_.FS))
        out = real_hold(copx, Fy)
        good = Fy > sp_.F_COP
        idx = np.where(good, np.arange(len(copx)), -1)
        np.maximum.accumulate(idx, out=idx)
        ahead = np.minimum(np.arange(len(copx)) + L, len(copx) - 1)
        use = np.where(good, np.arange(len(copx)), np.where(good[ahead], ahead, idx))
        return np.where(Fy > sp_.F_CONTACT, copx[np.where(use < 0, 0, use)], out)

    sp_.lowpass, sp_.hold_cop, sp_.deriv = patched_lowpass, patched_hold, patched_deriv
    try:
        return ex.prepare(subj, trial)
    finally:
        sp_.lowpass, sp_.hold_cop, sp_.deriv = real_lowpass, real_hold, real_deriv


def solve_and_score(d, ref_fR=None, warmup=0):
    S = ex.newton_sum(d, 1.0)
    pr = ex.vpp_prior(d, CAL["h_vpp"])
    fR, _, _ = ex.solve(d, S, fc_smooth=CAL["fc_smooth"], prior=pr,
                        w_prior=CAL["w_prior"], w_torque=0.0)
    grf = d["grf"]
    trim = d["trim"].copy()
    trim[:warmup] = False          # the kernel is still filling from the edge pad before this
    on = (grf["R"]["Fy"] > sp_.F_CONTACT) & trim
    BW = sp_.MASS * 9.81
    out = dict(stance=100 * np.sqrt(np.mean((fR[on] - grf["R"]["Fx"][on]) ** 2)) / BW)
    if ref_fR is not None:
        m = min(len(fR), len(ref_fR))
        o = on[:m]
        out["vs_batch"] = 100 * np.sqrt(np.mean((fR[:m][o] - ref_fR[:m][o]) ** 2)) / BW
    return out, fR


def self_test(fs=None):
    """The model reproduces polynomials of degree below its order exactly, so the velocity kernel
    must return the slope of a ramp and the acceleration kernel the curvature of a parabola. That
    is a sharper check on the kernels than any tolerance on the gains."""
    fs = fs or sp_.FS
    n, bad = 8000, []
    t = np.arange(n) / fs
    for L in (0.0, 0.05, 0.15):
        (kp, kv, ka), Lf, W = kernel_for(6.0, fs, L)
        m = slice(W + 50, n - W - 50)
        v = apply_kernel(3.7 * t, kv, Lf, W)[m].mean()
        a = apply_kernel(0.5 * 2.9 * t ** 2, ka, Lf, W)[m].mean()
        ok = abs(kp.sum() - 1) < 1e-9 and abs(v - 3.7) < 1e-4 and abs(a - 2.9) < 1e-3
        say(f"    L = {1000*L:5.0f} ms  {len(kp):4d} taps  sum {kp.sum():.10f}  "
            f"ramp {v:.6f} (3.7)  parabola {a:.6f} (2.9)  {'ok' if ok else 'FAIL'}")
        if not ok:
            bad.append(L)
    return not bad


# --------------------------------------------------------------------------------------------
def main(n_trials=12, looks=(0.0, 0.02, 0.05, 0.10, 0.15, 0.30)):
    fs = sp_.FS
    n_past = n_past_for(fs)
    say(f"design: order {ORDER} ({-20*ORDER} dB/decade, matching butter(4) applied twice); "
        f"only q/r matters, r normalised to 1")
    for fc in (6.0, 20.0):
        q, target, w = q_for(fc, fs)
        kp, _, _ = smoother_kernels(fs, q, w, w)
        say(f"  filtfilt(butter(4, {fc:4.0f} Hz)) turns over at {target:5.2f} Hz "
            f"(not {fc:.0f}: the response is squared) -> q/r = {q:.4g}, "
            f"smoother {_f3db(kp, fs):5.2f} Hz, settles in {w/fs:.2f} s")
    say(f"  past window for the chain: {n_past} frames ({n_past/fs:.2f} s); scoring starts after "
        f"it, so every arm is judged on the same settled frames")
    say("  self test (the model is exact on polynomials below its order):")
    if not self_test():
        say("  SELF TEST FAILED -- the kernels are not trustworthy, stopping")
        return None, None

    trials = [t for t in available_trials() if t[0] not in CAL_SUBJECTS][:n_trials]
    say(f"\n{len(trials)} test trials")

    # ---- part A: where does the cost of being causal come from? ----
    arms = [("marker",), ("force",), ("cop",), ("marker", "force", "cop")]
    rows, t0 = [], time.time()
    for subj, trial, *_ in trials:
        try:
            d0 = prepare_with(subj, trial, None)
        except Exception as e:
            say(f"  skip {subj} {trial}: {e!r}")
            continue
        sp_.configure(subj, trial)
        base, ref = solve_and_score(d0, warmup=n_past)
        rows.append(dict(subject=subj, trial=trial, arm="batch", **base, vs_batch=0.0))
        for which in arms:
            d = prepare_with(subj, trial, 0.0, which, design="truncated")
            sp_.configure(subj, trial)
            m, _ = solve_and_score(d, ref, warmup=n_past)
            rows.append(dict(subject=subj, trial=trial, arm="+".join(which), **m))
        say(f"  decomposition {subj} {trial} [{(time.time()-t0)/60:.1f} min]")
    dec = pd.DataFrame(rows)
    dec.to_csv(os.path.join(sp_.OUT, "causal_design_decomp.csv"), index=False)
    b = dec[dec.arm == "batch"].stance.mean()
    say("\ncost of removing the future, element by element (look-ahead 0, truncated filter)")
    say(f"{'element made causal':>28s} {'stance':>9s} {'penalty':>9s}")
    say(f"{'none (whole record)':>28s} {b:9.3f} {0.0:9.3f}")
    for which in arms:
        g = dec[dec.arm == "+".join(which)]
        if len(g):
            say(f"{'+'.join(which):>28s} {g.stance.mean():9.3f} {g.stance.mean()-b:9.3f}")

    # ---- part B: truncated filter vs designed chain, over look-ahead ----
    rows = []
    for subj, trial, *_ in trials:
        try:
            d0 = prepare_with(subj, trial, None)
        except Exception as e:
            say(f"  skip {subj} {trial}: {e!r}")
            continue
        sp_.configure(subj, trial)
        base, ref = solve_and_score(d0, warmup=n_past)
        rows.append(dict(subject=subj, trial=trial, design="batch", look_s=np.nan,
                         **base, vs_batch=0.0))
        for design in ("truncated", "designed"):
            for L in looks:
                d = prepare_with(subj, trial, L, ("marker", "force", "cop"), design=design)
                sp_.configure(subj, trial)
                m, _ = solve_and_score(d, ref, warmup=n_past)
                rows.append(dict(subject=subj, trial=trial, design=design, look_s=L, **m))
        say(f"  sweep {subj} {trial} [{(time.time()-t0)/60:.1f} min]")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(sp_.OUT, "causal_design_sweep.csv"), index=False)

    bb = df[df.design == "batch"].stance.mean()
    say(f"\nwhole-record pipeline: {bb:.3f} %BW\n")
    say(f"{'look-ahead':>12s} {'truncated':>11s} {'designed':>10s} {'gain':>8s} {'vs batch':>10s}")
    for L in looks:
        t_ = df[(df.design == "truncated") & (df.look_s == L)].stance.mean()
        g = df[(df.design == "designed") & (df.look_s == L)]
        say(f"{1000*L:9.0f} ms {t_:11.3f} {g.stance.mean():10.3f} "
            f"{t_-g.stance.mean():8.3f} {g.vs_batch.mean():10.3f}")
    LOG.close()
    return dec, df


if __name__ == "__main__":
    main()

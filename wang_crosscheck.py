"""Reference cross-check on the Wang 2023 data set.

Question: does our planar inverse dynamics agree with an independent 3-D implementation?
Wang et al. published, per trial, the OpenSim inverse-dynamics result computed from marker
kinematics and the instrumented-treadmill GRF (`...ID.sto`), and the wearable counterpart
computed from IMU kinematics and insole forces (`...ID_Portable.sto`).

Two comparisons:
  1. our reference (lab markers + treadmill 3-D GRF, planar ID)  vs  Wang's ID.sto
     -> validates our reference pipeline (the same role as the check on the Camargo data)
  2. Wang's ID_Portable.sto (their wearable result)  vs  their own ID.sto
     -> reproduces their published wearable accuracy in our metric, so that the SoleID numbers
        in the application layer can be placed on the same scale.

Sign convention: OpenSim reports generalised forces about the joint coordinate (dorsiflexion,
knee flexion, hip flexion). Our planar moments are about +z of the lab frame. The comparison is
therefore reported with the sign that maximises the correlation, and the flip is recorded.
"""
import os
import sys

import numpy as np
import pandas as pd

import soleid_planar as sp_
import wang_adapter as wa
import wang_run as wr

COLS = dict(ankle="ankle_angle_r_moment", knee="knee_angle_r_moment", hip="hip_flexion_r_moment")


def compare(a, b, trim):
    a, b = a[trim], b[trim]
    r = float(np.corrcoef(a, b)[0, 1])
    s = 1.0 if r >= 0 else -1.0
    return dict(r=abs(r), sign_flip=s < 0,
                rmse=float(np.sqrt(np.mean((s * a - b) ** 2))),
                peak_ours=float(np.max(np.abs(a))), peak_theirs=float(np.max(np.abs(b))))


def run_trial(subj, trial, info):
    d = wa.load_trial(subj, trial)
    mass, height = info[subj]["mass"], info[subj]["height"]
    n = len(d["t"])
    sp_.MASS, sp_.HEIGHT, sp_.FS = mass, height, wr.FS

    trc = wa.read_trc(d["paths"]["trc"])
    n = min(n, len(trc[0]))
    Plab, pelvis_lab, lab_ok = wr.planar_from_markers(trc, n)
    if not lab_ok:
        raise ValueError("marker gaps")
    Plab = {k: (sp_.lowpass(v, 6.0, wr.FS) if isinstance(v, np.ndarray) else v) for k, v in Plab.items()}
    pelvis_lab = sp_.lowpass(pelvis_lab, 6.0, wr.FS)
    Plab["R.Hip"] = Plab["L.Hip"] = wr.hip_from_pelvis_planar(pelvis_lab, height)
    segs = wr.build(Plab, mass, height, n)

    grf = {s: {k: v[:n] for k, v in d["ref"][s].items()} for s in ("R", "L")}
    ours = sp_.inverse_dynamics(segs, "R", grf["R"]["Fx"], grf["R"]["Fy"], grf["R"]["copx"])

    id_lab, id_wear = d["id_lab"], d["id_wear"]
    m = min(n, len(id_lab), len(id_wear))
    trim = np.zeros(m, bool)
    trim[int(wr.FS):m - int(wr.FS)] = True
    onR = grf["R"]["Fy"][:m] > sp_.F_CONTACT
    trim &= onR                                  # compare during stance, where the moments are defined

    row = dict(subject=subj, trial=trial, speed=d["speed"], mass=mass, n=m)
    for j, col in COLS.items():
        theirs_lab = id_lab[col].values[:m] / mass          # N m -> N m / kg
        theirs_wear = id_wear[col].values[:m] / mass
        c1 = compare(ours[j][:m] / mass, theirs_lab, trim)
        c2 = compare(theirs_wear, theirs_lab, trim)
        for k, v in c1.items():
            row[f"ours_vs_theirs_{j}_{k}"] = v
        for k, v in c2.items():
            row[f"wang_wearable_{j}_{k}"] = v
    return row


def main():
    info = wa.subject_info()
    subs = sorted([x for x in os.listdir(wa.ROOT) if x.startswith("Subj")])
    rows = []
    for s in subs:
        for t in wa.WALKS:
            try:
                rows.append(run_trial(s, t, info))
                print(f"  {s} {t}: ankle r={rows[-1]['ours_vs_theirs_ankle_r']:.3f} "
                      f"knee r={rows[-1]['ours_vs_theirs_knee_r']:.3f} "
                      f"hip r={rows[-1]['ours_vs_theirs_hip_r']:.3f}", flush=True)
            except Exception as e:
                print(f"  {s} {t}: skip ({e!r})", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(sp_.OUT, "wang_crosscheck.csv"), index=False)
    summarize(df)
    return df


def summarize(df):
    print(f"\nWANG CROSS-CHECK: {df.subject.nunique()} subjects, {len(df)} trials")
    print("\n1) our planar reference vs Wang's 3-D OpenSim ID (lab kinematics + treadmill GRF)")
    for j in COLS:
        print(f"   {j:6s}: r = {df[f'ours_vs_theirs_{j}_r'].mean():.3f} "
              f"[{df[f'ours_vs_theirs_{j}_r'].min():.3f}, {df[f'ours_vs_theirs_{j}_r'].max():.3f}]  "
              f"RMSE = {df[f'ours_vs_theirs_{j}_rmse'].mean():.3f} Nm/kg  "
              f"(sign flipped in {100 * df[f'ours_vs_theirs_{j}_sign_flip'].mean():.0f}% of trials)")
    print("\n2) Wang's own wearable ID vs their lab ID (their published comparison, our metric)")
    for j in COLS:
        print(f"   {j:6s}: r = {df[f'wang_wearable_{j}_r'].mean():.3f}  "
              f"RMSE = {df[f'wang_wearable_{j}_rmse'].mean():.3f} Nm/kg")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        summarize(pd.read_csv(os.path.join(sp_.OUT, "wang_crosscheck.csv")))
    else:
        main()

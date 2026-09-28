"""Adapter for the Wang et al. 2023 dataset (Zenodo 6457662) -> SoleID planar pipeline.

Inputs per trial (already synchronised at 100 Hz by the authors):
  Xsens/<trial>.mvnx            segment positions (23 segments) + centre of mass  [wearable kinematics]
  OS/DataFiles/<S><trial>.trc   marker trajectories                                [lab kinematics]
  OS/DataFiles/..XLD.mot        treadmill GRF, 3D per foot + COP                   [reference]
  OS/DataFiles/..XLD_INSOLE.mot insole vertical force + COP per foot               [SoleID input]
  OS/DataFiles/..ID.sto         Wang's lab inverse dynamics                        [cross-check]
  OS/DataFiles/..ID_Portable.sto Wang's wearable inverse dynamics                  [their baseline]

Frames: MVNX is X forward, Y left, Z up (metres, global, drifting on a treadmill).
OpenSim ground frame here is X forward, Y up, Z right. The adapter converts MVNX to the
OpenSim convention, removes the treadmill drift (linear trend in X) and aligns the horizontal
origin so that the mean stance-foot position matches the mean insole COP. Both steps are data
alignment, not parameter tuning.
"""
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import config

ROOT = config.WANG_PROCESSED
CACHE = config.WANG_CACHE
os.makedirs(CACHE, exist_ok=True)

WALKS = ["walk_09", "walk_18", "walk_27", "walk_36", "walk_45", "walk_54"]   # km/h * 10
SPEED = {w: int(w.split("_")[1]) / 10 / 3.6 for w in WALKS}                    # m/s
NS = "{http://www.xsens.com/mvn/mvnx}"

# MVNX segments we need for a sagittal 7-segment model
SEG_NEEDED = ["Pelvis", "RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToe",
              "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToe", "T8", "Head"]


def subject_info():
    """mass and height per subject from subjs_info.txt"""
    txt = open(os.path.join(ROOT, "subjs_info.txt")).read()
    out = {}
    for m in re.finditer(r"subj(\d+):\s*subjMass\s*=\s*([\d.]+);\s*subjHeight\s*=\s*([\d.]+);", txt, re.S):
        out[f"Subj{int(m.group(1)):02d}"] = dict(mass=float(m.group(2)), height=float(m.group(3)))
    return out


def parse_mvnx(path, cache=True):
    """Return t (s), segment positions, centre-of-mass position/velocity/acceleration and the
    sensors' gravity-free accelerations, all converted to the OpenSim-like frame.

    MVN 2021 writes <centerOfMass> as 9 numbers: position, velocity, acceleration. Xsens'
    global *position* drifts on a treadmill, but the COM acceleration and the sensors' free
    accelerations come from the sensors themselves and are not affected by that drift.
    """
    key = os.path.join(CACHE, os.path.basename(os.path.dirname(os.path.dirname(path))) + "_" +
                       os.path.basename(path).replace(".mvnx", "_v2.npz"))
    if cache and os.path.exists(key):
        z = np.load(key, allow_pickle=True)
        return dict(t=z["t"], pos={k: z[f"p_{k}"] for k in SEG_NEEDED if f"p_{k}" in z},
                    com=z["com"], com_vel=z["com_vel"], com_acc=z["com_acc"],
                    free_acc={k[2:]: z[k] for k in z.files if k.startswith("a_")})
    labels, sensors = [], []
    positions, coms, accs, times = [], [], [], []
    for ev, el in ET.iterparse(path, events=("end",)):
        tag = el.tag.replace(NS, "")
        if tag == "segments":
            labels = [s.get("label") for s in el.findall(f"{NS}segment")]
            el.clear()
        elif tag == "sensors":
            sensors = [s.get("label") for s in el.findall(f"{NS}sensor")]
            el.clear()
        elif tag == "frame":
            if el.get("type") != "normal":
                el.clear()
                continue
            p = el.find(f"{NS}position")
            c = el.find(f"{NS}centerOfMass")
            a = el.find(f"{NS}sensorFreeAcceleration")
            if p is None:
                el.clear()
                continue
            positions.append(np.fromstring(p.text, sep=" "))
            v = np.fromstring(c.text, sep=" ") if c is not None else np.full(9, np.nan)
            coms.append(np.pad(v, (0, max(0, 9 - len(v))))[:9])
            accs.append(np.fromstring(a.text, sep=" ") if a is not None else np.full(3 * len(sensors), np.nan))
            times.append(float(el.get("time")) / 1000.0)
            el.clear()
    P = np.array(positions).reshape(len(positions), -1, 3)
    C = np.array(coms)
    A = np.array(accs).reshape(len(accs), -1, 3)
    t = np.array(times) - times[0]
    # MVNX: x forward, y left, z up  ->  OpenSim-like: x forward, y up, z right
    conv = lambda a: np.column_stack([a[..., 0], a[..., 2], -a[..., 1]])
    pos = {lab: conv(P[:, i, :]) for i, lab in enumerate(labels) if lab in SEG_NEEDED}
    free_acc = {lab: conv(A[:, i, :]) for i, lab in enumerate(sensors)}
    out = dict(t=t, pos=pos, com=conv(C[:, 0:3]), com_vel=conv(C[:, 3:6]), com_acc=conv(C[:, 6:9]),
               free_acc=free_acc)
    if cache:
        np.savez_compressed(key, t=t, com=out["com"], com_vel=out["com_vel"], com_acc=out["com_acc"],
                            **{f"p_{k}": v for k, v in pos.items()},
                            **{f"a_{k}": v for k, v in free_acc.items()})
    return out


def read_mot(path):
    """OpenSim .mot/.sto -> DataFrame (header auto-detected)."""
    with open(path) as f:
        lines = f.readlines()
    i = next(k for k, l in enumerate(lines) if l.strip().lower() == "endheader")
    cols = lines[i + 1].split()
    data = np.array([[float(x) for x in l.split()] for l in lines[i + 2:] if l.strip()])
    return pd.DataFrame(data, columns=cols[:data.shape[1]])


def read_trc(path):
    """OpenSim .trc -> dict of marker -> (N,3), plus time. Units taken from the header
    (these files are already in metres)."""
    with open(path) as f:
        lines = f.readlines()
    units = lines[2].split()[4] if len(lines[2].split()) > 4 else "m"
    scale = 1.0 if units.lower().startswith("m") else 1e-3
    names = lines[3].split()[2:]
    data = np.array([[float(x) if x not in ("", "NaN") else np.nan for x in l.split()]
                     for l in lines[6:] if l.strip()])
    t = data[:, 1]
    out = {}
    for i, n in enumerate(names):
        xyz = data[:, 2 + 3 * i: 5 + 3 * i]
        if xyz.shape[1] == 3:
            out[n] = xyz * scale
    return t, out


def trial_paths(subj, trial):
    base = os.path.join(ROOT, subj, "OS", "DataFiles", f"{subj}{trial}")
    return dict(mvnx=os.path.join(ROOT, subj, "Xsens", f"{trial}.mvnx"),
                trc=base + ".trc", xld=base + "XLD.mot", insole=base + "XLD_INSOLE.mot",
                id_lab=base + "ID.sto", id_wear=base + "ID_Portable.sto",
                ik_lab=base + "IK.mot", ik_imu=base + "IK_IMU.mot")


def load_trial(subj, trial):
    p = trial_paths(subj, trial)
    xld, ins = read_mot(p["xld"]), read_mot(p["insole"])
    out = dict(subj=subj, trial=trial, speed=SPEED[trial], t=xld.time.values,
               ref=dict(R=dict(Fx=xld.ground_force_vx.values, Fy=xld.ground_force_vy.values,
                               copx=xld.ground_force_px.values),
                        L=dict(Fx=xld.l_ground_force_vx.values, Fy=xld.l_ground_force_vy.values,
                               copx=xld.l_ground_force_px.values)),
               insole=dict(R=dict(Fy=ins.ground_force_vy.values, copx=ins.ground_force_px.values),
                           L=dict(Fy=ins.l_ground_force_vy.values, copx=ins.l_ground_force_px.values)),
               id_lab=read_mot(p["id_lab"]), id_wear=read_mot(p["id_wear"]), paths=p)
    return out


def resample_mvnx(mv, n, fs=100.0, tau=0.0):
    """MVNX runs at 250 Hz and starts before the force recording. Resample every channel onto the
    force time base (n samples at fs), shifted by tau seconds."""
    t_src = mv["t"]
    t_dst = np.arange(n) / fs + tau
    f = lambda a: np.column_stack([np.interp(t_dst, t_src, a[:, i]) for i in range(a.shape[1])])
    out = dict(t=np.arange(n) / fs,
               pos={k: f(v) for k, v in mv["pos"].items()},
               com=f(mv["com"]), com_vel=f(mv["com_vel"]), com_acc=f(mv["com_acc"]),
               free_acc={k: f(v) for k, v in mv.get("free_acc", {}).items()})
    return out


def align_time(mv, Fy_total, mass, n, fs=100.0, span=4.0, coarse=0.02, fine=0.004):
    """Find the MVNX time offset by matching the vertical COM-derived force to the measured one:
    m (a_y + g) = sum Fy. Returns (tau, correlation)."""
    def score(tau):
        r = resample_mvnx(mv, n, fs, tau)
        pred = mass * (r["com_acc"][:, 1] + 9.81)
        sl = slice(int(fs), n - int(fs))
        a, b = pred[sl], Fy_total[sl]
        if np.std(a) < 1e-6 or np.std(b) < 1e-6:
            return -1.0
        return float(np.corrcoef(a, b)[0, 1])
    taus = np.arange(-span, span + 1e-9, coarse)
    best = max(taus, key=score)
    taus = np.arange(best - coarse, best + coarse + 1e-9, fine)
    best = max(taus, key=score)
    return float(best), score(best)


def xsens_planar(subj, trial, n, fs=100.0, Fy_total=None, mass=None, tau=None):
    """Planar (x, y) joint positions, COM and sensor accelerations from MVNX, resampled to the
    force time base, time-aligned, treadmill drift removed and walking direction normalised."""
    mv_raw = parse_mvnx(trial_paths(subj, trial)["mvnx"])
    if tau is None and Fy_total is not None and mass is not None:
        tau, r_align = align_time(mv_raw, Fy_total, mass, n, fs)
    else:
        tau, r_align = (tau or 0.0), np.nan
    mv = resample_mvnx(mv_raw, n, fs, tau)
    pos = mv["pos"]
    m = n
    # remove the forward drift: the treadmill keeps the subject in place, Xsens integrates travel
    t = mv["t"][:m]
    ref = pos["Pelvis"][:m, 0]
    a, b = np.polyfit(t, ref, 1)
    # the subject may travel along -x in the MVNX frame; rotate 180 deg about the vertical so
    # that the walking direction is +x, as in the OpenSim/treadmill frame
    sgn = 1.0 if a >= 0 else -1.0
    out = {}
    for k, v in pos.items():
        v = v[:m].copy()
        v[:, 0] = sgn * (v[:, 0] - (a * t + b))    # drift removed, direction normalised
        out[k] = v[:, :2]
    com = mv["com"][:m].copy()
    com[:, 0] = sgn * (com[:, 0] - (a * t + b))
    flip = np.array([sgn, 1.0, 1.0])
    return dict(pos=out, com=com[:, :2], speed_est=abs(a), n=m, flipped=sgn < 0, tau=tau,
                r_align=r_align,
                com_acc=(mv["com_acc"][:m] * flip)[:, :2],
                com_vel=(mv["com_vel"][:m] * flip)[:, :2],
                free_acc={k: (v[:m] * flip)[:, :2] for k, v in mv.get("free_acc", {}).items()})


if __name__ == "__main__":
    info = subject_info()
    subs = sorted([d for d in os.listdir(ROOT) if d.startswith("Subj")])
    print("subjects:", subs)
    print("info:", {k: info[k] for k in subs if k in info})
    d = load_trial(subs[0], "walk_36")
    print("trial:", d["subj"], d["trial"], "speed", d["speed"], "n", len(d["t"]))
    for s in ("R", "L"):
        print(f"  ref {s}: Fy max {d['ref'][s]['Fy'].max():.0f} N, Fx range {d['ref'][s]['Fx'].min():.0f}..{d['ref'][s]['Fx'].max():.0f}")
        print(f"  insole {s}: Fy max {d['insole'][s]['Fy'].max():.0f} N, cop range {d['insole'][s]['copx'].min():.3f}..{d['insole'][s]['copx'].max():.3f}")
    k = xsens_planar(subs[0], "walk_36", len(d["t"]))
    print("mvnx frames:", k["n"], "forward speed estimate", round(k["speed_est"], 3), "m/s")
    for seg in ("Pelvis", "RightFoot", "RightToe"):
        p = k["pos"][seg]
        print(f"  {seg}: x {p[:, 0].min():.3f}..{p[:, 0].max():.3f}  y {p[:, 1].min():.3f}..{p[:, 1].max():.3f}")

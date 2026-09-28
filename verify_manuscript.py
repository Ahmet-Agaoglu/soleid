"""Re-derive every number of the paper from results/ and compare it with the value the paper prints.

Each check pairs a value recomputed from the saved results with the value as printed in the paper,
rounded the same way; a FAIL means the code and the paper disagree. If config.MANUSCRIPT_TEX points
to the manuscript source, the script also checks that every printed value occurs in the text.
Literature values, method parameters and data-set design facts are not checked.
"""
import io
import json
import os

import numpy as np
import pandas as pd

import config

R = config.RESULTS
TEX = config.MANUSCRIPT_TEX

if TEX and os.path.exists(TEX):
    body = io.open(TEX, encoding="utf-8").read().split(r"\begin{document}")[1]
    # digit groups are written 19\,378 in the text; compare against the bare digits
    flat = body.replace("\\,", "")
else:
    flat = None                         # no manuscript source: compare with the printed values only

t = pd.read_csv(os.path.join(R, "test_all_trials.csv"))
cam = pd.read_csv(os.path.join(R, "camargo_treadmill_strides.csv"))
lg = pd.read_csv(os.path.join(R, "camargo_levelground_steps.csv"))
ramp = pd.read_csv(os.path.join(R, "camargo_ramp_steps.csv"))
w = pd.read_csv(os.path.join(R, "wang_application.csv"))
cc = pd.read_csv(os.path.join(R, "wang_crosscheck.csv"))
dg = pd.read_csv(os.path.join(R, "final_degradation_trials.csv"))
mf = pd.read_csv(os.path.join(R, "mhe_fukuchi.csv"))
mw = pd.read_csv(os.path.join(R, "mhe_wang.csv"))
q = pd.read_csv(os.path.join(R, "fukuchi_quality.csv"))
st = json.load(open(os.path.join(R, "statistics.json")))
_calj = json.load(open(os.path.join(R, "calibration.json")))
_skip = pd.read_csv(os.path.join(R, "test_skipped.csv"))
nc = json.load(open(os.path.join(R, "norm_constants.json")))
_fc = json.load(open(os.path.join(R, "friction_cone.json")))
_sn = json.load(open(os.path.join(R, "sole_normal.json")))
_vs = pd.read_csv(os.path.join(R, "vpp_split_test.csv"))
sp = pd.read_csv(os.path.join(R, "split_sensitivity.csv"))
_pv = pd.read_csv(os.path.join(R, "phase_vpp_test.csv"))
# the phase-dependent pivot was searched on two grids: the first one, then an extended one
_pv_grid_size = sum(len(pd.read_csv(os.path.join(R, f)))
                    for f in ("phase_vpp_grid_narrow.csv", "phase_vpp_grid.csv"))
_cc = pd.read_csv(os.path.join(R, "causal_chain.csv"))
_cc_batch = float(_cc[_cc.look_s.isna()].stance.mean())
_cd = pd.read_csv(os.path.join(R, "causal_design_decomp.csv"))
_cd_batch = float(_cd[_cd.arm == "batch"].stance.mean())
_cv = pd.read_csv(os.path.join(R, "causal_variants.csv"))
# systematic hip offset, the Wang 2x2
_hb = json.load(open(os.path.join(R, "hip_bias_decomp.json")))
_hbd = pd.read_csv(os.path.join(R, "hip_bias_decomp.csv"))
_wf = pd.read_csv(os.path.join(R, "wang_factorial.csv"))
_wf = _wf[_wf.lab_ok]
_wfs = json.load(open(os.path.join(R, "wang_factorial_stats.json")))
from comparison_tables import table_camargo as _tcam      # the same source the table is built from
_cam = {r["study"]: r for r in _tcam() if r["study"].startswith("SoleID")}
# ablation, quality control, statistics, reference agreement
_ab = json.load(open(os.path.join(R, "ablation_qp.json")))
_qc = json.load(open(os.path.join(R, "qc_audit.json")))
_sr = json.load(open(os.path.join(R, "stats_revision.json")))
_hb2 = json.load(open(os.path.join(R, "hip_bias_breakdown.json")))
_ra = json.load(open(os.path.join(R, "reference_agreement.json")))
_rwang = _ra["wang"]
_px = json.load(open(os.path.join(R, "stats_paired_extra.json")))
_sq = _px["square"]
_srp = _sr["paired"]


def _sqd(a, b, k, nd=3):
    return float((_wf[f"{a}_{k}"] - _wf[f"{b}_{k}"]).mean())
_q = pd.read_csv(os.path.join(R, "fukuchi_quality.csv"))
_qr = _q[_q.ok]
_per = _qr.groupby("subject").bw_ratio
_dg = pd.read_csv(os.path.join(R, "final_degradation_trials.csv"))
_dgs = _dg[_dg.arm == "SoleID"]
_dgc = _dgs[_dgs.condition == "clean"].set_index(["subject", "trial"])
_dgw = _dgs[_dgs.condition == "wearable"].set_index(["subject", "trial"])
_PARTS = [("marker_noise_mm", 10), ("kin_bias_mm", 20), ("marker_fps", 60), ("fy_scale_pct", 8),
          ("fy_noise_pctBW", 2), ("cop_shift_mm", 10), ("sync_frames", 1)]


def _deg_sum(met):
    return sum((_dgs[(_dgs.condition == c) & (_dgs.level == lv)].set_index(["subject", "trial"])[met]
                - _dgc[met]).mean() for c, lv in _PARTS)


from statsmodels.regression.mixed_linear_model import MixedLM as _MixedLM
_t2 = t.assign(gain=t.SoleID_v1_stance - t.SoleID_stance)
_trim = _t2[_t2.newton_check <= _t2.newton_check.quantile(0.95)]
_mt = _MixedLM.from_formula("gain ~ newton_check", _trim, groups=_trim["subject"]).fit(reml=True, method="lbfgs")
_gs = _t2.groupby("subject")[["gain", "newton_check"]].mean()


def _cam3(cond):
    r = _cam[f"SoleID ({cond})"]
    return (r["ankle"] + r["knee"] + r["hip"]) / 3


def _same_sign(col):
    b = _hbd[col].values
    return int(np.sum(np.sign(b) == np.sign(b.mean())))


def _pen(arm):
    return float(_cd[_cd.arm == arm].stance.mean()) - _cd_batch


def _var(mode, L=0.05):
    return float(_cv[(_cv["mode"] == mode) & (_cv.look_s == L)].stance.mean())


def _look(L, col="stance"):
    return float(_cc[_cc.look_s == L][col].mean())

pv_frozen, pv_phase = _pv[_pv.variant == "frozen"], _pv[_pv.variant == "phase"]
_vj = _vs.merge(t[["subject", "trial", "SoleID_stance", "SoleID_hip"]],
                on=["subject", "trial"])

_K = ["h_vpp", "w_prior", "w_torque", "fc_smooth"]
_pc = pd.read_csv(os.path.join(R, "split_propagated_camargo.csv"))
_wt = sp.groupby(_K).size()
if "n_strides" in _pc:          # one row per trial: weight each trial mean by its strides
    _g = _pc.assign(stance=_pc.stance * _pc.n_strides, hip=_pc.hip * _pc.n_strides) \
        .groupby(_K + ["subject"])
    _subj = _g[["stance", "hip"]].sum().div(_g.n_strides.sum(), axis=0)
else:                           # one row per stride
    _subj = _pc.groupby(_K + ["subject"])[["stance", "hip"]].mean()
_PW = (_subj.groupby(_K).mean()
       .join((_wt / _wt.sum()).rename("weight")))


_pwang = pd.read_csv(os.path.join(R, "split_propagated_wang.csv"))
_WW = (_pwang.groupby(_K + ["subject"])[["SoleID_stance", "SoleID_hip"]].mean().groupby(_K).mean()
       .join((_wt / _wt.sum()).rename("weight")))


def _ww(col):
    v, w = _WW[col].values, _WW.weight.values
    m = float(np.sum(v * w))
    sd = float(np.sqrt(np.sum(w * (v - m) ** 2)))
    fr = float(_WW.loc[(0.2, 1.0, 0.0, 12.0), col])
    return m, sd, float(v.min()), float(v.max()), fr, 100 * float(np.sum(w[v > fr]))


def _pw(col):
    """Weighted mean, sd, range, the frozen configuration's value, and its percentile."""
    v, w = _PW[col].values, _PW.weight.values
    m = float(np.sum(v * w))
    sd = float(np.sqrt(np.sum(w * (v - m) ** 2)))
    fr = float(_PW.loc[(0.2, 1.0, 0.0, 12.0), col])
    return m, sd, float(v.min()), float(v.max()), fr, 100 * float(np.sum(w[v > fr]))


def deg(cond, lv, arm, col="stance"):
    c, l = ("clean", 0) if lv == 0 else (cond, lv)
    return dg[(dg.condition == c) & (dg.level == l) & (dg.arm == arm)][col].mean()


CHECKS = [
    # (what, value from the data, how it is written in the tex)
    ("test subjects", t.subject.nunique(), "31"),
    ("test trials", len(t), "231"),
    ("calibration trials", _calj["n_trials"], "40"),
    ("quality: total trials", len(q), "382"),
    ("quality: failing", int((~q.ok).sum()), "105"),
    ("quality: dead belt", int((q.reason == "dead belt").sum()), "54"),
    ("quality: scale error", int(q.reason.fillna("").str.startswith("force/mass").sum()), "23"),
    ("quality: handrail", int((q.reason == "handrail").sum()), "28"),

    ("SoleID stance", t.SoleID_stance.mean(), "3.13"),
    ("SoleID ds", t.SoleID_ds.mean(), "4.01"),
    ("SoleID hip", t.SoleID_hip.mean(), "0.199"),
    ("SoleID knee", t.SoleID_knee.mean(), "0.110"),
    ("SoleID ankle", t.SoleID_ankle.mean(), "0.020"),
    ("Newton stance", t.Newton_prop_stance.mean(), "6.35"),
    ("Newton ds", t.Newton_prop_ds.mean(), "9.73"),
    ("Newton hip", t.Newton_prop_hip.mean(), "0.398"),
    ("Kin-only stance", t.Kin_only_stance.mean(), "6.36"),
    ("Kin-only hip", t.Kin_only_hip.mean(), "0.793"),
    ("no-prior stance", t.SoleID_v1_stance.mean(), "4.45"),
    ("no-prior hip", t.SoleID_v1_hip.mean(), "0.281"),
    ("win rate vs Newton", 100 * (t.SoleID_stance < t.Newton_prop_stance).mean(), "98.3"),
    ("win rate vs no-prior", 100 * (t.SoleID_stance < t.SoleID_v1_stance).mean(), "87.4"),
    ("Fukuchi dynamics residual", t.newton_check.mean(), "4.5"),

    ("prior gain slope", st["gain_vs_newton_check"]["slope"], "0.519"),
    ("prior gain r", st["gain_vs_newton_check"]["r"], "0.76"),
    ("gain Fukuchi", (t.SoleID_v1_stance - t.SoleID_stance).mean(), "1.32"),
    ("gain Camargo tm", (cam.SoleID_noVPP_stance - cam.SoleID_stance).mean(), "2.65"),
    ("gain Camargo lg", (lg.SoleID_noVPP_stance - lg.SoleID_stance).mean(), "3.26"),
    ("gain Camargo ramp", (ramp.SoleID_noVPP_stance - ramp.SoleID_stance).mean(), "3.78"),

    ("Camargo tm stance", cam.SoleID_stance.mean(), "3.60"),
    ("Camargo tm hip", cam.SoleID_hip.mean(), "0.235"),
    ("Camargo tm Newton stance", cam.Newton_prop_stance.mean(), "8.59"),
    ("Camargo tm Newton hip", cam.Newton_prop_hip.mean(), "0.561"),
    ("Camargo tm residual", cam.newton_check.mean(), "6.6"),
    ("Camargo lg stance", lg.SoleID_stance.mean(), "4.46"),
    ("Camargo lg hip", lg.SoleID_hip.mean(), "0.373"),
    ("Camargo lg residual", lg.newton_check.mean(), "9.2"),
    ("Camargo ramp stance", ramp.SoleID_stance.mean(), "5.15"),
    ("Camargo ramp hip", ramp.SoleID_hip.mean(), "0.412"),
    ("Camargo ramp residual", ramp.newton_check.mean(), "10.3"),
    ("ramp win vs no-prior", 100 * (ramp.SoleID_stance < ramp.SoleID_noVPP_stance).mean(), "99.3"),
    ("Camargo strides", len(cam), "19378"),
    ("Camargo lg steps", len(lg), "3240"),
    ("Camargo ramp steps", len(ramp), "13717"),

    ("Wang SoleID stance", w.SoleID_stance.mean(), "3.44"),
    ("Wang SoleID hip", w.SoleID_hip.mean(), "0.430"),
    ("Wang lab-kin hip", w.SoleID_lab_kin_hip.mean(), "0.281"),
    ("Wang lab-kin stance", w.SoleID_lab_kin_stance.mean(), "2.97"),
    ("Wang NoShear hip", w.NoShear_hip.mean(), "0.609"),
    ("Wang NoShear stance", w.NoShear_stance.mean(), "7.13"),
    ("Wang Newton hip", w.Newton_prop_hip.mean(), "0.595"),
    # the published pipeline's own figures are now the harmonized ones of reference_agreement.json
    # (checked below); the old wang_crosscheck values 0.170 and 0.22 only matched by coincidence
    ("Wang slow NoShear", w[w.speed == 0.25].NoShear_stance.mean(), "2.27"),
    ("Wang slow SoleID", w[w.speed == 0.25].SoleID_stance.mean(), "2.00"),
    ("Wang fast NoShear", w[w.speed == 1.5].NoShear_stance.mean(), "12.26"),
    ("Wang fast SoleID", w[w.speed == 1.5].SoleID_stance.mean(), "5.08"),

    ("deg: wearable SoleID", deg("wearable", 1, "SoleID"), "3.97"),
    ("deg: wearable Newton", deg("wearable", 1, "Newton_prop"), "7.55"),
    ("deg: wearable SoleID hip", deg("wearable", 1, "SoleID", "hip"), "0.303"),
    ("deg: wearable Newton hip", deg("wearable", 1, "Newton_prop", "hip"), "0.563"),
    ("deg: noise20 SoleID", deg("marker_noise_mm", 20, "SoleID"), "6.07"),
    ("deg: noise20 Newton", deg("marker_noise_mm", 20, "Newton_prop"), "12.30"),
    ("spec: noise10 shear", deg("marker_noise_mm", 10, "SoleID"), "4.13"),
    ("spec: noise10 hip", deg("marker_noise_mm", 10, "SoleID", "hip"), "0.261"),
    ("spec: sync2 shear", deg("sync_frames", 2, "SoleID"), "3.16"),
    ("spec: sync2 hip", deg("sync_frames", 2, "SoleID", "hip"), "0.291"),
    ("spec: scale12 shear", deg("fy_scale_pct", 12, "SoleID"), "3.43"),
    ("spec: cop15 shear", deg("cop_shift_mm", 15, "SoleID"), "3.36"),

    ("MHE vs batch (0.2/50)", mf[(mf.horizon_s == 0.2) & (mf.lookahead_s == 0.05)].vs_batch_pctBW.mean(), "0.022"),
    ("MHE no look-ahead", mf[mf.lookahead_s == 0.0].vs_batch_pctBW.mean(), "0.70"),
    # Table X is the look-ahead 50 ms row, so filter on it rather than averaging every sweep
    ("solve time H=0.2 mean", mf[(mf.horizon_s == 0.2) & (mf.lookahead_s == 0.05)].solve_ms_mean.mean(), "0.044"),
    ("solve time H=0.2 p95", mf[(mf.horizon_s == 0.2) & (mf.lookahead_s == 0.05)].solve_ms_p95.mean(), "0.058"),
    ("MHE Wang vs batch", mw[(mw.horizon_s == 0.2) & (mw.lookahead_s == 0.05)].vs_batch_pctBW.mean(), "0.0159"),
    # the sample the moving-horizon sweep ran on, and its TOST (IV-G)
    ("MHE trials, Fukuchi", len(mf[(mf.horizon_s == 0.2) & (mf.lookahead_s == 0.05)]), "24"),
    ("MHE subjects, Fukuchi", mf.subject.nunique(), "8"),
    ("MHE trials, Wang", len(mw[(mw.horizon_s == 0.2) & (mw.lookahead_s == 0.05)]), "18"),
    ("MHE subjects, Wang", mw.subject.nunique(), "6"),
    ("MHE TOST, largest |90% CI bound|", max(abs(st[k][b]) for k in ("tost_mhe_fukuchi", "tost_mhe_wang")
                                             for b in ("ci_lo", "ci_hi")), "0.01"),
    ("latency pilot trials", len(_cc[_cc.look_s.isna()]), "12"),
    ("solve time H=0.3 mean", mf[(mf.horizon_s == 0.3) & (mf.lookahead_s == 0.05)].solve_ms_mean.mean(), "0.065"),
    ("solve time H=0.3 p95", mf[(mf.horizon_s == 0.3) & (mf.lookahead_s == 0.05)].solve_ms_p95.mean(), "0.082"),
    ("solve time H=0.4 mean", mf[(mf.horizon_s == 0.4) & (mf.lookahead_s == 0.05)].solve_ms_mean.mean(), "0.085"),
    ("solve time H=0.4 p95", mf[(mf.horizon_s == 0.4) & (mf.lookahead_s == 0.05)].solve_ms_p95.mean(), "0.102"),
    ("solve time H=0.6 mean", mf[(mf.horizon_s == 0.6) & (mf.lookahead_s == 0.05)].solve_ms_mean.mean(), "0.130"),
    ("solve time H=0.6 p95", mf[(mf.horizon_s == 0.6) & (mf.lookahead_s == 0.05)].solve_ms_p95.mean(), "0.155"),

    ("SPM SoleID % of cycle", st["spm"]["SoleID_shear"]["percent_of_cycle"], "41"),
    ("SPM vertical-only %", st["spm"]["NoShear_shear"]["percent_of_cycle"], "58"),
    ("SPM shear mean |diff|", st["spm_effect_size"]["shear_mean_abs"], "1.00"),
    ("SPM shear max |diff|", st["spm_effect_size"]["shear_max_abs"], "3.75"),
    ("SPM hip mean |diff|", st["spm_effect_size"]["hip_mean_abs"], "0.080"),
    ("SPM hip max |diff|", st["spm_effect_size"]["hip_max_abs"], "0.311"),

    # --- calibration-split sensitivity (Sec. III.D), from split_sensitivity.csv
    ("split: mean test stance", sp.test_stance.mean(), "3.17"),
    ("split: sd test stance", sp.test_stance.std(), "0.10"),
    ("split: min test stance", sp.test_stance.min(), "2.94"),
    ("split: max test stance", sp.test_stance.max(), "3.54"),
    ("split: mean test hip", sp.test_hip.mean(), "0.2024"),
    ("split: sd test hip", sp.test_hip.std(), "0.0067"),
    ("split: h=0.3 share (%)", 100 * (sp.h_vpp == 0.3).mean(), "70"),
    ("split: fc=12 share (%)", 100 * (sp.fc_smooth == 12.0).mean(), "97"),
    ("split: w_prior=1 share (%)", 100 * (sp.w_prior == 1.0).mean(), "55"),

    # --- phase-dependent pivot sensitivity (Sec. V.E), from phase_vpp_test.csv
    ("phase: stance gain (%)", 100 * (1 - pv_phase.stance.mean() / pv_frozen.stance.mean()), "6.2"),
    ("phase: hip gain (%)", 100 * (1 - pv_phase.hip.mean() / pv_frozen.hip.mean()), "6.2"),
    ("phase: bias reduction (%)",
     100 * (1 - pv_phase.hip_bias_stance.mean() / pv_frozen.hip_bias_stance.mean()), "13"),
    ("phase: grid size", _pv_grid_size, "82"),

    # --- reporting convention (Sec. III-E): shear over stance, and what that is over the cycle.
    ("stance RMSE expressed over the cycle", t.SoleID_stance.mean() * np.sqrt(
        nc["fukuchi"]["_meta"]["stance_fraction"]), "2.43"),

    # --- split uncertainty carried to the external layer (Sec. III.D)
    ("prop: weighted mean stance", _pw("stance")[0], "3.80"),
    ("prop: sd stance", _pw("stance")[1], "0.27"),
    ("prop: min stance", _pw("stance")[2], "3.21"),
    ("prop: max stance", _pw("stance")[3], "4.49"),
    ("prop: weighted mean hip", _pw("hip")[0], "0.2486"),
    ("prop: sd hip", _pw("hip")[1], "0.0179"),
    ("prop: min hip", _pw("hip")[2], "0.210"),
    ("prop: max hip", _pw("hip")[3], "0.293"),
    ("prop: frozen stance", _pw("stance")[4], "3.59"),
    ("prop: frozen hip", _pw("hip")[4], "0.2348"),
    ("prop: frozen percentile", _pw("stance")[5], "74"),
    ("prop: n configurations", 13, "13"),
    ("prop: top-five share (%)", 100 * _PW.weight.nlargest(5).sum(), "87"),

    # --- the same propagation on the application layer (Table in Sec. III.D)
    ("wang prop: mean stance", _ww("SoleID_stance")[0], "3.22"),
    ("wang prop: sd stance", _ww("SoleID_stance")[1], "0.10"),
    ("wang prop: min stance", _ww("SoleID_stance")[2], "2.94"),
    ("wang prop: max stance", _ww("SoleID_stance")[3], "3.49"),
    ("wang prop: frozen stance", _ww("SoleID_stance")[4], "3.44"),
    ("wang prop: frozen pct", _ww("SoleID_stance")[5], "6"),
    ("wang prop: mean hip", _ww("SoleID_hip")[0], "0.4170"),
    ("wang prop: sd hip", _ww("SoleID_hip")[1], "0.0196"),
    ("wang prop: frozen hip", _ww("SoleID_hip")[4], "0.4287"),
    ("wang prop: frozen hip pct", _ww("SoleID_hip")[5], "18"),

    # --- how often the friction cone binds (Sec. II.C)
    ("cone: frames on the bound (%)", 100 * _fc["frac_frames_active"], "0.006"),
    ("cone: mean peak ratio", _fc["peak_ratio_mean"], "0.41"),
    ("cone: largest ratio", _fc["peak_ratio_max"], "0.80"),

    # --- the insole reads load normal to the sole, taken as vertical (Sec. II-B), sole_normal.py
    ("sole pitch, force-weighted (deg)", _sn["pitch_deg_force_weighted"], "11.5"),
    ("sole-normal reading vs vertical", _sn["vertical_error_pctBW"], "1.56"),

    # --- the pivot-point split arm (Sec. III-C, Table III, Sec. IV-A)
    ("split arm: shear stance", _vs.stance.mean(), "4.18"),
    ("split arm: shear ds", _vs.ds.mean(), "5.14"),
    ("split arm: vertical stance", _vs.vert_stance.mean(), "16.69"),
    ("split arm: ankle", _vs.ankle.mean(), "0.284"),
    ("split arm: knee", _vs.knee.mean(), "0.220"),
    ("split arm: hip", _vs.hip.mean(), "0.308"),
    ("split arm: SoleID better, shear (%)",
     100 * (_vj.SoleID_stance < _vj.stance).mean(), "85.7"),
    # bounded-look-ahead preprocessing (soleid/causal_chain.py)
    ("causal chain: whole record", _cc_batch, "3.06"),
    ("causal chain: 0 ms", _look(0.0), "3.70"),
    ("causal chain: 0 ms vs batch", _look(0.0, "vs_batch"), "1.30"),
    ("causal chain: 50 ms", _look(0.05), "2.97"),
    ("causal chain: 50 ms vs batch", _look(0.05, "vs_batch"), "0.28"),
    ("causal chain: 150 ms", _look(0.15), "3.05"),
    ("causal chain: 150 ms vs batch", _look(0.15, "vs_batch"), "0.09"),
    ("causal chain: 300 ms", _look(0.30), "3.06"),
    ("causal chain: 300 ms vs batch", _look(0.30, "vs_batch"), "0.02"),
    ("causal chain: cost of being causal", _look(0.0) - _cc_batch, "0.64"),
    # which element the look-ahead is spent on (soleid/causal_design.py)
    ("causal cost: marker filter", _pen("marker"), "0.32"),
    ("causal cost: force filter", _pen("force"), "0.38"),
    # (holding the center of pressure, -0.001, left the text when IV-G was simplified on 25 Sep)
    # designed smoother vs truncated filter at 50 ms (soleid/causal_variants.py)
    ("designed smoother, 50 ms", _var("state"), "3.3"),
    ("truncated filter, 50 ms", _var("truncated"), "3.1"),
    # A4: hip bias share, windows matched (soleid/hip_bias_decomp.py)
    ("hip bias, stance", _hb["stance"]["bias_mean"], "0.129"),
    ("hip bias, cycle", _hb["cycle"]["bias_mean"], "0.081"),
    ("hip bias, stance (V-B)", _hb["stance"]["bias_mean"], "0.13"),
    ("hip bias same sign, stance", _same_sign("hip_bias_stance"), "204"),
    ("hip bias same sign, cycle", _same_sign("hip_bias_cycle"), "208"),
    ("hip bias share, cycle %", _hb["cycle"]["per_trial"], "24"),
    ("hip bias share, stance %", _hb["stance"]["per_trial"], "37"),
    # A2: the pooled comparison with the learned models now lives in the last row of Table VI,
    # which comparison_tables.py builds from the same source; the three-joint averages left the text
    # A5: the factorial (soleid/wang_factorial.py, wang_factorial_stats.py)
    ("factorial imu/insole hip", _wf.imu_insole_hip.mean(), "0.430"),
    ("factorial imu/plate hip", _wf.imu_plate_hip.mean(), "0.336"),
    ("factorial lab/insole hip", _wf.lab_insole_hip.mean(), "0.281"),
    ("factorial lab/plate hip", _wf.lab_plate_hip.mean(), "0.170"),
    ("factorial imu/insole knee", _wf.imu_insole_knee.mean(), "0.229"),
    ("factorial imu/plate knee", _wf.imu_plate_knee.mean(), "0.178"),
    ("factorial lab/insole knee", _wf.lab_insole_knee.mean(), "0.201"),
    ("factorial lab/plate knee", _wf.lab_plate_knee.mean(), "0.095"),
    ("factorial imu/insole ankle", _wf.imu_insole_ankle.mean(), "0.219"),
    ("factorial imu/plate ankle", _wf.imu_plate_ankle.mean(), "0.103"),
    ("factorial lab/insole ankle", _wf.lab_insole_ankle.mean(), "0.173"),
    ("factorial lab/plate ankle", _wf.lab_plate_ankle.mean(), "0.047"),
    ("factorial imu/insole shear", _wf.imu_insole_stance.mean(), "3.45"),
    ("factorial imu/plate shear", _wf.imu_plate_stance.mean(), "2.83"),
    ("factorial lab/insole shear", _wf.lab_insole_stance.mean(), "2.99"),
    ("factorial lab/plate shear", _wf.lab_plate_stance.mean(), "2.48"),
    ("kin cost hip", _wf.imu_insole_hip.mean() - _wf.lab_insole_hip.mean(), "0.148"),
    ("kin cost knee", _wf.imu_insole_knee.mean() - _wf.lab_insole_knee.mean(), "0.028"),
    ("kin cost ankle", _wf.imu_insole_ankle.mean() - _wf.lab_insole_ankle.mean(), "0.046"),
    ("kin cost shear", _wf.imu_insole_stance.mean() - _wf.lab_insole_stance.mean(), "0.46"),
    # single support and the ingredient ablation
    ("single support, no prior", t.SoleID_v1_ss.mean(), "3.32"),
    ("single support, SoleID", t.SoleID_ss.mean(), "2.52"),
    ("ablation: prior alone, stance", _ab["prior"]["stance"], "4.29"),
    ("ablation: prior smoothed, stance", _ab["prior_smooth"]["stance"], "4.07"),
    ("ablation: blend, stance", _ab["blend"]["stance"], "3.26"),
    ("ablation: full, stance", _ab["full"]["stance"], "3.13"),
    ("ablation: no cone = full, stance", _ab["no_cone"]["stance"], "3.13"),
    ("ablation: prior alone, hip", _ab["prior"]["hip"], "0.270"),
    ("ablation: prior smoothed, hip", _ab["prior_smooth"]["hip"], "0.257"),
    ("ablation: blend, hip", _ab["blend"]["hip"], "0.207"),
    # quality control
    ("QC subjects remaining", _qc["subjects_kept"], "36"),
    ("QC subjects total", _qc["subjects_total"], "51"),
    ("QC handrail runs", _qc["excluded_runs"]["handrail"]["n"], "25"),
    ("QC handrail stance", _qc["excluded_runs"]["handrail"]["stance"], "4.65"),
    ("QC scale runs", _qc["excluded_runs"]["force scale"]["n"], "23"),
    ("QC scale stance", _qc["excluded_runs"]["force scale"]["stance"], "4.24"),
    ("QC all kept, stance", _qc["test_plus_excluded"]["stance"], "3.36"),
    ("QC all kept, hip", _qc["test_plus_excluded"]["hip"], "0.209"),
    ("QC band 0.60-1.80 keeps", _qc["band_sensitivity"]["0.60-1.80"], "277"),
    ("QC band 0.85-1.10 keeps", _qc["band_sensitivity"]["0.85-1.10"], "276"),
    ("bw ratio subject min", _per.mean().min(), "0.78"),
    ("bw ratio subject max", _per.mean().max(), "0.97"),
    ("bw ratio sd within", float(np.sqrt(_per.var().mean())), "0.006"),
    ("bw ratio sd between", float(_per.mean().std()), "0.030"),
    ("bw ratio normalization %", 100 * (1 / _qr.bw_ratio.median() - 1), "7"),
    # statistics
    ("speed interaction", _sr["arm_by_speed"]["interaction"], "2.90"),
    ("speed interaction lo", _sr["arm_by_speed"]["lo"], "2.64"),
    ("speed interaction hi", _sr["arm_by_speed"]["hi"], "3.16"),
    ("ratio by speed", _sr["ratio_by_speed"]["slope"], "0.45"),
    ("ratio by speed lo", _sr["ratio_by_speed"]["lo"], "0.35"),
    ("ratio by speed hi", _sr["ratio_by_speed"]["hi"], "0.55"),
    ("age effect", _sr["age"]["estimate"], "0.27"),
    ("age effect lo", -_sr["age"]["lo95"], "0.28"),
    ("age effect hi", _sr["age"]["hi95"], "0.82"),
    ("older subjects", _sr["age"]["n_subjects"]["Older"], "11"),
    ("bootstrap over subjects lo", _sr["bootstrap_subjects"]["SoleID_stance"]["lo"], "2.91"),
    ("bootstrap over subjects hi", _sr["bootstrap_subjects"]["SoleID_stance"]["hi"], "3.37"),
    ("SPM shear, published %", _sr["spm_bonferroni"]["results"]["SoleID_shear"]["published_percent"], "41"),
    ("SPM shear, Bonferroni %", _sr["spm_bonferroni"]["results"]["SoleID_shear"]["percent_of_cycle"], "28"),
    ("Newton - SoleID gap", st["mixed_stance"]["contrasts"]["Newton_prop"]["estimate"], "3.2"),
    ("trial-weighted cam - fukuchi", cam.SoleID_stance.mean() - t.SoleID_stance.mean(), "0.47"),
    ("half slope, trimmed", _mt.params["newton_check"], "0.565"),
    ("half slope, trimmed lo", _mt.conf_int().loc["newton_check", 0], "0.497"),
    ("half slope, trimmed hi", _mt.conf_int().loc["newton_check", 1], "0.634"),
    ("half slope, subject means", np.polyfit(_gs.newton_check, _gs.gain, 1)[0], "0.38"),
    ("half slope, subject r", np.corrcoef(_gs.newton_check, _gs.gain)[0, 1], "0.72"),
    # degradation interactions
    ("combined increment, stance", (_dgw["stance"] - _dgc["stance"]).mean(), "0.84"),
    ("sum of increments, stance", _deg_sum("stance"), "1.30"),
    ("combined increment, hip", (_dgw["hip"] - _dgc["hip"]).mean(), "0.104"),
    ("sum of increments, hip", _deg_sum("hip"), "0.141"),
    # hip bias breakdown
    ("hip bias R stance", _hb2["limb"]["R"]["stance"], "0.13"),
    ("hip bias L stance", _hb2["limb"]["L"]["stance"], "0.15"),
    ("hip bias subjects positive", _hb2["subject"]["positive"], "26"),
    ("hip bias slowest bin", _hb2["speed"]["<0.8"], "0.12"),
    ("hip bias fastest bin", _hb2["speed"][">1.6"], "0.15"),
    ("hip bias early R", _hb2["phase"]["early"]["R"], "0.22"),
    ("hip bias early L", _hb2["phase"]["early"]["L"], "0.25"),
    ("hip bias later min", min(_hb2["phase"][p][s_] for p in ("mid", "late") for s_ in ("R", "L")), "0.06"),
    ("hip bias later max", max(_hb2["phase"][p][s_] for p in ("mid", "late") for s_ in ("R", "L")), "0.12"),
    # the reference itself, and the Wang comparison against either reference
    ("Wang wearable vs planar, ankle", _rwang["wangwear_vs_planar"]["ankle"]["rmse"], "0.108"),
    ("Wang wearable vs planar, knee", _rwang["wangwear_vs_planar"]["knee"]["rmse"], "0.270"),
    ("Wang wearable vs planar, hip", _rwang["wangwear_vs_planar"]["hip"]["rmse"], "0.604"),
    ("Wang wearable vs planar, knee r", _rwang["wangwear_vs_planar"]["knee"]["r"], "0.44"),
    ("Wang wearable vs lab, hip", _rwang["wangwear_vs_lab"]["hip"]["rmse"], "0.603"),
    ("Wang wearable vs lab, ankle", _rwang["wangwear_vs_lab"]["ankle"]["rmse"], "0.141"),
    ("SoleID vs Wang lab, hip", _rwang["soleid_vs_lab"]["hip"]["rmse"], "0.420"),
    ("SoleID vs Wang lab, ankle", _rwang["soleid_vs_lab"]["ankle"]["rmse"], "0.145"),
    ("SoleID vs planar (Wang), hip", _rwang["soleid_vs_planar"]["hip"]["rmse"], "0.430"),
    ("hip improvement, planar %", 100 * (1 - _rwang["soleid_vs_planar"]["hip"]["rmse"]
                                          / _rwang["wangwear_vs_planar"]["hip"]["rmse"]), "29"),
    ("hip improvement, lab %", 100 * (1 - _rwang["soleid_vs_lab"]["hip"]["rmse"]
                                       / _rwang["wangwear_vs_lab"]["hip"]["rmse"]), "30"),
    ("refs: Fukuchi hip", _ra["fukuchi"]["planar_vs_lab"]["hip"]["rmse"], "0.234"),
    ("refs: Camargo hip", _ra["camargo"]["planar_vs_lab"]["hip"]["rmse"], "0.143"),
    ("refs: Wang hip", _ra["wang"]["planar_vs_lab"]["hip"]["rmse"], "0.199"),
    # hip correlations between the references, as the footnote of Table X gives them
    ("refs: Fukuchi hip r", _ra["fukuchi"]["planar_vs_lab"]["hip"]["r"], "0.804"),
    ("refs: Camargo hip r", _ra["camargo"]["planar_vs_lab"]["hip"]["r"], "0.946"),
    ("refs: Wang hip r", _ra["wang"]["planar_vs_lab"]["hip"]["r"], "0.835"),
    ("hip bias by data set: Fukuchi", _ra["fukuchi"]["soleid_vs_planar"]["hip"]["bias"], "0.08"),
    ("hip bias by data set: Camargo", _ra["camargo"]["soleid_vs_planar"]["hip"]["bias"], "0.04"),
    ("hip bias by data set: Wang", _ra["wang"]["soleid_vs_planar"]["hip"]["bias"], "0.19"),
    # second round: paired contrasts as the primary result (stance contrasts as Table IV rounds them)
    ("paired stance, kin-only", _srp["stance"]["Kin_only"]["estimate"], "3.30"),
    ("paired stance, kin-only lo", _srp["stance"]["Kin_only"]["lo"], "2.90"),
    ("paired stance, kin-only hi", _srp["stance"]["Kin_only"]["hi"], "3.69"),
    ("paired stance, Newton", _srp["stance"]["Newton_prop"]["estimate"], "3.26"),
    ("paired stance, Newton lo", _srp["stance"]["Newton_prop"]["lo"], "2.87"),
    ("paired stance, Newton hi", _srp["stance"]["Newton_prop"]["hi"], "3.65"),
    ("paired stance, no prior", _srp["stance"]["SoleID_v1"]["estimate"], "1.39"),
    ("paired stance, no prior lo", _srp["stance"]["SoleID_v1"]["lo"], "1.03"),
    ("paired stance, no prior hi", _srp["stance"]["SoleID_v1"]["hi"], "1.74"),
    ("paired ds, Newton", _srp["ds"]["Newton_prop"]["estimate"], "5.786"),
    ("paired ds, Newton lo", _srp["ds"]["Newton_prop"]["lo"], "5.184"),
    ("paired hip, kin-only (table)", _srp["hip"]["Kin_only"]["estimate"], "0.600"),
    ("paired hip, Newton (table)", _srp["hip"]["Newton_prop"]["estimate"], "0.200"),
    ("paired hip, no prior (table)", _srp["hip"]["SoleID_v1"]["estimate"], "0.085"),
    ("kin-only vertical, stance", _px["kin_only_vertical_stance"]["mean"], "25.15"),
    ("Wang kin contrast (paired)", -_px["wang_kin_contrast"]["hip"]["estimate"], "0.148"),
    ("Wang kin contrast lo", -_px["wang_kin_contrast"]["hip"]["hi"], "0.060"),
    ("Wang kin contrast hi", -_px["wang_kin_contrast"]["hip"]["lo"], "0.235"),
    ("square: insole@IMU ankle", _sq["ankle"]["insole_with_imu"]["estimate"], "0.116"),
    ("square: insole@IMU ankle lo", _sq["ankle"]["insole_with_imu"]["lo"], "0.087"),
    ("square: insole@IMU ankle hi", _sq["ankle"]["insole_with_imu"]["hi"], "0.146"),
    ("square: kin@insole hip", _sq["hip"]["kin_with_insole"]["estimate"], "0.148"),
    ("square: kin@insole hip lo", _sq["hip"]["kin_with_insole"]["lo"], "0.060"),
    ("square: kin@insole hip hi", _sq["hip"]["kin_with_insole"]["hi"], "0.235"),
    ("square: hip difference p", _sq["hip"]["kin_minus_insole"]["p"], "0.31"),
    ("square: ankle difference p", _sq["ankle"]["kin_minus_insole"]["p"], "0.14"),
    ("square: knee interaction", -_sq["knee"]["interaction"]["estimate"], "0.055"),
    ("square: knee interaction lo", -_sq["knee"]["interaction"]["hi"], "0.010"),
    ("square: knee interaction hi", -_sq["knee"]["interaction"]["lo"], "0.100"),
    ("square row: kin@plate hip", _sqd("imu_plate", "lab_plate", "hip"), "0.165"),
    ("square row: kin@plate knee", _sqd("imu_plate", "lab_plate", "knee"), "0.083"),
    ("square row: kin@plate ankle", _sqd("imu_plate", "lab_plate", "ankle"), "0.056"),
    ("square row: insole@lab hip", _sqd("lab_insole", "lab_plate", "hip"), "0.111"),
    ("square row: insole@lab knee", _sqd("lab_insole", "lab_plate", "knee"), "0.106"),
    ("square row: insole@lab ankle", _sqd("lab_insole", "lab_plate", "ankle"), "0.126"),
    ("square row: insole@IMU hip", _sqd("imu_insole", "imu_plate", "hip"), "0.094"),
    ("square row: insole@IMU shear", _sqd("imu_insole", "imu_plate", "stance"), "0.62"),
]

# ------------------------------------------------------------------------------------------------
# Remaining table cells and values of the text. Values that come from the data are checked here;
# literature values, method parameters and data-set design facts (speeds, sampling rates) are not.
from protocol import CAL_SUBJECTS as _CAL
_fr = json.load(open(os.path.join(R, "final_read_checks.json")))      # final_read_checks.py
_ang = json.load(open(os.path.join(R, "ramp_angles.json")))["angles_deg"]   # ramp_angles.py
_f2 = json.load(open(os.path.join(R, "fig2_trial.json")))             # make_figures.py
_pvg = pd.read_csv(os.path.join(R, "phase_vpp_grid.csv"))
_pvg = _pvg[(_pvg.h_single == _pvg.h_double) & (_pvg.blend == 0)].set_index("h_single")
_grid = pd.read_csv(os.path.join(R, "grid_all_trials.csv"))
_gsub = _grid.groupby(_K + ["subject"])[["stance", "hip"]].mean().reset_index()
_gfro = _gsub[(_gsub.h_vpp == 0.2) & (_gsub.w_prior == 1.0) & (_gsub.w_torque == 0.0)
              & (_gsub.fc_smooth == 12.0) & ~_gsub.subject.isin(_CAL)]
_ours = _gfro[["stance", "hip"]].mean()
_oracle = _gsub.groupby(_K).stance.mean().min()
_lgtm = lg[lg.plate.str.startswith("Treadmill")]
_agree = _lgtm.plate.str[-1] == _lgtm.side
_mh = lambda df, H, L=0.05: df[(df.horizon_s == H) & (df.lookahead_s == L)].vs_batch_pctBW.mean()
_scale = q.reason.fillna("").str.startswith("force/mass")

CHECKS += [
    # Table III
    ("T3 kinematics-only, double support", t.Kin_only_ds.mean(), "9.80"),   # the table said 9.81
    ("T3 no prior, double support", t.SoleID_v1_ds.mean(), "6.03"),
    ("T3 kinematics-only, single support", t.Kin_only_ss.mean(), "3.51"),
    ("T3 Newton, single support", t.Newton_prop_ss.mean(), "3.52"),
    ("T3 kinematics-only ankle", t.Kin_only_ankle.mean(), "0.445"),
    ("T3 Newton ankle", t.Newton_prop_ankle.mean(), "0.047"),
    ("T3 no prior ankle", t.SoleID_v1_ankle.mean(), "0.032"),
    ("T3 kinematics-only knee", t.Kin_only_knee.mean(), "0.582"),
    ("T3 Newton knee", t.Newton_prop_knee.mean(), "0.221"),
    ("T3 no prior knee", t.SoleID_v1_knee.mean(), "0.155"),
    ("T3 hip contrast, kin-only lo", _srp["hip"]["Kin_only"]["lo"], "0.557"),
    ("T3 hip contrast, kin-only hi", _srp["hip"]["Kin_only"]["hi"], "0.643"),
    ("T3 hip contrast, Newton lo", _srp["hip"]["Newton_prop"]["lo"], "0.176"),
    ("T3 hip contrast, Newton hi", _srp["hip"]["Newton_prop"]["hi"], "0.225"),
    ("T3 hip contrast, no prior lo", _srp["hip"]["SoleID_v1"]["lo"], "0.063"),
    ("T3 hip contrast, no prior hi", _srp["hip"]["SoleID_v1"]["hi"], "0.107"),
    # IV-A
    ("bootstrap over subjects, hip lo", _sr["bootstrap_subjects"]["SoleID_hip"]["lo"], "0.185"),
    ("bootstrap over subjects, hip hi", _sr["bootstrap_subjects"]["SoleID_hip"]["hi"], "0.215"),
    ("paired ds, Newton hi", _srp["ds"]["Newton_prop"]["hi"], "6.388"),
    ("error slope on speed", st["mixed_stance"]["speed_slope"]["estimate"], "3.466"),
    ("pivot split vertical error", _vs.vert_stance.mean(), "16.7"),
    ("pivot split vertical, % of stride range", _fr["vertical_error_of_range_pct"], "14.5"),
    ("SPM shear, largest difference at % cycle", _fr["spm_max_at_pct"], "2"),
    ("Fig. 2 trial speed", _f2["speed"], "1.12"),
    # IV-B
    ("gain slope lo", st["gain_vs_newton_check"]["lo"], "0.478"),
    ("calibration: prior bias (N)", _pvg.loc[0.2, "prior_bias_N"], "-19.6"),
    ("calibration: right-foot shear bias (N)", _pvg.loc[0.2, "sol_bias_N"], "-10.1"),
    ("calibration: hip offset, stance", _pvg.loc[0.2, "hip_bias_stance"], "0.13"),
    ("calibration: shear bias at h = 0.4 (N)", _pvg.loc[0.4, "sol_bias_N"], "-8.6"),
    ("Wang dynamics residual", w.newton_check_pctBW.mean(), "3.9"),
    # IV-C and Table IV
    ("combined increment, 95% lo", _fr["combined_increment_ci"][0], "0.76"),
    ("combined increment, 95% hi", _fr["combined_increment_ci"][1], "0.92"),
    ("hip criterion", t.SoleID_hip.mean() + 0.10, "0.299"),
    ("spec: bias 30 mm shear", deg("kin_bias_mm", 30, "SoleID"), "3.16"),
    ("spec: bias 30 mm hip", deg("kin_bias_mm", 30, "SoleID", "hip"), "0.244"),
    ("spec: 30 Hz shear", deg("marker_fps", 30, "SoleID"), "3.15"),
    ("spec: 30 Hz hip", deg("marker_fps", 30, "SoleID", "hip"), "0.195"),
    ("spec: scale 12% hip", deg("fy_scale_pct", 12, "SoleID", "hip"), "0.203"),
    ("spec: force noise 5 shear", deg("fy_noise_pctBW", 5, "SoleID"), "3.14"),
    ("spec: force noise 5 hip", deg("fy_noise_pctBW", 5, "SoleID", "hip"), "0.208"),
    ("spec: CoP 15 mm hip", deg("cop_shift_mm", 15, "SoleID", "hip"), "0.179"),
    # IV-D, Table VI and the ramp inclinations
    ("TOST generalisation", st["tost_generalisation"]["mean"], "0.380"),
    ("TOST generalisation lo", st["tost_generalisation"]["ci_lo"], "0.094"),
    ("TOST generalisation hi", st["tost_generalisation"]["ci_hi"], "0.667"),
    ("level ground, slow", lg[lg.speed_class == "slow"].SoleID_stance.mean(), "3.62"),
    ("level ground, normal", lg[lg.speed_class == "normal"].SoleID_stance.mean(), "4.62"),
    ("level ground, fast", lg[lg.speed_class == "fast"].SoleID_stance.mean(), "5.16"),
    ("ramp vertical-only, shallowest", ramp[ramp.incline == 1].NoShear_stance.mean(), "12.2"),
    ("ramp vertical-only, steepest", ramp[ramp.incline == 6].NoShear_stance.mean(), "10.4"),
    ("ramp inclination 1", _ang["1"], "5.3"),
    ("ramp inclination 2", _ang["2"], "7.4"),
    ("ramp inclination 3", _ang["3"], "8.9"),
    ("ramp inclination 4", _ang["4"], "10.7"),
    ("ramp inclination 5", _ang["5"], "12.4"),
    ("ramp inclination 6", _ang["6"], "17.9"),
    ("T6 treadmill kin-only shear", cam.Kin_only_stance.mean(), "10.06"),
    ("T6 treadmill no prior shear", cam.SoleID_noVPP_stance.mean(), "6.25"),
    ("T6 treadmill kin-only hip", cam.Kin_only_hip.mean(), "0.938"),
    ("T6 treadmill no prior hip", cam.SoleID_noVPP_hip.mean(), "0.409"),
    ("T6 level vertical-only shear", lg.NoShear_stance.mean(), "10.92"),
    ("T6 level Newton shear", lg.Newton_prop_stance.mean(), "9.14"),
    ("T6 level no prior shear", lg.SoleID_noVPP_stance.mean(), "7.72"),
    ("T6 level vertical-only hip", lg.NoShear_hip.mean(), "0.909"),
    ("T6 level Newton hip", lg.Newton_prop_hip.mean(), "0.763"),
    ("T6 level no prior hip", lg.SoleID_noVPP_hip.mean(), "0.646"),
    ("T6 ramp vertical-only shear", ramp.NoShear_stance.mean(), "11.73"),
    ("T6 ramp Newton shear", ramp.Newton_prop_stance.mean(), "10.63"),
    ("T6 ramp no prior shear", ramp.SoleID_noVPP_stance.mean(), "8.93"),
    ("T6 ramp vertical-only hip", ramp.NoShear_hip.mean(), "0.938"),
    ("T6 ramp Newton hip", ramp.Newton_prop_hip.mean(), "0.850"),
    ("T6 ramp no prior hip", ramp.SoleID_noVPP_hip.mean(), "0.715"),
    # IV-E, Tables VII and VIII
    ("T7 SoleID ankle", w.SoleID_ankle.mean(), "0.219"),
    ("T7 SoleID knee", w.SoleID_knee.mean(), "0.229"),
    ("T7 lab kin ankle", w.SoleID_lab_kin_ankle.mean(), "0.173"),
    ("T7 lab kin knee", w.SoleID_lab_kin_knee.mean(), "0.201"),
    ("T7 vertical-only ankle", w.NoShear_ankle.mean(), "0.245"),
    ("T7 Newton ankle", w.Newton_prop_ankle.mean(), "0.231"),
    ("T7 vertical-only knee", w.NoShear_knee.mean(), "0.340"),
    ("T7 Newton knee", w.Newton_prop_knee.mean(), "0.322"),
    ("T7 Newton shear", w.Newton_prop_stance.mean(), "5.33"),
    ("T7 Wang ankle r", _rwang["wangwear_vs_planar"]["ankle"]["r"], "0.98"),
    ("T7 Wang hip r", _rwang["wangwear_vs_planar"]["hip"]["r"], "0.03"),
    ("T7 vertical-only ankle r", w.NoShear_ankle_r.mean(), "0.94"),
    ("T7 vertical-only knee r", w.NoShear_knee_r.mean(), "0.32"),
    ("T7 vertical-only hip r", w.NoShear_hip_r.mean(), "0.73"),
    ("T7 Newton ankle r", w.Newton_prop_ankle_r.mean(), "0.95"),
    ("T7 Newton knee r", w.Newton_prop_knee_r.mean(), "0.59"),
    ("T7 Newton hip r", w.Newton_prop_hip_r.mean(), "0.56"),
    ("T7 SoleID ankle r", w.SoleID_ankle_r.mean(), "0.95"),
    ("T7 SoleID knee r", w.SoleID_knee_r.mean(), "0.71"),
    ("T7 SoleID hip r", w.SoleID_hip_r.mean(), "0.38"),
    ("T7 lab kin ankle r", w.SoleID_lab_kin_ankle_r.mean(), "0.97"),
    ("T7 lab kin knee r", w.SoleID_lab_kin_knee_r.mean(), "0.81"),
    ("T7 lab kin hip r", w.SoleID_lab_kin_hip_r.mean(), "0.68"),
    ("T8 kinematics with plate, shear", _sqd("imu_plate", "lab_plate", "stance"), "0.36"),
    ("T8 insole with IMU, knee", _sqd("imu_insole", "imu_plate", "knee"), "0.051"),
    ("T8 insole with lab, shear", _sqd("lab_insole", "lab_plate", "stance"), "0.51"),
    ("factorial trials", len(_wf), "53"),
    # IV-F, Table IX
    ("T9 Fukuchi planar vs lab ankle", _ra["fukuchi"]["planar_vs_lab"]["ankle"]["rmse"], "0.218"),
    ("T9 Fukuchi planar vs lab knee", _ra["fukuchi"]["planar_vs_lab"]["knee"]["rmse"], "0.193"),
    ("T9 Fukuchi SoleID vs lab ankle", _ra["fukuchi"]["soleid_vs_lab"]["ankle"]["rmse"], "0.221"),
    ("T9 Fukuchi SoleID vs lab knee", _ra["fukuchi"]["soleid_vs_lab"]["knee"]["rmse"], "0.209"),
    ("T9 Fukuchi SoleID vs lab hip", _ra["fukuchi"]["soleid_vs_lab"]["hip"]["rmse"], "0.298"),
    ("T9 Fukuchi SoleID vs planar ankle", _ra["fukuchi"]["soleid_vs_planar"]["ankle"]["rmse"], "0.019"),
    ("T9 Fukuchi SoleID vs planar knee", _ra["fukuchi"]["soleid_vs_planar"]["knee"]["rmse"], "0.101"),
    ("T9 Fukuchi SoleID vs planar hip", _ra["fukuchi"]["soleid_vs_planar"]["hip"]["rmse"], "0.183"),
    ("T9 Camargo planar vs lab ankle", _ra["camargo"]["planar_vs_lab"]["ankle"]["rmse"], "0.093"),
    ("T9 Camargo planar vs lab knee", _ra["camargo"]["planar_vs_lab"]["knee"]["rmse"], "0.101"),
    ("T9 Camargo SoleID vs lab ankle", _ra["camargo"]["soleid_vs_lab"]["ankle"]["rmse"], "0.092"),
    ("T9 Camargo SoleID vs lab knee", _ra["camargo"]["soleid_vs_lab"]["knee"]["rmse"], "0.153"),
    ("T9 Camargo SoleID vs lab hip", _ra["camargo"]["soleid_vs_lab"]["hip"]["rmse"], "0.243"),
    ("T9 Camargo SoleID vs planar ankle", _ra["camargo"]["soleid_vs_planar"]["ankle"]["rmse"], "0.032"),
    ("T9 Camargo SoleID vs planar knee", _ra["camargo"]["soleid_vs_planar"]["knee"]["rmse"], "0.128"),
    ("T9 Camargo SoleID vs planar hip", _ra["camargo"]["soleid_vs_planar"]["hip"]["rmse"], "0.237"),
    ("T9 Wang planar vs lab ankle", _ra["wang"]["planar_vs_lab"]["ankle"]["rmse"], "0.117"),
    ("T9 Wang planar vs lab knee", _ra["wang"]["planar_vs_lab"]["knee"]["rmse"], "0.114"),
    ("T9 Wang SoleID vs lab knee", _ra["wang"]["soleid_vs_lab"]["knee"]["rmse"], "0.255"),
    ("T9 Wang SoleID vs planar ankle", _ra["wang"]["soleid_vs_planar"]["ankle"]["rmse"], "0.219"),
    ("T9 Wang SoleID vs planar knee", _ra["wang"]["soleid_vs_planar"]["knee"]["rmse"], "0.229"),
    ("T9 Wang wearable vs lab knee", _ra["wang"]["wangwear_vs_lab"]["knee"]["rmse"], "0.261"),
    ("T9 Fukuchi reference r, ankle", _ra["fukuchi"]["planar_vs_lab"]["ankle"]["r"], "0.934"),
    ("T9 Fukuchi reference r, knee", _ra["fukuchi"]["planar_vs_lab"]["knee"]["r"], "0.740"),
    ("T9 Camargo reference r, ankle", _ra["camargo"]["planar_vs_lab"]["ankle"]["r"], "0.996"),
    ("T9 Camargo reference r, knee", _ra["camargo"]["planar_vs_lab"]["knee"]["r"], "0.956"),
    ("T9 Wang reference r, ankle", _ra["wang"]["planar_vs_lab"]["ankle"]["r"], "0.990"),
    ("T9 Wang reference r, knee", _ra["wang"]["planar_vs_lab"]["knee"]["r"], "0.857"),
    # IV-G, Table X
    ("T10 Fukuchi H 0.2", _mh(mf, 0.2), "0.0219"),
    ("T10 Fukuchi H 0.3", _mh(mf, 0.3), "0.0220"),
    ("T10 Fukuchi H 0.4", _mh(mf, 0.4), "0.0220"),
    ("T10 Fukuchi H 0.6", _mh(mf, 0.6), "0.0220"),
    ("T10 Wang H 0.3", _mh(mw, 0.3), "0.0159"),
    ("T10 Wang H 0.4", _mh(mw, 0.4), "0.0159"),
    ("T10 Wang H 0.6", _mh(mw, 0.6), "0.0159"),
    ("T10 Fukuchi no look-ahead", mf[mf.lookahead_s == 0.0].vs_batch_pctBW.mean(), "0.6999"),
    ("T10 Wang no look-ahead", mw[mw.lookahead_s == 0.0].vs_batch_pctBW.mean(), "0.2630"),
    # III-C1 and Table II
    ("split: trials in the grid", _grid[["subject", "trial"]].drop_duplicates().shape[0], "271"),
    ("split: draws", len(sp), "500"),
    ("split: our draw, stance", _ours["stance"], "3.21"),
    ("split: our draw, hip", _ours["hip"], "0.2069"),
    ("split: better than, stance %", 100 * (sp.test_stance > _ours["stance"]).mean(), "38"),
    ("split: better than, hip %", 100 * (sp.test_hip > _ours["hip"]).mean(), "28"),
    ("split: best over all subjects", _oracle, "3.08"),
    ("split: cost of the calibration set %", 100 * (_ours["stance"] - _oracle) / _ours["stance"], "3.9"),
    ("split: min test hip", sp.test_hip.min(), "0.185"),
    ("split: max test hip", sp.test_hip.max(), "0.225"),
    ("prop: frozen hip percentile", _pw("hip")[5], "74"),
    ("wang prop: min hip", _ww("SoleID_hip")[2], "0.387"),
    ("wang prop: max hip", _ww("SoleID_hip")[3], "0.456"),
    # III-D
    ("Wang time offset min (s)", w.tau.min(), "0.54"),
    ("Wang time offset max (s)", w.tau.max(), "0.90"),
    ("Wang alignment correlation", w.r_align.mean(), "0.93"),
    ("dead plate ratio", q[q.reason == "dead belt"].bw_ratio.mean(), "0.5"),
    ("scale-error ratio min", q[_scale].bw_ratio.min(), "1.82"),
    ("scale-error ratio max", q[_scale].bw_ratio.max(), "1.89"),
    ("retained ratio median", _qc["retained_ratio"]["median"], "0.93"),
    ("retained ratio p5", _qc["retained_ratio"]["p5"], "0.90"),
    ("retained ratio p95", _qc["retained_ratio"]["p95"], "0.96"),
    ("Wang without the outlier, shear", _fr["wang_without_outlier"]["stance"], "2.94"),
    ("Wang without the outlier, hip", _fr["wang_without_outlier"]["hip"], "0.408"),
    ("plate label, counter-clockwise contacts", int((_lgtm.direction == "ccw").sum()), "802"),
    ("plate label, clockwise contacts", int((_lgtm.direction == "cw").sum()), "391"),
    # II-D, V and counts
    ("smallest lever of the prior (m)", _fr["smallest_lever_m"], "0.94"),
    ("propulsive peak underestimate (N)", -t.SoleID_prop_peak_err.mean(), "15.5"),
    ("mass rescaling raises the residual %",
     100 * (_fr["mass_rescaled_residual"] / _fr["residual_from_timeseries"] - 1), "1"),
    ("hip reduction against Newton %", 100 * (1 - t.SoleID_hip.mean() / t.Newton_prop_hip.mean()), "50"),
    ("Camargo subjects", cam.subject.nunique(), "22"),
    ("Wang subjects", w.subject.nunique(), "9"),
    ("Wang trials", len(w), "54"),
    ("subjects in three laboratories", t.subject.nunique() + cam.subject.nunique() + w.subject.nunique(), "62"),
]

fails, timing = [], []
for name, val, written in CHECKS:
    nd = len(written.split(".")[1]) if "." in written else 0
    got = f"{float(val):.{nd}f}"
    if got != written:
        # solve times are wall-clock measurements that depend on the machine and its load: a re-run
        # lists them separately instead of failing
        (timing if name.startswith("solve time") else fails).append(
            f"  {name:28s} data {got:>10s}   paper {written:>10s}")
    if flat is not None and written not in flat:
        fails.append(f"  {name:28s} value {written} NOT FOUND in the manuscript")
print(f"{len(CHECKS)} numbers checked, {len(fails)} problems")
for f in fails:
    print(f)
if timing:
    print(f"\n{len(timing)} solve times differ from the paper's (wall-clock, machine-dependent; "
          "not counted as problems):")
    for f in timing:
        print(f)

# derived claims stated as percentages in the text
print()
for name, val, written in [
    ("hip vs published Wang, % better (harmonized)",
     100 * (1 - _rwang["soleid_vs_planar"]["hip"]["rmse"] / _rwang["wangwear_vs_planar"]["hip"]["rmse"]), "29"),
    ("Wang fast ratio", w[w.speed == 1.5].NoShear_stance.mean() / w[w.speed == 1.5].SoleID_stance.mean(), "2.4"),
    ("kin-only / SoleID, shear", t.Kin_only_stance.mean() / t.SoleID_stance.mean(), "2"),
    ("kin-only / SoleID, hip", t.Kin_only_hip.mean() / t.SoleID_hip.mean(), "4"),
    ("Newton / SoleID, hip", t.Newton_prop_hip.mean() / t.SoleID_hip.mean(), "2"),
]:
    nd = len(written.split(".")[1]) if "." in written else 0
    got = f"{float(val):.{nd}f}"
    print(f"  {'OK ' if got == written else 'FAIL'} {name:34s} {got} (text says {written})")
    if got != written:
        fails.append(f"  derived claim {name}: data {got}, text {written}")

# claims that are not a single rounded number
_contrast_p = max(_srp[k][a]["p"] for k in ("stance", "hip") for a in ("Kin_only", "Newton_prop", "SoleID_v1"))
_bins = pd.cut(t.speed, [0, 0.8, 1.2, 1.6, 3.0])
_by_bin = t.groupby(_bins, observed=True)[["Kin_only_stance", "Newton_prop_stance", "SoleID_v1_stance",
                                            "SoleID_stance"]].mean()
_by_cond = dg.groupby(["condition", "level", "arm"]).stance.mean().unstack("arm")
for name, ok in [
    ("Holm: largest adjusted p written as 7e-14",
     f"{_sr['holm']['max_adjusted_p']:.0e}" == "7e-14" and (flat is None or r"7\times10^{-14}" in flat)),
    ("Table III contrasts all p < 1e-13", _contrast_p < 1e-13 and (flat is None or r"p < 10^{-13}" in flat)),
    ("Fig. 2 shows a test subject, not a calibration one", _f2["subject"] not in _CAL),
    ("plate label: every counter-clockwise contact agrees", bool(_agree[_lgtm.direction == "ccw"].all())),
    ("plate label: no clockwise contact agrees", not bool(_agree[_lgtm.direction == "cw"].any())),
    ("Wang outlier also has the worst alignment", _fr["wang_outlier"]["worst_alignment"]),
    ("Wang outlier shear more than twice the median", _fr["wang_outlier"]["shear_over_median"] > 2),
    ("IV-A: SoleID lowest in every speed bin", bool((_by_bin.idxmin(axis=1) == "SoleID_stance").all())),
    ("IV-C: SoleID lowest under every perturbation", bool((_by_cond.idxmin(axis=1) == "SoleID").all())),
    ("II-D: the 0.2 m floor on the lever is never active", _fr["smallest_lever_m"] > 0.2),
    ("II-E: six test trials, and no calibration trial, fail the marker check",
     len(_skip) == 6 and _calj["n_trials"] == 40),
]:
    print(f"  {'OK ' if ok else 'FAIL'} {name}")
    if not ok:
        fails.append(f"  claim failed: {name}")

print()
print("FAIL" if fails else "every number checked here matches the paper")
if fails:
    raise SystemExit(1)

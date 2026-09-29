"""Run the whole analysis, from the prepared data sets to every number and figure of the paper.

    python run_all.py              # everything, in order (several hours on one CPU)
    python run_all.py --from NAME  # resume at the first step whose script is NAME
    python run_all.py --list       # print the steps and stop

Each step is one script invocation; the order respects what each script reads from results/.
The last step, verify_manuscript.py, recomputes every number the paper prints from the results
and compares it with the printed value. The data sets must be downloaded and prepared first
(README.md, "Data").
"""
import subprocess
import sys
import time

STEPS = [
    # ---- controlled layer: Fukuchi et al. (2018)
    ("protocol.py", ["calibrate"], "parameter grid on the five calibration subjects; frozen choice"),
    ("protocol.py", ["test"], "frozen estimator and all arms on the 31 test subjects"),
    ("design_loop.py", [], "design options rejected on the calibration subjects"),
    ("vpp_split.py", [], "pivot-point split arm"),
    ("final_degradation.py", [], "sensitivity to input degradation (Table IV)"),
    ("curves.py", [], "stride-normalised curves for SPM"),
    ("ablation_qp.py", [], "what each ingredient of the program contributes"),
    ("friction_cone.py", [], "how often the friction bound is active"),
    ("sole_normal.py", [], "sole-normal reading taken as vertical force"),
    ("hip_bias_decomp.py", [], "systematic hip offset"),
    ("phase_vpp.py", [], "phase-dependent pivot height: first grid"),
    ("phase_vpp.py", ["--wide", "--test"], "phase-dependent pivot height: extended grid, test set"),
    ("split_sensitivity.py", [], "which five calibration subjects: 108 configurations x 271 trials"),
    ("causal_chain.py", [], "latency pilot with bounded look-ahead filtering"),
    ("causal_design.py", [], "where the look-ahead is spent"),
    ("causal_variants.py", [], "fixed-lag smoother against truncated filter"),
    ("moving_horizon.py", ["fukuchi"], "moving-horizon form, controlled layer"),
    # ---- external layer: Camargo et al. (2021)
    ("camargo_run.py", [], "treadmill"),
    ("camargo_overground.py", ["levelground"], "level ground"),
    ("camargo_overground.py", ["ramp"], "ramps"),
    ("ramp_angles.py", [], "ramp inclinations measured from the plates"),
    # ---- application layer: Wang et al. (2023)
    ("wang_run.py", [], "real insoles and IMU suit"),
    ("wang_crosscheck.py", [], "the published wearable pipeline, rescored"),
    ("wang_factorial.py", [], "kinematics x force source, 2x2"),
    ("moving_horizon.py", ["wang"], "moving-horizon form, application layer"),
    ("split_propagate.py", [], "calibration-draw uncertainty carried to Camargo"),
    ("split_propagate.py", ["--wang"], "calibration-draw uncertainty carried to Wang"),
    # ---- across layers
    ("reference_agreement.py", ["fukuchi"], "planar reference against Visual3D"),
    ("reference_agreement.py", ["camargo"], "planar reference against OpenSim"),
    ("reference_agreement.py", ["wang"], "planar reference against OpenSim"),
    ("norms.py", [], "normalisation constants"),
    ("statistics.py", [], "mixed models, TOST, SPM"),
    ("stats_revision.py", [], "paired contrasts, Holm, bootstrap over subjects"),
    ("wang_factorial_stats.py", [], "intervals for the 2x2"),
    ("stats_paired_extra.py", [], "remaining paired contrasts"),
    ("hip_bias_breakdown.py", [], "hip offset by subject, limb, phase, speed"),
    ("qc_audit.py", [], "trial flow and exclusions"),
    ("final_read_checks.py", ["--lever"], "remaining single numbers of the paper"),
    ("comparison_tables.py", [], "summary of every setting and published results, as LaTeX tables"),
    ("make_figures.py", [], "single-panel figures (fig2 to fig5)"),
    ("multipanel_figures.py", [], "multi-panel figures (mp_overview to mp_online)"),
    ("verify_manuscript.py", [], "every number of the paper against the results"),
]


def main():
    if "--list" in sys.argv:
        for i, (script, args, what) in enumerate(STEPS, 1):
            print(f"{i:2d}. python {script} {' '.join(args):<14s} {what}")
        return
    start = 0
    if "--from" in sys.argv:
        name = sys.argv[sys.argv.index("--from") + 1]
        start = next(i for i, s in enumerate(STEPS) if s[0] == name)
    t0 = time.time()
    for i, (script, args, what) in enumerate(STEPS[start:], start + 1):
        print(f"\n[{i}/{len(STEPS)}] python {script} {' '.join(args)}   ({what})", flush=True)
        t = time.time()
        r = subprocess.run([sys.executable, script] + args)
        if r.returncode != 0:
            sys.exit(f"step {i} ({script}) failed with exit code {r.returncode}")
        print(f"      done in {(time.time() - t) / 60:.1f} min", flush=True)
    print(f"\nall steps done in {(time.time() - t0) / 3600:.1f} h")


if __name__ == "__main__":
    main()

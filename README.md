# SoleID

Code and results for the paper

> A. Ağaoğlu, "Unknown-Input Estimation of Shear Ground Reaction Forces and Sagittal-Plane Joint
> Torques from Insole Vertical Force and Kinematics," manuscript, 2026.

An instrumented insole measures the vertical ground reaction force and the center of pressure but
not the shear (anteroposterior) component, which joint moments need. SoleID treats the missing
shear as an unknown input and recovers it, per foot, with a convex quadratic program that combines
whole-body horizontal dynamics, a friction cone, unilateral contact, the measured vertical force and
center of pressure, and a virtual-pivot-point direction prior. Sagittal joint moments then follow by
inverse dynamics. The estimator has four parameters, selected once on five subjects and then frozen;
it is evaluated on three public data sets.

Everything in this repository is computed from those three data sets. `run_all.py` recreates every
result file from the data, and `verify_manuscript.py` recomputes every number the paper prints from
the result files and compares it with the printed value.

## Contents

| Path | What |
|---|---|
| `soleid_planar.py` | planar seven-segment model, inverse dynamics, the estimator, the baselines |
| `experiments.py` | data preparation shared by the analyses; the direction prior |
| `protocol.py` | quality control, calibration on five subjects, test run on the controlled layer |
| `camargo_*.py`, `wang_*.py` | adapters and runs for the external and application layers |
| `moving_horizon.py` | sliding-window (online) form of the estimator |
| other `*.py` | one analysis each (table below) |
| `download_*.py`, `camargo_to_csv.m` | data download and conversion |
| `run_all.py` | the whole analysis, in order |
| `verify_manuscript.py` | every number of the paper, recomputed from `results/` |
| `config.py` | all paths (data, results, figures) |
| `results/` | the result files the paper draws on |
| `figures/` | Figs. 2-5 as written by `make_figures.py` (Fig. 1 is a drawing) |

## Requirements

* Python 3.14 and the packages in `requirements.txt` (`pip install -r requirements.txt`), the
  versions the results were produced with.
* MATLAB (R2025b was used), once, to convert the Camargo data from `.mat` tables to CSV.
* An extractor for `.rar` archives (7-Zip, WinRAR or `unrar`) for the Wang data.

## Data

The three data sets are public and are not redistributed here. The code expects them under `./data`
unless the environment variables `SOLEID_WBDS`, `SOLEID_CAMARGO` and `SOLEID_WANG` point elsewhere
(`config.py`):

```
data/
  WBDS/                    Fukuchi et al.: WBDSinfo.csv and the text files of the treadmill trials
  Camargo2021/
    raw/                   <subject>/<date>/<mode>/<sensor>/*.mat, as in the data set
    csv/                   the same tables as CSV (camargo_to_csv.m)
    SubjectInfo.mat        subject table of the data set
    SubjectInfo.csv        the same as CSV (camargo_to_csv.m)
  Wang2023/
    Processed_data/        extracted from Processed_data.rar
    cache/                 written by wang_adapter.py on first use
```

### Controlled layer: Fukuchi et al. (2018)

Treadmill walking at eight speeds with a force plate under each belt; the release used here
(figshare version 6, retrieved September 2026) has 51 volunteers. License: CC BY 4.0.

* Record: figshare, <https://doi.org/10.6084/m9.figshare.5722711>
* `python download_wbds.py` fetches `WBDSinfo.csv` and, for every treadmill trial, the marker
  (`mkr`), force-plate (`grf`) and Visual3D joint-kinetics (`knt`) text files. The trial files are
  read out of the 687 MB archive `WBDSascii.zip` by HTTP range requests, so the archive itself is not
  downloaded (about 1.2 GB in total).

### External layer: Camargo et al. (2021)

Treadmill, level-ground and ramp walking of 22 subjects, recorded in a second laboratory.
License: CC BY 4.0.

* Records: Mendeley Data, in three parts,
  <https://doi.org/10.17632/fcgm3chfff.1> (part 1),
  <https://doi.org/10.17632/k9kvm5tn3f.1> (part 2) and
  <https://doi.org/10.17632/jj3r5f9pnf.1> (part 3);
  also linked from the authors' laboratory page,
  <http://www.epic.gatech.edu/opensource-biomechanics-camargo-et-al>.
* `python download_camargo.py` fetches, for the treadmill, level-ground and ramp modes, the marker,
  force-plate, condition and inverse-dynamics tables into `data/Camargo2021/raw/` (again by range
  requests into the archives), and the subject table `SubjectInfo.mat`.
* Convert to CSV (this also writes `SubjectInfo.csv`):
  `matlab -batch "camargo_to_csv('data/Camargo2021/raw', 'data/Camargo2021/csv')"`

### Application layer: Wang et al. (2023)

Nine subjects with pressure insoles and an IMU suit, and an instrumented treadmill with optical
motion capture as reference. License: CC BY 3.0 NL.

* Record: Zenodo, <https://doi.org/10.5281/zenodo.6457662>
* `python download_wang2023.py` fetches `Processed_data.rar` (17.9 GB; resumable, MD5-checked).
  Extract it so that the subject folders sit in `data/Wang2023/Processed_data/`.

## Running

```
python run_all.py --list      # the 41 steps and what each produces
python run_all.py             # all of them, in order; several hours on one CPU
python verify_manuscript.py   # the check of the paper's numbers alone, in seconds
```

`verify_manuscript.py` recomputes each number of the paper from `results/` and compares it with the
value printed in the paper, rounded the same way; it exits with a non-zero status on any mismatch.
Values taken from the literature, method parameters and facts of the data-set design are not
checked. The analyses are deterministic (every random draw has a fixed seed), so a full run
reproduces the committed result files, with one exception: the solve times in Table X are
wall-clock measurements that vary with the machine and its load, and `verify_manuscript.py` lists
them separately instead of counting them as mismatches.

`results/` holds every file the checks and the figures read. Large intermediate files (the per-trial
time series and metrics of the controlled layer, the stride-normalized reference curves) are left
out and are recreated by `run_all.py`; of the time series, only the trial shown in Fig. 2 is included.

`audit_core.py` (the physics and the estimator on cases with a known answer) and `audit_data.py`
(protocol integrity and physical plausibility of each data set) are additional checks that can be run
at any time.

## Where each result of the paper comes from

| Paper | Scripts | Result files |
|---|---|---|
| Table I, data sets | `protocol.py`, `camargo_run.py`, `camargo_overground.py`, `wang_run.py` | `fukuchi_quality.csv`, `test_all_trials.csv`, `camargo_*`, `wang_application.csv` |
| Table II, other calibration draws | `split_sensitivity.py`, `split_propagate.py` | `grid_all_trials.csv`, `split_sensitivity.csv`, `split_propagated_*.csv` |
| Table III, controlled layer | `protocol.py`, `vpp_split.py`, `stats_revision.py`, `stats_paired_extra.py` | `test_all_trials.csv`, `vpp_split_test.csv`, `stats_revision.json`, `stats_paired_extra.json` |
| Table IV, input degradation | `final_degradation.py`, `final_read_checks.py` | `final_degradation_trials.csv`, `final_read_checks.json` |
| Table V, external layer | `camargo_run.py`, `camargo_overground.py` | `camargo_treadmill_strides.csv`, `camargo_levelground_steps.csv`, `camargo_ramp_steps.csv` |
| Table VI, published results on the Camargo data | `comparison_tables.py` | `tables_comparison.tex`, `comparison_tables.md` |
| Table VII, real insoles and IMU suit | `wang_run.py`, `reference_agreement.py` | `wang_application.csv`, `reference_agreement.json` |
| Table VIII, kinematics x force source | `wang_factorial.py`, `wang_factorial_stats.py` | `wang_factorial.csv`, `wang_factorial_stats.json` |
| Table IX, the references | `reference_agreement.py` | `reference_agreement.json` |
| Table X, moving-horizon form | `moving_horizon.py`, `statistics.py` | `mhe_fukuchi.csv`, `mhe_wang.csv`, `statistics.json` |
| Table XI, latency pilot | `causal_chain.py`, `causal_design.py`, `causal_variants.py` | `causal_chain.csv`, `causal_design_*.csv`, `causal_variants.csv` |
| Fig. 2 (`fig2_timeseries`) | `protocol.py`, `make_figures.py` | `fig2_trial.json` and that trial's `*_timeseries.csv` |
| Fig. 3 (`fig3_speed`) | `make_figures.py` | `test_all_trials.csv`, `wang_application.csv` |
| Fig. 4 (`fig4_spm`) | `curves.py`, `statistics.py`, `make_figures.py` | `curves_fukuchi.npz`, `statistics.json` |
| Fig. 5 (`fig5_ramp`) | `camargo_overground.py`, `ramp_angles.py`, `make_figures.py` | `camargo_ramp_steps.csv`, `ramp_angles.json` |
| Other values in the text | `ablation_qp.py`, `friction_cone.py`, `sole_normal.py`, `hip_bias_decomp.py`, `hip_bias_breakdown.py`, `phase_vpp.py`, `design_loop.py`, `norms.py`, `qc_audit.py`, `final_read_checks.py` | the files of the same names in `results/` |

## Citation

If you use this code, please cite the paper above and the three data sets:

* C. A. Fukuchi, R. K. Fukuchi, and M. Duarte, "A public dataset of overground and treadmill walking
  kinematics and kinetics in healthy individuals," *PeerJ*, vol. 6, e4640, 2018,
  doi:10.7717/peerj.4640. Data: doi:10.6084/m9.figshare.5722711.
* J. Camargo, A. Ramanathan, W. Flanagan, and A. Young, "A comprehensive, open-source dataset of
  lower limb biomechanics in multiple conditions of stairs, ramps, and level-ground ambulation and
  transitions," *Journal of Biomechanics*, vol. 119, 110320, 2021,
  doi:10.1016/j.jbiomech.2021.110320. Data: doi:10.17632/fcgm3chfff.1, doi:10.17632/k9kvm5tn3f.1,
  doi:10.17632/jj3r5f9pnf.1.
* H. Wang, A. Basu, G. Durandau, and M. Sartori, "A wearable real-time kinetic measurement sensor
  setup for human locomotion," *Wearable Technologies*, vol. 4, e11, 2023, doi:10.1017/wtc.2023.7.
  Data: H. Wang, A. Basu, G. Durandau, and M. Sartori, "Comprehensive Kinetic and EMG Dataset of
  Daily Locomotion with 6 types of Sensors," Zenodo, 2022, doi:10.5281/zenodo.6457662.

The method and the code also build on:

* H.-M. Maus, S. W. Lipfert, M. Günther, J. Rummel, and A. Seyfarth, "Upright human gait did not
  provide a major mechanical challenge for our ancestors," *Nature Communications*, vol. 1, 70,
  2010, doi:10.1038/ncomms1073 (virtual pivot point).
* D. A. Winter, *Biomechanics and Motor Control of Human Movement*, 4th ed. John Wiley & Sons, 2009
  (segment parameters).
* M. E. Harrington, A. B. Zavatsky, S. E. M. Lawson, Z. Yuan, and T. N. Theologis, "Prediction of the
  hip joint centre in adults, children, and patients with cerebral palsy based on magnetic resonance
  imaging," *Journal of Biomechanics*, vol. 40, no. 3, pp. 595-602, 2007,
  doi:10.1016/j.jbiomech.2006.02.003 (hip joint center).
* B. Stellato, G. Banjac, P. Goulart, A. Bemporad, and S. Boyd, "OSQP: an operator splitting solver
  for quadratic programs," *Mathematical Programming Computation*, vol. 12, no. 4, pp. 637-672, 2020,
  doi:10.1007/s12532-020-00179-2 (the solver).
* T. C. Pataky, "Generalized n-dimensional biomechanical field analysis using statistical parametric
  mapping," *Journal of Biomechanics*, vol. 43, no. 10, pp. 1976-1982, 2010,
  doi:10.1016/j.jbiomech.2010.03.008 (SPM, via spm1d).

## License

The code is released under the MIT License (`LICENSE`). The data sets remain under their authors'
licenses (above); the files in `results/` are derived from them.

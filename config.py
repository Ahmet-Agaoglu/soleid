"""Where the data, the results and the figures live.

Every path in the code comes from here. The defaults follow the layout in README.md: the three
public data sets under ./data, results in ./results and figures in ./figures, all next to this file.
Two ways to put them elsewhere:

  * environment variables SOLEID_DATA, SOLEID_RESULTS, SOLEID_FIGURES (and, per data set,
    SOLEID_WBDS, SOLEID_CAMARGO, SOLEID_WANG), or
  * a file config_local.py next to this one that assigns any of the names below; it is not tracked.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

DATA = os.environ.get("SOLEID_DATA", os.path.join(HERE, "data"))
RESULTS = os.environ.get("SOLEID_RESULTS", os.path.join(HERE, "results"))
FIGURES = os.environ.get("SOLEID_FIGURES", os.path.join(HERE, "figures"))

# Fukuchi et al. (2018), the WBDS text files and WBDSinfo.csv
WBDS = os.environ.get("SOLEID_WBDS", os.path.join(DATA, "WBDS"))
# Camargo et al. (2021): raw/ (as downloaded), csv/ (after camargo_to_csv.m), SubjectInfo.csv
CAMARGO = os.environ.get("SOLEID_CAMARGO", os.path.join(DATA, "Camargo2021"))
# Wang et al. (2023): Processed_data/ (extracted from Processed_data.rar) and cache/
WANG = os.environ.get("SOLEID_WANG", os.path.join(DATA, "Wang2023"))

# LaTeX copy of the comparison table; the manuscript source is not part of this repository
TABLES_TEX = os.path.join(RESULTS, "tables_comparison.tex")
# If set to the manuscript's .tex file, verify_manuscript.py also checks that every number it
# re-derives is written in the text; otherwise it compares with the values of the paper alone
MANUSCRIPT_TEX = None

try:                                   # local overrides, never committed
    from config_local import *         # noqa: F401,F403
except ImportError:
    pass

CAMARGO_CSV = os.path.join(CAMARGO, "csv")
CAMARGO_INFO = os.path.join(CAMARGO, "SubjectInfo.csv")
WANG_PROCESSED = os.path.join(WANG, "Processed_data")
WANG_CACHE = os.path.join(WANG, "cache")

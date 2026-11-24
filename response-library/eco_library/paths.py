"""Locations used by the library.  Everything is relative to this folder (response-library) and to
the frozen baseline beside it (../baseline); nothing is written under the baseline."""
from pathlib import Path

LIBRARY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = LIBRARY_ROOT.parent
BASELINE = PROJECT_ROOT/"baseline"
BASELINE_OUTPUTS = BASELINE/"eco_outputs_final"
NOTEBOOK = BASELINE/"eco-waveform-final.ipynb"
RUNS = LIBRARY_ROOT/"runs"            # simulation output (not tracked by Git)
RESULTS = LIBRARY_ROOT/"results"      # the committed results

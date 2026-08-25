"""Phase 3 cleaning-ablation package.

This package is deliberately kept outside ``tfm_dirty_values`` because the
Phase 2 authorization artifact binds a SHA-256 identity over every file in
``src/tfm_dirty_values/**/*.py``. Adding a module there would invalidate the
completed Phase 2 authorization and damage its provenance record.

Phase 3 imports the frozen ``tfm_dirty_values`` package for data loading,
splits, corruption injection, model adapters, and metrics, so the ablation
provably reuses identical machinery.
"""

from .cleaning import CLEANER_ID, CLEANER_PARAMETERS, apply_cleaner, fit_cleaner

__all__ = ["CLEANER_ID", "CLEANER_PARAMETERS", "apply_cleaner", "fit_cleaner"]

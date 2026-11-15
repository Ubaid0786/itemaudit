"""itemaudit: audit an agent benchmark from its results matrix alone."""

from .irt import IRTFit, fit_2pl
from .items import dif, guttman_errors, item_stats, kr20, system_fit
from .leaderboard import Leaderboard, leaderboard
from .matrix import ResultsMatrix, load_results
from .report import report
from .stats import (
    PairedComparison,
    compare,
    holm,
    mcnemar_exact,
    min_detectable_difference,
    min_significant_difference,
    paired_bootstrap_ci,
    paired_se,
)

__version__ = "0.1.0"

__all__ = [
    "ResultsMatrix",
    "load_results",
    "compare",
    "PairedComparison",
    "leaderboard",
    "Leaderboard",
    "item_stats",
    "system_fit",
    "dif",
    "guttman_errors",
    "kr20",
    "fit_2pl",
    "IRTFit",
    "report",
    "mcnemar_exact",
    "holm",
    "paired_se",
    "paired_bootstrap_ci",
    "min_detectable_difference",
    "min_significant_difference",
]

"""
Top-level re-exports to keep the public API stable after reorganizing into
implementation/feature_selection/* and other subpackages.

This mirrors the earlier Prompt-8 surface that's used by tests.
"""

from .feature_selection.mi import EqualWidthMI, MIRankItem
from .feature_selection.mrmr import CorrelationDiversity
from .feature_selection.dedup import PreScoreDeduper
from .feature_selection.mean_imputer import MeanImputer
from .feature_selection.standard_scaler import StandardScaler
from .feature_selection.hygiene import train_only_fit_apply

equal_width_bins = EqualWidthMI.equal_width_bins
mi_equal_width = EqualWidthMI.mi_equal_width
rank_by_mi_with_ties = EqualWidthMI.rank_by_mi_with_ties

pearsonr_stable = CorrelationDiversity.pearsonr_stable
select_diverse_by_corr = CorrelationDiversity.select_diverse_by_corr

canonical_struct_key = PreScoreDeduper.canonical_struct_key
numeric_fingerprint = PreScoreDeduper.numeric_fingerprint
dedup = PreScoreDeduper.dedup

from .complexity import Complexity
from .interpretability import Criterion, Rubric, CalibratorS

__all__ = [
	"EqualWidthMI", "MIRankItem", "equal_width_bins", "mi_equal_width", "rank_by_mi_with_ties",
	"CorrelationDiversity", "pearsonr_stable", "select_diverse_by_corr",
	"PreScoreDeduper", "canonical_struct_key", "numeric_fingerprint", "dedup",
	"MeanImputer", "StandardScaler", "train_only_fit_apply",
	"Complexity", "Criterion", "Rubric", "CalibratorS",
]


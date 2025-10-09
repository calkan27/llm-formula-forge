"""
Public API (function-shaped re-exports for tests and callers)
------------------------------------------------------------
equal_width_bins(x, b=16)              -> ndarray[int]
mi_equal_width(x, y, b=16)             -> float
rank_by_mi_with_ties(features, y, b=16)-> List[MIRankItem]

select_diverse_by_corr(X, ranked_indices, top_k, corr_cutoff=0.9) -> List[int]
pearsonr_stable(a, b)                 -> float

canonical_struct_key(text)            -> str
numeric_fingerprint(text, n_points=8) -> str
dedup(texts)                          -> (unique_texts, meta_map)

MeanImputer, StandardScaler           -> estimators (class-per-file)
train_only_fit_apply(Xtr, Xva, Xte)   -> (Xtr2, Xva2, Xte2, imputer, scaler)
"""

from .mi import EqualWidthMI, MIRankItem
from .mrmr import CorrelationDiversity
from .dedup import PreScoreDeduper
from .mean_imputer import MeanImputer
from .standard_scaler import StandardScaler
from .hygiene import train_only_fit_apply


equal_width_bins = EqualWidthMI.equal_width_bins
mi_equal_width = EqualWidthMI.mi_equal_width
rank_by_mi_with_ties = EqualWidthMI.rank_by_mi_with_ties

pearsonr_stable = CorrelationDiversity.pearsonr_stable
select_diverse_by_corr = CorrelationDiversity.select_diverse_by_corr

canonical_struct_key = PreScoreDeduper.canonical_struct_key
numeric_fingerprint = PreScoreDeduper.numeric_fingerprint
dedup = PreScoreDeduper.dedup

__all__ = [
	"EqualWidthMI", "MIRankItem", "equal_width_bins", "mi_equal_width", "rank_by_mi_with_ties",
	"CorrelationDiversity", "pearsonr_stable", "select_diverse_by_corr",
	"PreScoreDeduper", "canonical_struct_key", "numeric_fingerprint", "dedup",
	"MeanImputer", "StandardScaler", "train_only_fit_apply",
]


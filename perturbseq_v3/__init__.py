from __future__ import absolute_import

__version__ = 0.1

from .cell_population import CellPopulation, MeanPopulation, fancy_dendrogram, fit_dendrogram, correlation_heatmap, metaapply
from .expression_normalization import z_normalize_expression, normalize_to_control, normalize_matrix_to_control, strip_low_expression, log_normalize_expression, equalize_UMI_counts, normalize_to_gemgroup_control, inherit_normalized_matrix, normalize_matrix_by_key
from .cell_cycle import get_cell_phase_genes, add_cell_cycle_scores, cell_cycle_position_heatmap
from .transformers import PCAReducer, ICAReducer, PCATSNEReducer, PCAUMAPReducer
from .differential_expression import ks_de, ad_de, boruta_de, tree_selector, BorutaDEResult, TreeSelectorResult, find_noisy_genes
from .util import upper_triangle, nzflat, gini
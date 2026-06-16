"""
profiler_imports.py — Profiler Desktop · app/utils/
=====================================================
Bulk imports for the desktop version (offline).
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import io
import time
import datetime

# ── Core scientific stack (toujours en mémoire, pas de gain à lazy-iser) ──────
import streamlit as st
import numpy as np
import pandas as pd
import openpyxl
import joblib

# ── Visualisation ─────────────────────────────────────────────────────────────
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap

# ── Scikit-learn : modules fréquents — import direct ─────────────────────────
import shap
import eli5

from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, silhouette_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler
)
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, BaggingClassifier,
    ExtraTreesClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier, StackingClassifier, VotingClassifier,
    IsolationForest,
)
from sklearn.linear_model import (
    SGDClassifier, LogisticRegression, RidgeClassifier,
    PassiveAggressiveClassifier, Perceptron, Lasso,
)
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.dummy import DummyClassifier
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.cluster import KMeans
from sklearn.feature_selection import VarianceThreshold

# ── Stats / imbalanced ────────────────────────────────────────────────────────
from scipy.stats import ttest_ind, shapiro, kstest
from statsmodels.stats.multitest import multipletests
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from statannotations.Annotator import Annotator

# ── Misc ─────────────────────────────────────────────────────────────────────
from itertools import combinations
import networkx as nx
from upsetplot import UpSet
import gseapy as gp
import dask.dataframe as dd


# ════════════════════════════════════════════════════════════════════════════
# LAZY IMPORT HELPERS
# Chaque fonction importe le module lourd UNE SEULE FOIS (variable module-level)
# et le retourne à chaque appel suivant sans re-importer.
# ════════════════════════════════════════════════════════════════════════════

# ── TensorFlow / Keras ────────────────────────────────────────────────────────
_tf_module = None
def lazy_tensorflow():
    """Import tensorflow uniquement quand le tab Deep Learning est utilisé."""
    global _tf_module
    if _tf_module is None:
        import tensorflow as tf
        _tf_module = tf
    return _tf_module

_tf_callback_cls = None
def lazy_tf_callback():
    """Import tensorflow.keras.callbacks.Callback (pour profiler_DL)."""
    global _tf_callback_cls
    if _tf_callback_cls is None:
        from tensorflow.keras.callbacks import Callback
        _tf_callback_cls = Callback
    return _tf_callback_cls

# ── UMAP ──────────────────────────────────────────────────────────────────────
_umap_module = None
def lazy_umap():
    """Import umap uniquement quand une projection UMAP est demandée."""
    global _umap_module
    if _umap_module is None:
        import umap.umap_ as _umap
        _umap_module = _umap
    return _umap_module

# ── t-SNE (sklearn, léger mais lazy par cohérence) ───────────────────────────
_tsne_cls = None
def lazy_tsne():
    """Import sklearn.manifold.TSNE."""
    global _tsne_cls
    if _tsne_cls is None:
        from sklearn.manifold import TSNE
        _tsne_cls = TSNE
    return _tsne_cls

# ── Lifelines (Survival) ──────────────────────────────────────────────────────
_lifelines_modules = None
def lazy_lifelines():
    """Import lifelines (KaplanMeier, CoxPH, logrank) — onglet Survival."""
    global _lifelines_modules
    if _lifelines_modules is None:
        from lifelines import KaplanMeierFitter, CoxPHFitter
        from lifelines.statistics import logrank_test
        _lifelines_modules = {
            'KaplanMeierFitter': KaplanMeierFitter,
            'CoxPHFitter':       CoxPHFitter,
            'logrank_test':      logrank_test,
        }
    return _lifelines_modules

# ── Fastcluster (heatmaps hiérarchiques) ─────────────────────────────────────
_fastcluster_module = None
def lazy_fastcluster():
    """Import fastcluster uniquement pour les heatmaps hiérarchiques."""
    global _fastcluster_module
    if _fastcluster_module is None:
        import fastcluster
        _fastcluster_module = fastcluster
    return _fastcluster_module

# ── PyOpenMS (mzML / mzXML) ───────────────────────────────────────────────────
_pyopenms_modules = None
def lazy_pyopenms():
    """Import pyopenms uniquement quand des fichiers mzML/mzXML sont chargés."""
    global _pyopenms_modules
    if _pyopenms_modules is None:
        from pyopenms import MSExperiment, MzMLFile, MzXMLFile
        _pyopenms_modules = {
            'MSExperiment': MSExperiment,
            'MzMLFile':     MzMLFile,
            'MzXMLFile':    MzXMLFile,
        }
    return _pyopenms_modules


# ════════════════════════════════════════════════════════════════════════════
# PROFILER PACKAGE IMPORTS  (app.* — structure Desktop v1.2)
# ════════════════════════════════════════════════════════════════════════════

# data/
from app.data.profiler_conversion import convert_raw_to_mzml
from app.data.profiler_data_loading import (
    load_uploaded_files, add_group, lazy_import, safe_load_data,
    get_data_for_source, finalize_data_load, render_mzml_chromatogram_sidebar,
)
from app.data.profiler_structured_data_file import (
    maxquant_data, load_structured_data, update_class_names, get_data,
    cluster_index_to_letter, diann_data, perseus_data,
    get_meta_columns, get_omics_columns, get_target_column_options,
    CLASS_ALIASES, ID_ALIASES, detect_omics_format,
    spectronaut_protein_data, spectronaut_peptide_data,
    fragpipe_data, proteome_discoverer_data, progenesis_data,
    peaks_data, maxquant_peptide_data, diann_peptide_data,
    rnaseq_counts_data, salmon_kallisto_data, featurecounts_data,
    star_counts_data, htseq_counts_data, metaboanalyst_data,
    xcms_mzmine_data, load_omics_auto, render_tabular_loader,
)

# core/
from app.core.profiler_preprocessing import (
    load_and_preprocess_data, preprocess_data, apply_binning_to_mass_range,
    preprocess_data_dask, apply_binning_to_mass_range_dask,
)
from app.core.profiler_training import (
    train_models, plot_learning_curve, compare_models,
    train_regression_models, compare_regression_models, plot_roc_curves,
)
from app.core.profiler_DL import (
    build_mlp, build_cnn, build_rnn,
    train_DL, compare_DL, display_model_results, display_global_results,
)

# analysis/
from app.analysis.profiler_sampling import apply_sampling
from app.analysis.profiler_unsupervised import plot_umap, plot_tsne, apply_pca, plot_pca
from app.analysis.profiler_visualization import (
    display_data_section, display_model_results, plot_mean_spectrum,
    plot_individual_spectra, plot_protein_expression_bubble, plot_heatmap,
    calculate_maximal_intersections, calculate_all_intersections,
)
from app.data.profiler_data_exploration import (
    plot_feature_distribution, plot_venn_diagram, plot_upset,
    plot_multiple_features_line, plot_multiple_features_radar,
    plot_multiple_features_distribution,
)
from app.analysis.profiler_features_importance import (
    eli5_feature_importance, plot_shap_values, boxplot_significant_features,
    violinplot_significant_features, plot_heatmap_samples, eli5_format_to_dataframe,
    calculate_volcano_data, plot_volcano, detect_peaks,
    barplot_significant_features, plot_significant_features,
    _resolve_features, render_heatmap_dendrogram_widget
)
from app.analysis.profiler_genes_enrichment import (
    perform_gsea, perform_gsea_offline,
    load_gene_sets, load_gene_sets_offline,
    render_enrichment_tab,
)
from app.analysis.profiler_survival import (
    create_cox_pipeline, detect_delimiter, detect_collinearity, infer_time_unit,
)
from app.analysis.profiler_rt import (
    load_all_rt, convert_raw_to_mzml_rt, load_data_single_file_rt,
    preprocess_data_rt, apply_binning_to_mass_range_rt, apply_svd_rt, decision_rt,
    visualize_predictions_circles_rt, convert_raw_to_mzml_rt_multi_format_with_zip,
)
from app.analysis.profiler_normality import (
    diagnose_normality, display_class_info, calculate_missing_values,
    perform_shapiro_wilk_test, display_distribution,
    plot_missing_heatmap, plot_missing_per_class,
    plot_zero_inflation_per_class, plot_feature_completeness_rank,
)
from app.analysis.profiler_longitudinal import (
    validate_longitudinal_df,
    plot_trajectory,
    plot_multi_trajectory,
    delta_features,
    run_lmm,
    run_rm_anova,
    volcano_longitudinal,
    plot_longitudinal_heatmap,
    summarise_longitudinal,
    render_longitudinal_tab,
)

import streamlit as st
import os
import datetime
import numpy as np
import pandas as pd
import openpyxl
import joblib
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import dask.dataframe as dd
import shap
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import TruncatedSVD
import eli5
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, HistGradientBoostingClassifier, StackingClassifier, VotingClassifier
)
from sklearn.linear_model import (
    SGDClassifier, LogisticRegression, RidgeClassifier, PassiveAggressiveClassifier, Perceptron, Lasso
)
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.dummy import DummyClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import matplotlib

# from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from statannotations.Annotator import Annotator
from itertools import combinations
from matplotlib.colors import LinearSegmentedColormap
import umap.umap_ as umap
from sklearn.manifold import TSNE
from tensorflow.keras.callbacks import Callback
import tensorflow as tf
import fastcluster
from pyopenms import MSExperiment, MzMLFile, MzXMLFile
import gseapy as gp
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import time
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import io
from upsetplot import UpSet
import networkx as nx
from scipy.stats import ttest_ind, shapiro, kstest
from sklearn.impute import KNNImputer

# ─── Import Profiler modules (new package structure v1.2) ────────────────────
# data/
from app.data.profiler_conversion import convert_raw_to_mzml
# from app.data.profiler_data_loading import save_uploaded_file, load_uploaded_files, add_group, lazy_import, safe_load_data, get_data_for_source, finalize_data_load
from app.data.profiler_data_loading import (
    load_uploaded_files, add_group, lazy_import, safe_load_data,
    get_data_for_source, finalize_data_load, render_mzml_chromatogram_sidebar,
)
from app.data.profiler_structured_data_file import (
    maxquant_data, load_structured_data, update_class_names, get_data,
    cluster_index_to_letter, diann_data, perseus_data,
    get_meta_columns, get_omics_columns, get_target_column_options,
    CLASS_ALIASES, ID_ALIASES,
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
# from app.core.profiler_training import plot_roc_curve_cv, plot_sensitivity_specificity_threshold, plot_precision_recall_curve_cv
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
    visualize_predictions_circles_rt, convert_raw_to_mzml_rt_multi_format_with_zip
)
from app.analysis.profiler_normality import (
    diagnose_normality, display_class_info, calculate_missing_values,
    perform_shapiro_wilk_test, display_distribution,
    plot_missing_heatmap, plot_missing_per_class,
    plot_zero_inflation_per_class, plot_feature_completeness_rank,
)

# Extended omics format parsers


# Longitudinal / repeated-measures analysis module
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
"""
Software Name: Profiler
Module name : impots dependences
Author: Yanis Zirem
Email : yanis.zirem@yahoo.com / yanis.zirem@univ-lille.fr
Creation Date: 15/01/2025
Last Updated: 23/10/2025
Version: 1.0.0

Context:
This module is part of the "Profiler" project, originally developed for a web version (https://prism-profiler.univ-lille.fr) and now adapted for a desktop version (profiler_desktop_GUI).
It is designed for archiving on Zenodo and integration into GitHub releases.

License: l’Agence pour la Protection des Programmes IDDN (InterDeposit Digital Number) : FR2 .0013 .0300044 .0005 .S6 .C7 .20258 .0009 .312301
Citation:
If Profiler or this module (a part of Profiler) is used in a publication, please cite:
Zirem, Y. (2025). Profiler: an open web platform for multi-omics analysis. Journal of Bioinformatics. [DOI or Zenodo/GitHub link available in the article].

Links:
- GitHub temporary Repository: https://github.com/yanisZirem/Profiler_v1_requests_datatests
"""

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

# Import Profiler modules (developped modules by Yanis Zirem)
from profiler_conversion_desk import convert_raw_to_mzml
from profiler_data_loading import save_uploaded_file, load_uploaded_files, add_group, get_data, lazy_import,safe_load_data, get_data_for_source, finalize_data_load
from profiler_preprocessing import load_and_preprocess_data, preprocess_data, apply_binning_to_mass_range
from profiler_sampling import apply_sampling
from profiler_unsupervised import (
    plot_umap, plot_tsne, apply_pca, plot_pca
)
from profiler_ML import train_models, plot_learning_curve, compare_models
from profiler_visualization import (
    display_data_section, display_model_results, plot_mean_spectrum, 
    plot_individual_spectra, plot_heatmap, calculate_maximal_intersections,calculate_all_intersections
)
from profiler_data_exploration import plot_feature_distribution, plot_venn_diagram, plot_upset, plot_multiple_features_line, plot_multiple_features_radar, plot_multiple_features_distribution 
from profiler_features_importance import (
    eli5_feature_importance, plot_shap_values, boxplot_significant_features, 
    violinplot_significant_features, plot_heatmap_samples, eli5_format_to_dataframe, 
    calculate_volcano_data, plot_volcano, detect_peaks, barplot_significant_features, plot_significant_features
)
from profiler_DL import build_mlp, build_cnn, build_rnn, train_DL, compare_DL, display_model_results, display_global_results
from profiler_structured_data_file import (
    maxquant_data, load_structured_data, update_class_names, get_data, 
    cluster_index_to_letter, diann_data, perseus_data
)
from profiler_genes_enrichement import perform_gsea,load_gene_sets
from profiler_survival import create_cox_pipeline, detect_delimiter, detect_collinearity, infer_time_unit
from profiler_rt import (
    load_all_rt, convert_raw_to_mzml_rt, load_data_single_file_rt, 
    preprocess_data_rt, apply_binning_to_mass_range_rt, apply_svd_rt, decision_rt, 
    visualize_predictions_circles_rt
)
from profiler_normality import(
    diagnose_normality, display_class_info,calculate_missing_values,perform_shapiro_wilk_test, display_distribution
)

from offline_enrichr import(
    load_gene_sets_offline, perform_gsea_offline  
)
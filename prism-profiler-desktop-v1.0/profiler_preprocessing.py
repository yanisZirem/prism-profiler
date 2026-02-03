"""
Software Name: Profiler
Module Name: Preprocessing
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

import pandas as pd
import numpy as np
import streamlit as st
from scipy.stats import rankdata
import dask.dataframe as dd
import numexpr as ne
from pandarallel import pandarallel
import gc



# def preprocess_data(all_data, normalization_type=None, _progress_bar=None):
#     st.info("Applying Normalization...")

#     if all_data.empty:
#         st.error("DataFrame is empty.")
#         return pd.DataFrame()

#     if _progress_bar:
#         _progress_bar.progress(0.1)

#     # Colonnes fixes à garder
#     fixed_columns = ['Class', 'File', 'RT', 'Sum']
#     if not all(col in all_data.columns for col in fixed_columns):
#         st.error("Missing one or more required columns.")
#         return pd.DataFrame()

#     fixed_data = all_data[fixed_columns].reset_index(drop=True)
#     intensity_df = all_data.drop(columns=fixed_columns)
#     intensity_cols = intensity_df.columns
#     intensity_data = intensity_df.fillna(0).astype(np.float32).values

#     if _progress_bar:
#         _progress_bar.progress(0.3)

#     # Normalisation
#     if normalization_type:
#         try:
#             if normalization_type == 'Log Normalization':
#                 intensity_data = np.log1p(intensity_data)
#             elif normalization_type == 'Log10':
#                 intensity_data = np.log10(intensity_data + 1)
#             elif normalization_type == 'Log2':
#                 intensity_data = np.log2(intensity_data + 1)
#             elif normalization_type == 'TIC':
#                 sums = fixed_data['Sum'].replace(0, np.nan).values.astype(np.float32).reshape(-1, 1)
#                 intensity_data = intensity_data / sums
#             elif normalization_type == 'RMS':
#                 rms_vals = np.sqrt(np.sum(np.square(intensity_data), axis=1)).reshape(-1, 1)
#                 intensity_data = intensity_data / np.where(rms_vals == 0, np.nan, rms_vals)
#             elif normalization_type == 'BasePeak':
#                 max_vals = np.max(intensity_data, axis=1).reshape(-1, 1)
#                 intensity_data = intensity_data / np.where(max_vals == 0, np.nan, max_vals)
#             elif normalization_type == 'QNorm':
#                 intensity_data = quantile_normalization(intensity_data)
#         except Exception as e:
#             st.error(f"Normalization error: {e}")
#             return pd.DataFrame()

#     if _progress_bar:
#         _progress_bar.progress(0.9)

#     # Nettoyage final
#     intensity_data = np.nan_to_num(intensity_data, nan=0.0, posinf=0.0, neginf=0.0)
#     intensity_df = pd.DataFrame(intensity_data, columns=intensity_cols, dtype=np.float32)
#     del intensity_data
#     gc.collect()

#     if _progress_bar:
#         _progress_bar.progress(1.0)

#     return pd.concat([fixed_data, intensity_df], axis=1)

# def quantile_normalization(data):
#     sorted_vals = np.sort(data, axis=0)
#     mean_vals = np.mean(sorted_vals, axis=1).astype(np.float32)

#     ranks = np.apply_along_axis(rankdata, 0, data).astype(int) - 1
#     norm_data = np.zeros_like(data, dtype=np.float32)

#     for i in range(data.shape[1]):
#         norm_data[:, i] = mean_vals[ranks[:, i]]

#     del sorted_vals, mean_vals, ranks
#     gc.collect()

#     return norm_data

# def load_and_preprocess_data(mzml_dirs, class_names, normalization_type, apply_binning_option, bin_width, mass_range):
#     try:
#         data = load_data(mzml_dirs, class_names)

#         if apply_binning_option:
#             data = apply_binning_to_mass_range(data, bin_width, mass_range)

#         if normalization_type and normalization_type != 'None':
#             data = preprocess_data(data, normalization_type)

#         return data
#     except Exception as e:
#         st.error(f"An error occurred during data loading and preprocessing: {e}")
#         return None
    
# def apply_binning_to_mass_range(data, bin_width=0.1, mass_range=(600, 1000)):
#     if bin_width <= 0:
#         raise ValueError("bin_width must be greater than zero.")

#     min_mass, max_mass = mass_range
#     if min_mass >= max_mass:
#         raise ValueError("Invalid mass range.")

#     fixed_cols = ['Class', 'File', 'RT', 'Sum']
#     if not all(col in data.columns for col in fixed_cols):
#         raise ValueError("Missing one or more fixed columns.")

#     bins = np.arange(min_mass, max_mass + bin_width, bin_width)
#     bin_centers = ((bins[:-1] + bins[1:]) / 2).round(3)
#     binned_array = np.zeros((len(data), len(bin_centers)), dtype=np.float32)

#     numerical_cols = [col for col in data.columns if col not in fixed_cols]

#     for col in numerical_cols:
#         try:
#             mass = float(col)
#             bin_idx = np.digitize(mass, bins) - 1
#             if 0 <= bin_idx < len(bin_centers):
#                 binned_array[:, bin_idx] += data[col].astype(np.float32).values
#         except ValueError:
#             continue  # skip non-numeric columns

#     binned_df = pd.DataFrame(binned_array, columns=bin_centers, index=data.index)

#     del binned_array
#     gc.collect()

#     return pd.concat([data[fixed_cols].reset_index(drop=True), binned_df.reset_index(drop=True)], axis=1)


def preprocess_data(all_data, normalization_type=None, _progress_bar=None):
    st.info("Applying Normalization...")

    if all_data.empty:
        st.error("DataFrame is empty.")
        return pd.DataFrame()

    if _progress_bar:
        _progress_bar.progress(0.1)

    # Colonnes fixes à garder
    fixed_columns = ['Class', 'File', 'RT', 'Sum']
    if not all(col in all_data.columns for col in fixed_columns):
        st.error("Missing one or more required columns.")
        return pd.DataFrame()

    fixed_data = all_data[fixed_columns].reset_index(drop=True)
    intensity_df = all_data.drop(columns=fixed_columns)
    intensity_cols = intensity_df.columns
    intensity_data = intensity_df.fillna(0).astype(np.float32).values

    if _progress_bar:
        _progress_bar.progress(0.3)


    if normalization_type:
        try:
            if normalization_type == 'Log1p':
                intensity_data = np.log1p(intensity_data)

            elif normalization_type == 'Log10':
                intensity_data = np.log10(intensity_data + 1)

            elif normalization_type == 'Log2':
                intensity_data = np.log2(intensity_data + 1)


            elif normalization_type == 'RMS':
                rms_vals = np.sqrt(np.sum(np.square(intensity_data), axis=1)).reshape(-1, 1)
                intensity_data = intensity_data / np.where(rms_vals == 0, np.nan, rms_vals)

            elif normalization_type == 'BasePeak':
                max_vals = np.max(intensity_data, axis=1).reshape(-1, 1)
                intensity_data = intensity_data / np.where(max_vals == 0, np.nan, max_vals)

            elif normalization_type == 'QNorm':
                intensity_data = quantile_normalization(intensity_data)

            elif normalization_type == 'Median of Ratios (Deseq2-like)':
                intensity_data = median_of_ratios_norm(intensity_data)

            elif normalization_type == 'TMM (Deseq2-like)':
                intensity_data = tmm_norm(intensity_data)

            elif normalization_type == 'CPM':
                intensity_data = cpm_norm(intensity_data, log=False)

            elif normalization_type == 'logCPM':
                intensity_data = cpm_norm(intensity_data, log=True)

            elif normalization_type == 'VST':
                intensity_data = vst_norm(intensity_data)

            elif normalization_type == 'Median':
                intensity_data = median_norm(intensity_data)

            elif normalization_type == 'Mean':
                intensity_data = mean_norm(intensity_data)

            elif normalization_type == 'Total Intensity':
                intensity_data = total_intensity_norm(intensity_data)

        except Exception as e:
            st.error(f"Normalization error: {e}")
            return pd.DataFrame()


    if _progress_bar:
        _progress_bar.progress(0.9)

    # Nettoyage final
    intensity_data = np.nan_to_num(intensity_data, nan=0.0, posinf=0.0, neginf=0.0)
    intensity_df = pd.DataFrame(intensity_data, columns=intensity_cols, dtype=np.float32)

    # Libération mémoire
    del intensity_data
    gc.collect()

    if _progress_bar:
        _progress_bar.progress(1.0)

    return pd.concat([fixed_data, intensity_df], axis=1)

def quantile_normalization(data):
    sorted_vals = np.sort(data, axis=0)
    mean_vals = np.mean(sorted_vals, axis=1).astype(np.float32)

    ranks = np.apply_along_axis(rankdata, 0, data).astype(int) - 1
    norm_data = np.zeros_like(data, dtype=np.float32)

    for i in range(data.shape[1]):
        norm_data[:, i] = mean_vals[ranks[:, i]]

    del sorted_vals, mean_vals, ranks
    gc.collect()

    return norm_data

def load_and_preprocess_data(mzml_dirs, class_names, normalization_type, apply_binning_option, bin_width, mass_range):
    try:
        data = load_data(mzml_dirs, class_names)

        if apply_binning_option:
            data = apply_binning_to_mass_range(data, bin_width, mass_range)

        if normalization_type and normalization_type != 'None':
            data = preprocess_data(data, normalization_type)

        return data
    except Exception as e:
        st.error(f"An error occurred during data loading and preprocessing: {e}")
        return None
    
def apply_binning_to_mass_range(data, bin_width=0.1, mass_range=(600, 1000)):
    if bin_width <= 0:
        raise ValueError("bin_width must be greater than zero.")

    min_mass, max_mass = mass_range
    if min_mass >= max_mass:
        raise ValueError("Invalid mass range.")

    fixed_cols = ['Class', 'File', 'RT', 'Sum']
    if not all(col in data.columns for col in fixed_cols):
        raise ValueError("Missing one or more fixed columns.")

    bins = np.arange(min_mass, max_mass + bin_width, bin_width)
    bin_centers = ((bins[:-1] + bins[1:]) / 2).round(3)
    binned_array = np.zeros((len(data), len(bin_centers)), dtype=np.float32)

    numerical_cols = [col for col in data.columns if col not in fixed_cols]

    for col in numerical_cols:
        try:
            mass = float(col)
            bin_idx = np.digitize(mass, bins) - 1
            if 0 <= bin_idx < len(bin_centers):
                binned_array[:, bin_idx] += data[col].astype(np.float32).values
        except ValueError:
            continue  # skip non-numeric columns

    binned_df = pd.DataFrame(binned_array, columns=bin_centers, index=data.index)

    del binned_array
    gc.collect()

    return pd.concat([data[fixed_cols].reset_index(drop=True), binned_df.reset_index(drop=True)], axis=1)



def median_of_ratios_norm(X):
    # X shape = (samples, features)
    X = X.astype(np.float32) + 1.0

    geo_means = np.exp(np.mean(np.log(X), axis=0))
    geo_means[geo_means == 0] = np.nan

    ratios = X / geo_means
    size_factors = np.nanmedian(ratios, axis=1)
    size_factors[size_factors == 0] = 1.0

    return X / size_factors[:, None]



def tmm_norm(X):
    X = X.astype(np.float32) + 1.0

    lib_sizes = X.sum(axis=1)
    ref_idx = np.argsort(lib_sizes)[len(lib_sizes) // 2]
    ref = X[ref_idx]

    factors = np.ones(X.shape[0], dtype=np.float32)

    for i in range(X.shape[0]):
        logR = np.log2(X[i] / ref)
        logA = 0.5 * np.log2(X[i] * ref)

        mask = np.isfinite(logR) & np.isfinite(logA)
        logR = logR[mask]
        logA = logA[mask]

        if len(logR) == 0:
            continue

        lo, hi = np.percentile(logA, [30, 70])
        keep = (logA > lo) & (logA < hi)

        if keep.sum() > 0:
            factors[i] = 2 ** np.median(logR[keep])

    factors[factors == 0] = 1.0
    return X / factors[:, None]



def cpm_norm(X, log=True):
    lib_sizes = X.sum(axis=1).astype(np.float32)
    lib_sizes[lib_sizes == 0] = np.nan

    cpm = (X / lib_sizes[:, None]) * 1e6
    if log:
        cpm = np.log2(cpm + 1)

    return cpm


def vst_norm(X):
    X = X.astype(np.float32)
    mean = np.mean(X, axis=0)
    var = np.var(X, axis=0)

    a = np.maximum(var - mean, 1e-6)
    return np.log2(X + np.sqrt(X * a) + a)

def median_norm(X):
    med = np.median(X, axis=1)
    med[med == 0] = 1.0
    return X / med[:, None]


def mean_norm(X):
    mean = np.mean(X, axis=1)
    mean[mean == 0] = 1.0
    return X / mean[:, None]


def total_intensity_norm(X):
    total = np.sum(X, axis=1)
    total[total == 0] = 1.0
    return X / total[:, None]
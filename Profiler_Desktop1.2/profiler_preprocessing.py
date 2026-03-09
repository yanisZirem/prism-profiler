import pandas as pd
import numpy as np
import streamlit as st
from scipy.stats import rankdata
import dask.dataframe as dd

import numexpr as ne
from pandarallel import pandarallel
import gc
# Initialiser pandarallel
# ── pandarallel: 3 workers max — 10 workers × N users = CPU saturation ───────
pandarallel.initialize(progress_bar=False, nb_workers=3)


def preprocess_data(all_data, normalization_type=None, _progress_bar=None):
    st.info("Applying Normalization...")

    if all_data.empty:
        st.error("DataFrame is empty.")
        return pd.DataFrame()

    if _progress_bar:
        _progress_bar.progress(0.1)

    # ── Preserve ALL non-numeric columns: hard metadata + _meta cols + any categorical ──
    _NON_FEAT_BASE = {'Class', 'File', 'RT', 'Sum', 'ID', 'Original_index'}
    fixed_columns = [
        c for c in all_data.columns
        if c in _NON_FEAT_BASE
        or str(c).endswith('_meta')
        or not pd.api.types.is_numeric_dtype(all_data[c])
    ]
    if not {'Class'}.issubset(all_data.columns):
        st.error("Missing required column: 'Class'.")
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



########Paralellization dask ########
import numexpr as ne
import dask.dataframe as dd

ne.set_num_threads(4)  # aligned with BLAS thread limit in Profiler.py

def preprocess_data_dask(ddf, normalization_type=None, _progress_bar=None):
    if _progress_bar:
        _progress_bar.progress(60)

    # Convertir les colonnes d’intensité
    columns_to_exclude = ['Class', 'File', 'RT', 'Sum']
    intensity_columns = [col for col in ddf.columns if col not in columns_to_exclude]

    if normalization_type:
        if normalization_type == 'Log Normalization':
            ddf[intensity_columns] = ddf[intensity_columns].map_partitions(lambda df: np.log1p(df))
        elif normalization_type == 'Log10':
            ddf[intensity_columns] = ddf[intensity_columns].map_partitions(lambda df: np.log10(df + 1))
        elif normalization_type == 'Log2':
            ddf[intensity_columns] = ddf[intensity_columns].map_partitions(lambda df: np.log2(df + 1))
        elif normalization_type == 'TIC':
            ddf[intensity_columns] = ddf[intensity_columns].div(ddf['Sum'], axis=0)
        elif normalization_type == 'RMS':
            def rms_norm(df):
                rms = np.sqrt((df[intensity_columns] ** 2).sum(axis=1))
                return df[intensity_columns].div(rms, axis=0)
            ddf[intensity_columns] = ddf.map_partitions(rms_norm)
        elif normalization_type == 'BasePeak':
            def basepeak(df):
                max_peak = df[intensity_columns].max(axis=1)
                return df[intensity_columns].div(max_peak, axis=0)
            ddf[intensity_columns] = ddf.map_partitions(basepeak)
        elif normalization_type == 'QNorm':
            # Pas de support direct de QNorm avec Dask, fallback Pandas
            ddf = ddf.compute()
            ddf[intensity_columns] = quantile_normalization(ddf[intensity_columns])
            ddf = dd.from_pandas(ddf, npartitions=10)

    # Remplacer inf/NaN
    ddf = ddf.replace([np.inf, -np.inf], np.nan).fillna(0)

    if _progress_bar:
        _progress_bar.progress(80)

    return ddf


def apply_binning_to_mass_range_dask(ddf, bin_width=0.1, mass_range=(600, 1000)):
    min_mass, max_mass = mass_range
    bins = np.arange(min_mass, max_mass, bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_labels = [round(center, 3) for center in bin_centers]

    columns_to_exclude = ['Class', 'File', 'RT', 'Sum']
    numeric_cols = [col for col in ddf.columns if col not in columns_to_exclude]

    def bin_partition(df):
        binned = pd.DataFrame(0, index=df.index, columns=bin_labels)
        for col in numeric_cols:
            mass = float(col)
            bin_idx = np.digitize(mass, bins) - 1
            if 0 <= bin_idx < len(bin_centers):
                binned[bin_labels[bin_idx]] += df[col]
        return pd.concat([df[columns_to_exclude], binned], axis=1)

    return ddf.map_partitions(bin_partition)
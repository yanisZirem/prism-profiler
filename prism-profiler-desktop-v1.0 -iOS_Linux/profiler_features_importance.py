"""
Software Name: Profiler
Module Name: Features importance
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

# --- Standard library ---
import io
import gc
from itertools import combinations

# --- Scientific computing ---
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, f_oneway
from scipy.signal import find_peaks
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed

# --- Machine learning ---
import tensorflow as tf
import shap
import eli5
from eli5 import explain_prediction, explain_weights
from eli5.formatters import format_as_dataframe

from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.linear_model import (
    SGDClassifier, LogisticRegression, RidgeClassifier,
    PassiveAggressiveClassifier, Perceptron, Lasso
)
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
)
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, BaggingClassifier,
    ExtraTreesClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier, StackingClassifier, VotingClassifier
)

# --- Visualization ---
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from matplotlib.colors import LinearSegmentedColormap
from statannotations.Annotator import Annotator

# --- Web and parsing ---
from bs4 import BeautifulSoup

# --- Streamlit app ---
import streamlit as st


def eli5_feature_importance(model, label_encoder, data, top_features=50):
    y = data['Class']

    # Remove unwanted columns
    cols_to_drop = [col for col in ['Class', 'File', 'RT', 'Sum']
                    if col in data.columns[:4] or col in data.columns[-4:]]
    X = data.drop(columns=cols_to_drop, errors='ignore')

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # Extract the estimator if inside a pipeline
    if hasattr(model, 'named_steps'):
        fitted_model = model.named_steps[model.steps[-1][0]]
    else:
        fitted_model = model

    # Get feature and target names
    feature_names = [str(col) for col in X.columns]
    target_names = list(label_encoder.classes_)

    # Explain the weights
    explanation = explain_weights(
        fitted_model,
        feature_names=feature_names,
        top=top_features,
        target_names=target_names
    )

    html_contribution = eli5.format_as_html(explanation)
    df_contribution = format_as_dataframe(explanation)

    # Clean up the HTML output
    html_contribution = html_contribution.split('<div class="caveats">')[0] + '</div>'

    return html_contribution, df_contribution


def st_shap(plot, height=None):
    import streamlit.components.v1 as components
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height or 400)

def plot_shap_values(model, X, class_colors, class_names): 


    # Handle NaNs
    X = X.fillna(0)

    # Extract model
    if hasattr(model, 'named_steps'):
        fitted_model = model.named_steps['model']
        if 'preprocessing' in model.named_steps:
            X_transformed = model.named_steps['preprocessing'].transform(X)
        else:
            X_transformed = X
    else:
        fitted_model = model
        X_transformed = X

    # Unsupported
    unsupported_models = (
        "AdaBoostClassifier", "BaggingClassifier", "SVC", "NuSVC", "LinearSVC",
        "GaussianNB", "BernoulliNB", "DummyClassifier", "NearestCentroid",
        "KNeighborsClassifier", "QuadraticDiscriminantAnalysis"
    )
    if type(fitted_model).__name__ in unsupported_models:
        st.error(f"SHAP not supported for: {type(fitted_model).__name__}")
        return

    # Explainer
    if isinstance(fitted_model, (
        RandomForestClassifier, ExtraTreesClassifier, DecisionTreeClassifier,
        ExtraTreeClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
    )):
        explainer = shap.TreeExplainer(fitted_model)
    elif isinstance(fitted_model, LGBMClassifier):
        explainer = shap.TreeExplainer(fitted_model, data=X_transformed)
    elif isinstance(fitted_model, (
        LogisticRegression, RidgeClassifier, SGDClassifier, Perceptron,
        PassiveAggressiveClassifier, Lasso,LinearDiscriminantAnalysis
    )):
        explainer = shap.LinearExplainer(fitted_model, X_transformed)
    elif isinstance(fitted_model, MLPClassifier) or "tensorflow" in str(type(fitted_model)).lower():
        explainer = shap.DeepExplainer(fitted_model, np.array(X_transformed))
    else:
        st.warning("Model not supported for SHAP.")
        return

    # SHAP values
    shap_values = explainer.shap_values(X_transformed)

    # Select class
    if isinstance(shap_values, list) and len(shap_values) > 1:
        class_index = 0 
        shap_values_class = shap_values[class_index]

    elif isinstance(shap_values, list):
        class_index = 0
        shap_values_class = shap_values[0]
    else:
        class_index = None
        shap_values_class = shap_values

    feature_names = X.columns if hasattr(X, "columns") else [f"Feature {i}" for i in range(X.shape[1])]

    # Beeswarm
    st.markdown("**SHAP Beeswarm Plot**")
    fig1, ax1 = plt.subplots()
    shap.summary_plot(shap_values_class, X_transformed, feature_names=feature_names, plot_type="dot", show=False)
    st.pyplot(fig1)
    plt.clf()

    # Bar plot
    st.markdown("**SHAP Bar Plot**")
    fig2, ax2 = plt.subplots()
    shap.summary_plot(shap_values_class, X_transformed, feature_names=feature_names, plot_type="bar", show=False)
    st.pyplot(fig2)
    plt.clf()





def boxplot_significant_features(data, mz_values, class_colors=None, test='Kruskal', loc='inside', show_scatter=False, use_log2=False):
    data.columns = data.columns.astype(str)
    label = 'Class'
    order = sorted(data[label].unique())
    box_pairs = list(combinations(order, 2))

    custom_palette = {class_label: class_colors.get(class_label, 'blue') for class_label in order}

    num_mz_values = len(mz_values)
    num_cols = int(np.ceil(np.sqrt(num_mz_values)))
    num_rows = int(np.ceil(num_mz_values / num_cols))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 5 * num_rows), dpi=200, squeeze=False)

    progress_bar = st.progress(0)
    progress_step = 1.0 / num_mz_values

    for i, mz in enumerate(mz_values):
        row, col = divmod(i, num_cols)
        ax = axes[row, col]

        y_data = np.log2(data[mz]) if use_log2 else data[mz]
        y_label = f'{mz}'

        sns.boxplot(data=data, x=label, y=y_data, order=order, ax=ax, palette=custom_palette, hue=label, legend=False)
        ylabel = "log2(Intensity)" if use_log2 else "Intensity"
        ax.set_ylabel(ylabel)

        if show_scatter:
            sns.swarmplot(data=data, x=label, y=y_data, order=order, ax=ax, color=".25")

        unique_values = y_data.unique()
        if len(unique_values) > 1:
            groups = [y_data[data[label] == cls] for cls in order]
            try:
                stat, pvalue = f_oneway(*groups)
            except Exception as e:
                st.warning(f"ANOVA failed for feature {mz}: {e}")
                stat, pvalue = None, None

            if pvalue is not None:
                annotation = f"ANOVA p={pvalue:.3e}"
                ymax = y_data.max()
                y_offset = (y_data.max() - y_data.min()) * 0.1
                ax.text(0.5, ymax + y_offset, annotation, ha='center', va='bottom', fontsize=12, color='red', transform=ax.get_xaxis_transform())

            pairwise_test = 't-test_ind' if test == 'ANOVA' else test

            annotator = Annotator(ax, box_pairs, data=data, x=label, y=y_data, order=order)
            annotator.configure(test=pairwise_test, text_format='star', loc=loc, verbose=0)
            results = annotator.apply_and_annotate()

            if results and isinstance(results, list):
                for box_pair, stat_result in results:
                    if isinstance(stat_result, dict) and 'pvalue_text' in stat_result:
                        if "ns" in stat_result['pvalue_text'].lower():
                            stat_result['pvalue_text'] = f'ns ({stat_result["pvalue"]:.3f})'
        else:
            st.warning(f"All values for feature {mz} are identical. Skipping statistical test.")

        ax.set_xticklabels(order, fontsize=12)
        ax.set_title(y_label, fontsize=20)

        progress_bar.progress((i + 1) * progress_step)

    for i in range(num_mz_values, num_rows * num_cols):
        fig.delaxes(axes.flatten()[i])

    plt.tight_layout()
    st.pyplot(fig,dpi=300)



def violinplot_significant_features(data, mz_values, class_colors=None, test='Kruskal', loc='inside', show_scatter=False, use_log2=False):
    data.columns = data.columns.astype(str)
    label = 'Class'
    order = sorted(data[label].unique())
    box_pairs = list(combinations(order, 2))

    custom_palette = {class_label: class_colors.get(class_label, 'blue') for class_label in order}

    num_mz_values = len(mz_values)
    num_cols = int(np.ceil(np.sqrt(num_mz_values)))
    num_rows = int(np.ceil(num_mz_values / num_cols))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 5 * num_rows), dpi=200, squeeze=False)

    progress_bar = st.progress(0)
    progress_step = 1.0 / num_mz_values

    for i, mz in enumerate(mz_values):
        row, col = divmod(i, num_cols)
        ax = axes[row, col]

        y_data = np.log2(data[mz]) if use_log2 else data[mz]
        y_label = f'{mz}'

        sns.violinplot(data=data, x=label, y=y_data, order=order, ax=ax, palette=custom_palette, hue=label, legend=False)
        ylabel = "log2(Intensity)" if use_log2 else "Intensity"
        ax.set_ylabel(ylabel)
        if show_scatter:
            sns.swarmplot(data=data, x=label, y=y_data, order=order, ax=ax, color=".25")

        unique_values = y_data.unique()
        if len(unique_values) > 1:
            annotator = Annotator(ax, box_pairs, data=data, x=label, y=y_data, order=order)
            annotator.configure(test=test, text_format='star', loc=loc, verbose=0)
            results = annotator.apply_and_annotate()
            if results and isinstance(results, list):
                for box_pair, stat_result in results:
                    st.text(stat_result)  #  Affiche ce qui est retourné
                    if isinstance(stat_result, dict) and 'pvalue_text' in stat_result:
                        if "ns" in stat_result['pvalue_text'].lower():  # Vérifie si "ns" est dedans
                            stat_result['pvalue_text'] = f'ns ({stat_result["pvalue"]:.3f})'
        else:
            st.warning(f"All values for feature {mz} are identical. Skipping statistical test.")

        ax.set_xticklabels(order, fontsize=12)
        ax.set_title(y_label, fontsize=20)

        progress_bar.progress((i + 1) * progress_step)

    for i in range(num_mz_values, num_rows * num_cols):
        fig.delaxes(axes.flatten()[i])

    plt.tight_layout()
    st.pyplot(fig,dpi=300)

def barplot_significant_features(data, mz_values, class_colors=None, test='Kruskal', loc='inside', show_scatter=False, use_log2=False):
    data.columns = data.columns.astype(str)
    label = 'Class'
    order = sorted(data[label].unique())
    box_pairs = list(combinations(order, 2))

    custom_palette = {class_label: class_colors.get(class_label, 'blue') for class_label in order}

    num_mz_values = len(mz_values)
    num_cols = int(np.ceil(np.sqrt(num_mz_values)))
    num_rows = int(np.ceil(num_mz_values / num_cols))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 5 * num_rows), dpi=200, squeeze=False)

    progress_bar = st.progress(0)
    progress_step = 1.0 / num_mz_values

    for i, mz in enumerate(mz_values):
        row, col = divmod(i, num_cols)
        ax = axes[row, col]

        y_data = np.log2(data[mz]) if use_log2 else data[mz]
        y_label = f'{mz}'

        sns.barplot(data=data, x=label, y=y_data, order=order, ax=ax, palette=custom_palette, hue=label, legend=False)
        ylabel = "log2(Intensity)" if use_log2 else "Intensity"
        ax.set_ylabel(ylabel)
        if show_scatter:
            sns.swarmplot(data=data, x=label, y=y_data, order=order, ax=ax, color=".25")

        unique_values = y_data.unique()
        if len(unique_values) > 1:
            annotator = Annotator(ax, box_pairs, data=data, x=label, y=y_data, order=order)
            annotator.configure(test=test, text_format='star', loc=loc, verbose=0)
            results = annotator.apply_and_annotate()
            if results and isinstance(results, list):
                for box_pair, stat_result in results:
                    st.text(stat_result)  # DEBUG: Affiche ce qui est retourné
                    if isinstance(stat_result, dict) and 'pvalue_text' in stat_result:
                        if "ns" in stat_result['pvalue_text'].lower():  # Vérifie si "ns" est dedans
                            stat_result['pvalue_text'] = f'ns ({stat_result["pvalue"]:.3f})'
        else:
            st.warning(f"All values for feature {mz} are identical. Skipping statistical test.")

        ax.set_xticklabels(order, fontsize=12)
        ax.set_title(y_label, fontsize=20)

        progress_bar.progress((i + 1) * progress_step)

    for i in range(num_mz_values, num_rows * num_cols):
        fig.delaxes(axes.flatten()[i])

    plt.tight_layout()
    st.pyplot(fig,dpi=300)





def eli5_format_to_dataframe(eli5_html):
    """
    Convert an ELI5/LIME HTML table into a Pandas DataFrame, handling multi-row headers and duplicate column names.
    """
    if hasattr(eli5_html, "data"):  
        eli5_html = eli5_html.data  

    soup = BeautifulSoup(eli5_html, "html.parser")  
    table = soup.find("table")

    if not table:
        return pd.DataFrame()  

    header_rows = table.find_all("tr")[:2]  # First two rows are headers
    headers = [[th.get_text(strip=True) for th in row.find_all("th")] for row in header_rows]

    # Fill missing headers (if any)
    max_columns = max(len(h) for h in headers)
    headers = [row + [""] * (max_columns - len(row)) for row in headers]  

    # Ensure uniqueness by adding an index if needed
    combined_headers = []
    seen_headers = {}
    for i, col_group in enumerate(zip(*headers)):  
        main_header, sub_header = col_group
        main_header = main_header or f"Category_{i}"  # Assign unique names if empty

        column_name = f"{main_header} - {sub_header}".strip()

        # Ensure uniqueness by adding a counter if needed
        if column_name in seen_headers:
            seen_headers[column_name] += 1
            column_name += f"_{seen_headers[column_name]}"
        else:
            seen_headers[column_name] = 1

        combined_headers.append(column_name)

    # Extract data rows
    rows = []
    for tr in table.find_all("tr")[2:]:  # Skip header rows
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        rows.append(cells)

    # Normalize row lengths
    rows = [row + [None] * (max_columns - len(row)) for row in rows]

    # Create DataFrame with unique column names
    df = pd.DataFrame(rows, columns=combined_headers)

    return df


def calculate_volcano_data(
    data, class_column, features, 
    p_value_threshold=0.05, correction_method="fdr_bh"
):
    classes = data[class_column].unique()
    results = []

    grouped = {c: data.loc[data[class_column] == c, features] for c in classes}

    for i, control_class in enumerate(classes):
        for test_class in classes[i+1:]:
            control_data = grouped[control_class]
            test_data = grouped[test_class]

            t_stats, p_values = ttest_ind(
                control_data, test_data,
                axis=0, equal_var=False, nan_policy='omit'
            )

            # --- Multiple testing correction ---
            if correction_method and correction_method.lower() != "none":
                try:
                    reject, p_adj, _, _ = multipletests(
                        p_values, method=correction_method
                    )
                    p_values = p_adj
                except Exception as e:
                    print(f"Correction failed ({correction_method}): {e}")

            mean_control = np.nanmean(control_data, axis=0)
            mean_test = np.nanmean(test_data, axis=0)
            fold_change = np.divide(
                mean_control, mean_test,
                out=np.full_like(mean_control, np.nan), where=mean_test!=0
            )

            regulation = np.where(
                p_values < p_value_threshold,
                np.where(fold_change > 1, 'Upregulated',
                         np.where(fold_change < 1, 'Downregulated', 'Non-Significant')),
                'Non-Significant'
            )

            df = pd.DataFrame({
                'Feature': features,
                'Comparison': f"{control_class} vs {test_class}",
                'Fold Change': fold_change,
                'P-Value': p_values,
                'Regulation Type': regulation
            })
            results.append(df)

    df_all = pd.concat(results, ignore_index=True)
    df_all['Log2 Fold Change'] = np.log2(np.nan_to_num(df_all['Fold Change'], nan=np.nan, posinf=np.nan, neginf=np.nan))
    df_all['-Log10 P-Value'] = -np.log10(df_all['P-Value'].replace(0, np.nan))

    if len(classes) == 2:
        df_all['Color_Group'] = np.where(
            df_all['P-Value'] >= p_value_threshold, 'Non-Significant',
            np.where(df_all['Log2 Fold Change'] > 0, 'Upregulated', 'Downregulated')
        )
    else:
        df_all['Color_Group'] = np.where(df_all['P-Value'] >= p_value_threshold, 'Non-Significant', df_all['Comparison'])

    return df_all



def plot_volcano(volcano_data, highlight_features=True, p_value_threshold=0.05, fold_change_threshold=2.0):
    color_map = {
        'Upregulated': 'red',
        'Downregulated': 'blue',
        'Non-Significant': 'gray'
    }

    # Create a unique color for each comparison if there are multiple classes
    if 'Upregulated' not in volcano_data['Color_Group'].values:
        unique_comparisons = volcano_data['Comparison'].unique()
        color_map.update({
            comp: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
            for i, comp in enumerate(unique_comparisons)
        })

    fig = px.scatter(
        volcano_data, 
        x='Log2 Fold Change', 
        y='-Log10 P-Value',
        text='Feature' if highlight_features else None,
        color='Color_Group',
        title='Volcano Plot',
        hover_data=['Feature', 'Comparison', 'Regulation Type'],
        labels={
            'Log2 Fold Change': 'Log2 Fold Change',
            '-Log10 P-Value': '-Log10 P-Value'
        },
        color_discrete_map=color_map
    )

    # Customize points
    fig.update_traces(marker=dict(size=10, line=dict(width=0.5, color='black')))
    
    # Add significance cutoffs
    fig.add_hline(
        y=-np.log10(p_value_threshold),
        line_dash="dash", 
        annotation_text=f"p={p_value_threshold}",
        annotation_position="top left",
        annotation_font=dict(color="black", size=14)
    )
    fig.add_vline(
        x=fold_change_threshold, 
        line_dash="dash", 
        annotation_text=f"{fold_change_threshold}-fold",
        annotation_position="top right",
        annotation_font=dict(color="black", size=14)
    )
    fig.add_vline(
        x=-fold_change_threshold, 
        line_dash="dash", 
        annotation_text=f"-{fold_change_threshold}-fold",
        annotation_position="top left",
        annotation_font=dict(color="black", size=14)
    )

    # Adjust text labels for features
    if highlight_features:
        fig.update_traces(textposition='top center', textfont=dict(color='black', size=12))

    # Global layout: fonts and sizes
    # fig.update_layout(
    #     title=dict(font=dict(size=22, color="black")),
    #     font=dict(size=18, color="black"),  
    #     legend=dict(font=dict(size=16, color="black")),
    #     xaxis=dict(title_font=dict(size=20, color="black"), tickfont=dict(size=16, color="black")),
    #     yaxis=dict(title_font=dict(size=20, color="black"), tickfont=dict(size=16, color="black")),
    #     plot_bgcolor="white"
    # )
    fig.update_layout(
        title=dict(font=dict(size=24, color="black")),
        font=dict(size=20, color="black"),
        legend=dict(font=dict(size=18, color="black")),
        xaxis=dict(title_font=dict(size=22, color="black"), tickfont=dict(size=18, color="black")),
        yaxis=dict(title_font=dict(size=22, color="black"), tickfont=dict(size=18, color="black")),
        plot_bgcolor="white",
        width=1200,  # Taille en pixels pour un rendu plus net
        height=800,
    )
    return fig




def detect_peaks(data, intensity_threshold, show_stats=True):
    """
    Detects features (columns) in the dataset where peaks are present above a given intensity threshold.

    Parameters:
        data (pd.DataFrame): Input data with features as columns.
        intensity_threshold (float): Minimum intensity required to consider a peak.
        show_stats (bool): Whether to print min/max intensity info for each feature.

    Returns:
        list: Names of features with at least one detected peak above the threshold.
    """
    peak_features = []

    excluded_cols = {'Class', 'File', 'RT', 'Sum'}

    for col in data.columns:
        if col not in excluded_cols:
            try:
                intensities = data[col].astype(float).values
                peaks, properties = find_peaks(intensities, height=intensity_threshold)

                if show_stats:
                    min_intensity = np.min(intensities)
                    max_intensity = np.max(intensities)
                    print(f"{col}: Min={min_intensity:.4f}, Max={max_intensity:.4f}, Peaks={len(peaks)}")

                if len(peaks) > 0:
                    peak_features.append(col)

            except Exception as e:
                print(f"Skipping column {col} due to error: {e}")

    return peak_features






def plot_heatmap_samples(data, class_colors, selected_features, custom_colors, show_sample_names=True):

    data.columns = data.columns.astype(str)

    # Vérification que toutes les features sélectionnées sont présentes
    missing_features = [feature for feature in selected_features if feature not in data.columns]
    if missing_features:
        st.error(f"Invalid features: {', '.join(missing_features)}")
        return

    features = selected_features
    data.loc[:, features] = data[features].replace([np.inf, -np.inf], np.nan)
    data[features] = data[features].fillna(data[features].mean())

    scaler = StandardScaler()
    data[features] = scaler.fit_transform(data[features])

    if data[features].isnull().values.any() or np.isinf(data[features].values).any():
        st.error("Missing values after preprocessing.")
        return

    # --- Colormap vibrant ---
    cmap = LinearSegmentedColormap.from_list("custom_cmap", custom_colors)
    # Optionnel : renforcer le contraste
    vmin, vmax = data[features].min().min(), data[features].max().max()

    num_samples = data.shape[0]
    num_features = len(features)
    sample_labels = data['Class'].values

    fig_width = min(max(num_samples * 0.2, 10), 25)
    fig_height = min(max(num_features * 0.2, 10), 25)
    fontsize = max(10, 20 - max(num_samples, num_features) // 10)

    plt.figure(figsize=(fig_width, fig_height))
    g = sns.clustermap(
        data[features].T,
        cmap=cmap,
        square=True,
        col_cluster=True,
        row_cluster=True,
        center=0,
        z_score=None,  # déjà standardisé
        col_colors=data['Class'].map(class_colors),
        cbar_kws={'shrink': .8},
        yticklabels=False,
        xticklabels=False,
        vmin=vmin,
        vmax=vmax
    )

    # Noms des échantillons
    if show_sample_names and num_samples <= 15:
        col_order = g.dendrogram_col.reordered_ind
        ordered_labels = data['Class'].iloc[col_order].values
        g.ax_heatmap.set_xticks(np.arange(len(ordered_labels)) + 0.5)
        g.ax_heatmap.set_xticklabels(ordered_labels, rotation=90, fontsize=fontsize)

    # Noms des features
    if num_features <= 15:
        row_labels = g.ax_heatmap.get_yticklabels()
        g.ax_heatmap.set_yticklabels(row_labels, fontsize=fontsize)
    else:
        g.ax_heatmap.set_yticklabels([])

    st.pyplot(g.fig)

    # Export PNG
    buf = io.BytesIO()
    g.fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)  # dpi élevé pour netteté
    st.download_button("📥 Download Heatmap as PNG", buf.getvalue(), "heatmap.png")




def plot_significant_features(
    data, 
    mz_values, 
    class_colors=None, 
    test='Kruskal', 
    loc='inside',
    show_scatter=False, 
    use_log2=False, 
    plot_type='box', 
    pval_correction=None,
    significance_dict=None   
):

    data = data.copy()
    significance_dict = significance_dict or {} 

    data.columns = data.columns.astype(str)
    label = 'Class'
    order = sorted(data[label].unique())
    box_pairs = list(combinations(order, 2))

    palette = {cls: class_colors.get(cls, 'blue') for cls in order} if class_colors else None

    num_mz = len(mz_values)
    num_cols = int(np.ceil(np.sqrt(num_mz)))
    num_rows = int(np.ceil(num_mz / num_cols))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 5 * num_rows), dpi=200, squeeze=False)
    progress_bar = st.progress(0)
    step = 1.0 / num_mz

    for i, mz in enumerate(mz_values):
        row, col = divmod(i, num_cols)
        ax = axes[row, col]

        try:
            data["__ydata__"] = np.log2(data[mz]) if use_log2 else data[mz]
        except Exception as e:
            st.warning(f"⚠️ Could not apply log2 to feature {mz}: {e}")
            data["__ydata__"] = data[mz]

        ylabel = "log2(Intensity)" if use_log2 else "Intensity"


        if plot_type == 'box':
            sns.boxplot(data=data, x=label, y="__ydata__", order=order, ax=ax, palette=palette)
        elif plot_type == 'violin':
            sns.violinplot(data=data, x=label, y="__ydata__", order=order, ax=ax, palette=palette)
        elif plot_type == 'bar':
            sns.barplot(data=data, x=label, y="__ydata__", order=order, ax=ax, palette=palette)


        if show_scatter:
            sns.swarmplot(data=data, x=label, y="__ydata__", order=order, ax=ax, color=".25")

        ax.set_ylabel(ylabel)

        pval = significance_dict.get(mz, 1.0)
        suffix = "(S)" if pval < 0.05 else "(NS)"
        ax.set_title(f"{mz} {suffix}", fontsize=18)

        ax.set_xticklabels(order, fontsize=16)


        if len(np.unique(data["__ydata__"])) > 1:
            try:
                annotator = Annotator(ax, box_pairs, data=data, x=label, y="__ydata__", order=order)
                annotator.configure(test=test, text_format='star', loc=loc, verbose=0, pvalue_format_string="p = {:.3e}")
                annotator.apply_and_annotate()
            except Exception as e:
                st.warning(f"❌ Statistical annotation failed for {mz}: {e}")
        else:
            st.warning(f"⚠️ All values for {mz} are identical. Skipping annotation.")

        progress_bar.progress((i + 1) * step)


    for i in range(num_mz, num_rows * num_cols):
        fig.delaxes(axes.flatten()[i])

    plt.tight_layout()
    st.pyplot(fig)

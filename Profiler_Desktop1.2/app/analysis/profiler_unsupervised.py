"""
Software Name: Profiler
Module name : Unsupervised Learning
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




import profiler_perf  # noqa: F401 — budget CPU centralisé
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import umap.umap_ as umap
from sklearn.manifold import TSNE

_N_JOBS = profiler_perf.OUTER_JOBS  # budget partagé, laisse de la marge à l'UI Streamlit
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st



def _capture_matplotlib(fig, key: str):
    """Convert a Matplotlib figure to base64 PNG and store for the HTML report."""
    if fig is not None:
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=130)
            buf.seek(0)
            st.session_state[f"_report_{key}"] = ("b64", base64.b64encode(buf.read()).decode())
        except Exception:
            pass




def _capture_plotly(fig, key: str):
    """Store a Plotly figure in session_state for the HTML report."""
    if fig is not None:
        st.session_state[f"_report_{key}"] = ("plotly", fig)


def display_model_results(model_results):
    """Displays the model results."""
    for model_name, result in model_results.items():
        st.write(f"**{model_name}**")
        st.write(f"Accuracy: {result['accuracy']}")
        st.write(f"F1 Score: {result['f1_score']}")
        st.write("---")



def plot_feature_distribution(data, feature, class_colors, capture_name=None):
    if feature == 'Class':
        return px.histogram(data, x='Class', title=f"Distribution of {feature}", color='Class', color_discrete_map=class_colors)
    else:
        return px.histogram(data, x='Class', y=feature, histfunc='sum', title=f"Sum of {feature} Intensities by Class", color='Class', color_discrete_map=class_colors)
    
    if capture_name:
        _capture_plotly(fig, capture_name)
    return fig






import gc



def apply_pca(X, n_components,random_state=1):
    # Gérer les valeurs manquantes
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Appliquer PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_imputed)

    # Loadings
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    explained_variance = pca.explained_variance_ratio_

    return X_pca, (loadings, explained_variance)

















def plot_pca(reduced_data, class_labels, custom_colors=None, feature_intensity=None,
             X=None, capture_name=None, color_by=None, data_orig=None,
             explained_variance=None):
    """
    Parameters
    ----------
    color_by   : None | 'Class' | '<meta_col>'  — column used to colour points
    data_orig  : original DataFrame (needed to pull ID / _meta columns for hover)
    explained_variance : optional array-like of the PCA explained_variance_ratio_
                          (one value per component). When given, axis titles
                          become "PC1 (42.3%)" etc. instead of a bare "PCA1".
    """
    num_components = min(reduced_data.shape[1], 3)
    pca_df = pd.DataFrame(reduced_data, columns=[f'PCA{i+1}' for i in range(num_components)])
    pca_df['Class'] = class_labels.values if hasattr(class_labels, 'values') else class_labels
    pca_df['Index'] = pca_df.index

    if feature_intensity and feature_intensity != 'None' and X is not None and feature_intensity in X.columns:
        pca_df[feature_intensity] = X[feature_intensity].values

    n_samples = pca_df.shape[0]
    marker_size = max(4, 10 - np.log10(n_samples))

    # ── Axis labels with % of variance explained per component ──────────────
    def _pc_label(i):
        if explained_variance is not None and i < len(explained_variance):
            return f"PC{i+1} ({explained_variance[i] * 100:.1f}%)"
        return f"PC{i+1}"

    axis_labels = {f'PCA{i+1}': _pc_label(i) for i in range(num_components)}

    # ── Enrich hover + colour ────────────────────────────────────────────────
    _color_col = color_by if color_by and color_by != 'Class' else 'Class'
    if data_orig is not None:
        pca_df, hover_cols, color_kwargs = _build_hover_and_color(
            pca_df, data_orig, _color_col, custom_colors or {}
        )
    else:
        hover_cols = ['Index']
        color_kwargs = dict(color='Class', color_discrete_map=custom_colors)

    if num_components == 2:
        fig = px.scatter(pca_df, x='PCA1', y='PCA2', hover_data=hover_cols,
                          labels=axis_labels, **color_kwargs)
        if feature_intensity and feature_intensity != 'None':
            fig.update_traces(marker=dict(color=pca_df[feature_intensity],
                                          colorbar=dict(title=feature_intensity, x=-0.5),
                                          colorscale='jet', showscale=True))
        max_val = pca_df[['PCA1', 'PCA2']].abs().max().max()
        buffer = 0.05 * max_val
        fig.update_layout(
            xaxis=dict(range=[-max_val-buffer, max_val+buffer], scaleanchor="y", scaleratio=1),
            yaxis=dict(range=[-max_val-buffer, max_val+buffer], scaleanchor="x", scaleratio=1)
        )
    else:
        fig = px.scatter_3d(pca_df, x='PCA1', y='PCA2', z='PCA3',
                            hover_data=hover_cols, labels=axis_labels, **color_kwargs)
        if feature_intensity and feature_intensity != 'None':
            fig.update_traces(marker=dict(color=pca_df[feature_intensity],
                                          colorbar=dict(title=feature_intensity, x=-0.05),
                                          colorscale='jet', showscale=True))

    fig.update_traces(marker=dict(size=marker_size, opacity=0.8, line=dict(width=0.5, color='black')))
    fig = update_axes_style(fig, num_components=num_components, plot_type="PCA Scatter Plot")
    if capture_name:
        _capture_plotly(fig, capture_name)
    return fig


def plot_tsne(data, num_components=2, custom_colors=None, feature_intensity=None,
              random_state=1, capture_name=None, color_by=None, data_orig=None):
    if 'Class' not in data.columns:
        st.error("Class column not found in data.")
        return None

    num_components = min(num_components, 3)
    n_samples = data.shape[0]
    perplexity = max(2, min(int(np.log2(n_samples)), 100))

    features = data.drop('Class', axis=1)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features.values.astype('float32'))

    tsne = TSNE(n_components=num_components, perplexity=perplexity, random_state=random_state,
                n_jobs=_N_JOBS, method="barnes_hut", n_iter=500 if n_samples > 2000 else 1000)
    tsne_results = tsne.fit_transform(features_scaled)
    del features_scaled; gc.collect()

    df_tsne = pd.DataFrame(tsne_results, columns=[f't-SNE{i+1}' for i in range(num_components)])
    df_tsne['Class'] = data['Class'].values
    df_tsne['Index'] = df_tsne.index

    if feature_intensity and feature_intensity != 'None' and feature_intensity in data.columns:
        df_tsne[feature_intensity] = data[feature_intensity].values

    marker_size = max(4, 10 - np.log10(n_samples))
    cmax_value = data[feature_intensity].max() if feature_intensity and feature_intensity != 'None' else None

    # ── Enrich hover + colour ────────────────────────────────────────────────
    _color_col = color_by if color_by and color_by != 'Class' else 'Class'
    _orig = data_orig if data_orig is not None else data
    df_tsne, hover_cols, color_kwargs = _build_hover_and_color(
        df_tsne, _orig, _color_col, custom_colors or {}
    )

    if num_components == 2:
        fig = px.scatter(df_tsne, x='t-SNE1', y='t-SNE2', hover_data=hover_cols, **color_kwargs)
        if feature_intensity and feature_intensity != 'None':
            fig.update_traces(marker=dict(color=df_tsne[feature_intensity],
                                          colorbar=dict(title=feature_intensity, x=-0.5),
                                          colorscale='jet', showscale=True,
                                          cmin=0, cmax=cmax_value))
    else:
        fig = px.scatter_3d(df_tsne, x='t-SNE1', y='t-SNE2', z='t-SNE3',
                            hover_data=hover_cols, **color_kwargs)
        if feature_intensity and feature_intensity != 'None':
            fig.update_traces(marker=dict(color=df_tsne[feature_intensity],
                                          colorbar=dict(title=feature_intensity, x=-0.05),
                                          colorscale='jet', showscale=True,
                                          cmin=0, cmax=cmax_value))

    fig.update_traces(marker=dict(size=marker_size, opacity=0.8, line=dict(width=0.5, color='black')))
    fig = update_axes_style(fig, num_components=num_components, plot_type="t-SNE Scatter Plot")
    if capture_name:
        _capture_plotly(fig, capture_name)
    return fig


def plot_umap(data, num_components=2, custom_colors=None, feature_intensity=None,
              random_state=1, capture_name=None, color_by=None, data_orig=None):
    if 'Class' not in data.columns:
        st.error("Class column not found in data.")
        return None

    num_components = min(num_components, 3)
    n_samples = data.shape[0]
    n_neighbors = max(2, min(int(np.log2(n_samples)), 100))

    data_features = data.drop('Class', axis=1)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_features.values.astype('float32'))

    reducer = umap.UMAP(
        n_components=num_components, n_neighbors=n_neighbors,
        random_state=random_state, n_jobs=_N_JOBS,
        low_memory=False, verbose=False
    )
    umap_results = reducer.fit_transform(data_scaled)
    del data_scaled; gc.collect()

    df_umap = pd.DataFrame(umap_results, columns=[f'UMAP{i+1}' for i in range(num_components)])
    df_umap['Class'] = data['Class'].values
    df_umap['Index'] = df_umap.index

    if feature_intensity and feature_intensity != 'None' and feature_intensity in data.columns:
        df_umap[feature_intensity] = data[feature_intensity].values

    marker_size = max(4, 10 - np.log10(n_samples))

    # ── Enrich hover + colour ────────────────────────────────────────────────
    _color_col = color_by if color_by and color_by != 'Class' else 'Class'
    _orig = data_orig if data_orig is not None else data
    df_umap, hover_cols, color_kwargs = _build_hover_and_color(
        df_umap, _orig, _color_col, custom_colors or {}
    )

    if num_components == 2:
        fig = px.scatter(df_umap, x='UMAP1', y='UMAP2', hover_data=hover_cols, **color_kwargs)
        if feature_intensity and feature_intensity != 'None':
            fig.update_traces(marker=dict(color=df_umap[feature_intensity],
                                          colorbar=dict(title=feature_intensity, x=-0.5),
                                          colorscale='jet', showscale=True))
        max_range = max(df_umap[['UMAP1', 'UMAP2']].abs().max())
        fig.update_layout(
            xaxis=dict(range=[-max_range, max_range], scaleanchor="y", scaleratio=1),
            yaxis=dict(range=[-max_range, max_range], scaleanchor="x", scaleratio=1)
        )
    else:
        fig = px.scatter_3d(df_umap, x='UMAP1', y='UMAP2', z='UMAP3',
                            hover_data=hover_cols, **color_kwargs)
        if feature_intensity and feature_intensity != 'None':
            fig.update_traces(marker=dict(color=df_umap[feature_intensity],
                                          colorbar=dict(title=feature_intensity, x=-0.05),
                                          colorscale='jet', showscale=True))

    fig.update_traces(marker=dict(size=marker_size, opacity=0.8, line=dict(width=0.5, color='black')))
    fig = update_axes_style(fig, num_components=num_components, plot_type="UMAP Scatter Plot")
    if capture_name:
        _capture_plotly(fig, capture_name)
    return fig


def update_axes_style(fig, num_components=2, plot_type="Scatter Plot"):
    """
    Met à jour le style des axes et du fond pour des plots 2D ou 3D (Plotly >=5).

    Args:
        fig: figure Plotly (px.scatter, px.scatter_3d)
        num_components: 2 ou 3 selon la dimension du plot
        plot_type: titre du type de plot pour le graphique
    """
    # Couleur et taille de base
    title_font_size = 20
    tick_font_size = 16
    font_color = "black"
    
    # Layout commun
    if num_components == 2:
        fig.update_layout(
            title=dict(text=plot_type, font=dict(size=22)),
            font=dict(color=font_color, size=18),
            xaxis=dict(
                title=dict(text=fig.layout.xaxis.title.text if fig.layout.xaxis.title.text else 'X',
                           font=dict(size=title_font_size, color=font_color)),
                tickfont=dict(size=tick_font_size, color=font_color)
            ),
            yaxis=dict(
                title=dict(text=fig.layout.yaxis.title.text if fig.layout.yaxis.title.text else 'Y',
                           font=dict(size=title_font_size, color=font_color)),
                tickfont=dict(size=tick_font_size, color=font_color)
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(
                title=dict(font=dict(size=18, color=font_color)),
                font=dict(size=16, color=font_color)
            )
        )
    else:  # 3D plot
        _sx = fig.layout.scene.xaxis.title.text if fig.layout.scene.xaxis.title.text else 'X'
        _sy = fig.layout.scene.yaxis.title.text if fig.layout.scene.yaxis.title.text else 'Y'
        _sz = fig.layout.scene.zaxis.title.text if fig.layout.scene.zaxis.title.text else 'Z'
        fig.update_layout(autosize=True,
            title=dict(text=plot_type, font=dict(size=22)),
            font=dict(color=font_color, size=18),
            scene=dict(
                xaxis=dict(title=dict(text=_sx, font=dict(size=title_font_size, color=font_color))),
                yaxis=dict(title=dict(text=_sy, font=dict(size=title_font_size, color=font_color))),
                zaxis=dict(title=dict(text=_sz, font=dict(size=title_font_size, color=font_color))),
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(
                title=dict(font=dict(size=18, color=font_color)),
                font=dict(size=16, color=font_color)
            )
        )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: build rich hover + optional meta coloring for scatter plots
# ─────────────────────────────────────────────────────────────────────────────

def _build_hover_and_color(
    df_plot,          # DataFrame already containing embedding coords + 'Class' + 'Index'
    data_orig,        # original dataframe (for meta columns & ID column)
    color_col,        # 'Class' or a _meta column name
    custom_colors,    # class→color dict (used only when color_col == 'Class')
):
    """
    Injects ID and _meta columns into df_plot, builds hover_data list and
    colour args for px.scatter / px.scatter_3d.

    Returns
    -------
    df_plot         : enriched df
    hover_cols      : list of extra columns for hover_data
    color_kwargs    : dict to unpack into px.scatter(…)
    """
    meta_cols = [c for c in data_orig.columns if str(c).endswith('_meta')]

    # ── ID column ────────────────────────────────────────────────────────────
    _ID_ALIASES = {'ID', 'id', 'sample_id', 'SampleID', 'sample', 'name',
                   'Name', 'patient_id', 'PatientID', 'File'}
    id_col = next((c for c in _ID_ALIASES if c in data_orig.columns), None)
    if id_col and id_col not in df_plot.columns:
        df_plot[id_col] = data_orig[id_col].values

    # ── Meta columns ─────────────────────────────────────────────────────────
    for m in meta_cols:
        if m not in df_plot.columns:
            df_plot[m] = data_orig[m].values

    # ── Hover list ───────────────────────────────────────────────────────────
    hover_cols = ['Index']
    if id_col:
        hover_cols.append(id_col)
    hover_cols += [m for m in meta_cols if m not in hover_cols]

    # ── Colour kwargs ─────────────────────────────────────────────────────────
    if color_col == 'Class' or color_col not in df_plot.columns:
        color_kwargs = dict(color='Class', color_discrete_map=custom_colors)
    else:
        # Numeric meta → continuous colorscale; categorical meta → discrete
        col_vals = df_plot[color_col]
        if pd.api.types.is_numeric_dtype(col_vals):
            color_kwargs = dict(
                color=color_col,
                color_continuous_scale='Viridis',
            )
        else:
            # Auto-generate palette for unique values
            unique_vals = col_vals.dropna().unique()
            import plotly.express as _px
            palette = _px.colors.qualitative.Safe
            meta_color_map = {v: palette[i % len(palette)] for i, v in enumerate(sorted(unique_vals))}
            color_kwargs = dict(color=color_col, color_discrete_map=meta_color_map)

    return df_plot, hover_cols, color_kwargs

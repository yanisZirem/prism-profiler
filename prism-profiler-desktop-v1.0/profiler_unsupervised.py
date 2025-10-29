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




import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import umap.umap_ as umap
from sklearn.manifold import TSNE
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

def plot_umap(data, num_components=2, custom_colors=None, feature_intensity=None, random_state=1):
    if 'Class' not in data.columns:
        st.error("Class column not found in data.")
        return

    num_components = min(num_components, 3)

    n_samples = data.shape[0]
    n_neighbors = max(2, min(int(np.log2(n_samples)), 100))

    # Standardize the feature data
    data_features = data.drop(['Class'], axis=1)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_features)

    reducer = umap.UMAP(n_components=num_components, n_neighbors=n_neighbors, random_state=random_state, n_jobs=-1)
    umap_results = reducer.fit_transform(data_scaled)

    df_umap = pd.DataFrame(umap_results, columns=[f'UMAP{i+1}' for i in range(num_components)])
    df_umap['Class'] = data['Class'].values
    df_umap['Index'] = df_umap.index

    if feature_intensity and feature_intensity != 'None':
        df_umap[feature_intensity] = data[feature_intensity]

    # Adjust marker size based on the number of samples
    marker_size = max(4, 10 - np.log10(n_samples))

    if num_components == 2:
        fig = px.scatter(df_umap, x='UMAP1', y='UMAP2', color='Class',
                         color_discrete_map=custom_colors,
                         hover_data=['Index'])

        if feature_intensity and feature_intensity != 'None':
            fig.update_traces(marker=dict(color=df_umap[feature_intensity],
                                          colorbar=dict(title=feature_intensity, x=-0.5),
                                          colorscale='jet', showscale=True))

        fig.update_layout(
            xaxis=dict(
                title='UMAP1',
                titlefont=dict(size=24, color='black'),
                tickfont=dict(size=22, color='black'),
                showgrid=False,
                showline=True,
                linecolor='black',
                linewidth=2
            ),
            yaxis=dict(
                title='UMAP2',
                titlefont=dict(size=24, color='black'),
                tickfont=dict(size=22, color='black'),
                showgrid=False,
                showline=True,
                linecolor='black',
                linewidth=2
            ),
            legend=dict(
                font=dict(size=20, color='black')
            ),
            title=dict(
                text="UMAP Plot",
                font=dict(size=26, color='black')
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
    else:
        fig = px.scatter_3d(df_umap, x='UMAP1', y='UMAP2', z='UMAP3', color='Class',
                            color_discrete_map=custom_colors,
                            hover_data=['Index'])

        if feature_intensity and feature_intensity != 'None':
            fig.update_traces(marker=dict(color=df_umap[feature_intensity],
                                          colorbar=dict(title=feature_intensity, x=-0.05),
                                          colorscale='jet', showscale=True))

        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    title='UMAP1',
                    titlefont=dict(size=22, color='black'),
                    tickfont=dict(size=18, color='black'),
                    showgrid=False,
                    showline=True,
                    linecolor='black',
                    linewidth=2
                ),
                yaxis=dict(
                    title='UMAP2',
                    titlefont=dict(size=22, color='black'),
                    tickfont=dict(size=18, color='black'),
                    showgrid=False,
                    showline=True,
                    linecolor='black',
                    linewidth=2
                ),
                zaxis=dict(
                    title='UMAP3',
                    titlefont=dict(size=22, color='black'),
                    tickfont=dict(size=18, color='black'),
                    showgrid=False,
                    showline=True,
                    linecolor='black',
                    linewidth=2
                )
            ),
            legend=dict(
                font=dict(size=24, color='black')
            ),
            title=dict(
                text="UMAP Plot",
                font=dict(size=26, color='black')
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

    fig.update_traces(marker=dict(size=marker_size, opacity=0.8, line=dict(width=0.5, color='black')))
    st.plotly_chart(fig)





def plot_tsne(data, num_components=2, custom_colors=None, feature_intensity=None, random_state=1):
    """
    Generates and displays a t-SNE plot with optional feature intensity coloring.

    Args:
        data (pd.DataFrame): The input data containing features and a 'Class' column.
        num_components (int): The number of t-SNE components (2 or 3).
        custom_colors (dict, optional): A dictionary mapping class labels to colors. Defaults to None.
        feature_intensity (str, optional): The name of the feature to use for intensity coloring. Defaults to None.
        random_state (int, optional): Random seed for t-SNE. Defaults to 1.
    """
    if 'Class' not in data.columns:
        st.error("Class column not found in data.")
        return

    num_components = min(num_components, 3)

    n_samples = data.shape[0]
    perplexity = max(2, min(int(np.log2(n_samples)), 100))

    # Standardize the features
    features = data.drop('Class', axis=1)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    tsne = TSNE(n_components=num_components, perplexity=perplexity, random_state=random_state)
    tsne_results = tsne.fit_transform(features_scaled)

    df_tsne = pd.DataFrame(tsne_results, columns=[f't-SNE{i+1}' for i in range(num_components)])
    df_tsne['Class'] = data['Class']
    df_tsne['Index'] = df_tsne.index

    if feature_intensity and feature_intensity != 'None':
        df_tsne[feature_intensity] = data[feature_intensity]

    marker_size = max(4, 10 - np.log10(n_samples))

    cmax_value = data[feature_intensity].max() if feature_intensity and feature_intensity != 'None' else None

    if num_components == 2:
        fig = px.scatter(df_tsne, x='t-SNE1', y='t-SNE2', color='Class',
                         color_discrete_map=custom_colors,
                         hover_data=['Index'])

        if feature_intensity and feature_intensity != 'None':
            fig.update_traces(marker=dict(color=df_tsne[feature_intensity],
                                          colorbar=dict(title=feature_intensity, x=-0.5),
                                          colorscale='jet', showscale=True,
                                          cmin=0, cmax=cmax_value))

        fig.update_layout(
            xaxis=dict(title='t-SNE1', titlefont=dict(size=24, color='black'),
                       tickfont=dict(size=22, color='black'), showgrid=False,
                       showline=True, linecolor='black', linewidth=2),
            yaxis=dict(title='t-SNE2', titlefont=dict(size=24, color='black'),
                       tickfont=dict(size=22, color='black'), showgrid=False,
                       showline=True, linecolor='black', linewidth=2),
            legend=dict(font=dict(size=20, color='black')),
            title=dict(text="t-SNE Plot", font=dict(size=26, color='black')),
            plot_bgcolor='white', paper_bgcolor='white'
        )
    else:
        fig = px.scatter_3d(df_tsne, x='t-SNE1', y='t-SNE2', z='t-SNE3', color='Class',
                            color_discrete_map=custom_colors,
                            hover_data=['Index'])

        if feature_intensity and feature_intensity != 'None':
            fig.update_traces(marker=dict(color=df_tsne[feature_intensity],
                                          colorbar=dict(title=feature_intensity, x=-0.05),
                                          colorscale='jet', showscale=True,
                                          cmin=0, cmax=cmax_value))

        fig.update_layout(
            scene=dict(
                xaxis=dict(title='t-SNE1', titlefont=dict(size=22, color='black'),
                           tickfont=dict(size=18, color='black'), showgrid=False,
                           showline=True, linecolor='black', linewidth=2),
                yaxis=dict(title='t-SNE2', titlefont=dict(size=22, color='black'),
                           tickfont=dict(size=18, color='black'), showgrid=False,
                           showline=True, linecolor='black', linewidth=2),
                zaxis=dict(title='t-SNE3', titlefont=dict(size=22, color='black'),
                           tickfont=dict(size=18, color='black'), showgrid=False,
                           showline=True, linecolor='black', linewidth=2)
            ),
            legend=dict(font=dict(size=24, color='black')),
            title=dict(text="t-SNE Plot", font=dict(size=26, color='black')),
            plot_bgcolor='white', paper_bgcolor='white'
        )

    fig.update_traces(marker=dict(size=marker_size, opacity=0.8, line=dict(width=0.5, color='black')))
    st.plotly_chart(fig)



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



def plot_pca(reduced_data, class_labels, custom_colors=None, feature_intensity=None, X=None):
    num_components = min(reduced_data.shape[1], 3)

    pca_df = pd.DataFrame(reduced_data, columns=[f'PCA{i+1}' for i in range(num_components)])
    pca_df['Class'] = class_labels
    pca_df['Index'] = pca_df.index

    if feature_intensity and feature_intensity != 'None' and X is not None:
        pca_df[feature_intensity] = X[feature_intensity]

    n_samples = pca_df.shape[0]
    marker_size = max(4, 10 - np.log10(n_samples))

    if num_components == 2:
        fig = px.scatter(
            pca_df, x='PCA1', y='PCA2', color='Class',
            color_discrete_map=custom_colors,
            hover_data=['Index']
        )

        if feature_intensity and feature_intensity != 'None':
            fig.update_traces(marker=dict(color=pca_df[feature_intensity],
                                          colorbar=dict(title=feature_intensity, x=-0.5),
                                          colorscale='jet', showscale=True))

        fig.update_layout(
            xaxis=dict(
                title='PCA 1',
                titlefont=dict(size=24, color='black'),
                tickfont=dict(size=22, color='black'),
                showgrid=False,
                showline=True,
                linecolor='black',
                linewidth=2
            ),
            yaxis=dict(
                title='PCA 2',
                titlefont=dict(size=24, color='black'),
                tickfont=dict(size=22, color='black'),
                showgrid=False,
                showline=True,
                linecolor='black',
                linewidth=2
            ),
            legend=dict(
                font=dict(size=20, color='black')
            ),
            title=dict(
                text="PCA Scatter Plot",
                font=dict(size=26, color='black')
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
    else:
        fig = px.scatter_3d(
            pca_df, x='PCA1', y='PCA2', z='PCA3', color='Class',
            color_discrete_map=custom_colors,
            hover_data=['Index']
        )

        if feature_intensity and feature_intensity != 'None':
            fig.update_traces(marker=dict(color=pca_df[feature_intensity],
                                          colorbar=dict(title=feature_intensity, x=-0.05),
                                          colorscale='jet', showscale=True))

        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    title='PCA 1',
                    titlefont=dict(size=22, color='black'),
                    tickfont=dict(size=18, color='black'),
                    showgrid=False,
                    showline=True,
                    linecolor='black',
                    linewidth=2
                ),
                yaxis=dict(
                    title='PCA 2',
                    titlefont=dict(size=22, color='black'),
                    tickfont=dict(size=18, color='black'),
                    showgrid=False,
                    showline=True,
                    linecolor='black',
                    linewidth=2
                ),
                zaxis=dict(
                    title='PCA 3',
                    titlefont=dict(size=22, color='black'),
                    tickfont=dict(size=18, color='black'),
                    showgrid=False,
                    showline=True,
                    linecolor='black',
                    linewidth=2
                )
            ),
            legend=dict(
                font=dict(size=24, color='black')
            ),
            title=dict(
                text="PCA Scatter Plot (3D)",
                font=dict(size=26, color='black')
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

    fig.update_traces(marker=dict(size=marker_size, opacity=0.8, line=dict(width=0.5, color='black')))

    # Uniformiser les échelles X et Y
    if num_components == 2:
        max_range = max(pca_df[['PCA1', 'PCA2']].abs().max())
        fig.update_layout(
            xaxis=dict(range=[-max_range, max_range], scaleanchor="y", scaleratio=1),
            yaxis=dict(range=[-max_range, max_range], scaleanchor="x", scaleratio=1)
        )

    st.plotly_chart(fig)




def plot_pca(reduced_data, class_labels, custom_colors=None, feature_intensity=None, X=None):
    num_components = min(reduced_data.shape[1], 3)
    pca_df = pd.DataFrame(reduced_data, columns=[f'PCA{i+1}' for i in range(num_components)])
    pca_df['Class'] = class_labels
    pca_df['Index'] = pca_df.index

    if feature_intensity and feature_intensity != 'None' and X is not None and feature_intensity in X.columns:
        pca_df[feature_intensity] = X[feature_intensity]

    n_samples = pca_df.shape[0]
    marker_size = max(4, 10 - np.log10(n_samples))

    if num_components == 2:
        fig = px.scatter(pca_df, x='PCA1', y='PCA2', color='Class',
                         color_discrete_map=custom_colors, hover_data=['Index'])
        if feature_intensity and feature_intensity != 'None':
            fig.update_traces(marker=dict(color=pca_df[feature_intensity],
                                          colorbar=dict(title=feature_intensity, x=-0.5),
                                          colorscale='jet', showscale=True))
        max_range = max(pca_df[['PCA1', 'PCA2']].abs().max())
        fig.update_layout(xaxis=dict(range=[-max_range, max_range], scaleanchor="y", scaleratio=1),
                          yaxis=dict(range=[-max_range, max_range], scaleanchor="x", scaleratio=1))
    else:
        fig = px.scatter_3d(pca_df, x='PCA1', y='PCA2', z='PCA3', color='Class',
                            color_discrete_map=custom_colors, hover_data=['Index'])
        if feature_intensity and feature_intensity != 'None':
            fig.update_traces(marker=dict(color=pca_df[feature_intensity],
                                          colorbar=dict(title=feature_intensity, x=-0.05),
                                          colorscale='jet', showscale=True))

    fig.update_traces(marker=dict(size=marker_size, opacity=0.8, line=dict(width=0.5, color='black')))
    return fig


def plot_tsne(data, num_components=2, custom_colors=None, feature_intensity=None, random_state=1):
    if 'Class' not in data.columns:
        st.error("Class column not found in data.")
        return None

    num_components = min(num_components, 3)
    n_samples = data.shape[0]
    perplexity = max(2, min(int(np.log2(n_samples)), 100))

    features = data.drop('Class', axis=1)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    tsne = TSNE(n_components=num_components, perplexity=perplexity, random_state=random_state)
    tsne_results = tsne.fit_transform(features_scaled)

    df_tsne = pd.DataFrame(tsne_results, columns=[f't-SNE{i+1}' for i in range(num_components)])
    df_tsne['Class'] = data['Class']
    df_tsne['Index'] = df_tsne.index

    if feature_intensity and feature_intensity != 'None' and feature_intensity in data.columns:
        df_tsne[feature_intensity] = data[feature_intensity]

    marker_size = max(4, 10 - np.log10(n_samples))
    cmax_value = data[feature_intensity].max() if feature_intensity and feature_intensity != 'None' else None

    if num_components == 2:
        fig = px.scatter(df_tsne, x='t-SNE1', y='t-SNE2', color='Class',
                         color_discrete_map=custom_colors, hover_data=['Index'])
        if feature_intensity and feature_intensity != 'None':
            fig.update_traces(marker=dict(color=df_tsne[feature_intensity],
                                          colorbar=dict(title=feature_intensity, x=-0.5),
                                          colorscale='jet', showscale=True,
                                          cmin=0, cmax=cmax_value))
    else:
        fig = px.scatter_3d(df_tsne, x='t-SNE1', y='t-SNE2', z='t-SNE3', color='Class',
                            color_discrete_map=custom_colors, hover_data=['Index'])
        if feature_intensity and feature_intensity != 'None':
            fig.update_traces(marker=dict(color=df_tsne[feature_intensity],
                                          colorbar=dict(title=feature_intensity, x=-0.05),
                                          colorscale='jet', showscale=True,
                                          cmin=0, cmax=cmax_value))

    fig.update_traces(marker=dict(size=marker_size, opacity=0.8, line=dict(width=0.5, color='black')))
    return fig


def plot_umap(data, num_components=2, custom_colors=None, feature_intensity=None, random_state=1):
    if 'Class' not in data.columns:
        st.error("Class column not found in data.")
        return None

    num_components = min(num_components, 3)
    n_samples = data.shape[0]
    n_neighbors = max(2, min(int(np.log2(n_samples)), 100))

    data_features = data.drop('Class', axis=1)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_features)

    reducer = umap.UMAP(n_components=num_components, n_neighbors=n_neighbors,
                        random_state=random_state, n_jobs=-1)
    umap_results = reducer.fit_transform(data_scaled)

    df_umap = pd.DataFrame(umap_results, columns=[f'UMAP{i+1}' for i in range(num_components)])
    df_umap['Class'] = data['Class'].values
    df_umap['Index'] = df_umap.index

    if feature_intensity and feature_intensity != 'None' and feature_intensity in data.columns:
        df_umap[feature_intensity] = data[feature_intensity]

    marker_size = max(4, 10 - np.log10(n_samples))

    if num_components == 2:
        fig = px.scatter(df_umap, x='UMAP1', y='UMAP2', color='Class',
                         color_discrete_map=custom_colors, hover_data=['Index'])
        if feature_intensity and feature_intensity != 'None':
            fig.update_traces(marker=dict(color=df_umap[feature_intensity],
                                          colorbar=dict(title=feature_intensity, x=-0.5),
                                          colorscale='jet', showscale=True))
    else:
        fig = px.scatter_3d(df_umap, x='UMAP1', y='UMAP2', z='UMAP3', color='Class',
                            color_discrete_map=custom_colors, hover_data=['Index'])
        if feature_intensity and feature_intensity != 'None':
            fig.update_traces(marker=dict(color=df_umap[feature_intensity],
                                          colorbar=dict(title=feature_intensity, x=-0.05),
                                          colorscale='jet', showscale=True))

    fig.update_traces(marker=dict(size=marker_size, opacity=0.8, line=dict(width=0.5, color='black')))
    return fig




def update_axes_style(fig, num_components=2, plot_type="Scatter Plot"):
    """
    Uniformise le style des axes, titres, ticks et légendes pour les scatter plots 2D/3D
    et renomme automatiquement les axes selon le type de réduction (PCA, t-SNE, UMAP).
    """
    # Déterminer les labels des axes automatiquement
    if "PCA" in plot_type:
        axis_labels = [f"PCA {i+1}" for i in range(num_components)]
    elif "t-SNE" in plot_type:
        axis_labels = [f"t-SNE {i+1}" for i in range(num_components)]
    elif "UMAP" in plot_type:
        axis_labels = [f"UMAP {i+1}" for i in range(num_components)]
    else:
        axis_labels = [f"Component {i+1}" for i in range(num_components)]

    if num_components == 2:
        fig.update_layout(
            xaxis=dict(
                title=axis_labels[0],
                titlefont=dict(size=24, color='black', family="Arial, sans-serif"),
                tickfont=dict(size=22, color='black'),
                showgrid=False,
                showline=True,
                linecolor='black',
                linewidth=2
            ),
            yaxis=dict(
                title=axis_labels[1],
                titlefont=dict(size=24, color='black', family="Arial, sans-serif"),
                tickfont=dict(size=22, color='black'),
                showgrid=False,
                showline=True,
                linecolor='black',
                linewidth=2
            ),
            legend=dict(font=dict(size=20, color='black')),
            title=dict(text=plot_type, font=dict(size=26, color='black')),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
    else:  # 3D
        z_label = axis_labels[2] if num_components >= 3 else "Component 3"
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    title=axis_labels[0],
                    titlefont=dict(size=22, color='black', family="Arial, sans-serif"),
                    tickfont=dict(size=18, color='black'),
                    showgrid=False,
                    showline=True,
                    linecolor='black',
                    linewidth=2
                ),
                yaxis=dict(
                    title=axis_labels[1],
                    titlefont=dict(size=22, color='black', family="Arial, sans-serif"),
                    tickfont=dict(size=18, color='black'),
                    showgrid=False,
                    showline=True,
                    linecolor='black',
                    linewidth=2
                ),
                zaxis=dict(
                    title=z_label,
                    titlefont=dict(size=22, color='black', family="Arial, sans-serif"),
                    tickfont=dict(size=18, color='black'),
                    showgrid=False,
                    showline=True,
                    linecolor='black',
                    linewidth=2
                )
            ),
            legend=dict(font=dict(size=24, color='black')),
            title=dict(text=f"{plot_type} (3D)", font=dict(size=26, color='black')),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

    return fig

def plot_pca(reduced_data, class_labels, custom_colors=None, feature_intensity=None, X=None):
    num_components = min(reduced_data.shape[1], 3)
    pca_df = pd.DataFrame(reduced_data, columns=[f'PCA{i+1}' for i in range(num_components)])
    pca_df['Class'] = class_labels
    pca_df['Index'] = pca_df.index

    if feature_intensity and feature_intensity != 'None' and X is not None and feature_intensity in X.columns:
        pca_df[feature_intensity] = X[feature_intensity]

    n_samples = pca_df.shape[0]
    marker_size = max(4, 10 - np.log10(n_samples))

    if num_components == 2:
        fig = px.scatter(pca_df, x='PCA1', y='PCA2', color='Class',
                         color_discrete_map=custom_colors, hover_data=['Index'])
        if feature_intensity and feature_intensity != 'None':
            fig.update_traces(marker=dict(color=pca_df[feature_intensity],
                                          colorbar=dict(title=feature_intensity, x=-0.5),
                                          colorscale='jet', showscale=True))

        # ---------- FIX : auto-range avec buffer ----------
        max_val = pca_df[['PCA1', 'PCA2']].abs().max().max()  # max absolu global
        buffer = 0.05 * max_val  # 5% buffer pour que les points soient visibles
        fig.update_layout(
            xaxis=dict(range=[-max_val-buffer, max_val+buffer], scaleanchor="y", scaleratio=1),
            yaxis=dict(range=[-max_val-buffer, max_val+buffer], scaleanchor="x", scaleratio=1)
        )
    else:
        fig = px.scatter_3d(pca_df, x='PCA1', y='PCA2', z='PCA3', color='Class',
                            color_discrete_map=custom_colors, hover_data=['Index'])
        if feature_intensity and feature_intensity != 'None':
            fig.update_traces(marker=dict(color=pca_df[feature_intensity],
                                          colorbar=dict(title=feature_intensity, x=-0.05),
                                          colorscale='jet', showscale=True))

    fig.update_traces(marker=dict(size=marker_size, opacity=0.8, line=dict(width=0.5, color='black')))
    fig = update_axes_style(fig, num_components=num_components, plot_type="PCA Scatter Plot")
    return fig


def plot_tsne(data, num_components=2, custom_colors=None, feature_intensity=None, random_state=1):
    if 'Class' not in data.columns:
        st.error("Class column not found in data.")
        return None

    num_components = min(num_components, 3)
    n_samples = data.shape[0]
    perplexity = max(2, min(int(np.log2(n_samples)), 100))

    features = data.drop('Class', axis=1)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    tsne = TSNE(n_components=num_components, perplexity=perplexity, random_state=random_state)
    tsne_results = tsne.fit_transform(features_scaled)

    df_tsne = pd.DataFrame(tsne_results, columns=[f't-SNE{i+1}' for i in range(num_components)])
    df_tsne['Class'] = data['Class']
    df_tsne['Index'] = df_tsne.index

    if feature_intensity and feature_intensity != 'None' and feature_intensity in data.columns:
        df_tsne[feature_intensity] = data[feature_intensity]

    marker_size = max(4, 10 - np.log10(n_samples))
    cmax_value = data[feature_intensity].max() if feature_intensity and feature_intensity != 'None' else None

    if num_components == 2:
        fig = px.scatter(df_tsne, x='t-SNE1', y='t-SNE2', color='Class',
                         color_discrete_map=custom_colors, hover_data=['Index'])
        if feature_intensity and feature_intensity != 'None':
            fig.update_traces(marker=dict(color=df_tsne[feature_intensity],
                                          colorbar=dict(title=feature_intensity, x=-0.5),
                                          colorscale='jet', showscale=True,
                                          cmin=0, cmax=cmax_value))
    else:
        fig = px.scatter_3d(df_tsne, x='t-SNE1', y='t-SNE2', z='t-SNE3', color='Class',
                            color_discrete_map=custom_colors, hover_data=['Index'])
        if feature_intensity and feature_intensity != 'None':
            fig.update_traces(marker=dict(color=df_tsne[feature_intensity],
                                          colorbar=dict(title=feature_intensity, x=-0.05),
                                          colorscale='jet', showscale=True,
                                          cmin=0, cmax=cmax_value))

    fig.update_traces(marker=dict(size=marker_size, opacity=0.8, line=dict(width=0.5, color='black')))
    fig = update_axes_style(fig, num_components=num_components, plot_type="t-SNE Scatter Plot")
    return fig


def plot_umap(data, num_components=2, custom_colors=None, feature_intensity=None, random_state=1):
    if 'Class' not in data.columns:
        st.error("Class column not found in data.")
        return None

    num_components = min(num_components, 3)
    n_samples = data.shape[0]
    n_neighbors = max(2, min(int(np.log2(n_samples)), 100))

    data_features = data.drop('Class', axis=1)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_features)

    reducer = umap.UMAP(n_components=num_components, n_neighbors=n_neighbors,
                        random_state=random_state, n_jobs=-1)
    umap_results = reducer.fit_transform(data_scaled)

    df_umap = pd.DataFrame(umap_results, columns=[f'UMAP{i+1}' for i in range(num_components)])
    df_umap['Class'] = data['Class'].values
    df_umap['Index'] = df_umap.index

    if feature_intensity and feature_intensity != 'None' and feature_intensity in data.columns:
        df_umap[feature_intensity] = data[feature_intensity]

    marker_size = max(4, 10 - np.log10(n_samples))

    if num_components == 2:
        fig = px.scatter(df_umap, x='UMAP1', y='UMAP2', color='Class',
                         color_discrete_map=custom_colors, hover_data=['Index'])
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
        fig = px.scatter_3d(df_umap, x='UMAP1', y='UMAP2', z='UMAP3', color='Class',
                            color_discrete_map=custom_colors, hover_data=['Index'])
        if feature_intensity and feature_intensity != 'None':
            fig.update_traces(marker=dict(color=df_umap[feature_intensity],
                                          colorbar=dict(title=feature_intensity, x=-0.05),
                                          colorscale='jet', showscale=True))

    fig.update_traces(marker=dict(size=marker_size, opacity=0.8, line=dict(width=0.5, color='black')))
    fig = update_axes_style(fig, num_components=num_components, plot_type="UMAP Scatter Plot")
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
        fig.update_layout(
            title=dict(text=plot_type, font=dict(size=22)),
            font=dict(color=font_color, size=18),
            scene=dict(
                xaxis=dict(title=dict(text='X', font=dict(size=title_font_size, color=font_color))),
                yaxis=dict(title=dict(text='Y', font=dict(size=title_font_size, color=font_color))),
                zaxis=dict(title=dict(text='Z', font=dict(size=title_font_size, color=font_color))),
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(
                title=dict(font=dict(size=18, color=font_color)),
                font=dict(size=16, color=font_color)
            )
        )
    return fig

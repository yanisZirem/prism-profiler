"""
Software Name: Profiler
Module name : data_exploration
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
import seaborn as sns
import matplotlib.pyplot as plt
from venn import venn
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import gc
from upsetplot import UpSet, from_memberships
import pandas as pd



def plot_feature_distribution(data, feature, class_colors, histfunc='sum'):
    if feature == 'Class':
        fig = px.histogram(
            data,
            x='Class',
            title=f"Distribution of {feature}",
            color='Class',
            color_discrete_map=class_colors
        )
    else:
        fig = px.histogram(
            data,
            x='Class',
            y=feature,
            histfunc=histfunc,
            title=f"{histfunc.capitalize()} of {feature} intensities by class",
            color='Class',
            color_discrete_map=class_colors
        )

    # fig.update_layout(
    #     title_font_size=22,
    #     font=dict(color="black", size=18),
    #     legend=dict(
    #         title_font=dict(size=18, color='black'),
    #         font=dict(size=16, color='black')
    #     ),
    #     xaxis=dict(
    #         title='Class',
    #         titlefont=dict(size=20, color='black'),
    #         tickfont=dict(size=16, color='black')
    #     ),
    #     yaxis=dict(
    #         title='Intensity' if feature != 'Class' else 'Count',
    #         titlefont=dict(size=20, color='black'),
    #         tickfont=dict(size=16, color='black')
    #     ),
    #     plot_bgcolor='white',
    #     paper_bgcolor='white'
    # )
    fig.update_layout(
        title=dict(
            text=fig.layout.title.text,
            font=dict(size=22)
        ),
        font=dict(color="black", size=18),
        legend=dict(
            title=dict(font=dict(size=18, color='black')),
            font=dict(size=16, color='black')
        ),
        xaxis=dict(
            title=dict(text='Class', font=dict(size=20, color='black')),
            tickfont=dict(size=16, color='black')
        ),
        yaxis=dict(
            title=dict(text='Intensity' if feature != 'Class' else 'Count', font=dict(size=20, color='black')),
            tickfont=dict(size=16, color='black')
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    return fig


def plot_venn_diagram(data, class_column, color_palette, source=None):
    if source != "Raw Data":
        data = data.replace(0, np.nan)

    detected_features = {
        class_name: set(group.dropna(axis=1, how='all').columns) - {class_column}
        for class_name, group in data.groupby(class_column)
    }

    num_classes = len(detected_features)
    if num_classes < 2 or num_classes > 6:
        st.error("⚠️ Venn diagram supports only 2 to 6 classes.")
        return None

    venn_colors = [
        color_palette.get(class_name, '#000000')
        for class_name in detected_features.keys()
    ]

    # figure matplotlib explicite
    fig, ax = plt.subplots(figsize=(10, 8))
    venn_plot = venn(detected_features, cmap=venn_colors, ax=ax)
    ax.set_title(f"Venn Diagram ({source})" if source else "Venn Diagram")

    # --- Stockage des exclusifs et communs ---
    exclusive_features = {}
    all_classes = list(detected_features.keys())
    for cls in all_classes:
        current_features = detected_features[cls]
        other_features = set().union(*[detected_features[c] for c in all_classes if c != cls])
        truly_exclusive = current_features - other_features
        exclusive_features[cls] = truly_exclusive

    common_features = set.intersection(*detected_features.values())

    st.session_state['common_features'] = common_features
    st.session_state['exclusive_features'] = exclusive_features

    # la figure (pas plt)
    return fig



def plot_upset(data, class_column, source=None):

    try:
        if source != "Raw Data":
            data = data.replace(0, np.nan)

        # --- Nettoyage colonnes inutiles ---
        clean_data = data.drop(columns=['File', 'RT', 'Sum'], errors='ignore')

        if class_column not in clean_data.columns:
            st.error(f"Column '{class_column}' not found in the data.")
            del clean_data
            gc.collect()
            return None

        clean_data[class_column] = clean_data[class_column].astype(str)

        # --- Groupement et transformation ---
        class_feature = (
            clean_data
            .groupby(class_column)
            .apply(lambda df: df.drop(columns=[class_column], errors='ignore').notna().any())
            .astype(int)
        )

        # On évite les copies inutiles
        upset_input = class_feature.T
        del class_feature 
        gc.collect()

        if upset_input.empty:
            st.warning("⚠️ No features available for UpSet plot.")
            del upset_input, clean_data
            gc.collect()
            return None

        # --- Calcul des memberships ---
        memberships = [
            tuple(upset_input.columns[i] for i, present in enumerate(row) if present)
            for row in upset_input.values
        ]

        upset_data = from_memberships(
            memberships,
            data=pd.Series(upset_input.index, index=upset_input.index)
        )

        # Nettoyage 
        del memberships, upset_input, clean_data
        gc.collect()

        # --- Taille dynamique ---
        n_features, n_subsets = upset_data.shape[0], len(upset_data.columns) if hasattr(upset_data, "columns") else 5
        fig_width = min(40, 5 + n_subsets * 1.5)
        fig_height = min(30, 5 + n_features * 0.5)

        fig = plt.figure(figsize=(fig_width, fig_height))

        upset = UpSet(
            upset_data,
            subset_size='count',
            show_counts=True,
            sort_by='cardinality',
            sort_categories_by=None,
            min_subset_size=20,
            facecolor="black"
        )
        upset.plot(fig=fig)

        # Modifier les labels
        for ax in fig.axes:
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_fontsize(10)
            for label in ax.get_yticklabels():
                label.set_fontsize(10)

        fig.suptitle(f"UpSet Plot ({source})" if source else "UpSet Plot", fontsize=18)

        # 🔹 Nettoyage final avant de retourner la figure
        del upset_data, upset
        gc.collect()

        return fig

    except Exception as e:
        st.error(f"❌ Error while generating UpSet plot: {e}")
        gc.collect()
        return None


def plot_multiple_features_line(data, features, feature_colors, histfunc='sum', error_type=None):

    melted = data.melt(id_vars='Class', value_vars=features, var_name='Feature', value_name='Value')

    if histfunc == 'percentage':
        melted = melted.groupby(['Class', 'Feature'])['Value'].sum().reset_index()
        total_by_class = melted.groupby('Class')['Value'].transform('sum')
        melted['Value'] = (melted['Value'] / total_by_class) * 100
        error_df = None  # No error bars for percentage
    else:
        agg_df = melted.groupby(['Class', 'Feature'])['Value'].agg(histfunc).reset_index()
        if error_type == 'sem':
            error_df = melted.groupby(['Class', 'Feature'])['Value'].sem().reset_index().rename(columns={'Value': 'Error'})
        elif error_type == 'std':
            error_df = melted.groupby(['Class', 'Feature'])['Value'].std().reset_index().rename(columns={'Value': 'Error'})
        else:
            error_df = None
        melted = agg_df

    fig = go.Figure()

    for cls in melted['Class'].unique():
        df_c = melted[melted['Class'] == cls].sort_values(by='Feature')
        error_c = None
        if error_df is not None:
            error_c = error_df[error_df['Class'] == cls].sort_values(by='Feature')['Error']

        fig.add_trace(go.Scatter(
            x=df_c['Feature'],
            y=df_c['Value'],
            mode='lines+markers',
            name=f"Class {cls}",
            line=dict(color=feature_colors.get(cls, '#333333')),
            error_y=dict(
                type='data',
                array=error_c if error_c is not None else None,
                visible=error_c is not None
            )
        ))

    fig.update_layout(
        title=f"{'Percentage' if histfunc == 'percentage' else histfunc.capitalize()} Line Chart",
        font=dict(color='black', size=16),
        title_font=dict(size=20, color='black'),
        xaxis=dict(
            title='Feature',
            title_font=dict(size=16, color='black'),
            tickfont=dict(size=14, color='black'),
            linecolor='black',
            showline=True
        ),
        yaxis=dict(
            title=f"{'Percentage (%)' if histfunc == 'percentage' else histfunc.capitalize()}",
            title_font=dict(size=16, color='black'),
            tickfont=dict(size=14, color='black'),
            linecolor='black',
            showline=True
        ),
        legend=dict(
            title_font=dict(size=16, color='black'),
            font=dict(size=14, color='black')
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    return fig

def plot_multiple_features_radar(data, features, feature_colors, histfunc='sum'):
    melted = data.melt(id_vars='Class', value_vars=features, var_name='Feature', value_name='Value')

    if histfunc == 'percentage':
        melted = melted.groupby(['Class', 'Feature'])['Value'].sum().reset_index()
        total_by_class = melted.groupby('Class')['Value'].transform('sum')
        melted['Value'] = (melted['Value'] / total_by_class) * 100
    else:
        melted = melted.groupby(['Class', 'Feature'])['Value'].agg(histfunc).reset_index()

    fig = go.Figure()

    for cls in melted['Class'].unique():
        df_c = melted[melted['Class'] == cls].sort_values(by='Feature')
        fig.add_trace(go.Scatterpolar(
            r=df_c['Value'],
            theta=df_c['Feature'],
            fill='toself',
            name=f"Class {cls}",
            line=dict(color=feature_colors.get(cls, '#333333'))
        ))

    fig.update_layout(
        title=f"{'Percentage' if histfunc == 'percentage' else histfunc.capitalize()} Radar Chart",
        font=dict(color='black', size=16),
        title_font=dict(size=20, color='black'),
        polar=dict(
            radialaxis=dict(visible=True, color='black', tickfont=dict(color='black')),
            angularaxis=dict(tickfont=dict(color='black'))
        ),
        legend=dict(
            title_font=dict(size=16, color='black'),
            font=dict(size=14, color='black')
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    return fig

def plot_multiple_features_distribution(data, features, feature_colors, histfunc='sum'):
    melted = data.melt(id_vars='Class', value_vars=features, var_name='Feature', value_name='Value')

    if histfunc == 'percentage':
        melted = melted.groupby(['Class', 'Feature'])['Value'].sum().reset_index()
        total_by_class = melted.groupby('Class')['Value'].transform('sum')
        melted['Value'] = (melted['Value'] / total_by_class) * 100
    else:
        melted = melted.groupby(['Class', 'Feature'])['Value'].agg(histfunc).reset_index()

    fig = px.bar(
        melted,
        x='Class',
        y='Value',
        color='Feature',
        barmode='group',
        title=f"{'Percentage' if histfunc == 'percentage' else histfunc.capitalize()} Bar Chart",
        color_discrete_map=feature_colors,
        labels={
            'Class': 'Class',
            'Value': f"{'Percentage (%)' if histfunc == 'percentage' else histfunc.capitalize()}",
            'Feature': 'Feature'
        }
    )

    unique_classes = melted['Class'].nunique()

    if unique_classes <= 2:
        fig.update_layout(
            width=600,
            height=600,
            bargap=0.6
        )
        fig.update_traces(width=0.1)
    else:
        fig.update_layout(
            width=900,
            height=600,
            bargap=0.2
        )
        fig.update_traces(width=0.2)

    fig.update_layout(
        font=dict(color='black', size=22),
        title_font=dict(size=26, color='black'),
        xaxis=dict(
            title_font=dict(size=26, color='black'),
            tickfont=dict(size=22, color='black'),
            linecolor='black',
            showline=True,
            showgrid=False
        ),
        yaxis=dict(
            title_font=dict(size=26, color='black'),
            tickfont=dict(size=22, color='black'),
            linecolor='black',
            showline=True,
            showgrid=False
        ),
        legend=dict(
            title_font=dict(size=26, color='black'),
            font=dict(size=22, color='black')
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    return fig

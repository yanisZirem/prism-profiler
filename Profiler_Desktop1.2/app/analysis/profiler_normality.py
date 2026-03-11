"""
Software Name: Profiler
Author: Yanis Zirem
Email : yanis.zirem@yahoo.com / yanis.zirem@univ-lille.fr
Creation Date: 15/01/2025
Last Updated: 05/03/2026
Version: 1.2.0

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
import streamlit as st
from scipy.stats import ttest_ind, shapiro, kstest
import plotly.express as px

def diagnose_normality(p_value):
    if p_value > 0.05:
        return "Normal distribution (p-value > 0.05)"
    else:
        return "Non-normal distribution (p-value <= 0.05)"



def display_class_info(data):
    if 'Class' in data.columns:

        COLUMNS_TO_EXCLUDE = {'Class', 'ID', 'File', 'RT', 'Sum'}
        class_counts = data['Class'].value_counts()
        class_percentages = (class_counts / len(data)) * 100
        imbalance_ratio = class_counts.max() / class_counts.min()

        class_info_df = pd.DataFrame({
            "Class": class_counts.index,
            "Count": class_counts.values,
            "Percentage (%)": class_percentages.round(2).values
        })
        st.markdown("**Class Distribution Summary**")
        st.dataframe(class_info_df, use_container_width=True)

        color_map = {cls: st.session_state.get('class_colors', {}).get(cls, '#CCCCCC')
                     for cls in class_counts.index}
        fig = px.pie(
            names=class_counts.index, values=class_counts.values,
            color=class_counts.index, color_discrete_map=color_map,
            title="Class Proportion"
        )
        fig.update_traces(textposition='inside', textinfo='percent+label',
                          textfont_size=16)
        fig.update_layout(
            legend_title_text='Class',
            legend=dict(font=dict(size=14)),
            title=dict(font=dict(size=16)),
            margin=dict(l=10, r=10, t=50, b=10),
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)
        # Capture for HTML report
        st.session_state["_report_class_proportion"] = ("plotly", fig)

        st.markdown("**Interpretation**")
        if imbalance_ratio > 2:
            st.warning("⚠️ The dataset is imbalanced. Consider using over or undersampling techniques.")
        else:
            st.success("✅ The class distribution appears reasonably balanced.")

        return True
    else:
        st.warning("⚠️ No 'Class' column found in the dataset.")
        return False

def calculate_missing_values(data):
    COLUMNS_TO_EXCLUDE = {'Class', 'ID', 'File', 'RT', 'Sum'}
    relevant_columns = [
        col for col in data.columns
        if col not in COLUMNS_TO_EXCLUDE
        and not str(col).endswith('_meta')
        and pd.api.types.is_numeric_dtype(data[col])
    ]
    missing_values = data[relevant_columns].isnull().sum()
    missing_values_percentage = (missing_values / len(data)) * 100
    total_missing_values = missing_values.sum()
    total_missing_percentage = total_missing_values / (len(data) * len(relevant_columns)) * 100

    missing_values_df = pd.DataFrame({
        "Missing Values": missing_values,
        "Percentage (%)": missing_values_percentage
    })
    missing_values_df.loc['Total'] = [total_missing_values, total_missing_percentage]
    missing_values_df = missing_values_df.rename(index={'Total': 'All'})
    missing_values_df = pd.concat([pd.DataFrame(missing_values_df.loc['All']).T, missing_values_df.drop('All')])

    return relevant_columns, missing_values_df



def perform_shapiro_wilk_test(data, relevant_columns):
    shapiro_results = {}
    normal_count = 0
    total_p_value = 0
    valid_tests = 0

    for feature in relevant_columns:
        non_missing_data = data[feature].dropna()
        if len(non_missing_data) >= 3:
            shapiro_stat, shapiro_p_value = shapiro(non_missing_data)
            shapiro_results[feature] = (shapiro_stat, shapiro_p_value)
            total_p_value += shapiro_p_value
            valid_tests += 1
            if shapiro_p_value > 0.05:
                normal_count += 1

    avg_p_value = total_p_value / valid_tests if valid_tests > 0 else float('nan')
    normal_ratio = normal_count / len(relevant_columns) if len(relevant_columns) > 0 else 0

    st.markdown("**Shapiro-Wilk Normality Summary:**")
    st.info(f"- Number of features following normal distribution: {normal_count}/{len(relevant_columns)} ({normal_ratio:.2%})")
    # st.write(f"- Average P-value: {avg_p_value:.4f}")

    return avg_p_value, normal_ratio


def display_distribution(data, relevant_columns):
    options = ["Choose a feature"] + relevant_columns

    # Ensure selected_feature is initialized and valid
    if 'selected_feature' not in st.session_state:
        st.session_state.selected_feature = "Choose a feature"

    feature = st.selectbox("Select a Feature to Analyze", options, index=options.index(st.session_state.selected_feature), key="feature_select")
    st.session_state.selected_feature = feature

    if feature != "Choose a feature":
        plot_data = data[[feature, 'Class']].dropna()
        plot_data['Index'] = plot_data.index

        fig = px.histogram(plot_data, x=feature, color='Class', barmode='overlay',
                           title=f"Distribution of {feature}",
                           labels={'value': 'Value', 'count': 'Frequency'},
                           hover_data=['Index'])

        fig.update_layout(bargap=0.2)
        st.plotly_chart(fig, use_container_width=True)



# def display_class_info(data):
#     if 'Class' not in data.columns:
#         st.warning("⚠️ No 'Class' column found in the dataset.")
#         return False

#     # Colonnes à exclure pour le comptage des features
#     COLUMNS_TO_EXCLUDE = ['Class', 'File', 'RT', 'Sum']
#     num_features = len([col for col in data.columns if col not in COLUMNS_TO_EXCLUDE])

#     # Comptage des classes
#     class_counts = data['Class'].value_counts()
#     class_percentages = (class_counts / len(data)) * 100
#     imbalance_ratio = class_counts.max() / class_counts.min()

#     # Tableau de distribution
#     class_info_df = pd.DataFrame({
#         "Class": class_counts.index,
#         "Count": class_counts.values,
#         "Percentage (%)": class_percentages.round(2).values
#     })
#     st.markdown("**Class Distribution Summary**")
#     st.dataframe(class_info_df, use_container_width=True)

#     # Pie chart interactif avec Plotly
#     st.markdown("**Class Proportion**")
#     colors = [st.session_state['class_colors'].get(cls, '#CCCCCC') for cls in class_counts.index]  # fallback gris

#     import plotly.express as px
#     fig = px.pie(
#         names=class_counts.index,
#         values=class_counts.values,
#         color=class_counts.index,
#         color_discrete_map={cls: color for cls, color in zip(class_counts.index, colors)},
#         title="Class Proportion"
#     )
#     fig.update_traces(textposition='inside', textinfo='percent+label')
#     fig.update_layout(
#         legend_title_text='Class',
#         legend=dict(font=dict(size=14)),
#         title=dict(font=dict(size=20))
#     )
#     st.plotly_chart(fig, use_container_width=True)

#     # Interprétation automatique
#     st.markdown("**Interpretation**")
#     if imbalance_ratio > 2:
#         st.warning("⚠️ The dataset is imbalanced. Consider using over or undersampling techniques.")
#     else:
#         st.success("✅ The class distribution appears reasonably balanced. No major concern regarding imbalance.")

#     return True


def display_class_info(data):
    if 'Class' not in data.columns:
        st.warning("⚠️ No 'Class' column found in the dataset.")
        return False

    # Colonnes à exclure pour le comptage des features
    COLUMNS_TO_EXCLUDE = {'Class', 'ID', 'File', 'RT', 'Sum'}
    num_features = len([col for col in data.columns if col not in COLUMNS_TO_EXCLUDE and not str(col).endswith('_meta')])

    # Comptage des classes
    class_counts = data['Class'].value_counts()
    class_percentages = (class_counts / len(data)) * 100
    imbalance_ratio = class_counts.max() / class_counts.min()

    # Tableau de distribution
    class_info_df = pd.DataFrame({
        "Class": class_counts.index,
        "Count": class_counts.values,
        "Percentage (%)": class_percentages.round(2).values
    })
    st.markdown("**Class Distribution Summary**")
    st.dataframe(class_info_df, use_container_width=True)

    # Pie chart interactif avec Plotly
    # st.markdown("**Class Proportion**")
    colors = [st.session_state['class_colors'].get(cls, '#CCCCCC') for cls in class_counts.index]  # fallback gris

    fig = px.pie(
        names=class_counts.index,
        values=class_counts.values,
        color=class_counts.index,
        color_discrete_map={cls: color for cls, color in zip(class_counts.index, colors)},
        title="Class Proportion"
    )
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        textfont_size=20  # <-- taille du texte pourcentage + label
    )
    fig.update_layout(
        legend_title_text='Class',
        legend=dict(font=dict(size=18)),   # <-- taille de la légende
        title=dict(font=dict(size=18))     # <-- taille du titre
    )
    st.plotly_chart(fig, use_container_width=True)

    # Interprétation automatique
    st.markdown("**Interpretation**")
    if imbalance_ratio > 2:
        st.warning("⚠️ The dataset is imbalanced. Consider using over or undersampling techniques.")
    else:
        st.success("✅ The class distribution appears reasonably balanced. No major concern regarding imbalance.")

    return True

# def display_class_info(data):
#     if 'Class' not in data.columns:
#         st.warning("⚠️ No 'Class' column found in the dataset.")
#         return False

#     # Colonnes à exclure pour le comptage des features
#     COLUMNS_TO_EXCLUDE = ['Class', 'File', 'RT', 'Sum']
#     num_features = len([col for col in data.columns if col not in COLUMNS_TO_EXCLUDE])

#     # Comptage des classes
#     class_counts = data['Class'].value_counts()
#     class_percentages = (class_counts / len(data)) * 100
#     imbalance_ratio = class_counts.max() / class_counts.min()

#     # Tableau de distribution
#     class_info_df = pd.DataFrame({
#         "Class": class_counts.index,
#         "Count": class_counts.values,
#         "Percentage (%)": class_percentages.round(2).values
#     })
#     st.markdown("**Class Distribution Summary**")
#     st.dataframe(class_info_df, use_container_width=True)

#     # ✅ Utilisation directe du mapping global depuis session_state
#     color_map = st.session_state.get("class_colors", {})
#     # fallback si jamais une classe n'a pas de couleur attribuée
#     color_map = {cls: color_map.get(cls, "#CCCCCC") for cls in class_counts.index}

#     import plotly.express as px
#     fig = px.pie(
#         names=class_counts.index,
#         values=class_counts.values,
#         color=class_counts.index,
#         color_discrete_map=color_map,  # 👈 mapping central appliqué ici
#         title="Class Proportion"
#     )
#     fig.update_traces(
#         textposition='inside',
#         textinfo='percent+label',
#         textfont_size=20
#     )
#     fig.update_layout(
#         legend_title_text='Class',
#         legend=dict(font=dict(size=18)),
#         title=dict(font=dict(size=18))
#     )
#     st.plotly_chart(fig, use_container_width=True)

#     # Interprétation automatique
#     st.markdown("**Interpretation**")
#     if imbalance_ratio > 2:
#         st.warning("⚠️ The dataset is imbalanced. Consider using over or undersampling techniques.")
#     else:
#         st.success("✅ The class distribution appears reasonably balanced. No major concern regarding imbalance.")

#     return True


# ─────────────────────────────────────────────────────────────────────────────
# PUBLICATION-READY MISSING VALUE & ZERO-INFLATION PLOTS
# ─────────────────────────────────────────────────────────────────────────────
import plotly.graph_objects as go
import numpy as np

_PUB = dict(
    plot_bgcolor="white", paper_bgcolor="white",
    font=dict(family="Arial", color="black"),
)

def _axis(label, size=16):
    return dict(title=dict(text=label, font=dict(size=size, color="black", family="Arial Black")),
                tickfont=dict(size=14, color="black", family="Arial"),
                showline=True, linecolor="black", linewidth=1.5,
                showgrid=True, gridcolor="#ebebeb", zeroline=False)


def plot_missing_heatmap(data, max_features: int = 60):
    """
    Heatmap sample × feature: blue = present, white = missing.
    Publication-ready.
    """
    COLUMNS_TO_EXCLUDE = {'Class', 'ID', 'File', 'RT', 'Sum'}
    feat_cols = [c for c in data.columns if c not in COLUMNS_TO_EXCLUDE and not str(c).endswith('_meta') and pd.api.types.is_numeric_dtype(data[c])][:max_features]
    sub = data[feat_cols]
    missing_matrix = sub.isnull().astype(int).values  # 1 = missing

    # Sample labels: use ID if available, else index
    if 'ID' in data.columns:
        ylabels = data['ID'].astype(str).tolist()
    else:
        ylabels = [str(i) for i in data.index]

    fig = go.Figure(data=go.Heatmap(
        z=missing_matrix,
        x=[str(c) for c in feat_cols],
        y=ylabels,
        colorscale=[[0, "#2563eb"], [1, "#f1f5f9"]],
        showscale=True,
        colorbar=dict(
            title=dict(text="Missing", font=dict(size=13, color="black")),
            tickvals=[0, 1], ticktext=["Present", "Missing"],
            tickfont=dict(size=12, color="black")
        ),
        hovertemplate="Feature: %{x}<br>Sample: %{y}<br>Status: %{text}<extra></extra>",
        text=[["Missing" if v == 1 else "Present" for v in row] for row in missing_matrix]
    ))
    n_feat = len(feat_cols)
    fig.update_layout(
        title=dict(text="Missing Value Pattern (Sample × Feature)",
                   font=dict(size=20, color="black", family="Arial Black")),
        xaxis=dict(title=dict(text="Features", font=dict(size=15, color="black", family="Arial Black")),
                   tickfont=dict(size=10, color="black"), tickangle=-45,
                   showline=True, linecolor="black"),
        yaxis=dict(title=dict(text="Samples", font=dict(size=15, color="black", family="Arial Black")),
                   tickfont=dict(size=11, color="black"),
                   showline=True, linecolor="black"),
        height=max(400, min(800, len(ylabels) * 14 + 120)),
        margin=dict(l=80, r=60, t=60, b=120),
        **_PUB
    )
    return fig


def plot_missing_per_class(data):
    """
    Bar chart: % missing values per feature, grouped by class.
    Publication-ready.
    """
    COLUMNS_TO_EXCLUDE = {'Class', 'ID', 'File', 'RT', 'Sum'}
    feat_cols = [c for c in data.columns if c not in COLUMNS_TO_EXCLUDE and not str(c).endswith('_meta') and pd.api.types.is_numeric_dtype(data[c])]
    if 'Class' not in data.columns or not feat_cols:
        return None

    class_colors = st.session_state.get('class_colors', {})
    palette = px.colors.qualitative.Plotly

    rows = []
    for cls in data['Class'].dropna().unique():
        sub = data[data['Class'] == cls][feat_cols]
        miss_pct = sub.isnull().mean() * 100
        # Only keep top-20 most missing
        top = miss_pct.nlargest(20)
        for feat, pct in top.items():
            rows.append({"Feature": str(feat), "Class": cls, "Missing (%)": round(pct, 2)})

    if not rows:
        return None

    df_plot = pd.DataFrame(rows)
    color_map = {cls: class_colors.get(cls, palette[i % len(palette)])
                 for i, cls in enumerate(df_plot['Class'].unique())}

    fig = px.bar(df_plot, x="Feature", y="Missing (%)", color="Class",
                 barmode="group", color_discrete_map=color_map,
                 title="Top-20 Most Missing Features per Class")
    fig.update_layout(
        xaxis=_axis("Feature"),
        yaxis=_axis("Missing (%)"),
        legend=dict(font=dict(size=14, color="black", family="Arial"),
                    title=dict(font=dict(size=14, color="black"))),
        title=dict(font=dict(size=20, color="black", family="Arial Black")),
        height=480,
        margin=dict(l=60, r=30, t=60, b=120),
        **_PUB
    )
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=11))
    return fig


def plot_zero_inflation_per_class(data):
    """
    Violin + strip plot: % zeros per sample, coloured by class.
    Publication-ready.
    """
    COLUMNS_TO_EXCLUDE = {'Class', 'ID', 'File', 'RT', 'Sum'}
    feat_cols = [c for c in data.columns if c not in COLUMNS_TO_EXCLUDE and not str(c).endswith('_meta') and pd.api.types.is_numeric_dtype(data[c])]
    if 'Class' not in data.columns or not feat_cols:
        return None

    class_colors = st.session_state.get('class_colors', {})
    palette = px.colors.qualitative.Plotly

    sub = data[feat_cols].fillna(0)
    zero_pct = (sub == 0).mean(axis=1) * 100
    df_plot = pd.DataFrame({
        "Sample": data.get('ID', pd.Series(range(len(data)))).astype(str).values,
        "Class": data['Class'].values,
        "Zero-inflated (%)": zero_pct.values
    })

    color_map = {cls: class_colors.get(cls, palette[i % len(palette)])
                 for i, cls in enumerate(df_plot['Class'].unique())}

    fig = px.violin(df_plot, x="Class", y="Zero-inflated (%)", color="Class",
                    box=True, points="all", color_discrete_map=color_map,
                    hover_data=["Sample"],
                    title="Zero-Inflation Distribution per Class")
    fig.update_layout(
        xaxis=_axis("Class"),
        yaxis=_axis("% Zero values per sample"),
        legend=dict(font=dict(size=14, color="black", family="Arial"),
                    title=dict(font=dict(size=14, color="black"))),
        title=dict(font=dict(size=20, color="black", family="Arial Black")),
        showlegend=False,
        height=480,
        margin=dict(l=60, r=30, t=60, b=60),
        **_PUB
    )
    return fig


def plot_feature_completeness_rank(data, top_n: int = 40):
    """
    Horizontal bar: features ranked by completeness (% non-missing).
    Shows worst and best features. Publication-ready.
    """
    COLUMNS_TO_EXCLUDE = {'Class', 'ID', 'File', 'RT', 'Sum'}
    feat_cols = [c for c in data.columns if c not in COLUMNS_TO_EXCLUDE and not str(c).endswith('_meta') and pd.api.types.is_numeric_dtype(data[c])]
    if not feat_cols:
        return None

    completeness = (data[feat_cols].notna().mean() * 100).sort_values()
    # Show bottom N/2 and top N/2
    half = top_n // 2
    show = pd.concat([completeness.head(half), completeness.tail(half)]).drop_duplicates()

    colors = ["#dc2626" if v < 50 else "#16a34a" if v > 80 else "#f59e0b"
              for v in show.values]

    fig = go.Figure(go.Bar(
        x=show.values, y=[str(c) for c in show.index],
        orientation='h',
        marker_color=colors,
        text=[f"{v:.1f}%" for v in show.values],
        textposition='outside',
        textfont=dict(size=12, color="black", family="Arial")
    ))
    fig.update_layout(
        title=dict(text=f"Feature Completeness (top & bottom {half})",
                   font=dict(size=20, color="black", family="Arial Black")),
        xaxis=dict(title=dict(text="Completeness (%)", font=dict(size=15, color="black", family="Arial Black")),
                   tickfont=dict(size=13, color="black"), range=[0, 115],
                   showline=True, linecolor="black"),
        yaxis=dict(tickfont=dict(size=11, color="black"),
                   showline=True, linecolor="black"),
        height=max(400, len(show) * 22 + 120),
        margin=dict(l=200, r=80, t=60, b=60),
        **_PUB
    )
    # Threshold lines
    for thresh, color, label in [(50, "#dc2626", "50% threshold"), (80, "#16a34a", "80% threshold")]:
        fig.add_vline(x=thresh, line_dash="dash", line_color=color, line_width=1.5,
                      annotation_text=label,
                      annotation_font=dict(size=12, color=color))
    return fig

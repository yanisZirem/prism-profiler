
"""
Software Name: Profiler
Module name : Vizualisation
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

import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import shap
import pandas as pd
import seaborn as sns
import gc
from itertools import combinations
from collections import defaultdict


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



def display_dataframe_section(df):
    """
    Affiche tout le DataFrame sans restrictions sur les colonnes ou les lignes.
    
    :param df: DataFrame à afficher.
    """
    if df.empty:
        st.warning("DataFrame Empy.")
        return

    # Affiche tout le DataFrame
    st.write("Complete DataFrame overview:")
    st.dataframe(df, hide_index=True)  # Utilisation de st.dataframe pour permettre le défilement et l'exploration



def plot_mean_spectrum(df, selected_classes, class_colors, capture_name=None):
    """Plots the mean spectra for multiple selected classes using Plotly."""
    # Drop non-spectral columns if present
    columns_to_drop = [col for col in ['File', 'RT', 'Sum'] if col in df.columns]
    data_filtered = df.drop(columns=columns_to_drop)
    
    # Compute the mean spectrum for each class
    mean_spectra = data_filtered.groupby('Class').mean(numeric_only=True)
    
    # Create a Plotly figure
    fig = go.Figure()
    
    for selected_class in selected_classes:
        if selected_class in mean_spectra.index:
            mean_spectrum = mean_spectra.loc[selected_class]
            mz_values = mean_spectrum.index  # m/z values
            intensities = mean_spectrum.values  # Mean intensities
            
            fig.add_trace(go.Scatter(
                x=mz_values, 
                y=intensities, 
                mode='lines', 
                name=f"{selected_class}", 
                line=dict(color=class_colors.get(selected_class, 'black'))
            ))
    
    # fig.update_layout(
    #     title="Mean Spectra",
    #     title_font=dict(size=24, color="black"),
    #     font=dict(color="black", size=22),
    #     xaxis=dict(
    #         title="m/z",
    #         titlefont=dict(size=24, color='black'),
    #         tickfont=dict(size=22, color='black'),
    #         showgrid=False,  # Disable y-axis grid,
    #         showline=True
    #     ),
    #     yaxis=dict(
    #         title="Intensity",
    #         titlefont=dict(size=24, color='black'),
    #         tickfont=dict(size=22, color='black'),
    #         showgrid=False,  # Disable y-axis grid,
    #         showline=True
    #     ),
    #     legend=dict(
    #         font=dict(size=24, color='black')
    #     ),
    #     plot_bgcolor='white',
    #     paper_bgcolor='white'
    # )

    fig.update_layout(
        title=dict(
            text="Skyline",
            font=dict(size=24, color="black")
        ),
        font=dict(color="black", size=22),
        xaxis=dict(
            title=dict(
                text="Features",
                font=dict(size=24, color='black')
            ),
            tickfont=dict(size=22, color='black'),
            showgrid=False,
            showline=True
        ),
        yaxis=dict(
            title=dict(
                text="Intensity",
                font=dict(size=24, color='black')
            ),
            tickfont=dict(size=22, color='black'),
            showgrid=False,
            showline=True
        ),
        legend=dict(
            font=dict(size=24, color='black')
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )


    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True, 'modeBarButtonsToAdd': ['downloadImage'], 'toImageButtonOptions': {'format': 'png', 'scale': 2}})
    if capture_name:
        _capture_plotly(fig, capture_name)

def display_data_section(title, df):
    """
    Affiche tout le DataFrame avec Streamlit, de manière interactive.
    
    :param title: Titre de la section.
    :param df: DataFrame à afficher.
    """
    if df.empty:
        st.warning("The DataFrame is empty.")
        return

    st.write(title)
    # Affiche le DataFrame complet de manière interactive avec défilement
    st.dataframe(df, hide_index=True)


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


def plot_individual_spectra(data, class_colors, selected_indices, capture_name=None):
    spectra = data.drop(columns=['Class', 'File', 'RT', 'Sum'],errors='ignore')
    spectra['Index'] = spectra.index
    # spectra['File'] = data['File']
    spectra['Class'] = data['Class']
    spectra['Color'] = spectra['Class'].map(class_colors)

    # Filtrer pour ne garder que les indices sélectionnés
    spectra = spectra[spectra['Index'].isin(selected_indices)]

    # Conversion du format large en format long
    spectra_long = spectra.melt(id_vars=['Index', 'Class'], var_name='m/z', value_name='Intensity')

    # Vérifier si le DataFrame est vide après le filtrage
    if spectra_long.empty:
        st.warning("Aucun spectre disponible pour les indices sélectionnés.")
        return

  # Plot the spectrum with the correct format
    fig = px.line(spectra_long, x='m/z', y='Intensity', color='Class',
                  color_discrete_map=class_colors, title='Individual Spectra')

    fig.update_traces(mode='lines')
    # fig.update_layout(
    #     xaxis_title='Features',
    #     yaxis_title='Intensity',
    #     legend_title='Class',
    #     title="Skyline",
    #     title_font=dict(size=24, color="black"),
    #     font=dict(color="black", size=22),
    #     xaxis=dict(
    #         title="m/z",
    #         titlefont=dict(size=24, color='black'),
    #         tickfont=dict(size=22, color='black'),
    #         showgrid=False,  # Disable x-axis grid
    #         showline=True
    #     ),
    #     yaxis=dict(
    #         title="Intensity",
    #         titlefont=dict(size=24, color='black'),
    #         tickfont=dict(size=22, color='black'),
    #         showgrid=False,  # Disable y-axis grid
    #         showline=True
    #     ),
    #     legend=dict(
    #         font=dict(size=24, color='black')
    #     ),
    #     plot_bgcolor='white',
    #     paper_bgcolor='white'
    # )


    fig.update_layout(
        title=dict(
            text="Skyline",
            font=dict(size=24, color="black")
        ),
        font=dict(color="black", size=22),

        xaxis=dict(
            title=dict(
                text="Features",
                font=dict(size=24, color='black')
            ),
            tickfont=dict(size=22, color='black'),
            showgrid=False,
            showline=True
        ),

        yaxis=dict(
            title=dict(
                text="Intensity",
                font=dict(size=24, color='black')
            ),
            tickfont=dict(size=22, color='black'),
            showgrid=False,
            showline=True
        ),

        legend=dict(
            title=dict(
                text="Class",
                font=dict(size=24, color='black')
            ),
            font=dict(size=22, color='black')
        ),

        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True, 'modeBarButtonsToAdd': ['downloadImage'], 'toImageButtonOptions': {'format': 'png', 'scale': 2}})
    if capture_name:
        _capture_plotly(fig, capture_name)
    return fig

# Function to apply dark mode
def apply_dark_mode():
    dark_style = """
        <style>
            /* Global background & text */
            body, .stApp, .css-18e3th9, .css-1d391kg {
                background-color: #121212 !important;
                color: #E0E0E0 !important;
            }

            /* Inputs & Select boxes */
            .stTextInput>div>div>input,
            .stSelectbox>div>div>div,
            .stTextArea>div>textarea,
            .stNumberInput div div input {
                background-color: #1E1E1E !important;
                color: #F0A500 !important;
                border: 1px solid #F0A500 !important;
                border-radius: 8px !important;
                padding: 8px !important;
            }

            /* Buttons */
            .stButton>button {
                background-color: #2A2A2A !important;
                color: #FF4C4C !important;
                border: 1px solid #FF4C4C !important;
                font-size: 14px !important;
                padding: 8px 14px !important;
                border-radius: 8px !important;
                transition: all 0.3s ease-in-out;
            }

            .stButton>button:hover {
                background-color: #FF4C4C !important;
                color: white !important;
            }

            /* Tabs */
            .stTabs [data-baseweb="tab-list"], 
            .stTabs [data-baseweb="tab"] {
                background-color: #1A1A1A !important;
                color: #E0E0E0 !important;
                border-radius: 8px !important;
                padding: 5px !important;
            }

            /* Expanders */
            .st-expander {
                background-color: #222 !important;
                color: #E0E0E0 !important;
                border: 1px solid #E0E0E0 !important;
                border-radius: 8px !important;
                padding: 8px !important;
            }

            .st-expander-header {
                color: white !important;
                font-weight: bold !important;
            }

            /* Sliders & Checkboxes */
            .stSlider .stMarkdown,
            .stCheckbox div label {
                color: #E0E0E0 !important;
            }
        </style>
    """
    st.markdown(dark_style, unsafe_allow_html=True)






def plot_feature_distribution(data, feature, class_colors):
    if feature == 'Class':
        return px.histogram(data, x='Class', title=f"Distribution of {feature}", color='Class', color_discrete_map=class_colors)
    else:
        return px.histogram(data, x='Class', y=feature, histfunc='sum', title=f"Sum of {feature} Intensities by Class", color='Class', color_discrete_map=class_colors)
    



def plot_protein_expression_bubble(protein_list, data):
    """
    Plot a bubble plot showing the overexpression and underexpression of the specified proteins across conditions using Plotly.
    Args:
        protein_list (list): List of protein names to visualize.
        data (pd.DataFrame): Data containing expression values and conditions.
    """
    # Dynamically create a color mapping for each class
    unique_classes = data['Class'].unique()
    class_colors = {cls: color for cls, color in zip(unique_classes, px.colors.qualitative.Plotly)}

    # Create a DataFrame to store the bubble plot data
    bubble_data = []

    for protein in protein_list:
        # Filter data for the current protein
        protein_data = data.loc[:, [protein, 'Class']]  # Include the 'Class' column

        # Convert 'Expression' column to numeric
        protein_data['Expression'] = pd.to_numeric(protein_data[protein], errors='coerce')

        # Calculate mean expression for each condition
        mean_expression = protein_data.groupby('Class')['Expression'].mean()

        # Calculate z-scores for bubble sizes
        z_scores = (mean_expression - mean_expression.min()) / (mean_expression.max() - mean_expression.min())

        # Create bubbles for each condition
        for condition, z_score in zip(mean_expression.index, z_scores.values):
            bubble_data.append({'Protein': protein, 'Condition': condition, 'Expression': z_score})

    # Create a DataFrame from the bubble data
    bubble_df = pd.DataFrame(bubble_data)

    # Create the bubble plot using Plotly Express
    fig = px.scatter(bubble_df, x='Condition', y='Protein', size='Expression', color='Condition',
                     color_discrete_map=class_colors, labels={'Expression': 'Relative Expression'},
                     title='Protein Expression Bubble Plot', width=1600, height=1400)

    # Add color bar
    fig.update_layout(coloraxis_colorbar=dict(
        title='Condition',
        tickvals=[],
        ticktext=[]
    ))

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True, 'modeBarButtonsToAdd': ['downloadImage'], 'toImageButtonOptions': {'format': 'png', 'scale': 2}})



# # --- Fonction utilitaire pour afficher une heatmap ---
# def plot_heatmap(df, title, caption=None, capture_name=None):
#     n_classes = len(df.columns)
#     figsize = max(8, n_classes * 0.5)
#     fig, ax = plt.subplots(figsize=(figsize, figsize))
#     sns.heatmap(df, annot=True, fmt=".2f", cmap="jet", square=True, ax=ax)
#     ax.set_title(title, fontsize=14)
#     st.pyplot(fig)
#     if caption:
#         st.caption(caption)
#     del fig, ax
#     import gc
#     gc.collect()
#     if capture_name:
#         _capture_matplotlib(fig, capture_name)
    

# --- Fonction utilitaire pour afficher une heatmap ---
def plot_heatmap(df, title, caption=None, capture_name=None):
    import gc

    n_classes = len(df.columns)
    figsize = max(8, n_classes * 0.5)

    # Création de la figure
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="jet", square=True, ax=ax)

    # Titre
    ax.set_title(title, fontsize=14)

    # Affichage dans Streamlit
    st.pyplot(fig)

    # Affichage de la caption si fournie
    if caption:
        st.caption(caption)

    # Capture de la figure si nécessaire
    if capture_name:
        _capture_matplotlib(fig, capture_name)

    # Libération mémoire
    del fig, ax
    gc.collect()





# --- Fonction optimisée pour les intersections ---
def calculate_maximal_intersections(venn_data, classes):
    class_features = {}
    for cls in classes:
        cls_features = set(venn_data[venn_data['Class'] == cls].drop(columns=['Class']).columns)
        class_features[cls] = cls_features

    feature_counts = defaultdict(int)
    feature_to_classes = defaultdict(set)

    for cls, features in class_features.items():
        for feature in features:
            feature_counts[feature] += 1
            feature_to_classes[feature].add(cls)

    shared_features = {feature: classes for feature, classes in feature_to_classes.items() if len(classes) >= 2}

    maximal_intersections = defaultdict(set)
    for feature, cls_set in shared_features.items():
        for r in range(2, len(cls_set) + 1):
            for combo in combinations(cls_set, r):
                maximal_intersections[combo].add(feature)

    maximal_intersections_filtered = {}
    for combo, features in maximal_intersections.items():
        is_maximal = True
        for other_combo, other_features in maximal_intersections.items():
            if set(combo).issubset(other_combo) and features <= other_features and combo != other_combo:
                is_maximal = False
                break
        if is_maximal:
            maximal_intersections_filtered[combo] = features

    return maximal_intersections_filtered




def calculate_all_intersections(venn_data, classes):
    # Dictionnaire {classe: set(features non-nulles)}
    class_features = {}
    relevant_cols = [col for col in venn_data.columns if col != 'Class']
    
    for cls in classes:
        mask = venn_data['Class'] == cls
        cls_features = set(venn_data.loc[mask, relevant_cols].columns[venn_data.loc[mask, relevant_cols].notnull().any()])
        class_features[cls] = cls_features

    all_intersections = {}
    used_features = set()  # pour éviter les duplications

    # On parcourt d'abord les combinaisons les plus larges (N classes), puis descend
    for r in range(len(classes), 1, -1):
        for combo in combinations(classes, r):
            intersect = set.intersection(*(class_features[cls] for cls in combo))
            # On enlève ce qui a déjà été attribué à une intersection plus grande
            intersect -= used_features
            if intersect:
                all_intersections[combo] = intersect
                used_features.update(intersect)  # marquer ces features comme utilisées

    return all_intersections
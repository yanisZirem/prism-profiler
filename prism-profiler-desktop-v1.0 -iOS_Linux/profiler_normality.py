"""
Software Name: Profiler
Module name : data preparation
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
import streamlit as st
from scipy.stats import ttest_ind, shapiro, kstest
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

def diagnose_normality(p_value):
    if p_value > 0.05:
        return "Normal distribution (p-value > 0.05)"
    else:
        return "Non-normal distribution (p-value <= 0.05)"


def display_class_info(data):
    if 'Class' in data.columns:


        # Colonnes à exclure pour le comptage des features
        COLUMNS_TO_EXCLUDE = ['Class', 'File', 'RT', 'Sum']
        num_features = len([col for col in data.columns if col not in COLUMNS_TO_EXCLUDE])

        # Comptage des classes
        class_counts = data['Class'].value_counts()
        class_percentages = (class_counts / len(data)) * 100
        n_classes = class_counts.shape[0]
        imbalance_ratio = class_counts.max() / class_counts.min()



        # Tableau de distribution
        class_info_df = pd.DataFrame({
            "Class": class_counts.index,
            "Count": class_counts.values,
            "Percentage (%)": class_percentages.round(2).values
        })

        st.markdown("**Class Distribution Summary**")
        st.dataframe(class_info_df, use_container_width=True)


        st.markdown("**Class Proportion**")

        classes = class_counts.index
        colors = [st.session_state['class_colors'].get(cls, '#CCCCCC') for cls in classes]  

        fig2, ax2 = plt.subplots()
        ax2.pie(class_counts, labels=classes, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        ax2.axis('equal')
        st.pyplot(fig2)


        # Interprétation automatique
        st.markdown("**Interpretation**")
        if imbalance_ratio > 2:
            st.warning("⚠️ The dataset is imbalanced. Consider using over or undersampling techniques.")
        else:
            st.success("✅ The class distribution appears reasonably balanced. No major concern regarding imbalance.")

        return True
    else:
        st.warning("⚠️ No 'Class' column found in the dataset.")
        return False

def calculate_missing_values(data):
    COLUMNS_TO_EXCLUDE = ['Class', 'File', 'RT', 'Sum']
    relevant_columns = [col for col in data.columns if col not in COLUMNS_TO_EXCLUDE]
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




def display_class_info(data):
    if 'Class' not in data.columns:
        st.warning("⚠️ No 'Class' column found in the dataset.")
        return False

    # Colonnes à exclure pour le comptage des features
    COLUMNS_TO_EXCLUDE = ['Class', 'File', 'RT', 'Sum']
    num_features = len([col for col in data.columns if col not in COLUMNS_TO_EXCLUDE])

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
    colors = [st.session_state['class_colors'].get(cls, '#CCCCCC') for cls in class_counts.index]  

    import plotly.express as px
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
        textfont_size=20  
    )
    fig.update_layout(
        legend_title_text='Class',
        legend=dict(font=dict(size=18)),   
        title=dict(font=dict(size=18))     
    )
    st.plotly_chart(fig, use_container_width=True)

    # Interprétation automatique
    st.markdown("**Interpretation**")
    if imbalance_ratio > 2:
        st.warning("⚠️ The dataset is imbalanced. Consider using over or undersampling techniques.")
    else:
        st.success("✅ The class distribution appears reasonably balanced. No major concern regarding imbalance.")

    return True


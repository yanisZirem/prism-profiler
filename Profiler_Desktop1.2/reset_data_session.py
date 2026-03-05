import streamlit as st 
def reset_data_session_keys():
    keys_to_clear = [
        "data",
        "preprocessed_data",
        "compressed_data",
        "reduced_data",
        "models",
        "dl_models",
        "class_colors",
        "feature_distribution_plot",
        "mean_spectrum_plot",
        "individual_spectra_plot",
        "pc_index",
        "file_groups",
        "class_renaming",
        "rename_pending",
        "feature_data_source_1",
        "feature_data_source_2",
        "feature_data_source_3",
        "show_shap",
        "show_lime",
        "show_boxplots",
        "show_heatmap",
        "show_volcano",
        "class_column",
        "is_maxquant",
        "selected_feature_row",
        "selected_data_source",
        "latest_result",
        "processed_files",
        "monitoring",
        "normalization",
        "numeric_features",
        "label_encoder",
        "svd_model",
        "label_colors",
        "selected_feature",
        "survival_data"
    ]

    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    # Réinitialisation des clés obligatoires à vide
    st.session_state['file_groups'] = []
    st.session_state['class_renaming'] = {}
    st.session_state['class_colors'] = {}
    st.session_state['models'] = {}
    st.session_state['dl_models'] = {}
    st.session_state['data'] = None
    st.session_state['preprocessed_data'] = None

    st.session_state['compressed_data'] = None
    st.session_state['reduced_data'] = None
    st.session_state['feature_distribution_plot'] = None
    st.session_state['mean_spectrum_plot'] = None
    st.session_state['individual_spectra_plot'] = None
    st.session_state['pc_index'] = 0
    st.session_state['rename_pending'] = {}
    st.session_state['feature_data_source_1'] = 'None'
    st.session_state['feature_data_source_2'] = 'None'
    st.session_state['feature_data_source_3'] = 'None'
    st.session_state['show_shap'] = False
    st.session_state['show_lime'] = False
    st.session_state['show_boxplots'] = False
    st.session_state['show_heatmap'] = False
    st.session_state['show_volcano'] = False
    st.session_state['class_column'] = 'Class'
    st.session_state['is_maxquant'] = False
    st.session_state['selected_feature_row'] = "Choose an option"
    st.session_state['selected_data_source'] = 'None'
    st.session_state['latest_result'] = None
    st.session_state['processed_files'] = set()
    st.session_state['monitoring'] = False
    st.session_state['normalization'] = "None"
    st.session_state['numeric_features'] = []
    st.session_state['label_encoder'] = None
    st.session_state['svd_model'] = None
    st.session_state['label_colors'] = {}
    st.session_state['selected_feature'] = None
    st.session_state['survival_data'] = None
    st.session_state['expand_load_data'] = True
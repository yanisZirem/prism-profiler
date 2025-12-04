import os
import time
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from pyopenms import MSExperiment, MzMLFile
from scipy.signal import find_peaks, detrend
from scipy.stats import mode
import subprocess
import plotly.graph_objects as go
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import os
import zipfile
import subprocess
import streamlit as st

# # Load models, features, and label encoder
# def load_all_rt(model_files, feature_file, label_encoder_file):
#     if model_files:
#         st.session_state['models'] = [joblib.load(file) for file in model_files]
#         st.success("Models loaded successfully.")
#     if feature_file:
#         try:
#             numeric_features = joblib.load(feature_file)
#             st.session_state['numeric_features'] = numeric_features
#             st.success("Numeric features loaded successfully.")
#         except Exception as e:
#             st.error(f"Error loading numeric features: {e}")
#     if label_encoder_file:
#         try:
#             label_encoder = joblib.load(label_encoder_file)
#             st.session_state['label_encoder'] = label_encoder
#             st.success("Label encoder loaded successfully.")
#         except Exception as e:
#             st.error(f"Error loading label encoder: {e}")

def load_all_rt(model_files, feature_file, label_encoder_file):
    import joblib
    import streamlit as st

    # --- Charger les modèles ---
    if model_files:
        try:
            models = []
            for file in model_files:
                model_data = joblib.load(file)
                # ✅ compatibilité avec anciens/nouveaux formats
                if isinstance(model_data, dict) and 'model' in model_data:
                    model = model_data['model']  # extrait le vrai pipeline
                else:
                    model = model_data  # ancien format
                models.append(model)

            st.session_state['models'] = models
            st.success(f"✅ {len(models)} model(s) loaded successfully.")
        except Exception as e:
            st.error(f"❌ Error loading models: {e}")

    # --- Charger les features ---
    if feature_file:
        try:
            numeric_features = joblib.load(feature_file)
            st.session_state['numeric_features'] = numeric_features
            st.success("✅ Numeric features loaded successfully.")
        except Exception as e:
            st.error(f"❌ Error loading numeric features: {e}")

    # --- Charger le label encoder ---
    if label_encoder_file:
        try:
            label_encoder = joblib.load(label_encoder_file)
            st.session_state['label_encoder'] = label_encoder
            st.success("✅ Label encoder loaded successfully.")
        except Exception as e:
            st.error(f"❌ Error loading label encoder: {e}")




def convert_raw_to_mzml_rt(raw_dir, output_dir):
    raw_files = [f for f in os.listdir(raw_dir) if f.lower().endswith('.raw') and f not in st.session_state['processed_files']]
    if not raw_files:
        return False

    for raw_file in raw_files:
        raw_file_path = os.path.join(raw_dir, raw_file)
        command = [
            "msconvert", raw_file_path,
            "--filter", "lockmassRefiner mz=524 tol=1.0",
            "-o", output_dir, "--mzML"
        ]
        try:
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            st.session_state['processed_files'].add(raw_file)
            st.write(f"Converted {raw_file} to mzML.")
        except subprocess.CalledProcessError as e:
            st.error(f"Conversion failed: {e}")
    return True



def load_data_single_file_rt(file_path, class_name="Unknown"):
    all_data = pd.DataFrame()

    try:
        exp = MSExperiment()
        MzMLFile().load(file_path, exp)
    except RuntimeError as e:
        st.error(f"Error loading mzML file: {e}")
        return pd.DataFrame()

    chromatograms = exp.getChromatograms()
    if not chromatograms:
        st.warning(f"No chromatograms found in {file_path}")
        return pd.DataFrame()

    # Process chromatograms using peak detection logic
    chromatogram = chromatograms[0]
    times, intensities = chromatogram.get_peaks()
    if len(times) == 0 or len(intensities) == 0:
        st.warning(f"Chromatogram has no data in {file_path}")
        return pd.DataFrame()

    # Detrend the intensities (similar to the first function)
    intensities = detrend(intensities)

    # Peak detection based on chromatogram intensities
    peak_height_threshold = np.max(intensities) * 0.5
    peaks, _ = find_peaks(intensities, height=peak_height_threshold)

    spectra = exp.getSpectra()
    if not spectra:
        st.warning(f"No spectra found in {file_path}")
        return pd.DataFrame()

    # Select spectra based on the chromatogram peak retention times (RT)
    peak_rt_values = [times[i] for i in peaks]
    peak_spectra = []

    for spectrum in spectra:
        rt_value = spectrum.getRT()
        if any(abs(rt_value - rt) < 1 for rt in peak_rt_values):  # 1 is the RT tolerance threshold
            peak_spectra.append(spectrum)

    if not peak_spectra:
        st.warning(f"No peak spectra found in {file_path}")
        return pd.DataFrame()

    # Collect m/z values from all spectra
    all_mz_values = set()
    for spectrum in peak_spectra:
        mz_values, _ = spectrum.get_peaks()
        if len(mz_values) > 0:
            all_mz_values.update(mz_values)

    all_mz_values = sorted(all_mz_values)

    # Extract data for the DataFrame
    data = []
    for spectrum in peak_spectra:
        mz_values, intensities = spectrum.get_peaks()
        rt_value = spectrum.getRT()
        
        # Create a dictionary with m/z values as keys
        intensity_dict = dict(zip(mz_values, intensities))
        row_data = {mz: intensity_dict.get(mz, np.nan) for mz in all_mz_values}

        # Update with additional information
        row_data.update({
            'Class': class_name,
            'File': os.path.basename(file_path),
            'RT': rt_value,
            'Sum': sum(intensities)
        })
        data.append(row_data)

    # Convert to DataFrame and append to all_data
    file_data = pd.DataFrame(data)
    all_data = pd.concat([all_data, file_data], ignore_index=True)

    return all_data

def preprocess_data_rt(all_data, normalization_type=None):
    if all_data.empty:
        st.warning("Data is empty after loading.")
        return pd.DataFrame()

    intensity_columns = all_data.columns.drop(['Class', 'File', 'RT', 'Sum'], errors='ignore')

    if normalization_type == 'TIC':
        all_data[intensity_columns] = all_data[intensity_columns].div(all_data[intensity_columns].sum(axis=1), axis=0)
    elif normalization_type == 'RMS':
        all_data[intensity_columns] = all_data[intensity_columns].div(
            np.sqrt((all_data[intensity_columns] ** 2).sum(axis=1)), axis=0
        )
    elif normalization_type == 'BasePeak':
        all_data[intensity_columns] = all_data[intensity_columns].div(all_data[intensity_columns].max(axis=1), axis=0)

    return all_data

def apply_binning_to_mass_range_rt(data, bin_width=0.1, mass_range=(600, 1000)):
    """
    Auteur : Yanis Zirem . yanis.zirem@univ-lille.fr ot yanis.zirem2016@gmail.com
    Regroupe les colonnes numériques basées sur une plage de masse et une largeur de bin définie.
    - Le nom de chaque nouvelle colonne est le centre du bin.
    - La valeur correspondante est la somme des valeurs des colonnes appartenant au bin.

    :param data: DataFrame contenant les données.
    :param bin_width: Largeur de chaque bin (par exemple 0.1 Da).
    :param mass_range: Tuple définissant la plage de masse (min, max).
    :return: DataFrame avec les colonnes regroupées par bins.
    """
    if bin_width <= 0:
        raise ValueError("bin_width must be greater than zero.")
    
    min_mass, max_mass = mass_range
    if min_mass >= max_mass:
        raise ValueError("Invalid mass range. Ensure min_mass < max_mass.")
    
    # Colonnes fixes à ne pas modifier
    fixed_cols = ["Class", "File", "RT", "Sum"]
    fixed_data = data[fixed_cols]  # Les colonnes à conserver telles quelles

    # Colonnes numériques restantes
    numerical_cols = data.drop(columns=fixed_cols).columns

    # Créer les bins
    bins = np.arange(min_mass, max_mass, bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2  # Centres des bins
    bin_labels = [round(center, 3) for center in bin_centers]

    # Assigner chaque colonne numérique à un bin
    binned_data = pd.DataFrame(0, index=data.index, columns=bin_labels)  # Colonnes binned initialisées à 0
    for col in numerical_cols:
        mass = float(col)  # Convertir le nom de la colonne en float
        bin_index = np.digitize(mass, bins) - 1  # Trouver le bin correspondant (0-based index)
        if 0 <= bin_index < len(bin_centers):  # Vérifier si la masse est dans la plage
            binned_data[bin_labels[bin_index]] += data[col]  # Ajouter les valeurs à la bonne colonne bin

    # Combiner les colonnes fixes et les colonnes binées
    result = pd.concat([fixed_data, binned_data], axis=1)

    return result

def apply_svd_rt(data, n_components):
    data.columns = data.columns.astype(str)
    numeric_data = data.select_dtypes(include=[np.number])

    if numeric_data.shape[1] < 2:
        st.warning("Not enough numeric features for SVD. Please check your data.")
        return data

    imputer = SimpleImputer(strategy='mean')
    numeric_data_imputed = imputer.fit_transform(numeric_data)

    svd = TruncatedSVD(n_components=n_components)
    svd_data = svd.fit_transform(numeric_data_imputed)

    st.session_state['svd_model'] = svd
    return svd_data


def decision_rt(processed_data, use_svd=False):
    if st.session_state['numeric_features'] and st.session_state['models']:
        if use_svd and st.session_state['svd_model'] is not None:
            processed_data = st.session_state['svd_model'].transform(processed_data)
        else:
            processed_data = processed_data.reindex(columns=st.session_state['numeric_features'], fill_value=0)

        predictions = np.array([model.predict(processed_data) for model in st.session_state['models']])
        majority_predictions, counts = mode(predictions, axis=0)
        percentages = (counts / len(st.session_state['models'])) * 100

        if st.session_state['label_encoder'] is not None:
            categories = st.session_state['label_encoder'].inverse_transform(majority_predictions.flatten())
        else:
            categories = [str(label) for label in np.unique(majority_predictions)]

        result_df = pd.DataFrame({
            'Category': categories,
            'Confidence (%)': percentages.flatten()
        })

        return result_df
    else:
        st.warning("Numeric features or models are missing.")
        return pd.DataFrame({"Category": ["None"], "Confidence (%)": ["None"]})
# def decision_rt(processed_data, use_svd=False):
#     if st.session_state['numeric_features'] and st.session_state['models']:
#         if use_svd and st.session_state['svd_model'] is not None:
#             processed_data = st.session_state['svd_model'].transform(processed_data)
#         else:
#             processed_data = processed_data.reindex(columns=st.session_state['numeric_features'], fill_value=0)

#         # Gérer le cas où 'models' peut être un seul modèle ou plusieurs
#         predictions = []
        
#         # Si st.session_state['models'] contient plusieurs modèles
#         if isinstance(st.session_state['models'], list):
#             for model in st.session_state['models']:
#                 if hasattr(model, 'predict'):
#                     predictions.append(model.predict(processed_data))
#                 else:
#                     st.error(f"Le modèle {model} ne dispose pas de la méthode 'predict'.")
#                     predictions.append(np.nan)  # ou gérer autrement
#         # Si st.session_state['models'] contient un seul modèle
#         else:
#             if hasattr(st.session_state['models'], 'predict'):
#                 predictions.append(st.session_state['models'].predict(processed_data))
#             else:
#                 st.error(f"Le modèle {st.session_state['models']} ne dispose pas de la méthode 'predict'.")
#                 predictions.append(np.nan)

#         # Conversion des prédictions en tableau numpy et calcul de la majorité
#         predictions = np.array(predictions)
#         majority_predictions, counts = mode(predictions, axis=0)
#         percentages = (counts / len(st.session_state['models'])) * 100

#         # Décodage des prédictions avec le label encoder
#         if st.session_state['label_encoder'] is not None:
#             categories = st.session_state['label_encoder'].inverse_transform(majority_predictions.flatten())
#         else:
#             categories = [str(label) for label in np.unique(majority_predictions)]

#         result_df = pd.DataFrame({
#             'Category': categories,
#             'Confidence (%)': percentages.flatten()
#         })

#         return result_df
#     else:
#         st.warning("Numeric features or models are missing.")
#         return pd.DataFrame({"Category": ["None"], "Confidence (%)": ["None"]})

def visualize_predictions_circles_rt(result_df):
    # Calculate the majority prediction (mode)
    majority_label = result_df['Category'].mode()[0]  # Get the most frequent label
    majority_confidence = result_df[result_df['Category'] == majority_label]['Confidence (%)'].mean()

    # Find if there is a tie in the predictions
    prediction_counts = result_df['Category'].value_counts()
    is_tie = len(prediction_counts) > 1 and prediction_counts.iloc[0] == prediction_counts.iloc[1]

    fig, ax = plt.subplots(figsize=(8, 4))

    # Display only the majority label in a circle
    if not is_tie:
        color = st.session_state['label_colors'].get(majority_label, 'gray')
        display_label = majority_label  # Only show the label for the majority prediction

        # Create the circle for the majority prediction
        circle = plt.Circle((0, 0), majority_confidence * 0.1, color=color, alpha=0.5)
        ax.add_artist(circle)

        # Display the majority label
        ax.text(0, 0, display_label, ha='center', va='center', fontsize=12, color='black')
    else:
        # If there is a tie, display "Outlier" in black
        color = 'black'
        display_label = "Outlier"

        # Create the circle for the outlier
        circle = plt.Circle((0, 0), majority_confidence * 0.1, color=color, alpha=0.5)
        ax.add_artist(circle)

        # Display the outlier label
        ax.text(0, 0, display_label, ha='center', va='center', fontsize=12, color='black')

    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.2, 0.2)
    ax.axis('off')

    plt.title('Prediction Results', fontsize=16)
    st.pyplot(fig)





def convert_raw_to_mzml_rt_multi_format_with_zip(
    raw_dir,
    output_dir,
    file_type="thermo",
    mass_range=None,
    peak_picking=False,
    lock_mass=None,
    output_format='mzML'
):
    file_extensions = {
        "thermo": ".raw",
        "waters": ".raw",
        "bruker": ".d"
    }

    if file_type not in file_extensions:
        st.error(f"Unsupported file type: {file_type}")
        return False

    file_ext = file_extensions[file_type]
    raw_files = []

    # Décompresser les fichiers .zip dans le répertoire
    for f in os.listdir(raw_dir):
        full_path = os.path.join(raw_dir, f)

        # Si c'est un fichier zip, on le décompresse
        if f.lower().endswith('.zip'):
            try:
                with zipfile.ZipFile(full_path, 'r') as zip_ref:
                    zip_ref.extractall(raw_dir)  # Décompresser les fichiers dans le même dossier
                    # st.write(f"📦 {f} décompressé avec succès.")
                os.remove(full_path)  # Supprimer le fichier ZIP après extraction
            except zipfile.BadZipFile:
                st.error(f"❌ Erreur lors de la décompression de {f}.")
                continue

    # Maintenant, on cherche les fichiers .raw et .d extraits
    for f in os.listdir(raw_dir):
        full_path = os.path.join(raw_dir, f)
        if f in st.session_state.get('processed_files', []):
            continue
        if file_type in ["thermo", "waters"]:
            if f.lower().endswith(".raw"):
                raw_files.append(f)
        elif file_type == "bruker":
            if f.lower().endswith(".d"):
                raw_files.append(f)

    if not raw_files:
        # st.error(f"No file {file_ext} found in the folder.")
        return False

    st.write(f"Converting {len(raw_files)} files to {output_format}...")
    progress_bar = st.progress(0)

    def convert_file(raw_file):
        raw_file_path = os.path.join(raw_dir, raw_file)
        output_file_name = os.path.splitext(raw_file)[0] + ".mzML"
        output_file_path = os.path.join(output_dir, output_file_name)

        # Debug: Print the file paths
        # st.write(f"📝 Chemin du fichier RAW : {raw_file_path}")
        # st.write(f"📝 Chemin de sortie : {output_file_path}")

        # Vérifier si le fichier existe
        if not os.path.exists(raw_file_path):
            st.error(f"❌ File not found: {raw_file_path}")
            return

        docker_command = [
            "sudo", "docker", "run",
            "-v", f"{os.path.abspath(raw_dir)}:/data",
            "-v", f"{os.path.abspath(output_dir)}:/out",
            "chambm/pwiz-skyline-i-agree-to-the-vendor-licenses",
            "wine", "msconvert",
            f"/data/{raw_file}",
            f"--{output_format}",
            "--outfile", output_file_name,
            "-o", "/out"
        ]

        # Debug: Print the Docker command
        # st.write(f"🚀 Commande Docker : {' '.join(docker_command)}")

        if file_type == "waters" and lock_mass:
            docker_command += ["--filter", f"lockmassRefiner mz={lock_mass} tol=1.0"]

        if mass_range:
            docker_command += ["--filter", f"mzWindow {mass_range}"]

        if peak_picking:
            docker_command += ["--filter", "peakPicking true 1-"]

        try:
            result = subprocess.run(docker_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            # st.write(f"📤 Docker stdout ({raw_file}):")
            # st.code(result.stdout)

            # st.write(f"🐛 Docker stderr ({raw_file}):")
            # st.code(result.stderr)

            if result.returncode != 0:
                st.error(f"❌ Error with {raw_file}:\n{result.stderr}")
            else:
                # st.success(f"✅ {raw_file} converted.")
                if 'processed_files' not in st.session_state or not isinstance(st.session_state['processed_files'], list):
                    st.session_state['processed_files'] = []

                st.session_state['processed_files'].append(raw_file)
        except Exception as e:
            st.error(f"❌ Exception with {raw_file}: {e}")

    # Exécution séquentielle (pas de parallèle)
    for i, raw_file in enumerate(raw_files):
        convert_file(raw_file)
        progress_bar.progress((i + 1) / len(raw_files))

    return True
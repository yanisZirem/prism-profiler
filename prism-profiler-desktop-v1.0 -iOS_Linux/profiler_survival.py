"""
Software Name: Profiler
Module name : Survivla_meta_data
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



from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import io
from lifelines import CoxPHFitter
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.pipeline import make_pipeline


def create_cox_pipeline(df):
    # Vérification de la présence de 'Class'
    if 'Class' not in df.columns:
        raise ValueError("The column 'Class' is missing from the dataframe.")

    # Détection automatique des colonnes numériques (hors 'Overall survival' et 'State')
    num_cols = [col for col in df.select_dtypes(include=['number']).columns if col not in ['Overall survival', 'State']]
    
    # Détection des colonnes catégorielles
    cat_cols = [col for col in df.select_dtypes(exclude=['number']).columns if col != 'Class']
    cat_cols.append('Class')  # Assurer que 'Class' est bien inclus


    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), num_cols),
            ('cat', make_pipeline(SimpleImputer(strategy='most_frequent'),
                                OneHotEncoder(drop='first', handle_unknown='ignore')), cat_cols)
        ]
    )
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    return pipeline

def detect_delimiter(file_content, delimiters=[',', ';', '\t', '|']):
    """Détecte automatiquement le délimiteur dans un fichier CSV/TXT."""
    first_line = file_content.split('\n')[0]  # Lire uniquement la première ligne
    for delim in delimiters:
        if delim in first_line:
            return delim
    return ','  # Par défaut, utilise la virgule



# Fonction pour détecter la colinéarité
def detect_collinearity(X, threshold=5):
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return any(vif_data["VIF"] > threshold)  # Retourne True si une colinéarité est détectée

def infer_time_unit(survival_times):
    """Infers the time unit based on the median and distribution of survival times."""
    max_val = survival_times.max()
    median_val = survival_times.median()
    if max_val > 365:
        return "days", [30, 90, 180, 365]
    elif max_val > 24:
        return "months", [1, 3, 6, 12]
    else:
        return "years", [0.5, 1, 2, 5]
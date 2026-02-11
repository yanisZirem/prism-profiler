"""
Software Name: Profiler
Module name : Machine Learning
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

# Machine Learning Models
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, BaggingClassifier,
    ExtraTreesClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier, StackingClassifier, VotingClassifier
)
from sklearn.linear_model import (
    SGDClassifier, LogisticRegression, RidgeClassifier,
    PassiveAggressiveClassifier, Perceptron
)
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.dummy import DummyClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from lightgbm import LGBMClassifier

# Preprocessing and Utilities
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import clone

# Model Evaluation and Selection
from sklearn.model_selection import (
    StratifiedKFold, cross_val_predict, cross_val_score, learning_curve
)
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score
)
from sklearn.pipeline import Pipeline

# Data Handling and Visualization
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import gc


from sklearn.calibration import CalibratedClassifierCV

def train_models(X, y, n_splits=3, progress_bar=None):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = le.classes_

    preprocess_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler())
    ])
    X_processed = preprocess_pipeline.fit_transform(X)
    feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]

    # Adapter n_neighbors selon les données
    min_samples_per_class = np.min(np.bincount(y_encoded))
    adapted_n_neighbors = min(5, min_samples_per_class)
    if adapted_n_neighbors < 1:
        adapted_n_neighbors = 1

    models = {
        'RandomForest': RandomForestClassifier(n_estimators=50, max_depth=10),
        'AdaBoost': AdaBoostClassifier(n_estimators=50, algorithm="SAMME"),
        'SGD': SGDClassifier(),
        # 'LinearSVC': LinearSVC(dual=False, n_jobs=-1),
        'NaiveBayes_Gaussian': GaussianNB(),
        'NaiveBayes_Bernoulli': BernoulliNB(),
        'DecisionTree': DecisionTreeClassifier(max_depth=10),
        'LogisticRegression': LogisticRegression(max_iter=500, solver='lbfgs'),
        'Perceptron': Perceptron(),
        'RidgeClassifier': RidgeClassifier(),
        'PassiveAggressive': PassiveAggressiveClassifier(),
        'ExtraTree': ExtraTreeClassifier(max_depth=10),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=50, max_depth=10),
        'Bagging': BaggingClassifier(n_jobs=-1, n_estimators=10),
        'Dummy': DummyClassifier(),
        'NearestCentroid': NearestCentroid(),
        'KNeighbors': KNeighborsClassifier(n_neighbors=adapted_n_neighbors),
        'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
        'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis(),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=50),
        'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=50),
        'LGBMClassifier': LGBMClassifier(max_depth=5, n_estimators=50)
    }

    results = {}
    total_models = len(models)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)

    models_requiring_calibration = (
        RidgeClassifier,
        SGDClassifier,
        Perceptron,
        PassiveAggressiveClassifier
    )


    for i, (model_name, model) in enumerate(models.items()):
        # pipeline = Pipeline([
        #     ('model', model)
        # ])

        # --- Calibration uniquement si nécessaire ---
        if isinstance(model, models_requiring_calibration):
            calibrated_model = CalibratedClassifierCV(
                estimator=model,
                method="sigmoid",   # robuste pour petits datasets omics
                cv=3
            )
            pipeline = Pipeline([
                ('model', calibrated_model)
            ])
        else:
            pipeline = Pipeline([
                ('model', model)
            ])

        pipeline.fit(X_processed, y_encoded)

        pipeline.fit(X_processed, y_encoded)

        # # Prédictions et scores
        # y_pred = cross_val_predict(pipeline, X_processed, y_encoded, cv=cv, method='predict')
        # scores = cross_val_score(pipeline, X_processed, y_encoded, cv=cv)



        # confidence_scores = np.full(len(y_encoded), np.nan)

        # try:
        #     probas = cross_val_predict(
        #         pipeline,
        #         X_processed,
        #         y_encoded,
        #         cv=cv,
        #         method='predict_proba',
        #         n_jobs=-1
        #     )

        #     if probas.ndim == 2:
        #         confidence_scores = np.max(probas, axis=1)
        #     else:
        #         # cas binaire compressé
        #         confidence_scores = probas

        # except Exception:
        #     try:
        #         decision_scores = cross_val_predict(
        #             pipeline,
        #             X_processed,
        #             y_encoded,
        #             cv=cv,
        #             method='decision_function',
        #             n_jobs=-1
        #         )

        #         if decision_scores.ndim == 2:
        #             confidence_scores = np.max(
        #                 (decision_scores - decision_scores.min(axis=1, keepdims=True)) /
        #                 (decision_scores.max(axis=1, keepdims=True) - decision_scores.min(axis=1, keepdims=True) + 1e-9),
        #                 axis=1
        #             )
        #         else:
        #             confidence_scores = decision_scores

        #     except Exception:
        #         confidence_scores = np.full(len(y_encoded), np.nan)

       # Prédictions et scores
        y_pred = cross_val_predict(pipeline, X_processed, y_encoded, cv=cv, method='predict', n_jobs=4)
        scores = cross_val_score(pipeline, X_processed, y_encoded, cv=cv, n_jobs=4)



        confidence_scores = np.full(len(y_encoded), np.nan)

        try:
            probas = cross_val_predict(
                pipeline,
                X_processed,
                y_encoded,
                cv=cv,
                method='predict_proba',
                n_jobs=-1
            )

            if probas.ndim == 2:
                confidence_scores = np.max(probas, axis=1)
            else:
                # cas binaire compressé
                confidence_scores = probas

        except Exception:
            try:
                decision_scores = cross_val_predict(
                    pipeline,
                    X_processed,
                    y_encoded,
                    cv=cv,
                    method='decision_function',
                    n_jobs=-1
                )

                if decision_scores.ndim == 2:
                    confidence_scores = np.max(
                        (decision_scores - decision_scores.min(axis=1, keepdims=True)) /
                        (decision_scores.max(axis=1, keepdims=True) - decision_scores.min(axis=1, keepdims=True) + 1e-9),
                        axis=1
                    )
                else:
                    confidence_scores = decision_scores

            except Exception:
                confidence_scores = np.full(len(y_encoded), np.nan)

        report = classification_report(y_encoded, y_pred, target_names=class_names, zero_division=0, output_dict=True)
        cm = confusion_matrix(y_encoded, y_pred)
        f1 = f1_score(y_encoded, y_pred, average='weighted')
        accuracy = accuracy_score(y_encoded, y_pred)

        results[model_name] = {
            'classification_report': report,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'confusion_matrix': cm,
            'label_encoder': le,
            'model': pipeline,
            'f1_score': f1,
            'accuracy': accuracy,
            'features': feature_names,
            'confidence_scores': confidence_scores  # Ajout des scores de confiance
        }

        if progress_bar is not None:
            progress_bar.progress((i + 1) / total_models)

    del pipeline, model, y_pred, scores, cm
    gc.collect()
    return results



def compare_models(model_results):
    """Compares models based on their performance metrics."""
    metrics_df = pd.DataFrame({
        'Model': list(model_results.keys()),
        'Accuracy': [result['accuracy'] for result in model_results.values()],
        'F1 Score': [result['f1_score'] for result in model_results.values()]
    })
    fig = px.bar(metrics_df, x="Model", y=["Accuracy", "F1 Score"], barmode="group", title="Model Comparison")
    return fig




from sklearn.model_selection import StratifiedKFold, learning_curve

def plot_learning_curve(model, X, y, n_splits=5):

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)

    train_sizes, train_scores, test_scores = learning_curve(
        pipeline,
        X, y,
        cv=cv,
        scoring="f1_weighted",   # très conseillé
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 8)
    )

    # Création de la figure
    fig = go.Figure()

    # Ajout des courbes de score d'entraînement et de validation
    fig.add_trace(go.Scatter(
        x=train_sizes,
        y=train_scores.mean(axis=1),
        mode='lines+markers',
        name='Train Score',
        line=dict(color='blue', width=3),
        marker=dict(size=8)
    ))

    fig.add_trace(go.Scatter(
        x=train_sizes,
        y=test_scores.mean(axis=1),
        mode='lines+markers',
        name='Validation Score',
        line=dict(color='red', width=3),
        marker=dict(size=8)
    ))

    # Mise à jour de la mise en page
    fig.update_layout(
        title=dict(text="<b>Learning Curve</b>", font=dict(size=24, color='black', family="Arial, bold")),
        xaxis=dict(
            title=dict(text="<b>Training Set Size</b>", font=dict(size=18, color='black', family="Arial, bold")),
            tickfont=dict(size=16, color='black', family="Arial, bold")
        ),
        yaxis=dict(
            title=dict(text="<b>Score</b>", font=dict(size=18, color='black', family="Arial, bold")),
            tickfont=dict(size=16, color='black',  family="Arial, bold")
        ),
        plot_bgcolor='white',
        width=900,
        height=600,
        legend=dict(
            title_font=dict(size=16, color='black', family="Arial, bold"),
            font=dict(size=14, color='black')
        )
    )

    return fig




import pandas as pd
import plotly.express as px

def compare_models(model_results):
    """Compares models based on their performance metrics."""
    data = []
    
    for model_name, result in model_results.items():
        cm = result['confusion_matrix']
        
        if cm.shape == (2, 2):
            # Binaire - Déballage en tn, fp, fn, tp
            tn, fp, fn, tp = cm.ravel()

            # Calcul des métriques pour la classification binaire
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            # Multiclasse - Traitement de la matrice de confusion
            sensitivity = []
            specificity = []

            # Calcul des métriques pour chaque classe dans la classification multiclasse
            for i in range(cm.shape[0]):
                tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
                fp = np.sum(cm[:, i]) - cm[i, i]
                fn = np.sum(cm[i, :]) - cm[i, i]
                tp = cm[i, i]
                
                sensitivity.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
                specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0)

            # Moyenne des métriques pour la classification multiclasse
            sensitivity = np.mean(sensitivity)
            specificity = np.mean(specificity)

        # Ajout des métriques dans le résultat du modèle
        result['sensitivity'] = sensitivity
        result['specificity'] = specificity

        # Ajouter les données à la liste
        data.append({
            'Model': model_name,
            'Accuracy': result['accuracy'],
            'F1 Score': result['f1_score'],
            'Sensibilité': sensitivity,
            'Spécificité': specificity
        })

    # Création du DataFrame avec les métriques de chaque modèle
    metrics_df = pd.DataFrame(data)

    # Tri pour obtenir les top 3 modèles
    sorted_models = sorted(
        model_results.items(),
        key=lambda item: item[1].get('f1_score', 0),
        reverse=True
    )
    top_models = sorted_models[:3]

    # Création du message à afficher dans st.info
    top_models_info = "\n\n".join([ 
        f"**{i+1}. {name}**\n"
        f"- 🎯 F1 Score: `{info['f1_score']:.4f}`\n"
        f"- ✅ Accuracy: `{info['accuracy']:.4f}`\n"
        f"- 💓 Sensitivity (Recall): `{info['sensitivity']:.4f}`\n"
        f"- 🛡️ Specificity: `{info['specificity']:.4f}`"
        for i, (name, info) in enumerate(top_models)
    ])

    # Afficher les meilleurs modèles
    st.info(f"🏆 **Top 3 models (by F1 Score):**\n\n{top_models_info}")

    # Graphique comparatif des modèles
    fig = px.bar(
        metrics_df,
        x="Model",
        y=["Accuracy", "F1 Score"],
        barmode="group",
        title="All Models Comparison",
        text_auto=".2f"
    )

    # fig.update_layout(
    #     title_font_size=20,
    #     font=dict(color="black", size=14),
    #     legend_title_text='Metrics',
    #     legend=dict(font=dict(size=14, color='black')),
    #     xaxis=dict(title='Models', titlefont=dict(size=16, color='black'), tickfont=dict(size=12, color='black')),
    #     yaxis=dict(title='Score', titlefont=dict(size=16, color='black'), tickfont=dict(size=12, color='black')),
    #     plot_bgcolor='white',
    #     paper_bgcolor='white'
    # )

    fig.update_layout(
        title=dict(text=fig.layout.title.text if fig.layout.title.text else "Metrics Plot", font=dict(size=20)),
        font=dict(color="black", size=14),
        legend_title_text='Metrics',
        legend=dict(font=dict(size=14, color='black')),
        xaxis=dict(
            title=dict(text='Models', font=dict(size=16, color='black')),
            tickfont=dict(size=12, color='black')
        ),
        yaxis=dict(
            title=dict(text='Score', font=dict(size=16, color='black')),
            tickfont=dict(size=12, color='black')
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )


    fig.update_traces(marker_line_width=1.2)

    return fig


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



import tensorflow as tf
# Mixed precision: 2x faster on GPU, less VRAM
try:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
except Exception:
    pass
# Limit GPU memory growth (prevents OOM)
try:
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, BatchNormalization, GlobalAveragePooling1D, LSTM, GRU
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
import seaborn as sns
import plotly.graph_objects as go


class ProgressBarCallback(Callback):
    def __init__(self, progress_bar):
        super().__init__()
        self.progress_bar = progress_bar

    def on_epoch_end(self, epoch, logs=None):
        self.progress_bar.progress((epoch + 1) / self.params['epochs'])



def build_mlp(input_shape, num_classes, learning_rate):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_cnn(input_shape, num_classes,learning_rate):
    model = Sequential([
        Conv1D(32, kernel_size=3, activation='relu', input_shape=(input_shape, 1)),
        BatchNormalization(),
        Conv1D(64, kernel_size=3, activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_rnn(input_shape, num_classes,learning_rate):
    model = Sequential([
        LSTM(64, input_shape=(input_shape, 1), return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model



def compare_DL(models_results):
    df = pd.DataFrame({
        'Model': list(models_results.keys()),
        'Accuracy': [res['accuracy_mean'] for res in models_results.values()],
        'F1 Score': [res['f1_mean'] for res in models_results.values()]
    })
    fig = px.bar(df, x="Model", y=["Accuracy", "F1 Score"], barmode="group", title="Deep Learning Model Comparison")
    return fig



def display_model_results(model, X_test, y_test, history, label_encoder):
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_test_decoded = label_encoder.inverse_transform(y_test)
    y_pred_decoded = label_encoder.inverse_transform(y_pred)

    st.write("Classification Report:")
    st.text(classification_report(y_test_decoded, y_pred_decoded))

    st.write("Confusion Matrix:")
    cm = confusion_matrix(y_test_decoded, y_pred_decoded, labels=label_encoder.classes_)
    fig_cm = px.imshow(
        cm, x=list(label_encoder.classes_), y=list(label_encoder.classes_),
        labels=dict(x="Predicted", y="Actual", color="Count"),
        color_continuous_scale="Blues", text_auto=True, aspect="auto",
        title="Confusion Matrix"
    )
    fig_cm.update_layout(height=400, margin=dict(l=10, r=10, t=50, b=10),
                         plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig_cm, use_container_width=True)
    st.session_state["_report_dl_confusion_matrix"] = ("plotly", fig_cm)

    accuracy = accuracy_score(y_test_decoded, y_pred_decoded)
    st.text(f"**Accuracy**: {accuracy:.4f}")


    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(y=history.history['loss'], mode='lines', name='Train Loss'))
    fig_loss.add_trace(go.Scatter(y=history.history['val_loss'], mode='lines', name='Test Loss'))
    fig_loss.update_layout(title='Model Loss', xaxis_title='Epoch', yaxis_title='Loss', template="plotly_dark")
    st.plotly_chart(fig_loss, use_container_width=True)

    fig_accuracy = go.Figure()
    fig_accuracy.add_trace(go.Scatter(y=history.history['accuracy'], mode='lines', name='Train Accuracy'))
    fig_accuracy.add_trace(go.Scatter(y=history.history['val_accuracy'], mode='lines', name='Test Accuracy'))
    fig_accuracy.update_layout(title='Model Accuracy', xaxis_title='Epoch', yaxis_title='Accuracy', template="plotly_dark")
    st.plotly_chart(fig_accuracy, use_container_width=True)

def train_DL(X, y, n_splits=2, epochs=10, batch_size=32, learning_rate=0.001):
    le = LabelEncoder()
    y = le.fit_transform(y)  # int array
    class_names = le.classes_
    num_classes = len(class_names)

    scaler = StandardScaler()
    # ⚡ float32 → 2x less RAM for neural network inputs
    X_scaled = scaler.fit_transform(X).astype('float32')
    X_scaled = np.expand_dims(X_scaled, axis=-1)  # For CNN and RNN

    models = {
        'MLP': build_mlp(X.shape[1], num_classes, learning_rate),
        'CNN': build_cnn(X.shape[1], num_classes, learning_rate),
        'RNN': build_rnn(X.shape[1], num_classes, learning_rate)
    }
    feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
    results = {}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)

    all_y_test = []
    all_y_pred = []

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        accuracy_list, f1_list = [], []

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            progress_bar = st.progress(0)
            callbacks = [ProgressBarCallback(progress_bar)]
            # ⚡ verbose=0 → évite I/O stdout sur serveur
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=callbacks, verbose=0)

            y_pred = np.argmax(model.predict(X_test), axis=1)

            accuracy_list.append(accuracy_score(y_test, y_pred))
            f1_list.append(f1_score(y_test, y_pred, average='weighted'))

            all_y_test.extend(y_test)
            all_y_pred.extend(y_pred)
            
            accuracy = accuracy_score(y_test, y_pred)
        results[model_name] = {
            'accuracy_mean': np.mean(accuracy_list),
            'accuracy_std': np.std(accuracy_list),
            'accuracy': accuracy,
            'f1_mean': np.mean(f1_list),
            'f1_std': np.std(f1_list),
            'label_encoder': le,
            'model': model,
            'X_test': X_test,
            'y_test': y_test,
            'history': history,
            'all_y_test': all_y_test,
            'all_y_pred': all_y_pred,
            'features': feature_names 
        }
    return results

def display_global_results(model_results):
    all_y_test = model_results['all_y_test']
    all_y_pred = model_results['all_y_pred']
    label_encoder = model_results['label_encoder']

    y_test_decoded = label_encoder.inverse_transform(all_y_test)
    y_pred_decoded = label_encoder.inverse_transform(all_y_pred)

    st.markdown("**k-fold Classification Report:**")
    st.text(classification_report(y_test_decoded, y_pred_decoded))

    st.markdown("**k-fold Confusion Matrix:**")
    cm = confusion_matrix(y_test_decoded, y_pred_decoded, labels=label_encoder.classes_)
    fig_cm = px.imshow(
        cm, x=list(label_encoder.classes_), y=list(label_encoder.classes_),
        labels=dict(x="Predicted", y="Actual", color="Count"),
        color_continuous_scale="Blues", text_auto=True, aspect="auto",
        title="k-fold Confusion Matrix"
    )
    fig_cm.update_layout(height=400, margin=dict(l=10, r=10, t=50, b=10),
                         plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig_cm, use_container_width=True)
    st.session_state["_report_dl_kfold_confusion"] = ("plotly", fig_cm)

    accuracy = accuracy_score(y_test_decoded, y_pred_decoded)
    st.text(f"**Global Accuracy**: {accuracy:.4f}")

# Import necessary libraries
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sentence_transformers import SentenceTransformer
from scipy.sparse import hstack, csr_matrix, save_npz, load_npz
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.calibration import calibration_curve

import warnings
warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output

# Load CSV and clean specific numeric columns
def load_and_clean_data(path):
    df = pd.read_csv(path, encoding="ISO-8859-1")
    df[['runtimeMinutes', 'startYear']] = df[['runtimeMinutes', 'startYear']].replace('\\N', np.nan)
    df['runtimeMinutes'] = pd.to_numeric(df['runtimeMinutes'], errors='coerce')
    df['startYear'] = pd.to_numeric(df['startYear'], errors='coerce')
    df['runtimeMinutes'] = df['runtimeMinutes'].fillna(df['runtimeMinutes'].median())
    df['startYear'] = df['startYear'].fillna(df['startYear'].median())
    return df

# Generate binary labels based on avg_rating column
def generate_labels(df):
    df = df[df['avg_rating'].notna()]
    df['avg_rating'] = df['avg_rating'].astype(int)
    df['label'] = df['avg_rating'].apply(lambda x: 0 if x <= 2 else 1)
    return df

# Load or create BERT embeddings from the plot column
def load_or_generate_bert_embeddings(df, path="bert_mpnet_embeddings.npy"):
    if os.path.exists(path):
        print(f"Loading BERT embeddings from {path}...")
        return np.load(path)
    else:
        print("Generating BERT embeddings...")
        model = SentenceTransformer('all-mpnet-base-v2')
        plots = df['plot'].fillna('').tolist()
        embeddings = model.encode(plots, show_progress_bar=True).astype(np.float32)
        np.save(path, embeddings)
        print(f"Embeddings saved to {path}.")
        return embeddings

# Convert categorical column into multi-hot encoded format
# Collapse rare values into 'Others' if top_k is specified
def multi_hot_encode_with_others(df, column, top_k=None):
    df[column + '_list'] = df[column].fillna('').apply(lambda x: [s.strip() for s in x.split(',') if s.strip()])
    all_items = df[column + '_list'].explode()
    if top_k:
        top_items = set(all_items.value_counts().nlargest(top_k).index)
        df[column + '_list'] = df[column + '_list'].apply(lambda lst: [x if x in top_items else 'Others' for x in lst])
    mlb = MultiLabelBinarizer()
    return pd.DataFrame(mlb.fit_transform(df[column + '_list']), columns=[f"{column}_{c}" for c in mlb.classes_])

# Combine text embeddings, categorical multi-hot features, and numerical features
def assemble_features(df, X_text):
    genres = multi_hot_encode_with_others(df, 'genres')
    actors = multi_hot_encode_with_others(df, 'actors', top_k=100)
    writers = multi_hot_encode_with_others(df, 'writer', top_k=50)
    directors = multi_hot_encode_with_others(df, 'director', top_k=20)
    countries = multi_hot_encode_with_others(df, 'country', top_k=20)
    languages = multi_hot_encode_with_others(df, 'language', top_k=10)

    X_cat_df = pd.concat([genres, actors, writers, directors, countries, languages], axis=1)
    X_cat = csr_matrix(X_cat_df.values)
    X_num_df = df[['runtimeMinutes', 'startYear', 'num_rating']]
    X_num = StandardScaler().fit_transform(X_num_df)

    feature_names = (
        [f"text_{i}" for i in range(X_text.shape[1])] +
        list(X_cat_df.columns) +
        list(X_num_df.columns)
    )

    return hstack([csr_matrix(X_text), X_cat, X_num]), feature_names

# Apply SMOTE for class balancing (or load from cache if available)
def apply_smote(X_train, y_train, X_file="X_resampled.npz", y_file="y_resampled.npy"):
    if os.path.exists(X_file) and os.path.exists(y_file):
        print("Loading SMOTE-resampled data...")
        return load_npz(X_file), np.load(y_file)
    else:
        print("Applying SMOTE...")
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        save_npz(X_file, X_res)
        np.save(y_file, y_res)
        return X_res, y_res

# Compute ensemble results by averaging predicted probabilities
def ensemble_and_evaluate(y_test, probas):
    ensemble_proba = np.mean(probas, axis=0)
    ensemble_pred = (ensemble_proba >= 0.5).astype(int)
    acc = accuracy_score(y_test, ensemble_pred)
    f1 = f1_score(y_test, ensemble_pred, average='macro')
    plot_confusion_matrix(y_test, ensemble_pred, labels=[0, 1], title="Ensemble Confusion Matrix")
    print(f"\n[Ensemble] Accuracy: {acc:.4f}")
    print(f"[Ensemble] Macro F1-score: {f1:.4f}")
    return acc, f1

# Plot a confusion matrix as a heatmap
def plot_confusion_matrix(y_true, y_pred, labels, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Plot top-n feature importances from tree-based models
def plot_feature_importance(model, feature_names, top_n=10):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices], align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Importance")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.show()

# Plot ROC curves for multiple models
def plot_roc_curve_multi(y_test, probas_dict):
    plt.figure(figsize=(7, 6))
    for model_name, proba in probas_dict.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (All Models)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot bar chart comparing accuracy and F1 score of models
def plot_model_performance(results):
    df_perf = pd.DataFrame([
        {"Model": name, "Accuracy": res['accuracy'], "F1-score": res['f1_score']}
        for name, res in results.items()
    ])
    df_perf.set_index("Model")[["Accuracy", "F1-score"]].plot(kind="bar", figsize=(8, 5), ylim=(0, 1))
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.show()

# Plot Precision-Recall curves for multiple models
def plot_pr_curve_multi(y_test, probas_dict):
    plt.figure(figsize=(7, 6))
    for model_name, proba in probas_dict.items():
        precision, recall, _ = precision_recall_curve(y_test, proba)
        plt.plot(recall, precision, lw=2, label=model_name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (All Models)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot Calibration Curves to show how well predicted probabilities reflect true likelihood
def plot_calibration_curve_multi(y_test, probas_dict):
    plt.figure(figsize=(7, 6))
    for model_name, proba in probas_dict.items():
        prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', label=model_name)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve (All Models)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# =============================================================
# CANCER TYPE CLASSIFICATION FROM GENE EXPRESSION DATA
# Dataset: TCGA RNA-Seq
# 5 cancer types | 800 samples | 20,530 genes
# =============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# =============================================================
# FILE PATH SETUP
# =============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================
# 1. DATA LOADING
# =============================================================
# Gene expression datasets are extremely high-dimensional.
# Each row represents a patient sample.
# Each column represents the expression level of a gene.
#
# The goal:
# Learn whether gene expression patterns alone can identify
# which cancer type a patient belongs to.
# =============================================================

data = pd.read_csv(
    os.path.join(BASE_DIR, 'data.csv'),
    index_col=0
)

labels = pd.read_csv(
    os.path.join(BASE_DIR, 'labels.csv')
)

X = data.values
y = labels['Class'].values

print("Data shape:", data.shape)
print("Labels shape:", labels.shape)

# =============================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================
# Biological motivation:
# Different cancer types alter cellular behaviour differently.
# These alterations appear as distinct gene expression patterns.
#
# Before modelling, we inspect:
# - class balance
# - expression distributions
# - variance across genes
# - separability between cancer types
# =============================================================

# -------------------------------------------------------------
# CLASS DISTRIBUTION
# -------------------------------------------------------------
# Important because highly imbalanced datasets can bias models
# toward predicting majority cancer types.
# -------------------------------------------------------------

class_counts = labels['Class'].value_counts()

print("\nCancer type distribution:")
print(class_counts)

balance_ratio = class_counts.max() / class_counts.min()

print(f"\nClass imbalance ratio: {balance_ratio:.2f}")

plt.figure(figsize=(8, 5))

plt.bar(
    class_counts.index,
    class_counts.values,
    color='steelblue',
    edgecolor='black'
)

plt.title('Cancer Type Distribution')
plt.xlabel('Cancer Type')
plt.ylabel('Number of Samples')

for i, value in enumerate(class_counts.values):
    plt.text(
        i,
        value + 2,
        str(value),
        ha='center',
        fontweight='bold'
    )

plt.tight_layout()

plt.savefig(
    os.path.join(
        BASE_DIR,
        'cancer_type_distribution.png'
    ),
    dpi=300
)

plt.show()

# -------------------------------------------------------------
# MISSING VALUE CHECK
# -------------------------------------------------------------
# Missing values are dangerous in biological datasets because
# many ML models cannot handle NaNs directly.
# -------------------------------------------------------------

missing_values = data.isnull().sum().sum()

print(f"\nTotal missing values: {missing_values}")

# -------------------------------------------------------------
# OVERALL GENE EXPRESSION RANGE
# -------------------------------------------------------------
# Helps understand whether genes operate on vastly different
# numerical scales before standardisation.
# -------------------------------------------------------------

print("\nExpression value statistics:")
print(f"Minimum: {data.values.min():.3f}")
print(f"Maximum: {data.values.max():.3f}")
print(f"Mean: {data.values.mean():.3f}")

# -------------------------------------------------------------
# GENE EXPRESSION DISTRIBUTION FOR ONE PATIENT
# -------------------------------------------------------------
# Each patient has thousands of gene expression values.
# This plot shows how expression levels are distributed within
# a single tumour sample.
# -------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    data.iloc[1],
    bins=50,
    color='steelblue',
    edgecolor='black'
)

plt.title('Gene Expression Distribution for One Patient')
plt.xlabel('Expression Value')
plt.ylabel('Frequency')

plt.tight_layout()

plt.savefig(
    os.path.join(
        BASE_DIR,
        'single_patient_expression_distribution.png'
    ),
    dpi=300
)

plt.show()

# -------------------------------------------------------------
# EXPRESSION DISTRIBUTION FOR ONE GENE
# -------------------------------------------------------------
# This shows how one specific gene behaves across all patients.
# Some genes are stable; others vary dramatically between
# cancer types.
# -------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    data.iloc[:, 1],
    bins=50,
    color='tomato',
    edgecolor='black'
)

plt.title('Expression Distribution for One Gene')
plt.xlabel('Expression Value')
plt.ylabel('Frequency')

plt.tight_layout()

plt.savefig(
    os.path.join(
        BASE_DIR,
        'single_gene_expression_distribution.png'
    ),
    dpi=300
)

plt.show()

# -------------------------------------------------------------
# HIGHEST VARIANCE GENES
# -------------------------------------------------------------
# Genes with high variance often carry the strongest biological
# signal because they behave differently across patients.
# Low-variance genes usually contribute less information.
# -------------------------------------------------------------

gene_variance = data.var(axis=0).sort_values(ascending=False)

print("\nTop 5 highest variance genes:")
print(gene_variance.head())

plt.figure(figsize=(15, 5))

plt.bar(
    range(50),
    gene_variance[:50].values,
    color='seagreen'
)

plt.title('Top 50 Highest Variance Genes')
plt.xlabel('Gene Rank')
plt.ylabel('Variance')

plt.tight_layout()

plt.savefig(
    os.path.join(
        BASE_DIR,
        'top_50_highest_variance_genes.png'
    ),
    dpi=300
)

plt.show()

# -------------------------------------------------------------
# MEAN EXPRESSION DISTRIBUTION BY CANCER TYPE
# -------------------------------------------------------------
# Different cancers often activate different biological
# pathways, shifting global expression patterns.
# -------------------------------------------------------------

eda_df = data.copy()
eda_df['cancer_type'] = labels['Class'].values

plt.figure(figsize=(10, 5))

for cancer in eda_df['cancer_type'].unique():

    subset = eda_df[
        eda_df['cancer_type'] == cancer
    ]

    mean_expression = subset.drop(
        'cancer_type',
        axis=1
    ).mean(axis=1)

    plt.hist(
        mean_expression,
        bins=30,
        alpha=0.5,
        label=cancer
    )

plt.title('Mean Gene Expression by Cancer Type')
plt.xlabel('Mean Expression Value')
plt.ylabel('Frequency')

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        BASE_DIR,
        'mean_expression_by_cancer_type.png'
    ),
    dpi=300
)

plt.show()

# =============================================================
# 3. PREPROCESSING: STANDARDISATION + PCA
# =============================================================
# RNA-Seq datasets contain tens of thousands of genes.
# Most ML algorithms struggle with extremely high-dimensional
# data due to:
#
# - noise
# - redundancy
# - curse of dimensionality
#
# PCA compresses the dataset into a smaller number of
# informative components while preserving most variance.
#
# IMPORTANT:
# We split BEFORE scaling/PCA to prevent data leakage.
# =============================================================

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------------------------------------
# STANDARDISATION
# -------------------------------------------------------------
# Gene expression ranges vary significantly across genes.
# StandardScaler ensures every gene contributes equally.
# -------------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

print("\nScaling completed.")
print("Scaled train shape:", X_train_scaled.shape)

# -------------------------------------------------------------
# PRINCIPAL COMPONENT ANALYSIS (PCA)
# -------------------------------------------------------------
# PCA identifies directions of maximal variation in gene
# expression space.
#
# Instead of 20,530 genes, we use 200 compressed features.
# -------------------------------------------------------------

pca = PCA(n_components=200)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("\nPCA completed.")
print("Reduced train shape:", X_train_pca.shape)
print("Reduced test shape:", X_test_pca.shape)

# -------------------------------------------------------------
# EXPLAINED VARIANCE
# -------------------------------------------------------------
# Measures how much biological information is preserved after
# dimensionality reduction.
# -------------------------------------------------------------

explained = pca.explained_variance_ratio_

print(f"\nVariance captured by PC1: {explained[0]*100:.2f}%")
print(f"Variance captured by PC2: {explained[1]*100:.2f}%")

print(
    f"Total variance captured by 200 PCs: "
    f"{np.sum(explained)*100:.2f}%"
)

# -------------------------------------------------------------
# PCA CUMULATIVE VARIANCE PLOT
# -------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    np.cumsum(explained) * 100,
    linewidth=2
)

plt.axhline(
    y=80,
    color='red',
    linestyle='--',
    label='80% Variance'
)

plt.axhline(
    y=90,
    color='orange',
    linestyle='--',
    label='90% Variance'
)

plt.title('Cumulative Explained Variance from PCA')
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance (%)')

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        BASE_DIR,
        'pca_cumulative_variance.png'
    ),
    dpi=300
)

plt.show()

# -------------------------------------------------------------
# PCA VISUALISATION IN 2D
# -------------------------------------------------------------
# If cancer types cluster separately in PCA space, it suggests
# gene expression alone contains strong discriminatory signal.
# -------------------------------------------------------------

pca_df = pd.DataFrame(
    X_train_pca[:, :2],
    columns=['PC1', 'PC2']
)

pca_df['Cancer Type'] = y_train

colors = {
    'BRCA': 'blue',
    'KIRC': 'red',
    'COAD': 'green',
    'LUAD': 'orange',
    'PRAD': 'purple'
}

plt.figure(figsize=(10, 7))

for cancer, color in colors.items():

    subset = pca_df[
        pca_df['Cancer Type'] == cancer
    ]

    plt.scatter(
        subset['PC1'],
        subset['PC2'],
        label=cancer,
        color=color,
        alpha=0.7,
        s=35
    )

plt.title('PCA Projection of Cancer Types')

plt.xlabel(
    f'PC1 ({explained[0]*100:.1f}% variance)'
)

plt.ylabel(
    f'PC2 ({explained[1]*100:.1f}% variance)'
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        BASE_DIR,
        'pca_2d_projection.png'
    ),
    dpi=300
)

plt.show()

# =============================================================
# 4. MODEL TRAINING
# =============================================================
# We compare three fundamentally different classifiers:
#
# Logistic Regression:
# Linear probabilistic classifier
#
# Random Forest:
# Ensemble tree-based classifier
#
# Support Vector Machine:
# Margin-maximising nonlinear classifier
# =============================================================

models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    'SVM': SVC(
        kernel='rbf',
        random_state=42
    )
}

results = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train_pca, y_train)

    predictions = model.predict(X_test_pca)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average='macro'
    )

    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'predictions': predictions
    }

    print(
        f"{name} | "
        f"Accuracy: {accuracy:.4f} | "
        f"Macro F1: {macro_f1:.4f}"
    )

# =============================================================
# 5. EVALUATION
# =============================================================
# Accuracy alone is insufficient for multiclass biomedical
# datasets.
#
# Macro F1 gives equal importance to all cancer types.
# =============================================================

print("\nTrain label distribution:")
print(pd.Series(y_train).value_counts())

print("\nTest label distribution:")
print(pd.Series(y_test).value_counts())

# -------------------------------------------------------------
# CROSS-VALIDATION
# -------------------------------------------------------------
# Evaluates model stability across multiple train/test splits.
# -------------------------------------------------------------

print("\nCross-validating Logistic Regression...")

lr_cv = LogisticRegression(
    max_iter=1000,
    random_state=42
)

cv_scores = cross_val_score(
    lr_cv,
    X_train_pca,
    y_train,
    cv=5,
    scoring='accuracy'
)

print("CV scores:", cv_scores)

print(
    f"Mean CV accuracy: "
    f"{np.mean(cv_scores):.4f}"
)

print(
    f"CV standard deviation: "
    f"{np.std(cv_scores):.4f}"
)

# -------------------------------------------------------------
# BEST MODEL SELECTION
# -------------------------------------------------------------

best_model_name = max(
    results,
    key=lambda x: results[x]['macro_f1']
)

best_model = results[best_model_name]['model']

best_predictions = results[best_model_name]['predictions']

print(f"\nBest model: {best_model_name}")

# -------------------------------------------------------------
# CLASSIFICATION REPORT
# -------------------------------------------------------------
# Precision:
# How reliable positive predictions are.
#
# Recall:
# How many true cancer cases are detected.
#
# F1-score:
# Harmonic mean of precision and recall.
# -------------------------------------------------------------

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        best_predictions
    )
)

# -------------------------------------------------------------
# CONFUSION MATRIX
# -------------------------------------------------------------
# Shows which cancer types are confused with one another.
# -------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=best_model.classes_
)

plt.figure(figsize=(8, 6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=best_model.classes_
)

disp.plot(
    xticks_rotation=45,
    ax=plt.gca()
)

plt.title(
    f'Confusion Matrix - {best_model_name}'
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        BASE_DIR,
        'confusion_matrix.png'
    ),
    dpi=300
)

plt.show()

# -------------------------------------------------------------
# MODEL COMPARISON PLOT
# -------------------------------------------------------------
# Compares overall performance across all classifiers.
# -------------------------------------------------------------

model_names = list(results.keys())

accuracies = [
    results[name]['accuracy']
    for name in model_names
]

macro_f1_scores = [
    results[name]['macro_f1']
    for name in model_names
]

x = np.arange(len(model_names))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))

ax.bar(
    x - width / 2,
    accuracies,
    width,
    label='Accuracy',
    color='steelblue'
)

ax.bar(
    x + width / 2,
    macro_f1_scores,
    width,
    label='Macro F1',
    color='tomato'
)

ax.set_xticks(x)

ax.set_xticklabels(model_names)

ax.set_ylim(0, 1.05)

ax.set_ylabel('Score')

ax.set_title('Model Performance Comparison')

ax.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        BASE_DIR,
        'model_comparison.png'
    ),
    dpi=300
)

plt.show()

print("\nPipeline execution completed successfully.")
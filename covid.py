import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import anndata as ad
import scanpy as sc
from collections import Counter
import os
from processing import *

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix

warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-whitegrid')
N_SPLITS = 10
RANDOM_STATE = 42
OUTPUT_DIR = "cv_results_covid"

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Results will be saved in the '{OUTPUT_DIR}/' directory.")

print("--- Step 1: Loading and Preparing Data ---")
adata_full = ad.read_h5ad("your_path")

X_all = adata_full.X
y_all = adata_full.obs['infected'].values
gene_names = adata_full.var.index

print(f"Original dataset shape: {X_all.shape}")

print("Filtering genes that are zero for all columns...")
df_temp = pd.DataFrame(X_all.T, index=gene_names)
initial_gene_count = df_temp.shape[0]

df_temp = df_temp.loc[(df_temp != 0).any(axis=1)]

X_all = df_temp.values.T 
gene_names = df_temp.index

print(f"Genes remaining: {df_temp.shape[0]}")
print(f"Dropped {initial_gene_count - df_temp.shape[0]} genes.")
print(f"New dataset shape: {X_all.shape}")

print(f"Target labels shape: {y_all.shape}")

unique_classes, class_counts = np.unique(y_all, return_counts=True)
print(f"\nOverall class distribution:")
for cls, cnt in zip(unique_classes, class_counts):
    print(f"  Class {cls}: {cnt} samples ({100*cnt/len(y_all):.2f}%)")

print(f"\n--- Step 2: Starting {N_SPLITS}-Fold Cross-Validation with Logistic Regression ---")

fold_results = []
all_test_cms = []

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

for fold, (train_idx, test_idx) in enumerate(tqdm(skf.split(X_all, y_all), total=N_SPLITS, desc="CV Folds")):
    print(f"\n===== FOLD {fold + 1}/{N_SPLITS} =====")

    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]

    df_train_in = pd.DataFrame(X_train.T, index=gene_names)
    df_train_processed = process(df_train_in, average=False)
    X_train_proc = df_train_processed.values.T  # Transpose back to (n_samples, n_genes)

    df_test_in = pd.DataFrame(X_test.T, index=gene_names)
    df_test_processed = process(df_test_in, average=False)
    X_test_proc = df_test_processed.values.T
    
    k_features = min(20000, X_train.shape[1])
    anova_selector = SelectKBest(score_func=f_classif, k=k_features)
    X_train_fs = anova_selector.fit_transform(X_train, y_train)
    X_test_fs = anova_selector.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_fs)
    X_test_scaled = scaler.transform(X_test_fs)

    n_pcs = 220
    pca = PCA(n_components=n_pcs, whiten=True, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    adata_train_pca = ad.AnnData(X_train_pca)
    sc.pp.neighbors(adata_train_pca, n_neighbors=15, use_rep='X')
    sc.tl.leiden(adata_train_pca, resolution=0.4, random_state=RANDOM_STATE)
    train_leiden_clusters = adata_train_pca.obs['leiden'].values
    num_clusters_found = len(np.unique(train_leiden_clusters))

    knn_cluster_annotator = KNeighborsClassifier(n_neighbors=15, metric='cosine', weights='distance')
    knn_cluster_annotator.fit(X_train_pca, train_leiden_clusters)
    test_cluster_labels = knn_cluster_annotator.predict(X_test_pca)

    best_model_objects = {}

    for cluster_id in sorted(np.unique(train_leiden_clusters)):
        train_mask = (train_leiden_clusters == cluster_id)
        X_train_c, y_train_c = X_train_pca[train_mask], y_train[train_mask]

        if X_train_c.shape[0] < 5 or len(np.unique(y_train_c)) < 2:
            continue

        unique_labels, counts = np.unique(y_train_c, return_counts=True)
        total_samples = len(y_train_c)
        n_classes = len(unique_labels)
        
        class_weights = {}
        for label, count in zip(unique_labels, counts):
            class_weights[label] = total_samples / (n_classes * count)
        
        print(f"  Cluster {cluster_id}: {len(y_train_c)} samples, class distribution: {dict(zip(unique_labels, counts))}")
        print(f"    Class weights: {class_weights}")

        model = LogisticRegression(
            random_state=RANDOM_STATE,
            class_weight='balanced',
            max_iter=1000,
            solver='lbfgs'
        )
        model.fit(X_train_c, y_train_c)
        best_model_objects[cluster_id] = model

    all_test_preds, all_test_true = [], []
    for cid, model in best_model_objects.items():
        mask = (test_cluster_labels == cid)
        if np.sum(mask) > 0:
            all_test_preds.extend(model.predict(X_test_pca[mask]))
            all_test_true.extend(y_test[mask])
    
    test_macro_f1 = f1_score(all_test_true, all_test_preds, average='macro', zero_division=0)
    test_weighted_f1 = f1_score(all_test_true, all_test_preds, average='weighted', zero_division=0)

    unique_labels = sorted(np.unique(y_all))
    cm = confusion_matrix(all_test_true, all_test_preds, labels=unique_labels, normalize='true')
    all_test_cms.append(cm)

    fold_results.append({
        "fold": fold + 1,
        "num_clusters": num_clusters_found,
        "test_macro_f1": test_macro_f1,
        "test_weighted_f1": test_weighted_f1
    })
    
    print(f"  Test Macro F1: {test_macro_f1:.4f}, Test Weighted F1: {test_weighted_f1:.4f}")

results_df = pd.DataFrame(fold_results)
mean_clusters = results_df['num_clusters'].mean()
std_clusters = results_df['num_clusters'].std()
mean_test_macro_f1 = results_df['test_macro_f1'].mean()
std_test_macro_f1 = results_df['test_macro_f1'].std()
mean_test_weighted_f1 = results_df['test_weighted_f1'].mean()
std_test_weighted_f1 = results_df['test_weighted_f1'].std()

summary_path = os.path.join(OUTPUT_DIR, "cross_validation_summary.txt")
with open(summary_path, 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("CROSS-VALIDATION SUMMARY - LOGISTIC REGRESSION\n")
    f.write("=" * 60 + "\n\n")
    f.write("Configuration:\n")
    f.write(f"  - Model: Logistic Regression (class_weight='balanced')\n")
    f.write(f"  - Number of CV Folds: {N_SPLITS}\n")
    f.write(f"  - Feature Selection: ANOVA (top 20,000 features)\n")
    f.write(f"  - Dimensionality Reduction: PCA (200 components)\n")
    f.write(f"  - Clustering: Leiden (resolution=0.4)\n\n")
    f.write("=" * 60 + "\n")
    f.write("OVERALL PERFORMANCE METRICS (Mean ± Std Dev)\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Number of Clusters Found: {mean_clusters:.2f} ± {std_clusters:.2f}\n")
    f.write(f"Test Macro F1 Score:      {mean_test_macro_f1:.4f} ± {std_test_macro_f1:.4f}\n")
    f.write(f"Test Weighted F1 Score:   {mean_test_weighted_f1:.4f} ± {std_test_weighted_f1:.4f}\n\n")
    f.write("Note: Macro F1 treats all classes equally.\n")
    f.write("      Weighted F1 accounts for class imbalance.\n")

print(f"✅ Saved summary report to: {summary_path}")

detailed_csv_path = os.path.join(OUTPUT_DIR, "detailed_fold_results.csv")
results_df.to_csv(detailed_csv_path, index=False)
print(f"✅ Saved detailed fold results to: {detailed_csv_path}")

mean_cm = np.mean(all_test_cms, axis=0)
std_cm = np.std(all_test_cms, axis=0)
cm_labels = sorted(np.unique(y_all))

np.save(os.path.join(OUTPUT_DIR, "mean_confusion_matrix.npy"), mean_cm)
np.save(os.path.join(OUTPUT_DIR, "std_dev_confusion_matrix.npy"), std_cm)
print(f"✅ Saved confusion matrix numerical data to .npy files.")

print("\n--- Process Complete ---")

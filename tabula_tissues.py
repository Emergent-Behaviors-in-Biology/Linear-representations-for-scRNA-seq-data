import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score, 
    accuracy_score, precision_score, recall_score, roc_auc_score
)
import warnings
warnings.filterwarnings("ignore")
import sctop as top
import anndata as ad
import os
import gc
import json
from datetime import datetime

from transcriptformer.datasets import tabula_sapiens


DATA_BASE_PATH = "your_path"
OUTPUT_FOLDER = "./tabula_tissues/"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

TISSUES_LIST = [
    "lymphnode", "heart", "ear", "endothelium", "epithelium", "germline",
    "immune", "neural", "stromal", "bladder", "blood", "bone_marrow",
    "eye", "fat", "kidney", "large_intestine", "liver", "lung",
    "lymph_node", "mammary", "muscle", "ovary", "pancreas", "prostate",
    "salivary_gland", "skin", "small_intestine", "spleen", "stomach",
    "testis", "thymus", "tongue", "trachea", "uterus", "vasculature"
]

CELL_TYPE_COLUMN = "cell_type"
MIN_CELLS_PER_TYPE = 100  
DONOR_THRESHOLD = 16 
ANOVA_K = 20000
PCA_COMPONENTS = 150
N_SPLITS = 5 
TEST_SIZE = 0.2 
BASE_RANDOM_STATE = 0
RANDOM_STATE = 0
CHUNK_SIZE = 10000 


def extract_per_class_metrics_from_report(report_str):
    """Extract per-class metrics from sklearn classification report"""
    report_lines = report_str.split('\n')
    per_class_data = []
    
    for line in report_lines[2:-5]: 
        if line.strip():
            parts = line.split()
            if len(parts) >= 5:
                cell_type = ' '.join(parts[:-4])
                precision, recall, f1, support = parts[-4:]
                per_class_data.append({
                    'cell_type': cell_type,
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1),
                    'support': int(support)
                })
    
    return per_class_data

def save_split_results(tissue_name, split_idx, results, output_folder):
    """Save results for a single train-test split"""
    tissue_folder = os.path.join(output_folder, tissue_name)
    split_folder = os.path.join(tissue_folder, f"split_{split_idx}")
    os.makedirs(split_folder, exist_ok=True)
    
    for pipeline_name, result in results.items():
        per_class_data = extract_per_class_metrics_from_report(result['report'])
        
        per_class_df = pd.DataFrame(per_class_data)
        per_class_file = os.path.join(split_folder, f"{pipeline_name}_per_cell_type.csv")
        per_class_df.to_csv(per_class_file, index=False)
        
        overall_metrics = {
            'tissue': tissue_name,
            'split': split_idx,
            'pipeline': pipeline_name,
            'accuracy': result['Accuracy'],
            'macro_f1': result['Macro F1'],
            'macro_precision': result['Precision'],
            'macro_recall': result['Recall'],
            'auroc': result['AUROC'],
            'n_cell_types': len(per_class_data),
            'total_test_samples': sum(per_class_data[i]['support'] for i in range(len(per_class_data)))
        }
        
        overall_file = os.path.join(split_folder, f"{pipeline_name}_overall_metrics.json")
        with open(overall_file, 'w') as f:
            json.dump(overall_metrics, f, indent=2)
        
        report_file = os.path.join(split_folder, f"{pipeline_name}_classification_report.txt")
        with open(report_file, 'w') as f:
            f.write(result['report'])
        
        cm = confusion_matrix(result['y_true'], result['preds'])
        cm_df = pd.DataFrame(cm, 
                             index=sorted(set(result['y_true'])),
                             columns=sorted(set(result['y_true'])))
        cm_file = os.path.join(split_folder, f"{pipeline_name}_confusion_matrix.csv")
        cm_df.to_csv(cm_file)
        
        predictions_df = pd.DataFrame({
            'true_label': result['y_true'],
            'predicted_label': result['preds'],
            'correct': result['y_true'] == result['preds']
        })
        predictions_file = os.path.join(split_folder, f"{pipeline_name}_predictions.csv")
        predictions_df.to_csv(predictions_file, index=False)

def save_aggregated_results(tissue_name, all_split_results, output_folder):
    """Aggregate and save results across all train-test splits"""
    tissue_folder = os.path.join(output_folder, tissue_name)
    
    pipeline_names = list(all_split_results[0].keys())
    
    for pipeline_name in pipeline_names:
        split_metrics = []
        all_per_class = []
        
        for split_idx, split_results in enumerate(all_split_results):
            result = split_results[pipeline_name]
            split_metrics.append({
                'split': split_idx,
                'accuracy': result['Accuracy'],
                'macro_f1': result['Macro F1'],
                'macro_precision': result['Precision'],
                'macro_recall': result['Recall'],
                'auroc': result['AUROC']
            })
            
            per_class_data = extract_per_class_metrics_from_report(result['report'])
            for item in per_class_data:
                item['split'] = split_idx
                all_per_class.append(item)
        
        split_metrics_df = pd.DataFrame(split_metrics)
        split_metrics_file = os.path.join(tissue_folder, f"{pipeline_name}_split_metrics.csv")
        split_metrics_df.to_csv(split_metrics_file, index=False)
        
        summary_stats = {
            'tissue': tissue_name,
            'pipeline': pipeline_name,
            'n_splits': N_SPLITS,
            'test_size': TEST_SIZE,
            'accuracy_mean': split_metrics_df['accuracy'].mean(),
            'accuracy_std': split_metrics_df['accuracy'].std(),
            'macro_f1_mean': split_metrics_df['macro_f1'].mean(),
            'macro_f1_std': split_metrics_df['macro_f1'].std(),
            'macro_precision_mean': split_metrics_df['macro_precision'].mean(),
            'macro_precision_std': split_metrics_df['macro_precision'].std(),
            'macro_recall_mean': split_metrics_df['macro_recall'].mean(),
            'macro_recall_std': split_metrics_df['macro_recall'].std(),
            'auroc_mean': split_metrics_df['auroc'].mean(),
            'auroc_std': split_metrics_df['auroc'].std()
        }
        
        summary_file = os.path.join(tissue_folder, f"{pipeline_name}_summary_stats.json")
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        per_class_df = pd.DataFrame(all_per_class)
        per_class_file = os.path.join(tissue_folder, f"{pipeline_name}_all_splits_per_cell_type.csv")
        per_class_df.to_csv(per_class_file, index=False)
        
        per_class_summary = per_class_df.groupby('cell_type').agg({
            'precision': ['mean', 'std'],
            'recall': ['mean', 'std'],
            'f1_score': ['mean', 'std'],
            'support': 'sum'
        }).round(4)
        per_class_summary.columns = ['_'.join(col).strip() for col in per_class_summary.columns]
        per_class_summary_file = os.path.join(tissue_folder, f"{pipeline_name}_per_cell_type_summary.csv")
        per_class_summary.to_csv(per_class_summary_file)
    
    print(f"✓ Saved aggregated results for {tissue_name}")

def save_cross_tissue_per_celltype_results(output_folder):
    """Aggregate per-cell-type results across all tissues"""
    print("\n" + "="*80)
    print("Aggregating per-cell-type results across all tissues...")
    print("="*80)
    
    tissue_folders = [d for d in os.listdir(output_folder) 
                      if os.path.isdir(os.path.join(output_folder, d)) and not d.startswith('.')]
    
    for pipeline in ['LogisticRegression']:
        all_tissues_per_celltype = []
        
        for tissue_folder in tissue_folders:
            tissue_path = os.path.join(output_folder, tissue_folder)
            per_celltype_file = os.path.join(tissue_path, f"{pipeline}_per_cell_type_summary.csv")
            
            if os.path.exists(per_celltype_file):
                df = pd.read_csv(per_celltype_file)
                df['tissue'] = tissue_folder
                df['pipeline'] = pipeline
                all_tissues_per_celltype.append(df)
        
        if all_tissues_per_celltype:
            combined_df = pd.concat(all_tissues_per_celltype, ignore_index=True)
            
            combined_file = os.path.join(output_folder, f"{pipeline}_cross_tissue_per_celltype.csv")
            combined_df.to_csv(combined_file, index=False)
            
            celltype_summary = combined_df.groupby('cell_type').agg({
                'precision_mean': ['mean', 'std', 'min', 'max', 'count'],
                'recall_mean': ['mean', 'std', 'min', 'max'],
                'f1_score_mean': ['mean', 'std', 'min', 'max'],
                'support_sum': 'sum'
            }).round(4)
            
            celltype_summary.columns = ['_'.join(col).strip() for col in celltype_summary.columns]
            celltype_summary_file = os.path.join(output_folder, f"{pipeline}_celltype_summary_across_tissues.csv")
            celltype_summary.to_csv(celltype_summary_file)
            
            print(f"✓ Saved cross-tissue per-cell-type results for {pipeline}")

def fit_lr_and_eval(Xtr, ytr, Xte, yte, pipeline_name):
    """Logistic Regression classification with balanced class weights."""
    le = LabelEncoder()
    ytr_enc = le.fit_transform(ytr)
    yte_enc = le.transform(yte)
    
    n_classes = len(le.classes_)
    binary = n_classes == 2

    lr_model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced',
        solver='lbfgs',
        multi_class='ovr' if binary else 'multinomial',
        n_jobs=-1
    )
    lr_model.fit(Xtr, ytr_enc)
    
    probs = lr_model.predict_proba(Xte)
    preds_enc = np.argmax(probs, axis=1)
    preds = le.inverse_transform(preds_enc)

    all_labels = sorted(list(set(ytr) | set(yte)))
    acc = accuracy_score(yte, preds)
    macro_f1 = f1_score(yte, preds, labels=all_labels, average='macro', zero_division=0)
    macro_precision = precision_score(yte, preds, labels=all_labels, average='macro', zero_division=0)
    macro_recall = recall_score(yte, preds, labels=all_labels, average='macro', zero_division=0)

    try:
        if binary:
            auroc = roc_auc_score(yte_enc, probs[:, 1])
        else:
            auroc = roc_auc_score(yte_enc, probs, multi_class='ovr', average='macro')
    except Exception:
        auroc = np.nan

    report_str = classification_report(
        yte, preds, digits=3, labels=all_labels, zero_division=0
    )
    
    result = {
        'model': lr_model,
        'preds': preds,
        'probs': probs,
        'y_true': yte,
        'encoder': le,
        'report': report_str,
        'Accuracy': acc,
        'Macro F1': macro_f1,
        'Precision': macro_precision,
        'Recall': macro_recall,
        'AUROC': auroc
    }
    
    return result


def process_single_tissue(tissue_name, data_base_path, output_folder, chunk_size=CHUNK_SIZE):
    print(f"\n{'='*80}")
    print(f"Processing Tissue: {tissue_name}")
    print(f"{'='*80}")

    np.random.seed(BASE_RANDOM_STATE)
    
    file_path = os.path.join(data_base_path, f"{tissue_name}.h5ad")

    try:
        print(f"Loading {tissue_name}...")
        
        adata_backed = tabula_sapiens(
            tissue=tissue_name, 
            version="v2", 
            path=file_path,
            force_download=False
        )
        
        obs = adata_backed.obs.copy()
        print(f"Total cells: {len(obs)}, features: {adata_backed.n_vars}")

        if 'donor_id' in obs.columns:
            try:
                if obs['donor_id'].dtype == object:
                     donor_numbers = obs['donor_id'].str.replace('TSP', '', regex=False).astype(int)
                else:
                    donor_numbers = obs['donor_id'].astype(int)
                obs = obs.loc[donor_numbers > DONOR_THRESHOLD]
            except Exception as e:
                print(f"Warning: Issue filtering donor_ids: {e}. Skipping donor filter.")
        else:
            print("Warning: 'donor_id' column not found. Skipping donor filter.")


        type_counts = obs[CELL_TYPE_COLUMN].value_counts()
        valid_types = type_counts[type_counts >= MIN_CELLS_PER_TYPE].index
        obs = obs[obs[CELL_TYPE_COLUMN].isin(valid_types)]

        if len(valid_types) < 2:
            print("⚠️ Not enough cell types after filtering — skipping tissue")
            return None

        valid_idx = obs.index
        print(f"Remaining cells: {len(valid_idx)} across {len(valid_types)} cell types.")

        print("\nExtracting valid data from disk...")
        celltype_to_data = {ct: [] for ct in valid_types}

        for start in tqdm(range(0, len(valid_idx), chunk_size)):
            batch_idx = valid_idx[start:start + chunk_size]
            
            
            try:
                X_chunk = adata_backed[batch_idx].X[:]
                
                if hasattr(X_chunk, "toarray"):
                    X_chunk = X_chunk.toarray()
                    
                df_chunk = pd.DataFrame(X_chunk, index=batch_idx, columns=adata_backed.var_names)

                obs_chunk = obs.loc[batch_idx]
                for ct in obs_chunk[CELL_TYPE_COLUMN].unique():
                    ids = obs_chunk.index[obs_chunk[CELL_TYPE_COLUMN] == ct]
                    if len(ids) == 0:
                        continue
                    celltype_to_data[ct].append(df_chunk.loc[ids])

                del X_chunk, df_chunk, obs_chunk
            except Exception as e:
                print(f"Error processing chunk: {e}")
                continue
            
            gc.collect()

        print("\nRunning top.process() per cell type...")
        X_all_list, y_all_list = [], []

        for ct, chunk_list in tqdm(celltype_to_data.items()):
            if not chunk_list:
                continue

            df_ct = pd.concat(chunk_list, axis=0)
            del chunk_list
            gc.collect()

            processed = top.process(df_ct.T, average=False, chunk_size=500)
            X_all_list.append(processed.T.values)
            y_all_list.extend([ct] * processed.shape[1])

            del df_ct, processed
            gc.collect()

        del celltype_to_data
        gc.collect()

        X_all = np.vstack(X_all_list)
        y_all = np.array(y_all_list)

        del X_all_list, y_all_list
        gc.collect()

        print(f"\nFinal dataset: {X_all.shape[0]} samples × {X_all.shape[1]} features")
        print(f"Cell types: {np.unique(y_all)}")

        all_split_results = []

        for split_idx in range(N_SPLITS):
            print(f"\nSplit {split_idx + 1}/{N_SPLITS}")

            X_train, X_test, y_train, y_test = train_test_split(
                X_all, y_all, test_size=TEST_SIZE,
                random_state=BASE_RANDOM_STATE + split_idx,
                stratify=y_all
            )

            k = min(ANOVA_K, X_train.shape[1])
            anova_selector = SelectKBest(score_func=f_classif, k=k)
            X_train_fs = anova_selector.fit_transform(X_train, y_train)
            X_test_fs = anova_selector.transform(X_test)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_fs)
            X_test_scaled = scaler.transform(X_test_fs)

            n_pca = min(PCA_COMPONENTS, X_train_scaled.shape[1])
            pca = PCA(n_components=n_pca, whiten=True, random_state=RANDOM_STATE)
            Xtr = pca.fit_transform(X_train_scaled)
            Xte = pca.transform(X_test_scaled)

            split_results = {
                "LogisticRegression": fit_lr_and_eval(Xtr, y_train, Xte, y_test, "LogisticRegression"),
            }

            save_split_results(tissue_name, split_idx, split_results, output_folder)
            all_split_results.append(split_results)

            del X_train, X_test, X_train_fs, X_test_fs, X_train_scaled, X_test_scaled, Xtr, Xte
            gc.collect()

        save_aggregated_results(tissue_name, all_split_results, output_folder)
        print(f"\n✓ Completed processing for {tissue_name}")
        return True

    except Exception as e:
        print(f"✗ Error processing {tissue_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Process tissues from the provided list"""
    print(f"Starting processing for {len(TISSUES_LIST)} tissues")
    print(f"Using {N_SPLITS} train-test splits with test_size={TEST_SIZE}")
    
    summary_file = os.path.join(OUTPUT_FOLDER, "processing_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Tissue Classification Pipeline (Train-Test Split)\n")
        f.write(f"Started: {datetime.now()}\n")
        f.write(f"Total tissues: {len(TISSUES_LIST)}\n")
        f.write(f"Number of splits: {N_SPLITS}\n")
        f.write("="*80 + "\n\n")
    
    success_count = 0
    failed_tissues = []
    
    
    all_tissues_results = []
    
    for tissue_name in TISSUES_LIST:
        
        result = process_single_tissue(tissue_name, DATA_BASE_PATH, OUTPUT_FOLDER)
        
        if result:
            success_count += 1
            tissue_folder = os.path.join(OUTPUT_FOLDER, tissue_name)
            for pipeline in ['LogisticRegression']:
                summary_file_path = os.path.join(tissue_folder, f"{pipeline}_summary_stats.json")
                if os.path.exists(summary_file_path):
                    with open(summary_file_path, 'r') as f:
                        stats = json.load(f)
                        all_tissues_results.append(stats)
        else:
            failed_tissues.append(tissue_name)
        
        with open(summary_file, 'a') as f:
            status = "✓ SUCCESS" if result else "✗ FAILED"
            f.write(f"{status}: {tissue_name}\n")
        
        gc.collect()
    
    if all_tissues_results:
        cross_tissue_df = pd.DataFrame(all_tissues_results)
        cross_tissue_file = os.path.join(OUTPUT_FOLDER, "cross_tissue_summary.csv")
        cross_tissue_df.to_csv(cross_tissue_file, index=False)
        
        overall_stats = {}
        for pipeline in ['LogisticRegression']:
            pipeline_data = cross_tissue_df[cross_tissue_df['pipeline'] == pipeline]
            overall_stats[pipeline] = {
                'n_tissues': len(pipeline_data),
                'n_splits': N_SPLITS,
                'test_size': TEST_SIZE,
                'accuracy_mean': pipeline_data['accuracy_mean'].mean(),
                'accuracy_std': pipeline_data['accuracy_mean'].std(),
                'macro_f1_mean': pipeline_data['macro_f1_mean'].mean(),
                'macro_f1_std': pipeline_data['macro_f1_mean'].std()
            }
        
        overall_stats_file = os.path.join(OUTPUT_FOLDER, "overall_statistics.json")
        with open(overall_stats_file, 'w') as f:
            json.dump(overall_stats, f, indent=2)
    
    save_cross_tissue_per_celltype_results(OUTPUT_FOLDER)
    
    print(f"\n{'='*80}")
    print(f"Processing Complete!")
    print(f"Successful: {success_count}/{len(TISSUES_LIST)}")
    if failed_tissues:
        print(f"Failed tissues: {', '.join(failed_tissues)}")
    print(f"Results saved to: {OUTPUT_FOLDER}")
    print(f"{'='*80}")
    
    # Save final summary
    with open(summary_file, 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Processing Complete\n")
        f.write(f"Finished: {datetime.now()}\n")
        f.write(f"Successful: {success_count}/{len(TISSUES_LIST)}\n")
        if failed_tissues:
            f.write(f"Failed: {', '.join(failed_tissues)}\n")

if __name__ == "__main__":
    main()
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import sctop as top
import scanpy as sc
import re
from sklearn.model_selection import train_test_split
from collections import defaultdict
import time
from transcriptformer.datasets import tabula_sapiens
import anndata as ad
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import scipy as sp
import anndata as ad
from typing import Literal, Any

import os
import pickle
from pathlib import Path

output_dir = Path("sctop_tabula_results")
output_dir.mkdir(parents=True, exist_ok=True)

tissues = [
    "lymphnode", "heart", "ear", "endothelium", "epithelium", "germline",
    "immune", "neural", "stromal", "bladder", "blood", "bone_marrow",
    "eye", "fat", "kidney", "large_intestine", "liver", "lung",
    "lymph_node", "mammary", "muscle", "ovary", "pancreas", "prostate",
    "salivary_gland", "skin", "small_intestine", "spleen", "stomach",
    "testis", "thymus", "tongue", "trachea", "uterus", "vasculature"
]

for tissue in tissues:
    print(f'--- Processing {tissue} ---')

    try:
        file_path = os.path.join('your_path', f"{tissue}.h5ad")
        
        adata = tabula_sapiens(tissue=tissue, version="v2", path=file_path)
        print(f'{tissue} imported successfully')
    except Exception as e:
        print(f'Could not import {tissue}: {e}')
        continue # Skip to next tissue if import fails

    atlas_metadata = adata.obs
    cell_type_column = "cell_type"
    type_counts = atlas_metadata[cell_type_column].value_counts().sort_index()
    
    threshold = 100 
    types_above_threshold = type_counts[type_counts >= threshold].index

    # Check if there are at least 2 types above threshold to create a basis
    if len(types_above_threshold) > 1:
        print(f"Generating basis for {tissue}...")
        results = top.create_basis(adata, 'cell_type', 100, do_anova=False, outer_chunks = 5, inner_chunk_size = 500, plot_results=False)

        # 2. Save the results dictionary
        save_path = output_dir / f"{tissue}_top_results.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Results for {tissue} saved to {save_path}")
    else:
        print(f"Skipping {tissue}: Not enough cell types above threshold.")
import pandas as pd
from pathlib import Path

combo_path = Path(r'e:/Desktop/Supplemental_Code_stMLnet-AnalysisCode/stMLnet-AnalysisCode-main/benchmark/OtherMethods/combo_only 100列.csv')
mtx_dir = Path(r'e:/Desktop/Supplemental_Code_stMLnet-AnalysisCode-stMLnet-AnalysisCode-main/benchmark/apply_in_stBC/Breast_Cancer_Block_A_Section_1/filtered_feature_bc_matrix_combo')

combo = pd.read_csv(combo_path)
lig = combo['combo'].str.split('|').str[0].str.split('__').str[0]
rec = combo['combo'].str.split('|').str[1].str.split('__').str[0]
combo_genes = pd.Index(lig.tolist() + rec.tolist()).unique()

feat = pd.read_csv(mtx_dir / 'features.tsv', sep='\t', header=None, names=['id','name','type'])
feat_genes = pd.Index(feat['name'])

missing = combo_genes.difference(feat_genes)
extra = feat_genes.difference(combo_genes)

print('combo genes', len(combo_genes))
print('features genes', len(feat_genes))
print('missing', len(missing))
print('first missing', missing[:10].tolist())
print('extra beyond combo', len(extra))

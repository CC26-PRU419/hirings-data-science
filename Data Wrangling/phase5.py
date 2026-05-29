import os
import numpy as np
from scipy.sparse import load_npz
import pandas as pd
import pickle

print("=" * 55)
print("   FINAL HANDOFF VALIDATION — CC26-PRU419 HIRINGS")
print("=" * 55)

# 1. CSV
df_check = pd.read_csv('output/linkedin_jobs_cleaned.csv')
print(f"\n✅ linkedin_jobs_cleaned.csv")
print(f"   Rows    : {len(df_check):,}")
print(f"   Columns : {df_check.shape}")
print(f"   Null in clean_description: {df_check['clean_description'].isnull().sum()}")

# 2. TF-IDF
tfidf_check = load_npz('output/tfidf_matrix.npz')
print(f"\n✅ tfidf_matrix.npz")
print(f"   Shape   : {tfidf_check.shape}")

# 3. Feature names
with open('output/tfidf_feature_names.pkl', 'rb') as f:
    fn = pickle.load(f)
print(f"\n✅ tfidf_feature_names.pkl")
print(f"   Total features: {len(fn)}")

# 4. Embeddings
emb_check = np.load('output/job_embeddings.npy')
print(f"\n✅ job_embeddings.npy")
print(f"   Shape   : {emb_check.shape}")

# 5. Data dictionary
dd_size = os.path.getsize('output/data_dictionary.md') / 1024
print(f"\n✅ data_dictionary.md")
print(f"   Size    : {dd_size:.1f} KB")

# 6. Row alignment
assert len(df_check) == tfidf_check.shape[0] == emb_check.shape[0], \
    "❌ MISMATCH! Fix before handoff."
print(f"\n✅ Row alignment: {len(df_check):,} rows — ALIGNED")

# 7. File sizes
print(f"\n📁 Output folder contents:")
for fname in os.listdir('output/'):
    fpath = f'output/{fname}'
    size_mb = os.path.getsize(fpath) / (1024 * 1024)
    print(f"   {fname}: {size_mb:.2f} MB")

print("\n" + "=" * 55)
print("✅ ALL CHECKS PASSED")
print("✅ DATASET READY FOR HANDOFF TO AINUR & ALYA")
print(f"✅ Deadline: April 27, 2026 — ON TRACK")
print("=" * 55)

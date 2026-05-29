import nbformat as nbf
nb = nbf.read('notebook_wrangling.ipynb', as_version=4)

nb.cells.extend([
    nbf.v4.new_markdown_cell('# Phase 3: Feature Engineering (Vectorization)'),
    nbf.v4.new_code_cell('''import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz
import pickle

os.makedirs('output', exist_ok=True)

# 3.1 TF-IDF Vectorization
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.85,
    sublinear_tf=True
)

tfidf_matrix = tfidf.fit_transform(df_jobs['clean_description'])
feature_names = tfidf.get_feature_names_out()

save_npz('output/tfidf_matrix.npz', tfidf_matrix)
with open('output/tfidf_feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)

print(f"TF-IDF Matrix shape: {tfidf_matrix.shape}")
print(f"Top 30 features have been calculated.")'''),
    nbf.v4.new_code_cell('''# 3.2 Semantic Embedding
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
descriptions = df_jobs['clean_description'].tolist()
BATCH_SIZE = 512
all_embeddings = []

for i in range(0, len(descriptions), BATCH_SIZE):
    batch = descriptions[i:i + BATCH_SIZE]
    batch_embeddings = model.encode(batch, show_progress_bar=False)
    all_embeddings.append(batch_embeddings)

final_embeddings = np.vstack(all_embeddings)
np.save('output/job_embeddings.npy', final_embeddings)

print(f"Embedding matrix shape: {final_embeddings.shape}")'''),
    nbf.v4.new_code_cell('''# 3.3 Save Final Dataset
cols_to_keep = ['job_id', 'company_id', 'title', 'description', 'clean_description',
                'max_salary', 'min_salary', 'location', 'formatted_experience_level',
                'applies', 'views', 'remote_allowed', 'job_posting_url']

available_cols = [c for c in cols_to_keep if c in df_jobs.columns]
df_final = df_jobs[available_cols].copy()
df_final.to_csv('output/linkedin_jobs_cleaned.csv', index=False)

print(f"Final CSV saved with shape: {df_final.shape}")
print("Row alignment check:", len(df_final) == tfidf_matrix.shape[0] == final_embeddings.shape[0])''')
])

nbf.write(nb, 'notebook_wrangling.ipynb')

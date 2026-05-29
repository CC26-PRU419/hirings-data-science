import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz
from sentence_transformers import SentenceTransformer
import pickle

os.makedirs('output', exist_ok=True)

print("Loading df_jobs_nlp.pkl...")
df_jobs = pd.read_pickle('df_jobs_nlp.pkl')

print("Step 3.1: TF-IDF Vectorization...")
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

sum_tfidf = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
top_30_indices = sum_tfidf.argsort()[-30:][::-1]
top_30_features = [(feature_names[i], sum_tfidf[i]) for i in top_30_indices]
print("\nTop 30 TF-IDF features:")
for feat, score in top_30_features:
    print(f"{feat}: {score:.2f}")

print("\nStep 3.2: Semantic Embedding...")
model = SentenceTransformer('all-MiniLM-L6-v2')
descriptions = df_jobs['clean_description'].tolist()
BATCH_SIZE = 512
all_embeddings = []

for i in range(0, len(descriptions), BATCH_SIZE):
    batch = descriptions[i:i + BATCH_SIZE]
    batch_embeddings = model.encode(batch, show_progress_bar=False)
    all_embeddings.append(batch_embeddings)
    if i % 10_000 == 0:
        print(f"Embedding progress: {i} / {len(descriptions)}")

final_embeddings = np.vstack(all_embeddings)
np.save('output/job_embeddings.npy', final_embeddings)

print(f"Embedding matrix shape: {final_embeddings.shape}")
print(f"Example vector (row 0, first 10 dims): {final_embeddings[0][:10]}")

print("\nStep 3.3: Saving Final Dataset...")
cols_to_keep = ['job_id', 'company_id', 'title', 'description', 'clean_description',
                'max_salary', 'min_salary', 'location', 'formatted_experience_level',
                'applies', 'views', 'remote_allowed', 'job_posting_url']

available_cols = [c for c in cols_to_keep if c in df_jobs.columns]
df_final = df_jobs[available_cols].copy()
df_final.to_csv('output/linkedin_jobs_cleaned.csv', index=False)

print(f"Final CSV saved with shape: {df_final.shape}")

csv_len = len(df_final)
tfidf_len = tfidf_matrix.shape[0]
emb_len = final_embeddings.shape[0]

print(f"\nAlignment check:")
print(f"CSV rows: {csv_len}, TF-IDF rows: {tfidf_len}, Embedding rows: {emb_len}")
if csv_len == tfidf_len == emb_len:
    print("Alignment OK: TRUE")
else:
    print("Alignment OK: FALSE")

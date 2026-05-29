# Data Dictionary — LinkedIn Jobs Cleaned Dataset
**Project:** Hirings (CC26-PRU419)
**Prepared by:** Arvin Demas Naryama
**Last Updated:** April 2026
**Source:** LinkedIn Job Postings 2024 (Kaggle, ~123K rows from 33k/postings.csv)

---

## File: linkedin_jobs_cleaned.csv
Total rows: 123.824 | Total columns: 13

| Column | Data Type | Description | Example Value | Category |
|--------|-----------|-------------|---------------|----------|
| job_id | string | Unique LinkedIn job posting ID | "3901987" | Original |
| company_id | string | LinkedIn company ID | "1009" | Original |
| title | string | Job position title | "Senior Data Engineer" | Original |
| description | string | Raw job description text | "We are looking for a..." | Original |
| clean_description | string | Description after full NLP pipeline (noise removal, lowercase, stopword removal, lemmatization) | "look data engineer python skill" | Engineered |
| max_salary | float64 | Maximum annual salary in USD. Null values filled with column median. | 150000.0 | Original |
| min_salary | float64 | Minimum annual salary in USD. Null values filled with column median. | 80000.0 | Original |
| location | string | Job location in City, State format | "New York, NY" | Original |
| formatted_experience_level | string | Seniority level of the role | "Mid-Senior level" | Original |
| applies | float64 | Number of applications submitted. Null filled with 0. | 342.0 | Original |
| views | float64 | Number of times the posting was viewed | 1200.0 | Original |
| remote_allowed | float64 | Remote work flag: 1=remote allowed, 0=not remote. Null filled with 0. | 1.0 | Original |
| job_posting_url | string | Direct URL to the LinkedIn job posting | "https://linkedin.com/jobs/view/..." | Original |

---

## File: tfidf_matrix.npz
- **Type:** Scipy sparse matrix (CSR format)
- **Shape:** (123824, 5000)
- **Description:** TF-IDF weighted matrix from clean_description. Each row = one job posting. Each column = one of 5000 n-gram features (unigrams + bigrams).
- **Parameters used:** max_features=5000, ngram_range=(1,2), min_df=5, max_df=0.85, sublinear_tf=True
- **Top features:** work, team, service, customer, skill, job, company, opportunity, ability, year
- **How to load:**
```python
from scipy.sparse import load_npz
tfidf_matrix = load_npz('output/tfidf_matrix.npz')
```

---

## File: tfidf_feature_names.pkl
- **Type:** Python list (pickle format)
- **Length:** 5000
- **Description:** List of all 5000 feature names (words/bigrams) corresponding to columns in tfidf_matrix.npz
- **How to load:**
```python
import pickle
with open('output/tfidf_feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)
```

---

## File: job_embeddings.npy
- **Type:** NumPy ndarray (float32)
- **Shape:** (123824, 384)
- **Description:** Semantic embedding vectors generated from clean_description using SentenceTransformer model all-MiniLM-L6-v2. Each row = one job posting. 384 dimensions capture semantic meaning.
- **Note:** Generate ulang di Google Colab menggunakan GPU untuk hasil embedding yang valid (kode murni tersedia di notebook_wrangling.ipynb).
- **How to load:**
```python
import numpy as np
embeddings = np.load('output/job_embeddings.npy')
```

---

## NLP Pipeline Summary
Steps applied to `description` column to produce `clean_description`:

| Step | Method | Library | Description |
|------|--------|---------|-------------|
| 1 | Noise Removal | re (Regex) | Removed HTML tags, URLs, emails, digits, special characters |
| 2 | Case Folding | Python built-in | Converted all text to lowercase |
| 3 | Tokenization | str.split() | Split sentences into individual word tokens |
| 4 | Stopword Removal | NLTK | Removed English stopwords + tokens shorter than 3 characters |
| 5 | Lemmatization | NLTK WordNetLemmatizer | Reduced words to their base/root form |

---

## Data Quality Summary

| Metric | Before Cleaning | After Cleaning |
|--------|----------------|----------------|
| Total rows | 123.849 | 123.824 |
| Duplicate rows | 0 | 0 |
| Null in description | 7 | 0 |
| Null in title | 0 | 0 |
| Null in max_salary | 75.94% | 0% (filled with median) |
| Null in min_salary | 75.94% | 0% (filled with median) |
| Null in remote_allowed | 87.69% | 0% (filled with 0) |
| Rows dropped (NLP empty) | - | 18 |

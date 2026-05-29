import nbformat as nbf
nb = nbf.v4.new_notebook()

nb.cells = [
    nbf.v4.new_markdown_cell('# Phase 1: Data Wrangling'),
    nbf.v4.new_code_cell('''import pandas as pd
import gc

# 1.1 Read Data Using Chunking
chunk_list = []
for chunk in pd.read_csv('33k/postings.csv', chunksize=100_000):
    chunk_list.append(chunk)
    gc.collect()

df_jobs = pd.concat(chunk_list, ignore_index=True)
del chunk_list
gc.collect()

df_skills = pd.read_csv('33k/jobs/job_skills.csv')
df_industries = pd.read_csv('33k/jobs/job_industries.csv')

print(f"job_postings shape: {df_jobs.shape}")
print(f"job_skills shape: {df_skills.shape}")
print(f"job_industries shape: {df_industries.shape}")'''),
    nbf.v4.new_code_cell('''# 1.2 Data Assessment
missing = df_jobs.isnull().sum().reset_index()
missing.columns = ['column', 'missing_count']
missing['missing_pct'] = (missing['missing_count'] / len(df_jobs) * 100).round(2)
missing = missing[missing['missing_count'] > 0].sort_values('missing_pct', ascending=False)
print("=== MISSING VALUES ===")
print(missing.to_string())

print(f"\\n=== DUPLICATES ===")
print(f"Total duplicate rows: {df_jobs.duplicated().sum()}")
print(f"Duplicate job_id: {df_jobs.duplicated(subset=['job_id']).sum()}")

print("\\n=== DATA TYPES ===")
print(df_jobs.dtypes)

print("\\n=== BASIC STATISTICS ===")
print(df_jobs.describe())'''),
    nbf.v4.new_code_cell('''# 1.3 Data Cleaning
# 1. Remove duplicate job_id
df_jobs.drop_duplicates(subset=['job_id'], keep='first', inplace=True)

# 2. Drop rows missing critical columns
df_jobs.dropna(subset=['description', 'title'], inplace=True)

# 3. Strip whitespace from text columns
df_jobs['title'] = df_jobs['title'].str.strip()
df_jobs['description'] = df_jobs['description'].str.strip()

# 4. Standardize data types
df_jobs['job_id'] = df_jobs['job_id'].astype(str)
df_jobs['company_id'] = df_jobs['company_id'].astype(str)

# 5. Handle salary nulls & remote nulls
df_jobs['max_salary'] = df_jobs['max_salary'].fillna(df_jobs['max_salary'].median())
df_jobs['min_salary'] = df_jobs['min_salary'].fillna(df_jobs['min_salary'].median())
# Assume missing remote_allowed means not remote (0)
df_jobs['remote_allowed'] = df_jobs['remote_allowed'].fillna(0.0)

# 6. Reset index
df_jobs.reset_index(drop=True, inplace=True)

print(f"Shape after cleaning: {df_jobs.shape}")''')
]

nbf.write(nb, 'notebook_wrangling.ipynb')

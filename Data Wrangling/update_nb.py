import nbformat as nbf
nb = nbf.read('notebook_wrangling.ipynb', as_version=4)

nb.cells.extend([
    nbf.v4.new_markdown_cell('# Phase 2: Text Preprocessing (NLP Pipeline)'),
    nbf.v4.new_code_cell('''import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()'''),
    nbf.v4.new_code_cell('''def full_nlp_pipeline(text: str) -> str:
    """
    Full NLP pipeline:
    1. Noise Removal (HTML, URLs, numbers, special chars)
    2. Case Folding (lowercase)
    3. Tokenization
    4. Stopword Removal
    5. Lemmatization
    """
    if not isinstance(text, str) or text.strip() == '':
        return ''

    # Step 1: Noise Removal
    text = re.sub(r'<[^>]+>', ' ', text)            # Remove HTML tags
    text = re.sub(r'http\S+|www\.\S+', ' ', text)   # Remove URLs
    text = re.sub(r'\S+@\S+', ' ', text)             # Remove emails
    text = re.sub(r'\d+', ' ', text)                 # Remove numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)         # Remove symbols
    text = re.sub(r'\s+', ' ', text).strip()         # Normalize whitespace

    # Step 2: Case Folding
    text = text.lower()

    # Step 3: Tokenization
    tokens = text.split()

    # Step 4: Stopword Removal (keep only meaningful words, min length 3)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

    # Step 5: Lemmatization
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return ' '.join(tokens)

# Apply pipeline
print("Running NLP pipeline on descriptions...")
df_jobs['clean_description'] = df_jobs['description'].apply(full_nlp_pipeline)

# Remove rows where clean_description ended up empty
df_jobs = df_jobs[df_jobs['clean_description'].str.len() > 10]
df_jobs.reset_index(drop=True, inplace=True)

print(f"Shape after NLP: {df_jobs.shape}")'''),
    nbf.v4.new_code_cell('''# Show 3 before/after examples
for i in range(3):
    print(f"\\n--- Example {i+1} ---")
    print(f"BEFORE: {df_jobs['description'].iloc[i][:300]}")
    print(f"AFTER : {df_jobs['clean_description'].iloc[i][:300]}")''')
])

nbf.write(nb, 'notebook_wrangling.ipynb')

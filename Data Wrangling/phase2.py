import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import sys

# Download directly inside the script
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def full_nlp_pipeline(text: str) -> str:
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

if __name__ == "__main__":
    print("Loading df_jobs_cleaned.pkl...")
    df_jobs = pd.read_pickle('df_jobs_cleaned.pkl')
    
    print("Running NLP pipeline on descriptions. This might take a bit...")
    df_jobs['clean_description'] = df_jobs['description'].apply(full_nlp_pipeline)

    original_shape = df_jobs.shape
    df_jobs = df_jobs[df_jobs['clean_description'].str.len() > 10]
    df_jobs.reset_index(drop=True, inplace=True)

    print(f"Shape before NLP filter: {original_shape}")
    print(f"Shape after NLP: {df_jobs.shape}")

    # Calculate average length
    avg_length = df_jobs['clean_description'].str.len().mean()
    print(f"Average length of clean_description: {avg_length:.2f} characters")
    
    # Save checkpoint
    df_jobs.to_pickle('df_jobs_nlp.pkl')
    print("✅ Checkpoint df_jobs_nlp.pkl saved.")

    # Show 3 before/after examples
    for i in range(3):
        print(f"\n--- Example {i+1} ---")
        print(f"BEFORE: {df_jobs['description'].iloc[i][:300]}")
        print(f"AFTER : {df_jobs['clean_description'].iloc[i][:300]}")

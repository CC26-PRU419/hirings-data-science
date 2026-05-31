# ============================================================
# Feature Engineering Tabular — Hirings (CC26-PRU419)
# Author  : Arvin Demas Naryama
# Created : May 2026
# Purpose : Tambahan fitur tabular di atas NLP features yang
#           sudah ada (TF-IDF + Semantic Embedding) untuk
#           meningkatkan performa model rekomendasi karier.
# ============================================================

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# ── 1. Load Dataset ──────────────────────────────────────────
print("=" * 55)
print("  FEATURE ENGINEERING TABULAR — CC26-PRU419 HIRINGS")
print("=" * 55)

CSV_PATH = "output/linkedin_jobs_cleaned.csv"
assert os.path.exists(CSV_PATH), f"File tidak ditemukan: {CSV_PATH}"

df = pd.read_csv(CSV_PATH)
print(f"\n✅ Dataset loaded: {df.shape}")
print(f"   Kolom awal: {df.columns.tolist()}")

# ── 2. Salary Features ───────────────────────────────────────
print("\n[STEP 2] Salary Features...")

# Midpoint & Range
df["salary_midpoint"] = (df["max_salary"] + df["min_salary"]) / 2
df["salary_range"]    = df["max_salary"] - df["min_salary"]

# Log transform (mengatasi skewness ekstrem)
df["log_salary_mid"]  = np.log1p(df["salary_midpoint"])

# Salary tier berdasarkan kuartil
q33 = df["salary_midpoint"].quantile(0.33)
q66 = df["salary_midpoint"].quantile(0.66)

def salary_tier(val):
    if val <= q33:
        return 0   # Low
    elif val <= q66:
        return 1   # Mid
    else:
        return 2   # High

df["salary_tier"] = df["salary_midpoint"].apply(salary_tier)

print(f"   salary_midpoint : min={df['salary_midpoint'].min():.0f}, max={df['salary_midpoint'].max():.0f}")
print(f"   salary_range    : min={df['salary_range'].min():.0f}, max={df['salary_range'].max():.0f}")
print(f"   salary_tier dist:\n{df['salary_tier'].value_counts().sort_index().to_string()}")

# ── 3. Engagement Features ───────────────────────────────────
print("\n[STEP 3] Engagement Features...")

# Application rate (applies / views) — proxy engagement user
df["applies"]       = df["applies"].fillna(0)
df["views"]         = df["views"].fillna(1)  # hindari div/0
df["engagement_rate"] = df["applies"] / df["views"]

# Log transform engagement rate
df["log_engagement"] = np.log1p(df["engagement_rate"])

# Popularity flag: top 25% views
view_q75 = df["views"].quantile(0.75)
df["is_popular"] = (df["views"] >= view_q75).astype(int)

print(f"   engagement_rate mean  : {df['engagement_rate'].mean():.4f}")
print(f"   is_popular (top 25%%) : {df['is_popular'].sum():,} lowongan")

# ── 4. Text Length Features ──────────────────────────────────
print("\n[STEP 4] Text Length Features...")

df["description_word_count"] = df["clean_description"].fillna("").str.split().str.len()
df["title_word_count"]        = df["title"].fillna("").str.split().str.len()
df["desc_title_ratio"]        = df["description_word_count"] / (df["title_word_count"] + 1)

# Panjang deskripsi: Short / Medium / Long
def desc_tier(wc):
    if wc < 50:   return 0   # Short
    elif wc < 150: return 1  # Medium
    else:          return 2  # Long

df["desc_length_tier"] = df["description_word_count"].apply(desc_tier)

print(f"   description_word_count mean : {df['description_word_count'].mean():.1f}")
print(f"   title_word_count mean       : {df['title_word_count'].mean():.1f}")

# ── 5. Remote Feature ────────────────────────────────────────
print("\n[STEP 5] Remote Feature...")

df["has_remote"] = df["remote_allowed"].fillna(0).astype(int)
print(f"   Remote allowed: {df['has_remote'].sum():,} lowongan ({df['has_remote'].mean()*100:.1f}%)")

# ── 6. Experience Level Encoding ─────────────────────────────
print("\n[STEP 6] Experience Level Encoding...")

EXP_ORDER = {
    "Internship":    0,
    "Entry level":   1,
    "Associate":     2,
    "Mid-Senior level": 3,
    "Director":      4,
    "Executive":     5,
}

df["exp_level_encoded"] = (
    df["formatted_experience_level"]
    .map(EXP_ORDER)
    .fillna(-1)   # -1 = unknown
    .astype(int)
)

print(f"   Level distribution:")
print(df["exp_level_encoded"].value_counts().sort_index().to_string())

# ── 7. Location Features ─────────────────────────────────────
print("\n[STEP 7] Location Features...")

# Ekstrak state dari 'City, STATE' format
df["state"] = df["location"].str.extract(r",\s*([A-Z]{2})\s*$")

# Top 10 states → encode, sisanya = "OTHER"
top_states = df["state"].value_counts().head(10).index.tolist()
df["state_group"] = df["state"].apply(lambda x: x if x in top_states else "OTHER")

# Label encode
le_state = LabelEncoder()
df["state_encoded"] = le_state.fit_transform(df["state_group"].fillna("OTHER"))

print(f"   Top states: {top_states}")

# ── 8. Normalize Numeric Features (MinMax) ──────────────────
print("\n[STEP 8] MinMax Normalization...")

COLS_TO_NORM = [
    "salary_midpoint", "salary_range", "log_salary_mid",
    "engagement_rate", "log_engagement",
    "description_word_count", "title_word_count",
]

scaler = MinMaxScaler()
norm_values = scaler.fit_transform(df[COLS_TO_NORM])
norm_df = pd.DataFrame(norm_values, columns=[f"{c}_norm" for c in COLS_TO_NORM])
df = pd.concat([df.reset_index(drop=True), norm_df.reset_index(drop=True)], axis=1)

print(f"   Kolom ternormalisasi: {norm_df.columns.tolist()}")

# ── 9. Summary ───────────────────────────────────────────────
NEW_FEATURES = [
    "salary_midpoint", "salary_range", "log_salary_mid", "salary_tier",
    "engagement_rate", "log_engagement", "is_popular",
    "description_word_count", "title_word_count", "desc_title_ratio", "desc_length_tier",
    "has_remote",
    "exp_level_encoded",
    "state_encoded",
] + [f"{c}_norm" for c in COLS_TO_NORM]

print(f"\n{'='*55}")
print(f"  TOTAL FITUR BARU : {len(NEW_FEATURES)}")
print(f"  Total kolom df   : {df.shape[1]}")
print(f"{'='*55}")

# ── 10. Save ─────────────────────────────────────────────────
os.makedirs("output", exist_ok=True)
OUTPUT_CSV = "output/linkedin_jobs_features.csv"
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Dataset dengan fitur baru disimpan ke: {OUTPUT_CSV}")
print(f"   Shape: {df.shape}")

# Feature summary table
summary = pd.DataFrame({
    "Feature":     NEW_FEATURES,
    "Type":        ["float64", "float64", "float64", "int (0-2)",
                    "float64", "float64", "int (0/1)",
                    "int", "int", "float64", "int (0-2)",
                    "int (0/1)",
                    "int (-1..5)",
                    "int",
                   ] + ["float64 (0-1)"] * len(COLS_TO_NORM),
    "Description": [
        "Rata-rata (max+min)/2 gaji tahunan USD",
        "Selisih max_salary - min_salary",
        "Log transform salary_midpoint",
        "0=Low, 1=Mid, 2=High (berdasarkan kuartil)",
        "applies / views (tingkat konversi)",
        "Log transform engagement_rate",
        "1 jika views >= kuartil ke-75",
        "Jumlah kata di clean_description",
        "Jumlah kata di title",
        "description_word_count / (title_word_count+1)",
        "0=Short(<50), 1=Med(<150), 2=Long(>=150 kata)",
        "1 jika remote_allowed=1",
        "Ordinal level karir (Internship=0 .. Executive=5)",
        "Encoded state/provinsi top-10",
    ] + [f"MinMax normalized {c}" for c in COLS_TO_NORM],
})

DICT_PATH = "output/feature_engineering_summary.md"
with open(DICT_PATH, "w", encoding="utf-8") as f:
    f.write("# Feature Engineering Summary — Hirings\n\n")
    f.write("**Author:** Arvin Demas Naryama  \n")
    f.write("**Project:** CC26-PRU419 Hirings  \n")
    f.write("**Date:** May 2026  \n\n---\n\n")
    f.write(summary.to_markdown(index=False))
    f.write("\n")

print(f"✅ Feature summary disimpan ke: {DICT_PATH}")
print("\n✅ FEATURE ENGINEERING SELESAI!\n")

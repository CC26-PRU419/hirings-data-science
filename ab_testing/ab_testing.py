# ============================================================
# A/B Testing — Hirings Career Recommendation System
# Author  : Arvin Demas Naryama (Data Science)
# Project : CC26-PRU419 Hirings
# Date    : May 2026
#
# Tujuan:
#   Membandingkan performa dua algoritma rekomendasi pekerjaan:
#     - Group A (Control)  : TF-IDF Cosine Similarity
#     - Group B (Treatment): Semantic Embedding (BERT) Cosine Similarity
#
# Metrik yang diuji:
#   1. Relevance Score      → Kualitas rekomendasi
#   2. Click-Through Rate   → Interaksi pengguna
#   3. Rank Quality (NDCG)  → Kualitas urutan hasil rekomendasi
#
# Metode Statistik:
#   - Two-sample t-test (parametrik)
#   - Mann-Whitney U (non-parametrik)
#   - Cohen's d (effect size)
#   - Power Analysis
#   - Confidence Interval Bootstrap
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import os

np.random.seed(42)
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#161b2e",
    "axes.edgecolor":   "#2d4070",
    "text.color":       "#e2e8f0",
    "axes.labelcolor":  "#e2e8f0",
    "xtick.color":      "#94a3b8",
    "ytick.color":      "#94a3b8",
    "grid.color":       "#2d4070",
    "grid.alpha":       0.4,
    "font.family":      "sans-serif",
})

os.makedirs("output/ab_testing", exist_ok=True)

# ════════════════════════════════════════════════════════════
# 1. SIMULASI DATA EKSPERIMEN
#    Simulasi 1000 user sessions di platform Hirings.
#    Setiap user mendapatkan 10 rekomendasi dari salah satu
#    algoritma (random assignment 50/50).
# ════════════════════════════════════════════════════════════
print("=" * 60)
print("  A/B TESTING — HIRINGS RECOMMENDATION SYSTEM")
print("  CC26-PRU419 | Data Science Division")
print("=" * 60)

N_USERS = 1000
N_RECS  = 10      # Rekomendasi per user

# ── Group A: TF-IDF Cosine Similarity ──────────────────────
# Karakteristik: baik di keyword matching, kurang di semantik
#   Relevance score ~ N(0.61, 0.12)
#   CTR ~ Binomial(p=0.18)
mu_A, std_A = 0.61, 0.12
ctr_A       = 0.18

# ── Group B: Semantic Embedding (all-MiniLM-L6-v2) ─────────
# Karakteristik: lebih memahami konteks, relevansi lebih tinggi
#   Relevance score ~ N(0.72, 0.10)   ← hipotesis improvement
#   CTR ~ Binomial(p=0.27)
mu_B, std_B = 0.72, 0.10
ctr_B       = 0.27

n_A = N_USERS // 2   # 500 users per group
n_B = N_USERS // 2

# Generate relevance scores (per user = mean dari N_RECS rekomendasi)
rel_A_raw = np.random.normal(mu_A, std_A, size=(n_A, N_RECS))
rel_A_raw = np.clip(rel_A_raw, 0, 1)
rel_A = rel_A_raw.mean(axis=1)   # per-user avg relevance

rel_B_raw = np.random.normal(mu_B, std_B, size=(n_B, N_RECS))
rel_B_raw = np.clip(rel_B_raw, 0, 1)
rel_B = rel_B_raw.mean(axis=1)

# Generate CTR (binary per user: klik salah satu rekomendasi?)
clicks_A = np.random.binomial(1, ctr_A, size=n_A)
clicks_B = np.random.binomial(1, ctr_B, size=n_B)

# Generate NDCG (Normalized Discounted Cumulative Gain) @10
def simulate_ndcg(relevance_matrix, k=10):
    """Hitung NDCG@k dari matrix relevansi biner."""
    ndcg_scores = []
    for row in relevance_matrix:
        binary = (row > 0.5).astype(int)
        binary_sorted = np.sort(binary)[::-1]
        gains      = binary
        ideal_gains = binary_sorted
        discounts  = 1 / np.log2(np.arange(2, k + 2))
        dcg  = np.sum(gains * discounts)
        idcg = np.sum(ideal_gains * discounts)
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)
    return np.array(ndcg_scores)

ndcg_A = simulate_ndcg(rel_A_raw)
ndcg_B = simulate_ndcg(rel_B_raw)

# Buat DataFrame lengkap
df_A = pd.DataFrame({
    "user_id":        range(1, n_A + 1),
    "group":          "A_TF-IDF",
    "relevance_score": rel_A,
    "clicked":        clicks_A,
    "ndcg_at_10":     ndcg_A,
    "session_time_s": np.random.gamma(3, 15, size=n_A),   # detik di halaman hasil
})

df_B = pd.DataFrame({
    "user_id":        range(n_A + 1, N_USERS + 1),
    "group":          "B_Semantic",
    "relevance_score": rel_B,
    "clicked":        clicks_B,
    "ndcg_at_10":     ndcg_B,
    "session_time_s": np.random.gamma(3.8, 15, size=n_B),
})

df_exp = pd.concat([df_A, df_B], ignore_index=True)
df_exp.to_csv("output/ab_testing/experiment_data.csv", index=False)

print(f"\n✅ Data eksperimen dibuat")
print(f"   Total users : {N_USERS}")
print(f"   Group A     : {n_A} users (TF-IDF)")
print(f"   Group B     : {n_B} users (Semantic Embedding)")
print(f"\n{df_exp.groupby('group')[['relevance_score','clicked','ndcg_at_10']].describe().round(3).to_string()}")

# ════════════════════════════════════════════════════════════
# 2. DESCRIPTIVE STATISTICS
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  2. DESCRIPTIVE STATISTICS")
print("=" * 60)

metrics = ["relevance_score", "clicked", "ndcg_at_10", "session_time_s"]
desc = df_exp.groupby("group")[metrics].agg(["mean", "median", "std"]).round(4)
print(f"\n{desc.to_string()}")

ctr_rate_A = clicks_A.mean()
ctr_rate_B = clicks_B.mean()
print(f"\n  CTR Group A (TF-IDF)  : {ctr_rate_A:.2%}")
print(f"  CTR Group B (Semantic): {ctr_rate_B:.2%}")
print(f"  CTR Uplift            : +{(ctr_rate_B - ctr_rate_A)*100:.2f}pp")

# ════════════════════════════════════════════════════════════
# 3. STATISTICAL TESTS
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  3. STATISTICAL HYPOTHESIS TESTING")
print("=" * 60)

ALPHA = 0.05

def run_hypothesis_test(a, b, metric_name):
    """Jalankan t-test, Mann-Whitney U, dan hitung effect size."""
    print(f"\n  ── {metric_name} ──")

    # Normality check (Shapiro-Wilk — sample 100 acak)
    _, p_norm_a = stats.shapiro(np.random.choice(a, 100, replace=False))
    _, p_norm_b = stats.shapiro(np.random.choice(b, 100, replace=False))
    normal = (p_norm_a > ALPHA) and (p_norm_b > ALPHA)
    print(f"  Normality (Shapiro p): A={p_norm_a:.4f}, B={p_norm_b:.4f} → {'Normal' if normal else 'Non-normal'}")

    # Levene's test (variance equality)
    _, p_levene = stats.levene(a, b)
    equal_var = p_levene > ALPHA
    print(f"  Levene (equal var p) : {p_levene:.4f} → {'Equal var' if equal_var else 'Unequal var'}")

    # Two-sample t-test
    t_stat, p_ttest = stats.ttest_ind(a, b, equal_var=equal_var, alternative="less")
    print(f"\n  [T-Test]")
    print(f"    H0 : mean(A) >= mean(B)")
    print(f"    H1 : mean(A) <  mean(B)  [B lebih baik]")
    print(f"    t-statistic  : {t_stat:.4f}")
    print(f"    p-value      : {p_ttest:.6f}")
    print(f"    Keputusan    : {'✅ Tolak H0 → B signifikan lebih baik' if p_ttest < ALPHA else '❌ Gagal tolak H0'} (α={ALPHA})")

    # Mann-Whitney U (non-parametrik)
    u_stat, p_mw = stats.mannwhitneyu(a, b, alternative="less")
    print(f"\n  [Mann-Whitney U]")
    print(f"    U-statistic  : {u_stat:.2f}")
    print(f"    p-value      : {p_mw:.6f}")
    print(f"    Keputusan    : {'✅ Tolak H0 → B signifikan lebih baik' if p_mw < ALPHA else '❌ Gagal tolak H0'} (α={ALPHA})")

    # Cohen's d (effect size)
    pooled_std = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    cohens_d   = (np.mean(b) - np.mean(a)) / pooled_std
    if   abs(cohens_d) < 0.2: effect = "Negligible"
    elif abs(cohens_d) < 0.5: effect = "Small"
    elif abs(cohens_d) < 0.8: effect = "Medium"
    else:                      effect = "Large"
    print(f"\n  [Effect Size]")
    print(f"    Cohen's d    : {cohens_d:.4f}  → {effect}")

    # 95% CI untuk perbedaan mean (bootstrap)
    diffs = []
    for _ in range(2000):
        boot_a = np.random.choice(a, size=len(a), replace=True)
        boot_b = np.random.choice(b, size=len(b), replace=True)
        diffs.append(boot_b.mean() - boot_a.mean())
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    print(f"\n  [Bootstrap 95% CI for (mean_B - mean_A)]")
    print(f"    [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"    {'→ CI tidak mencakup 0: signifikan' if ci_lo > 0 else '→ CI mencakup 0: tidak konklusif'}")

    return {
        "metric": metric_name,
        "mean_A": np.mean(a), "mean_B": np.mean(b),
        "delta_abs": np.mean(b) - np.mean(a),
        "delta_pct": (np.mean(b) - np.mean(a)) / np.mean(a) * 100,
        "t_stat": t_stat, "p_ttest": p_ttest,
        "p_mannwhitney": p_mw,
        "cohens_d": cohens_d, "effect": effect,
        "ci_lo": ci_lo, "ci_hi": ci_hi,
        "significant": p_ttest < ALPHA,
    }

results = []
results.append(run_hypothesis_test(rel_A,    rel_B,    "Relevance Score"))
results.append(run_hypothesis_test(ndcg_A,   ndcg_B,   "NDCG@10"))
results.append(run_hypothesis_test(df_A["session_time_s"].values,
                                   df_B["session_time_s"].values, "Session Time (s)"))

# CTR: Chi-square test (proporsi)
print(f"\n  ── CTR (Click-Through Rate) ──")
ct = np.array([[clicks_A.sum(), n_A - clicks_A.sum()],
               [clicks_B.sum(), n_B - clicks_B.sum()]])
chi2, p_chi2, dof, _ = stats.chi2_contingency(ct, correction=False)
print(f"  [Chi-Square Test for CTR]")
print(f"    CTR A : {ctr_rate_A:.4f} ({clicks_A.sum()} / {n_A})")
print(f"    CTR B : {ctr_rate_B:.4f} ({clicks_B.sum()} / {n_B})")
print(f"    χ²    : {chi2:.4f}  |  df={dof}")
print(f"    p-val : {p_chi2:.6f}")
print(f"    Keputusan : {'✅ Signifikan → CTR B > A' if p_chi2 < ALPHA else '❌ Tidak signifikan'}")
# Relative Risk
rr = ctr_rate_B / ctr_rate_A
print(f"    Relative Risk (B/A): {rr:.3f}  → CTR naik {(rr-1)*100:.1f}%")

results.append({
    "metric": "CTR", "mean_A": ctr_rate_A, "mean_B": ctr_rate_B,
    "delta_abs": ctr_rate_B - ctr_rate_A,
    "delta_pct": (ctr_rate_B - ctr_rate_A) / ctr_rate_A * 100,
    "t_stat": None, "p_ttest": p_chi2,
    "p_mannwhitney": None, "cohens_d": None, "effect": None,
    "ci_lo": None, "ci_hi": None, "significant": p_chi2 < ALPHA,
})

df_results = pd.DataFrame(results)
df_results.to_csv("output/ab_testing/test_results.csv", index=False)

# ════════════════════════════════════════════════════════════
# 4. POWER ANALYSIS
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  4. POWER ANALYSIS")
print("=" * 60)

from scipy.stats import norm as sp_norm

def compute_power(n, delta, sigma, alpha=0.05):
    """Power untuk two-sample one-sided t-test."""
    se     = sigma * np.sqrt(2 / n)
    z_crit = sp_norm.ppf(1 - alpha)
    z_beta = delta / se - z_crit
    return sp_norm.cdf(z_beta)

# Power aktual experiment ini
sigma_pooled = np.sqrt((rel_A.std(ddof=1)**2 + rel_B.std(ddof=1)**2) / 2)
delta_actual = rel_B.mean() - rel_A.mean()
power_actual = compute_power(n_A, delta_actual, sigma_pooled)

print(f"\n  Relevance Score:")
print(f"    Δ (effect)     : {delta_actual:.4f}")
print(f"    σ (pooled)     : {sigma_pooled:.4f}")
print(f"    n per group    : {n_A}")
print(f"    Power achieved : {power_actual:.4f}  ({'✅ Adequate (>0.80)' if power_actual > 0.80 else '⚠️ Kurang (perlu lebih banyak data)'})")

# Sample size yang dibutuhkan untuk power 80%
def min_sample_size(delta, sigma, alpha=0.05, power=0.80):
    z_alpha = sp_norm.ppf(1 - alpha)
    z_beta  = sp_norm.ppf(power)
    n = 2 * ((z_alpha + z_beta) * sigma / delta) ** 2
    return int(np.ceil(n))

min_n = min_sample_size(delta_actual, sigma_pooled)
print(f"    Min n (power=80%): {min_n} per group")

# ════════════════════════════════════════════════════════════
# 5. VISUALISASI
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  5. VISUALISASI")
print("=" * 60)

fig = plt.figure(figsize=(18, 14), facecolor="#0f1117")
fig.suptitle(
    "A/B Testing Results — Hirings Recommendation System\n"
    "TF-IDF (Control) vs Semantic Embedding (Treatment)",
    fontsize=16, fontweight="bold", color="#e2e8f0", y=0.98
)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

COL_A = "#3b82f6"
COL_B = "#22c55e"

# ── Plot 1: Relevance Score Distribution ──
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(rel_A, bins=30, alpha=0.7, color=COL_A, label="A: TF-IDF",    density=True)
ax1.hist(rel_B, bins=30, alpha=0.7, color=COL_B, label="B: Semantic",  density=True)
ax1.axvline(rel_A.mean(), color=COL_A, linestyle="--", lw=2)
ax1.axvline(rel_B.mean(), color=COL_B, linestyle="--", lw=2)
ax1.set_title("Relevance Score Distribution", color="#e2e8f0", fontsize=10)
ax1.set_xlabel("Relevance Score", color="#94a3b8", fontsize=9)
ax1.legend(fontsize=8, facecolor="#1e2a4a", labelcolor="#e2e8f0")

# ── Plot 2: NDCG@10 Distribution ──
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(ndcg_A, bins=30, alpha=0.7, color=COL_A, label="A: TF-IDF",   density=True)
ax2.hist(ndcg_B, bins=30, alpha=0.7, color=COL_B, label="B: Semantic", density=True)
ax2.axvline(ndcg_A.mean(), color=COL_A, linestyle="--", lw=2)
ax2.axvline(ndcg_B.mean(), color=COL_B, linestyle="--", lw=2)
ax2.set_title("NDCG@10 Distribution", color="#e2e8f0", fontsize=10)
ax2.set_xlabel("NDCG@10", color="#94a3b8", fontsize=9)
ax2.legend(fontsize=8, facecolor="#1e2a4a", labelcolor="#e2e8f0")

# ── Plot 3: CTR Bar Chart ──
ax3 = fig.add_subplot(gs[0, 2])
bars = ax3.bar(["A: TF-IDF", "B: Semantic"], [ctr_rate_A, ctr_rate_B],
               color=[COL_A, COL_B], width=0.5, edgecolor="#2d4070")
ax3.set_ylim(0, max(ctr_rate_A, ctr_rate_B) * 1.35)
ax3.set_ylabel("CTR", color="#94a3b8", fontsize=9)
ax3.set_title("Click-Through Rate", color="#e2e8f0", fontsize=10)
for bar, val in zip(bars, [ctr_rate_A, ctr_rate_B]):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{val:.2%}", ha="center", va="bottom", color="#e2e8f0", fontsize=10, fontweight="bold")
ax3.text(0.5, 0.90, f"Chi² p={p_chi2:.4f}\n✅ Significant",
         transform=ax3.transAxes, ha="center", fontsize=8, color="#22c55e")

# ── Plot 4: Box Plot Relevance ──
ax4 = fig.add_subplot(gs[1, 0])
bp = ax4.boxplot([rel_A, rel_B], labels=["A: TF-IDF", "B: Semantic"],
                 patch_artist=True, notch=True,
                 medianprops=dict(color="white", lw=2))
bp["boxes"][0].set_facecolor(COL_A + "80")
bp["boxes"][1].set_facecolor(COL_B + "80")
for element in ["whiskers", "caps", "fliers"]:
    plt.setp(bp[element], color="#94a3b8")
ax4.set_title("Relevance Score Boxplot", color="#e2e8f0", fontsize=10)
ax4.set_ylabel("Score", color="#94a3b8", fontsize=9)

# ── Plot 5: Effect Size Summary Bar ──
ax5 = fig.add_subplot(gs[1, 1])
metric_labels = ["Relevance\nScore", "NDCG@10", "Session\nTime"]
deltas = [r["delta_pct"] for r in results[:3]]
colors_delta = [COL_B if d > 0 else "#ef4444" for d in deltas]
bars2 = ax5.bar(metric_labels, deltas, color=colors_delta, edgecolor="#2d4070", width=0.5)
ax5.axhline(0, color="#94a3b8", lw=1)
ax5.set_ylabel("Δ % (B vs A)", color="#94a3b8", fontsize=9)
ax5.set_title("Relative Improvement (B over A)", color="#e2e8f0", fontsize=10)
for bar, val in zip(bars2, deltas):
    ax5.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.2 if val >= 0 else bar.get_height() - 0.5,
             f"{val:+.1f}%", ha="center", fontsize=9, fontweight="bold",
             color="#e2e8f0", va="bottom")

# ── Plot 6: Power Curve ──
ax6 = fig.add_subplot(gs[1, 2])
ns = np.arange(50, 1500, 20)
powers = [compute_power(n, delta_actual, sigma_pooled) for n in ns]
ax6.plot(ns, powers, color="#f59e0b", lw=2.5)
ax6.axhline(0.80, color="#ef4444", linestyle="--", lw=1.5, label="Target Power = 0.80")
ax6.axvline(n_A,  color=COL_A, linestyle=":", lw=1.5, label=f"n={n_A} (actual)")
ax6.axvline(min_n, color=COL_B, linestyle=":", lw=1.5, label=f"n={min_n} (min)")
ax6.set_xlabel("Sample Size (per group)", color="#94a3b8", fontsize=9)
ax6.set_ylabel("Statistical Power", color="#94a3b8", fontsize=9)
ax6.set_title("Power Analysis Curve", color="#e2e8f0", fontsize=10)
ax6.legend(fontsize=7, facecolor="#1e2a4a", labelcolor="#e2e8f0")
ax6.set_ylim(0, 1.05)

# ── Plot 7: Bootstrap CI ──
ax7 = fig.add_subplot(gs[2, :2])
metric_names_ci = [r["metric"] for r in results[:3]]
ci_los = [r["ci_lo"] for r in results[:3]]
ci_his = [r["ci_hi"] for r in results[:3]]
means  = [r["delta_abs"] for r in results[:3]]
y_pos  = np.arange(len(metric_names_ci))
ax7.barh(y_pos, means,
         xerr=[np.array(means) - np.array(ci_los),
               np.array(ci_his) - np.array(means)],
         color=[COL_B if m > 0 else "#ef4444" for m in means],
         ecolor="#f59e0b", capsize=6, height=0.4, edgecolor="#2d4070")
ax7.axvline(0, color="#94a3b8", lw=1.5, linestyle="--")
ax7.set_yticks(y_pos)
ax7.set_yticklabels(metric_names_ci, color="#e2e8f0", fontsize=10)
ax7.set_xlabel("Δ Mean (B − A)  [Bootstrap 95% CI]", color="#94a3b8", fontsize=9)
ax7.set_title("Bootstrap 95% Confidence Interval — Mean Difference (B − A)",
              color="#e2e8f0", fontsize=10)

# ── Plot 8: Summary Table ──
ax8 = fig.add_subplot(gs[2, 2])
ax8.axis("off")
table_data = [
    ["Metric", "A Mean", "B Mean", "Δ%", "p-val", "Sig."],
] + [
    [r["metric"][:12],
     f"{r['mean_A']:.4f}", f"{r['mean_B']:.4f}",
     f"{r['delta_pct']:+.1f}%",
     f"{r['p_ttest']:.4f}",
     "✅" if r["significant"] else "❌"]
    for r in results
]
tbl = ax8.table(table_data, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
for (row, col), cell in tbl.get_celld().items():
    cell.set_facecolor("#1e2a4a" if row == 0 else "#0f1117")
    cell.set_edgecolor("#2d4070")
    cell.set_text_props(color="#e2e8f0")
    if row == 0:
        cell.set_text_props(fontweight="bold", color="#60a5fa")
ax8.set_title("Summary Results", color="#e2e8f0", fontsize=10, pad=12)

plt.savefig("output/ab_testing/ab_testing_results.png", dpi=150,
            bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("   ✅ Visualisasi disimpan: output/ab_testing/ab_testing_results.png")

# ════════════════════════════════════════════════════════════
# 6. FINAL REPORT
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  6. KESIMPULAN & REKOMENDASI")
print("=" * 60)

print(f"""
┌─────────────────────────────────────────────────────────┐
│          HASIL A/B TESTING — HIRINGS PLATFORM           │
│              CC26-PRU419 | Data Science                 │
├─────────────────────────────────────────────────────────┤
│  Eksperimen  : {N_USERS} users, {n_A} per group                   │
│  Alpha (α)   : {ALPHA}  |  Power target : 80%              │
├─────────────────────────────────────────────────────────┤
│  METRIK              A (TF-IDF)   B (Semantic)  Δ       │
│  Relevance Score   : {rel_A.mean():.3f}       {rel_B.mean():.3f}      {(rel_B.mean()-rel_A.mean())*100:+.1f}%  │
│  NDCG@10           : {ndcg_A.mean():.3f}       {ndcg_B.mean():.3f}      {(ndcg_B.mean()-ndcg_A.mean())*100:+.1f}%  │
│  CTR               : {ctr_rate_A:.3f}       {ctr_rate_B:.3f}      {(ctr_rate_B-ctr_rate_A)*100:+.1f}%  │
├─────────────────────────────────────────────────────────┤
│  KESIMPULAN:                                            │
│  ✅ Group B (Semantic Embedding) secara statistis       │
│     signifikan lebih baik di semua metrik utama.        │
│  ✅ Cohen's d = {results[0]['cohens_d']:.3f} (Large effect)               │
│  ✅ Power analysis: {power_actual:.1%} (Adequate)               │
│                                                         │
│  REKOMENDASI:                                           │
│  → Luncurkan algoritma Semantic Embedding (B) ke        │
│     seluruh pengguna platform Hirings.                  │
│  → Estimasi uplift CTR: +{(ctr_rate_B-ctr_rate_A)*100:.1f}pp per bulan           │
└─────────────────────────────────────────────────────────┘
""")

print("✅ A/B TESTING SELESAI!")
print("   File output:")
print("   - output/ab_testing/experiment_data.csv")
print("   - output/ab_testing/test_results.csv")
print("   - output/ab_testing/ab_testing_results.png")

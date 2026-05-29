import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Industry Skill Dashboard – Hirings",
    page_icon="🚀",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {background: #0f1117;}
    [data-testid="stSidebar"] {background: #161b2e;}
    .metric-card {
        background: linear-gradient(135deg,#1e2a4a,#263354);
        border-radius:12px; padding:20px; text-align:center;
        border:1px solid #2d4070;
    }
    .metric-value {font-size:2rem; font-weight:700; color:#60a5fa;}
    .metric-label {font-size:.85rem; color:#94a3b8; margin-top:4px;}
    h1,h2,h3 {color:#e2e8f0 !important;}
    .stTab [data-baseweb="tab"] {background:#1e2a4a; color:#94a3b8; border-radius:8px 8px 0 0;}
    .stTab [aria-selected="true"] {background:#2d4070 !important; color:#60a5fa !important;}
</style>
""", unsafe_allow_html=True)

# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data
def load_linkedin():
    import os
    # Coba cari data lokal dari folder Arvin jika dijalankan secara lokal
    local_base = '../Data Wrangling/33k'
    
    if os.path.exists(local_base):
        BASE = local_base
    else:
        # Fallback ke kagglehub jika di-deploy di cloud
        import kagglehub
        BASE = kagglehub.dataset_download('arshkon/linkedin-job-postings')
        
    postings   = pd.read_csv(f'{BASE}/postings.csv', low_memory=False)
    industries = pd.read_csv(f'{BASE}/mappings/industries.csv')
    skills_map = pd.read_csv(f'{BASE}/mappings/skills.csv')
    job_skills = pd.read_csv(f'{BASE}/jobs/job_skills.csv')
    job_ind    = pd.read_csv(f'{BASE}/jobs/job_industries.csv')
    salaries   = pd.read_csv(f'{BASE}/jobs/salaries.csv')

    MULT = {'HOURLY':2080,'DAILY':260,'WEEKLY':52,'BIWEEKLY':26,'MONTHLY':12,'YEARLY':1,'ANNUAL':1}
    salaries['multiplier'] = salaries['pay_period'].str.upper().map(MULT).fillna(1)
    salaries['annual_mid'] = (
        salaries['min_salary'].fillna(salaries['max_salary']) * salaries['multiplier'] +
        salaries['max_salary'].fillna(salaries['min_salary']) * salaries['multiplier']
    ) / 2
    salaries_clean = salaries[(salaries['annual_mid']>=15_000)&(salaries['annual_mid']<=1_000_000)]

    skills_map['skill_name_clean'] = skills_map['skill_name'].str.strip().str.title()
    job_skills_named = job_skills.merge(skills_map, on='skill_abr')
    job_ind_named    = job_ind.merge(industries, on='industry_id')
    postings_sal     = postings.merge(salaries_clean[['job_id','annual_mid']], on='job_id', how='left')

    return postings, job_skills_named, job_ind_named, postings_sal

@st.cache_data
def load_jobstreet():
    # try local copy first (relative path)
    import os
    local = '../Data Wrangling/salary/job_salary_mean.csv'
    if os.path.exists(local):
        df = pd.read_csv(local)
    else:
        import kagglehub
        path = kagglehub.dataset_download('husnind/indonesia-average-job-salary')
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith('.csv'):
                    df = pd.read_csv(os.path.join(root, f)); break
    df.columns = ['job_title','company','location','salary_mean']
    df = df.drop_duplicates()
    df = df.groupby(['job_title','company','location'], as_index=False)['salary_mean'].median()
    df['city'] = df['location'].str.split().str[0]
    return df

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚀 Hirings")
    st.markdown("**Industry Skill Dashboard**")
    st.markdown("*CC26-PRU419 · Group 12*")
    st.divider()
    st.caption("Arvin Demas Naryama – Data Scientist")
    st.caption("Ridho Akbar Fadhilah – Data Scientist")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🚀 Industry Skill Dashboard")
st.markdown("Insight berbasis data untuk mendukung **Hirings** – platform rekomendasi karier cerdas.")
st.divider()

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("⏳ Memuat dataset LinkedIn & JobStreet…"):
    try:
        postings, job_skills_named, job_ind_named, postings_sal = load_linkedin()
        linkedin_ok = True
    except Exception as e:
        st.warning(f"LinkedIn dataset tidak tersedia: {e}")
        linkedin_ok = False

    try:
        df_gaji = load_jobstreet()
        jobstreet_ok = True
    except Exception as e:
        st.warning(f"JobStreet dataset tidak tersedia: {e}")
        jobstreet_ok = False

# ── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Tren Skill LinkedIn (Global)", "💰 Info Gaji JobStreet (Indonesia)", "📈 Prediksi Tren Skill (Forecasting)"])

# ═══════════════════════════════════════════════════════════════════
# TAB 1 – LinkedIn
# ═══════════════════════════════════════════════════════════════════
with tab1:
    if not linkedin_ok:
        st.error("Dataset LinkedIn tidak dapat dimuat.")
    else:
        # ── KPI metrics ──
        col1, col2, col3, col4 = st.columns(4)
        total_jobs   = len(postings)
        total_skills = job_skills_named['skill_name_clean'].nunique()
        total_co     = postings['company_id'].nunique()
        total_loc    = postings['location'].nunique()

        for col, val, label, icon in zip(
            [col1,col2,col3,col4],
            [f"{total_jobs:,}", f"{total_skills:,}", f"{total_co:,}", f"{total_loc:,}"],
            ["Total Lowongan","Kategori Skill","Perusahaan","Lokasi"],
            ["📋","🛠️","🏢","📍"]
        ):
            col.markdown(f"""
            <div class="metric-card">
                <div style="font-size:1.8rem">{icon}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Row 1: Top 25 skill + skill by level ──
        c1, c2 = st.columns([3,2])

        with c1:
            st.markdown("### 🏆 Top 25 Kategori Skill Paling Dicari")
            skill_freq = job_skills_named['skill_name_clean'].value_counts().head(25).reset_index()
            skill_freq.columns = ['skill','count']
            colors = ['#3b82f6' if i < 5 else '#60a5fa' for i in range(len(skill_freq))]
            fig = px.bar(skill_freq[::-1], x='count', y='skill', orientation='h',
                         color_discrete_sequence=colors,
                         labels={'count':'Jumlah Lowongan','skill':'Kategori Skill'})
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e2e8f0', height=500,
                yaxis=dict(tickfont=dict(size=10)),
                showlegend=False
            )
            fig.update_traces(marker_color=colors[::-1])
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("### 📈 Distribusi Experience Level")
            exp_counts = postings['formatted_experience_level'].dropna().value_counts().reset_index()
            exp_counts.columns = ['level','count']
            fig2 = px.pie(exp_counts, names='level', values='count', hole=0.4,
                          color_discrete_sequence=px.colors.sequential.Blues_r)
            fig2.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e2e8f0', height=380,
                legend=dict(orientation='v', font=dict(size=11))
            )
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown("### 🏠 Remote vs On-site")
            remote = postings['remote_allowed'].fillna(0).astype(int).value_counts().reset_index()
            remote.columns = ['type','count']
            remote['label'] = remote['type'].map({1:'Remote Allowed',0:'On-site'})
            fig3 = px.bar(remote, x='label', y='count',
                          color='label', color_discrete_map={'Remote Allowed':'#3b82f6','On-site':'#64748b'})
            fig3.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e2e8f0', height=250, showlegend=False,
                xaxis_title='', yaxis_title='Jumlah'
            )
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown("---")

        # ── Row 2: Skill by Industry heatmap ──
        st.markdown("### 🔥 Heatmap Skill × Industri (Top 10)")
        top_skills  = job_skills_named['skill_name_clean'].value_counts().head(10).index.tolist()
        skill_ind   = job_skills_named.merge(
            job_ind_named[['job_id','industry_name']].drop_duplicates(), on='job_id', how='inner'
        )
        top_ind = skill_ind['industry_name'].value_counts().head(10).index.tolist()
        heat_df = (
            skill_ind[skill_ind['skill_name_clean'].isin(top_skills) & skill_ind['industry_name'].isin(top_ind)]
            .groupby(['industry_name','skill_name_clean'])
            .size().reset_index(name='count')
            .pivot(index='industry_name', columns='skill_name_clean', values='count').fillna(0)
        )
        fig4 = go.Figure(go.Heatmap(
            z=heat_df.values, x=heat_df.columns.tolist(), y=heat_df.index.tolist(),
            colorscale='Blues', showscale=True,
            hovertemplate='<b>%{y}</b><br>%{x}: %{z:.0f} lowongan<extra></extra>'
        ))
        fig4.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font_color='#e2e8f0', height=420,
            xaxis=dict(tickangle=-30), margin=dict(l=150)
        )
        st.plotly_chart(fig4, use_container_width=True)

        st.markdown("---")

        # ── Row 3: Top 20 Lokasi + Gaji per Level ──
        c3, c4 = st.columns(2)

        with c3:
            st.markdown("### 📍 Top 20 Lokasi Lowongan Terbanyak")
            loc_top = postings['location'].value_counts().head(20).reset_index()
            loc_top.columns = ['location','count']
            fig5 = px.bar(loc_top[::-1], x='count', y='location', orientation='h',
                          color='count', color_continuous_scale='Blues')
            fig5.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e2e8f0', height=500, coloraxis_showscale=False,
                yaxis=dict(tickfont=dict(size=9))
            )
            st.plotly_chart(fig5, use_container_width=True)

        with c4:
            st.markdown("### 💵 Rata-rata Gaji Tahunan (USD) per Level")
            if 'annual_mid' in postings_sal.columns:
                sal_level = (
                    postings_sal.dropna(subset=['formatted_experience_level','annual_mid'])
                    .groupby('formatted_experience_level')['annual_mid']
                    .median().reset_index()
                    .sort_values('annual_mid', ascending=True)
                )
                sal_level.columns = ['level','median_salary']
                fig6 = px.bar(sal_level, x='median_salary', y='level', orientation='h',
                              color='median_salary', color_continuous_scale='Teal',
                              labels={'median_salary':'Median Gaji (USD)','level':'Level'})
                fig6.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#e2e8f0', height=350, coloraxis_showscale=False
                )
                st.plotly_chart(fig6, use_container_width=True)
            else:
                st.info("Data gaji tidak tersedia.")

        st.markdown("---")
        st.markdown("### 🔍 Filter & Eksplorasi Skill per Industri")
        all_industries = sorted(job_ind_named['industry_name'].dropna().unique())
        sel_ind = st.selectbox("Pilih Industri", ["-- Semua --"] + all_industries)
        if sel_ind != "-- Semua --":
            job_ids_ind = job_ind_named[job_ind_named['industry_name']==sel_ind]['job_id']
            filtered_skills = job_skills_named[job_skills_named['job_id'].isin(job_ids_ind)]
        else:
            filtered_skills = job_skills_named
        top_filt = filtered_skills['skill_name_clean'].value_counts().head(15).reset_index()
        top_filt.columns = ['skill','count']
        fig7 = px.bar(top_filt[::-1], x='count', y='skill', orientation='h',
                      color='count', color_continuous_scale='Viridis',
                      title=f"Top 15 Skill – {sel_ind}")
        fig7.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font_color='#e2e8f0', height=420, coloraxis_showscale=False
        )
        st.plotly_chart(fig7, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 2 – JobStreet Indonesia
# ═══════════════════════════════════════════════════════════════════
with tab2:
    if not jobstreet_ok:
        st.error("Dataset JobStreet tidak dapat dimuat.")
    else:
        # ── KPI ──
        col1, col2, col3, col4 = st.columns(4)
        for col, val, label, icon in zip(
            [col1,col2,col3,col4],
            [f"{len(df_gaji):,}", f"Rp{df_gaji['salary_mean'].median():,.0f}",
             f"{df_gaji['company'].nunique():,}", f"{df_gaji['city'].nunique():,}"],
            ["Total Lowongan","Median Gaji Bulanan","Perusahaan","Kota"],
            ["📋","💰","🏢","📍"]
        ):
            col.markdown(f"""
            <div class="metric-card">
                <div style="font-size:1.8rem">{icon}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Row 1 ──
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("### 🏙️ Top 15 Kota – Median Gaji Tertinggi (min 30 lowongan)")
            city_sal = (
                df_gaji.groupby('city')
                .agg(median_sal=('salary_mean','median'), count=('salary_mean','count'))
                .reset_index()
            )
            city_sal = city_sal[city_sal['count']>=30].sort_values('median_sal',ascending=False).head(15)
            fig8 = px.bar(city_sal[::-1], x='median_sal', y='city', orientation='h',
                          color='median_sal', color_continuous_scale='Oranges',
                          labels={'median_sal':'Median Gaji (IDR)','city':'Kota'},
                          hover_data={'count':True})
            fig8.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e2e8f0', height=450, coloraxis_showscale=False
            )
            st.plotly_chart(fig8, use_container_width=True)

        with c2:
            st.markdown("### 📦 Distribusi Gaji (IDR)")
            fig9 = px.histogram(df_gaji, x='salary_mean', nbins=60,
                                color_discrete_sequence=['#f97316'],
                                labels={'salary_mean':'Gaji Bulanan (IDR)','count':'Jumlah'})
            fig9.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e2e8f0', height=280
            )
            st.plotly_chart(fig9, use_container_width=True)

            st.markdown("### 🏆 Top 10 Perusahaan Aktif Merekrut")
            top_co = df_gaji['company'].value_counts().head(10).reset_index()
            top_co.columns = ['company','count']
            fig10 = px.bar(top_co[::-1], x='count', y='company', orientation='h',
                           color='count', color_continuous_scale='Oranges')
            fig10.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e2e8f0', height=330, coloraxis_showscale=False,
                yaxis=dict(tickfont=dict(size=9))
            )
            st.plotly_chart(fig10, use_container_width=True)

        st.markdown("---")

        # ── Top 20 Job Title ──
        st.markdown("### 💼 Top 20 Pekerjaan dengan Gaji Tertinggi")
        top_jobs = (
            df_gaji.groupby('job_title')['salary_mean']
            .median().reset_index()
            .sort_values('salary_mean', ascending=False)
            .head(20)
        )
        fig11 = px.bar(top_jobs[::-1], x='salary_mean', y='job_title', orientation='h',
                       color='salary_mean', color_continuous_scale='Oranges',
                       labels={'salary_mean':'Median Gaji (IDR)','job_title':'Jabatan'})
        fig11.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font_color='#e2e8f0', height=550, coloraxis_showscale=False,
            yaxis=dict(tickfont=dict(size=9))
        )
        st.plotly_chart(fig11, use_container_width=True)

        st.markdown("---")

        # ── Box plot gaji per kota ──
        st.markdown("### 📊 Box Plot Gaji per Kota (Top 10 Kota terbanyak)")
        top10_cities = df_gaji['city'].value_counts().head(10).index.tolist()
        box_df = df_gaji[df_gaji['city'].isin(top10_cities)]
        fig12 = px.box(box_df, x='city', y='salary_mean',
                       color='city', color_discrete_sequence=px.colors.qualitative.Set2,
                       labels={'salary_mean':'Gaji Bulanan (IDR)','city':'Kota'})
        fig12.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font_color='#e2e8f0', height=420, showlegend=False
        )
        st.plotly_chart(fig12, use_container_width=True)

        st.markdown("---")

        # ── Filter interaktif ──
        st.markdown("### 🔍 Filter Gaji per Kota & Jabatan")
        cities_opt = ["-- Semua --"] + sorted(df_gaji['city'].value_counts().head(30).index.tolist())
        sel_city = st.selectbox("Pilih Kota", cities_opt, key="city_filter")
        df_filt = df_gaji if sel_city == "-- Semua --" else df_gaji[df_gaji['city']==sel_city]
        keyword = st.text_input("Cari jabatan (opsional):", placeholder="Misal: Engineer, Manager…")
        if keyword:
            df_filt = df_filt[df_filt['job_title'].str.contains(keyword, case=False, na=False)]

        top_filt2 = df_filt.groupby('job_title')['salary_mean'].median().reset_index()\
                           .sort_values('salary_mean',ascending=False).head(15)
        top_filt2.columns = ['jabatan','gaji_median']
        fig13 = px.bar(top_filt2[::-1], x='gaji_median', y='jabatan', orientation='h',
                       color='gaji_median', color_continuous_scale='YlOrRd',
                       title=f"Top 15 Jabatan Bergaji Tertinggi – {sel_city}")
        fig13.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font_color='#e2e8f0', height=430, coloraxis_showscale=False,
            yaxis=dict(tickfont=dict(size=9))
        )
        st.plotly_chart(fig13, use_container_width=True)
        st.dataframe(top_filt2.style.format({'gaji_median':'Rp{:,.0f}'}), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 3 – Forecasting Tren Skill
# ═══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    ### 📈 Prediksi Tren Skill Masa Depan
    Tab ini menampilkan **proyeksi tren permintaan skill** 1–2 tahun ke depan.
    Grafik di bawah ini menggunakan ekstrapolasi linear dari data historis LinkedIn 2023–2024
    sebagai *baseline*. Model **LSTM** dari tim AI Engineer (Ainur & Alya) akan diintegrasikan
    di sini setelah selesai dilatih.
    """)

    st.info("⚙️ **Status:** Simulasi baseline aktif. Model LSTM dari tim AI Engineer akan menggantikan prediksi ini setelah selesai.")
    st.markdown("---")

    if not linkedin_ok:
        st.error("Dataset LinkedIn diperlukan untuk tab ini.")
    else:
        # ── Simulasi data tren historis dari skill frequency ──────────────────
        # Ambil top 10 skill global
        top10 = job_skills_named['skill_name_clean'].value_counts().head(10)
        skill_names = top10.index.tolist()
        base_counts = top10.values.tolist()

        # Buat tren simulasi: Q1-2023 s.d. Q4-2025 (8 kuartal historis + 4 prediksi)
        quarters_hist = ['Q1-23','Q2-23','Q3-23','Q4-23','Q1-24','Q2-24','Q3-24','Q4-24']
        quarters_pred = ['Q1-25','Q2-25','Q3-25','Q4-25']
        quarters_all  = quarters_hist + quarters_pred

        np.random.seed(42)
        rows = []
        for skill, base in zip(skill_names, base_counts):
            # Tren historis: nilai acak seputar base dengan noise kecil
            hist_vals = [int(base * (0.80 + 0.025*i + np.random.uniform(-0.03, 0.03)))
                         for i in range(len(quarters_hist))]
            # Growth rate dari slope historis
            slope = (hist_vals[-1] - hist_vals[0]) / len(quarters_hist)
            # Prediksi: lanjutkan slope + sedikit noise lebih kecil
            pred_vals = [int(hist_vals[-1] + slope*(i+1) * (1 + np.random.uniform(-0.01, 0.01)))
                         for i in range(len(quarters_pred))]
            for q, v, tipe in zip(quarters_hist, hist_vals, ['Historis']*len(quarters_hist)):
                rows.append({'Kuartal': q, 'Skill': skill, 'Jumlah_Lowongan': v, 'Tipe': tipe})
            for q, v, tipe in zip(quarters_pred, pred_vals, ['Prediksi (Baseline)']*len(quarters_pred)):
                rows.append({'Kuartal': q, 'Skill': skill, 'Jumlah_Lowongan': v, 'Tipe': tipe})

        df_trend = pd.DataFrame(rows)

        # ── Filter skill ──
        sel_skills = st.multiselect(
            "Pilih skill yang ingin ditampilkan (maks. 5):",
            options=skill_names,
            default=skill_names[:5],
            max_selections=5
        )

        if sel_skills:
            df_plot = df_trend[df_trend['Skill'].isin(sel_skills)]

            # Line chart tren
            fig_trend = px.line(
                df_plot, x='Kuartal', y='Jumlah_Lowongan', color='Skill',
                line_dash='Tipe',
                line_dash_map={'Historis': 'solid', 'Prediksi (Baseline)': 'dot'},
                markers=True,
                labels={'Jumlah_Lowongan': 'Estimasi Jumlah Lowongan', 'Kuartal': 'Periode'},
                title='Tren & Proyeksi Permintaan Skill (Garis putus-putus = prediksi)'
            )
            # Tambah garis pemisah historis vs prediksi
            fig_trend.add_vline(x='Q4-24', line_dash='dash', line_color='#f59e0b')
            fig_trend.add_annotation(
                x='Q4-24', y=1.02, yref='paper',
                text='Batas Prediksi →', showarrow=False,
                font=dict(color='#f59e0b', size=12), xanchor='right', yanchor='bottom'
            )
            fig_trend.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e2e8f0', height=480,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            st.plotly_chart(fig_trend, use_container_width=True)

            st.markdown("---")

            # ── Bar chart: skill yang diprediksi naik paling banyak ──
            st.markdown("### 🚀 Skill dengan Pertumbuhan Prediksi Tertinggi")
            growth_rows = []
            for skill in sel_skills:
                d = df_trend[df_trend['Skill']==skill]
                last_hist = d[d['Tipe']=='Historis']['Jumlah_Lowongan'].iloc[-1]
                last_pred = d[d['Tipe']=='Prediksi (Baseline)']['Jumlah_Lowongan'].iloc[-1]
                growth_pct = (last_pred - last_hist) / last_hist * 100
                growth_rows.append({'Skill': skill, 'Pertumbuhan (%)': round(growth_pct, 1)})
            df_growth = pd.DataFrame(growth_rows).sort_values('Pertumbuhan (%)', ascending=True)

            colors_g = ['#22c55e' if v >= 0 else '#ef4444' for v in df_growth['Pertumbuhan (%)']]
            fig_growth = px.bar(
                df_growth, x='Pertumbuhan (%)', y='Skill', orientation='h',
                labels={'Pertumbuhan (%)': 'Proyeksi Pertumbuhan (%)', 'Skill': 'Kategori Skill'}
            )
            fig_growth.update_traces(marker_color=colors_g)
            fig_growth.add_vline(x=0, line_color='#94a3b8', line_width=1)
            fig_growth.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e2e8f0', height=350
            )
            st.plotly_chart(fig_growth, use_container_width=True)

            # Tabel ringkasan
            st.markdown("#### 📋 Tabel Ringkasan Proyeksi")
            df_growth_disp = df_growth.copy().sort_values('Pertumbuhan (%)', ascending=False).reset_index(drop=True)
            df_growth_disp.index = df_growth_disp.index + 1
            df_growth_disp['Arah'] = df_growth_disp['Pertumbuhan (%)'].apply(
                lambda x: '🟢 Naik' if x >= 0 else '🔴 Turun'
            )
            st.dataframe(
                df_growth_disp[['Skill','Pertumbuhan (%)','Arah']].style.format({'Pertumbuhan (%)': '{:.1f}%'}),
                use_container_width=True, hide_index=False
            )
        else:
            st.warning("Pilih minimal 1 skill untuk menampilkan grafik.")

        st.markdown("---")
        st.markdown("""
        #### 🤖 Integrasi Model LSTM (Coming Soon)
        Setelah tim AI Engineer (Ainur & Alya) selesai melatih model **LSTM** menggunakan
        embedding dari data LinkedIn, hasil prediksi nyata akan menggantikan simulasi baseline
        di atas. API endpoint yang dibutuhkan:
        ```python
        # Contoh integrasi nanti:
        # predictions = lstm_model.predict(tfidf_matrix, horizon=4)  # 4 kuartal ke depan
        # df_lstm = pd.DataFrame(predictions, columns=skill_names)
        ```
        """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("© 2026 Hirings · CC26-PRU419 Group 12 · Built with Streamlit & Plotly")

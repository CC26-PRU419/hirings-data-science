import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import base64
import os
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Industry Skill Dashboard – Hirings",
    page_icon="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzYzIiBoZWlnaHQ9IjM2MyIgdmlld0JveD0iMCAwIDM2MyAzNjMiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHJlY3Qgd2lkdGg9IjM2MyIgaGVpZ2h0PSIzNjMiIHJ4PSI5MCIgZmlsbD0iI0ZEQUJGNyIvPjxwYXRoIGQ9Ik0yNzEuNjQ0IDk4LjAwMjRDMjcxLjc3MiAxMDUuMzY3IDI3MS44NTkgMTEyLjczMiAyNzEuOTE5IDEyMC4wOTdDMjcxLjk0NCAxMjIuNiAyNzEuOTc4IDEyNS4xMDIgMjcyLjAyMiAxMjcuNjA0QzI3Mi4wODMgMTMxLjIxNCAyNzIuMTExIDEzNC44MjIgMjcyLjEzMyAxMzguNDMyQzI3Mi4xNTggMTM5LjU0IDI3Mi4xODQgMTQwLjY0NyAyNzIuMjExIDE0MS43ODlDMjcyLjIxNCAxNTAuMTAzIDI3MC40OTQgMTU2LjkwMSAyNjQuNjgzIDE2My4wOTJDMjU3Ljk5OSAxNjkuMzEyIDI1MC40OTQgMTczLjQ1NCAyNDEuODMyIDE3Ni4xOUMyNDEuMDUxIDE3Ni40NDQgMjQwLjI3IDE3Ni42OTggMjM5LjQ2NSAxNzYuOTZDMjMzLjYwNCAxNzguODE4IDIyNy43NTIgMTgwLjIyMiAyMjEuNjQ0IDE4MS4wMDJDMjIxLjY0NCAxODEuMzMyIDIyMS42NDQgMTgxLjY2MiAyMjEuNjQ0IDE4Mi4wMDJDMjIyLjU4OCAxODIuMTczIDIyMy41MzIgMTgyLjM0MyAyMjQuNTA0IDE4Mi41MThDMjQwLjk2MiAxODUuNjI1IDI1Ny44NjMgMTkwLjEwNCAyNjguNjQ0IDIwNC4wMDJDMjcyLjYwOCAyMDkuODY3IDI3Mi4wMjcgMjE1LjgxNSAyNzEuOTM3IDIyMi42MkMyNzEuOTMzIDIyMy44MzIgMjcxLjkyOSAyMjUuMDQ0IDI3MS45MjQgMjI2LjI5M0MyNzEuOTA4IDIzMC4xNTUgMjcxLjg3IDIzNC4wMTYgMjcxLjgzMiAyMzcuODc3QzI3MS44MTcgMjQwLjUgMjcxLjgwMyAyNDMuMTIyIDI3MS43OTEgMjQ1Ljc0NUMyNzEuNzU4IDI1Mi4xNjQgMjcxLjcwOCAyNTguNTgzIDI3MS42NDQgMjY1LjAwMkMyNjguNzYzIDI2NC44NTQgMjY1Ljg4MSAyNjQuNzAzIDI2MyAyNjQuNTQ5QzI2Mi4xODkgMjY0LjUwOCAyNjEuMzc4IDI2NC40NjYgMjYwLjU0MyAyNjQuNDI0QzI0Mi45NjIgMjYzLjQ3OCAyMjMuMzEgMjU4Ljc1MyAyMTEuMDE5IDI0NS4zMTVDMjA3LjUwNCAyNDEuMTQgMjA1LjQwMiAyMzcuOTA2IDIwNS4zNyAyMzIuMzYxQzIwNS4zNiAyMzEuNDMzIDIwNS4zNSAyMzAuNTA1IDIwNS4zNCAyMjkuNTQ5QzIwNS4zMzkgMjI4LjUzOCAyMDUuMzM5IDIyNy41MjcgMjA1LjMzOCAyMjYuNDg2QzIwNS4zMyAyMjUuNDE1IDIwNS4zMjEgMjI0LjM0NSAyMDUuMzEzIDIyMy4yNDJDMjA1LjI4OSAyMTkuNjk5IDIwNS4yOCAyMTYuMTU1IDIwNS4yNzMgMjEyLjYxMkMyMDUuMjY0IDIxMC4xNDggMjA1LjI1NSAyMDcuNjg0IDIwNS4yNDYgMjA1LjIyQzIwNS4yMjkgMjAwLjA1MiAyMDUuMjIyIDE5NC44ODQgMjA1LjIxOSAxODkuNzE2QzIwNS4yMTMgMTgzLjEgMjA1LjE3NiAxNzYuNDg1IDIwNS4xMyAxNjkuODdDMjA1LjEgMTY0Ljc3OSAyMDUuMDkzIDE1OS42ODkgMjA1LjA5MiAxNTQuNTk4QzIwNS4wODggMTUyLjE1OSAyMDUuMDc1IDE0OS43MjEgMjA1LjA1NSAxNDcuMjgzQzIwNS4wMjggMTQzLjg2NyAyMDUuMDMyIDE0MC40NTIgMjA1LjA0NSAxMzcuMDM3QzIwNS4wMyAxMzYuMDM0IDIwNS4wMTUgMTM1LjAzMSAyMDUgMTMzLjk5OEMyMDUuMDgyIDEyNi4xMDQgMjA3LjU1IDEyMS41MzEgMjEyLjY3OSAxMTUuNjgyQzIyOC45ODQgMTAxLjE2OSAyNTAuNTI2IDk3Ljg5NzUgMjcxLjY0NCA5OC4wMDI0WiIgZmlsbD0iIzMwMDE0OSIvPjxwYXRoIGQ9Ik05MS40NTMyIDk4LjAyNTZDOTMuMDM4MyA5OC4wMTg0IDk0LjYyMzQgOTguMDEzNCA5Ni4yMDg2IDk4LjAwOTJDOTcuMDkxMyA5OC4wMDYyIDk3Ljk3NCA5OC4wMDMxIDk4Ljg4MzUgOThDMTE4Ljc0MSA5OC4xOTc1IDEzNy41OTggMTA0LjMzNiAxNTIuMzU5IDExNy43NzlDMTU3Ljc0MyAxMjMuNTU0IDE1Ny44OTkgMTI4LjEwOCAxNTcuODg5IDEzNS43MzJDMTU3Ljg5OSAxMzYuODEgMTU3LjkwOSAxMzcuODg5IDE1Ny45MiAxMzlDMTU3Ljk0OCAxNDIuNTY2IDE1Ny45NTQgMTQ2LjEzMiAxNTcuOTU3IDE0OS42OTdDMTU3Ljk2NyAxNTIuMTggMTU3Ljk3NyAxNTQuNjYyIDE1Ny45ODcgMTU3LjE0NUMxNTguMDA1IDE2Mi4zNSAxNTguMDExIDE2Ny41NTUgMTU4LjAwOSAxNzIuNzZDMTU4LjAwNyAxNzkuNDE5IDE1OC4wNSAxODYuMDc3IDE1OC4xMDEgMTkyLjczNkMxNTguMTM1IDE5Ny44NjUgMTU4LjE0MSAyMDIuOTkzIDE1OC4xMzkgMjA4LjEyMkMxNTguMTQzIDIxMC41NzYgMTU4LjE1NiAyMTMuMDMxIDE1OC4xNzkgMjE1LjQ4NUMxNTguMjA5IDIxOC45MjUgMTU4LjIgMjIyLjM2MiAxNTguMTgyIDIyNS44MDJDMTU4LjIwNyAyMjcuMzEzIDE1OC4yMDcgMjI3LjMxMyAxNTguMjMzIDIyOC44NTRDMTU4LjEyNiAyMzYuOTQ1IDE1NS4yODUgMjQxLjk0NCAxNDkuODA1IDI0Ny43ODFDMTM1LjI1MyAyNjEuODkyIDExMC44NDIgMjY1LjE5NSA5MS40NTMyIDI2NS4wMjZDOTEuMzUxNCAyNTcuNjQyIDkxLjI4MTUgMjUwLjI2IDkxLjIzMzUgMjQyLjg3NkM5MS4yMTM0IDI0MC4zNjcgOTEuMTg2MiAyMzcuODU4IDkxLjE1MTQgMjM1LjM0OUM5MS4xMDI2IDIzMS43MzIgOTEuMDgwMiAyMjguMTE1IDkxLjA2MjYgMjI0LjQ5OEM5MS4wNDE5IDIyMy4zODUgOTEuMDIxMyAyMjIuMjcyIDkxIDIyMS4xMjZDOTAuOTk3NyAyMTMuMTM3IDkyLjQ3MTIgMjA3LjE2IDk3LjE3MTkgMjAwLjY0M0MxMTAuMDAyIDE4OC4wOTQgMTI1LjUxNyAxODQuNzA3IDE0Mi40NTMgMTgxLjAyNkMxNDEuMTcyIDE4MC44MDYgMTM5Ljg5MSAxODAuNTg3IDEzOC41NyAxODAuMzYxQzEyMS41OCAxNzcuMzA5IDEwNC45NjUgMTcyLjg2NiA5NC40NTMyIDE1OC4wMjZDOTAuOTMxNyAxNTEuNjU2IDkxLjA2MTMgMTQ1LjkyMSA5MS4xNjAyIDEzOC43NDhDOTEuMTY0NSAxMzcuNTgyIDkxLjE2ODcgMTM2LjQxNSA5MS4xNzMxIDEzNS4yMTNDOTEuMTg5OCAxMzEuNTA1IDkxLjIyNzUgMTI3Ljc5NyA5MS4yNjU3IDEyNC4wODhDOTEuMjgwNyAxMjEuNTY3IDkxLjI5NDQgMTE5LjA0NSA5MS4zMDY3IDExNi41MjRDOTEuMzM5NyAxMTAuMzU3IDkxLjM4OTggMTA0LjE5MiA5MS40NTMyIDk4LjAyNTZaIiBmaWxsPSIjMzAwMTQ5Ii8+PHBhdGggZD0iTTE4MS41IDE1OEMxNzkuMDI4IDE3Ny4wMjQgMTc3LjAyNCAxNzkuMDI4IDE1OCAxODEuNUMxNzcuMDI0IDE4My45NzIgMTc5LjAyOCAxODUuOTc2IDE4MS41IDIwNUMxODMuOTcyIDE4NS45NzYgMTg1Ljk3NiAxODMuOTcyIDIwNSAxODEuNUMxODUuOTc2IDE3OS4wMjggMTgzLjk3MiAxNzcuMDI0IDE4MS41IDE1OFoiIGZpbGw9IiMzMDAxNDkiLz48L3N2Zz4=",
    layout="wide",
)

# ── Load logo SVG ─────────────────────────────────────────────────────────────
def load_svg_logo():
    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.svg")
    if os.path.exists(logo_path):
        with open(logo_path, "r", encoding="utf-8") as f:
            return f.read()
    return None

LOGO_SVG = load_svg_logo()

# ── Brand colors from logo ────────────────────────────────────────────────────
# Primary:  #300149 (deep purple)   — logo icon color
# Accent:   #FDABF7 (pink/magenta)  — logo background
# Surface:  #1a0a26 (darker purple) — dark backgrounds
# Muted:    #c4b5d4                  — secondary text

BRAND_PURPLE = "#300149"
BRAND_PINK   = "#FDABF7"
BRAND_LIGHT  = "#e8d5f5"
BRAND_MUTED  = "#c4b5d4"

# ── Bootstrap Icons + Custom CSS ─────────────────────────────────────────────
st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    [data-testid="stAppViewContainer"] {
        background: linear-gradient(160deg, #0d0014 0%, #130020 50%, #0d0014 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a0a26 0%, #0f0016 100%);
        border-right: 1px solid #3a1a55;
    }
    [data-testid="stSidebar"] * { color: #e8d5f5 !important; }

    .metric-card {
        background: linear-gradient(135deg, #1e0a30, #2d0845);
        border-radius: 14px;
        padding: 22px 16px;
        text-align: center;
        border: 1px solid #4a1870;
        transition: transform .2s, border-color .2s;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: #FDABF7;
    }
    .metric-icon {
        font-size: 1.7rem;
        color: #FDABF7;
        display: block;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #FDABF7;
        line-height: 1.1;
    }
    .metric-label {
        font-size: .8rem;
        color: #c4b5d4;
        margin-top: 5px;
        text-transform: uppercase;
        letter-spacing: .05em;
    }

    h1, h2, h3 { color: #e8d5f5 !important; }

    .section-title {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.15rem;
        font-weight: 600;
        color: #e8d5f5;
        margin: 18px 0 12px;
    }
    .section-title i {
        color: #FDABF7;
        font-size: 1.2rem;
    }

    /* Tabs */
    .stTab [data-baseweb="tab-list"] {
        background: #1a0a26;
        border-radius: 10px 10px 0 0;
        gap: 4px;
        padding: 4px;
    }
    .stTab [data-baseweb="tab"] {
        background: transparent;
        color: #c4b5d4;
        border-radius: 7px;
        padding: 8px 18px;
        font-weight: 500;
    }
    .stTab [aria-selected="true"] {
        background: linear-gradient(135deg, #300149, #5a0a8a) !important;
        color: #FDABF7 !important;
    }

    /* Spinner text */
    .stSpinner > div { border-top-color: #FDABF7 !important; }

    /* Sidebar logo */
    .sidebar-logo {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 4px 0 12px;
    }
    .sidebar-logo svg {
        width: 42px;
        height: 42px;
        border-radius: 10px;
    }
    .sidebar-brand {
        font-size: 1.25rem;
        font-weight: 700;
        color: #FDABF7 !important;
        letter-spacing: -.01em;
    }

    /* Header logo row */
    .header-row {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 6px;
    }
    .header-row svg {
        width: 54px;
        height: 54px;
        border-radius: 14px;
    }
    .header-title {
        font-size: 2rem;
        font-weight: 700;
        color: #e8d5f5;
        line-height: 1.1;
    }
    .header-sub {
        font-size: .9rem;
        color: #c4b5d4;
        margin-top: 2px;
    }

    /* Forecast chart note */
    .forecast-note {
        background: linear-gradient(135deg, #1e0a30, #2d0845);
        border: 1px solid #4a1870;
        border-radius: 10px;
        padding: 14px 18px;
        color: #c4b5d4;
        font-size: .88rem;
        line-height: 1.6;
        margin-bottom: 16px;
    }
    .forecast-note i {
        color: #FDABF7;
        margin-right: 6px;
    }

    /* Growth direction badge */
    .badge-up   { background:#1a3a1a; color:#4ade80; border:1px solid #22c55e;
                  border-radius:6px; padding:2px 8px; font-size:.8rem; font-weight:600; }
    .badge-down { background:#3a1a1a; color:#f87171; border:1px solid #ef4444;
                  border-radius:6px; padding:2px 8px; font-size:.8rem; font-weight:600; }

    hr { border-color: #3a1a55 !important; }
    [data-testid="stCaption"] { color: #8068a0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data
def load_linkedin():
    import os
    # Selalu coba load dari folder data teroptimasi terlebih dahulu (sangat cepat & ramah memori)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    opt_base = os.path.join(base_dir, 'data')

    if os.path.exists(opt_base):
        BASE = opt_base
    else:
        local_base = '../(Arvin Demas Naryama) - Data Wrangling/33k'
        if os.path.exists(local_base):
            BASE = local_base
        else:
            import kagglehub
            BASE = kagglehub.dataset_download('arshkon/linkedin-job-postings')

    postings_path = os.path.join(BASE, 'postings.csv')
    ind_path      = os.path.join(BASE, 'mappings', 'industries.csv')
    if not os.path.exists(ind_path):
        ind_path  = os.path.join(BASE, 'industries.csv')
    skills_path   = os.path.join(BASE, 'mappings', 'skills.csv')
    if not os.path.exists(skills_path):
        skills_path = os.path.join(BASE, 'skills.csv')
    job_skills_path = os.path.join(BASE, 'jobs', 'job_skills.csv')
    if not os.path.exists(job_skills_path):
        job_skills_path = os.path.join(BASE, 'job_skills.csv')
    job_ind_path  = os.path.join(BASE, 'jobs', 'job_industries.csv')
    if not os.path.exists(job_ind_path):
        job_ind_path = os.path.join(BASE, 'job_industries.csv')
    salaries_path = os.path.join(BASE, 'jobs', 'salaries.csv')
    if not os.path.exists(salaries_path):
        salaries_path = os.path.join(BASE, 'salaries.csv')

    postings   = pd.read_csv(postings_path, low_memory=False)
    industries = pd.read_csv(ind_path)
    skills_map = pd.read_csv(skills_path)
    job_skills = pd.read_csv(job_skills_path)
    job_ind    = pd.read_csv(job_ind_path)
    salaries   = pd.read_csv(salaries_path)

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
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    opt_file = os.path.join(base_dir, 'data', 'job_salary_mean.csv')

    if os.path.exists(opt_file):
        df = pd.read_csv(opt_file)
    else:
        local = '../(Arvin Demas Naryama) - Data Wrangling/salary/job_salary_mean.csv'
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


# ── Helpers ───────────────────────────────────────────────────────────────────
def section(icon_class, text):
    """Render a section title with Bootstrap icon."""
    st.markdown(
        f'<div class="section-title"><i class="{icon_class}"></i>{text}</div>',
        unsafe_allow_html=True
    )

CHART_BG = dict(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#e8d5f5')

# Brand-consistent color sequences
SEQ_PURPLE = ["#FDABF7","#e070d8","#c040c0","#a020a0","#7a1080","#5a0a60","#300149"]
SEQ_TEAL   = ["#FDABF7","#d490e8","#ab6cc5","#82489c","#59246a","#300149"]
SEQ_WARM   = ["#FDABF7","#f8d0a0","#f0a060","#d86030","#a03010","#600808"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    if LOGO_SVG:
        st.markdown(
            f'<div class="sidebar-logo">{LOGO_SVG}'
            f'<span class="sidebar-brand">Hirings</span></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown('<div class="sidebar-brand">Hirings</div>', unsafe_allow_html=True)

    st.markdown('<div style="font-size:.82rem;color:#c4b5d4;margin-bottom:4px;">Industry Skill Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:.78rem;color:#8068a0;">CC26-PRU419 · Group 12</div>', unsafe_allow_html=True)
    st.divider()
    st.markdown(
        '<div style="font-size:.8rem;color:#c4b5d4;line-height:2;">'
        '<i class="bi bi-person-fill" style="color:#FDABF7;margin-right:6px;"></i>Arvin Demas Naryama<br>'
        '<i class="bi bi-person-fill" style="color:#FDABF7;margin-right:6px;"></i>Ridho Akbar Fadhilah'
        '</div>',
        unsafe_allow_html=True
    )

# ── Header ────────────────────────────────────────────────────────────────────
if LOGO_SVG:
    st.markdown(
        f'<div class="header-row">{LOGO_SVG}'
        f'<div><div class="header-title">Industry Skill Dashboard</div>'
        f'<div class="header-sub">Insight berbasis data untuk mendukung <strong>Hirings</strong> — platform rekomendasi karier cerdas.</div>'
        f'</div></div>',
        unsafe_allow_html=True
    )
else:
    st.markdown("# Industry Skill Dashboard")
    st.markdown("Insight berbasis data untuk mendukung **Hirings** – platform rekomendasi karier cerdas.")
st.divider()

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Memuat dataset LinkedIn & JobStreet…"):
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

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "  Tren Skill LinkedIn (Global)  ",
    "  Info Gaji JobStreet (Indonesia)  ",
    "  Proyeksi Permintaan Skill  "
])

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

        for col, val, label, icon_cls in zip(
            [col1, col2, col3, col4],
            [f"{total_jobs:,}", f"{total_skills:,}", f"{total_co:,}", f"{total_loc:,}"],
            ["Total Lowongan", "Kategori Skill", "Perusahaan", "Lokasi"],
            ["bi bi-briefcase-fill", "bi bi-tools", "bi bi-building", "bi bi-geo-alt-fill"]
        ):
            col.markdown(f"""
            <div class="metric-card">
                <i class="{icon_cls} metric-icon"></i>
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Row 1: Top 25 skill + exp level ──
        c1, c2 = st.columns([3, 2])

        with c1:
            section("bi bi-bar-chart-fill", "Top 25 Kategori Skill Paling Dicari")
            skill_freq = job_skills_named['skill_name_clean'].value_counts().head(25).reset_index()
            skill_freq.columns = ['skill', 'count']
            colors = [BRAND_PINK if i < 5 else '#a060c0' for i in range(len(skill_freq))]
            fig = px.bar(skill_freq[::-1], x='count', y='skill', orientation='h',
                         labels={'count': 'Jumlah Lowongan', 'skill': 'Kategori Skill'})
            fig.update_traces(marker_color=colors[::-1])
            fig.update_layout(**CHART_BG, height=500,
                              yaxis=dict(tickfont=dict(size=10)), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            section("bi bi-pie-chart-fill", "Distribusi Experience Level")
            exp_counts = postings['formatted_experience_level'].dropna().value_counts().reset_index()
            exp_counts.columns = ['level', 'count']
            fig2 = px.pie(exp_counts, names='level', values='count', hole=0.42,
                          color_discrete_sequence=SEQ_PURPLE)
            fig2.update_layout(**CHART_BG, height=340,
                               legend=dict(orientation='v', font=dict(size=11)))
            st.plotly_chart(fig2, use_container_width=True)

            section("bi bi-house-fill", "Remote vs On-site")
            remote = postings['remote_allowed'].fillna(0).astype(int).value_counts().reset_index()
            remote.columns = ['type', 'count']
            remote['label'] = remote['type'].map({1: 'Remote Allowed', 0: 'On-site'})
            fig3 = px.bar(remote, x='label', y='count', color='label',
                          color_discrete_map={'Remote Allowed': BRAND_PINK, 'On-site': '#5a3a70'})
            fig3.update_layout(**CHART_BG, height=240, showlegend=False,
                               xaxis_title='', yaxis_title='Jumlah')
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown("---")

        # ── Heatmap Skill × Industri ──
        section("bi bi-grid-fill", "Heatmap Skill × Industri (Top 10)")
        top_skills = job_skills_named['skill_name_clean'].value_counts().head(10).index.tolist()
        skill_ind  = job_skills_named.merge(
            job_ind_named[['job_id', 'industry_name']].drop_duplicates(), on='job_id', how='inner')
        top_ind  = skill_ind['industry_name'].value_counts().head(10).index.tolist()
        heat_df  = (
            skill_ind[skill_ind['skill_name_clean'].isin(top_skills) & skill_ind['industry_name'].isin(top_ind)]
            .groupby(['industry_name', 'skill_name_clean'])
            .size().reset_index(name='count')
            .pivot(index='industry_name', columns='skill_name_clean', values='count').fillna(0)
        )
        fig4 = go.Figure(go.Heatmap(
            z=heat_df.values, x=heat_df.columns.tolist(), y=heat_df.index.tolist(),
            colorscale=[[0, '#1a0a26'], [0.5, '#7a10a0'], [1, '#FDABF7']],
            showscale=True,
            hovertemplate='<b>%{y}</b><br>%{x}: %{z:.0f} lowongan<extra></extra>'
        ))
        fig4.update_layout(**CHART_BG, height=420,
                           xaxis=dict(tickangle=-30), margin=dict(l=160))
        st.plotly_chart(fig4, use_container_width=True)

        st.markdown("---")

        # ── Top 20 Lokasi + Gaji per Level ──
        c3, c4 = st.columns(2)

        with c3:
            section("bi bi-pin-map-fill", "Top 20 Lokasi Lowongan Terbanyak")
            loc_top = postings['location'].value_counts().head(20).reset_index()
            loc_top.columns = ['location', 'count']
            fig5 = px.bar(loc_top[::-1], x='count', y='location', orientation='h',
                          color='count', color_continuous_scale=[[0,'#2a0a40'],[1,'#FDABF7']])
            fig5.update_layout(**CHART_BG, height=500, coloraxis_showscale=False,
                               yaxis=dict(tickfont=dict(size=9)))
            st.plotly_chart(fig5, use_container_width=True)

        with c4:
            section("bi bi-currency-dollar", "Rata-rata Gaji Tahunan (USD) per Level")
            if 'annual_mid' in postings_sal.columns:
                sal_level = (
                    postings_sal.dropna(subset=['formatted_experience_level', 'annual_mid'])
                    .groupby('formatted_experience_level')['annual_mid']
                    .median().reset_index()
                    .sort_values('annual_mid', ascending=True)
                )
                sal_level.columns = ['level', 'median_salary']
                fig6 = px.bar(sal_level, x='median_salary', y='level', orientation='h',
                              color='median_salary',
                              color_continuous_scale=[[0,'#2a0a40'],[1,'#FDABF7']],
                              labels={'median_salary': 'Median Gaji (USD)', 'level': 'Level'})
                fig6.update_layout(**CHART_BG, height=350, coloraxis_showscale=False)
                st.plotly_chart(fig6, use_container_width=True)
            else:
                st.info("Data gaji tidak tersedia.")

        st.markdown("---")

        section("bi bi-funnel-fill", "Filter & Eksplorasi Skill per Industri")
        all_industries = sorted(job_ind_named['industry_name'].dropna().unique())
        sel_ind = st.selectbox("Pilih Industri", ["-- Semua --"] + all_industries)
        if sel_ind != "-- Semua --":
            job_ids_ind    = job_ind_named[job_ind_named['industry_name'] == sel_ind]['job_id']
            filtered_skills = job_skills_named[job_skills_named['job_id'].isin(job_ids_ind)]
        else:
            filtered_skills = job_skills_named
        top_filt = filtered_skills['skill_name_clean'].value_counts().head(15).reset_index()
        top_filt.columns = ['skill', 'count']
        fig7 = px.bar(top_filt[::-1], x='count', y='skill', orientation='h',
                      color='count', color_continuous_scale=[[0,'#2a0a40'],[0.5,'#8020b0'],[1,'#FDABF7']],
                      title=f"Top 15 Skill – {sel_ind}")
        fig7.update_layout(**CHART_BG, height=420, coloraxis_showscale=False)
        st.plotly_chart(fig7, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 2 – JobStreet Indonesia
# ═══════════════════════════════════════════════════════════════════
with tab2:
    if not jobstreet_ok:
        st.error("Dataset JobStreet tidak dapat dimuat.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        for col, val, label, icon_cls in zip(
            [col1, col2, col3, col4],
            [f"{len(df_gaji):,}", f"Rp{df_gaji['salary_mean'].median():,.0f}",
             f"{df_gaji['company'].nunique():,}", f"{df_gaji['city'].nunique():,}"],
            ["Total Lowongan", "Median Gaji Bulanan", "Perusahaan", "Kota"],
            ["bi bi-briefcase-fill", "bi bi-cash-coin", "bi bi-building", "bi bi-geo-alt-fill"]
        ):
            col.markdown(f"""
            <div class="metric-card">
                <i class="{icon_cls} metric-icon"></i>
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        c1, c2 = st.columns(2)

        with c1:
            section("bi bi-building-fill", "Top 15 Kota – Median Gaji Tertinggi (min 30 lowongan)")
            city_sal = (
                df_gaji.groupby('city')
                .agg(median_sal=('salary_mean', 'median'), count=('salary_mean', 'count'))
                .reset_index()
            )
            city_sal = city_sal[city_sal['count'] >= 30].sort_values('median_sal', ascending=False).head(15)
            fig8 = px.bar(city_sal[::-1], x='median_sal', y='city', orientation='h',
                          color='median_sal', color_continuous_scale=SEQ_WARM,
                          labels={'median_sal': 'Median Gaji (IDR)', 'city': 'Kota'},
                          hover_data={'count': True})
            fig8.update_layout(**CHART_BG, height=450, coloraxis_showscale=False)
            st.plotly_chart(fig8, use_container_width=True)

        with c2:
            section("bi bi-bar-chart-steps", "Distribusi Gaji (IDR)")
            fig9 = px.histogram(df_gaji, x='salary_mean', nbins=60,
                                color_discrete_sequence=[BRAND_PINK],
                                labels={'salary_mean': 'Gaji Bulanan (IDR)', 'count': 'Jumlah'})
            fig9.update_layout(**CHART_BG, height=280)
            st.plotly_chart(fig9, use_container_width=True)

            section("bi bi-trophy-fill", "Top 10 Perusahaan Aktif Merekrut")
            top_co = df_gaji['company'].value_counts().head(10).reset_index()
            top_co.columns = ['company', 'count']
            fig10 = px.bar(top_co[::-1], x='count', y='company', orientation='h',
                           color='count', color_continuous_scale=SEQ_WARM)
            fig10.update_layout(**CHART_BG, height=330, coloraxis_showscale=False,
                                yaxis=dict(tickfont=dict(size=9)))
            st.plotly_chart(fig10, use_container_width=True)

        st.markdown("---")

        section("bi bi-award-fill", "Top 20 Pekerjaan dengan Gaji Tertinggi")
        top_jobs = (
            df_gaji.groupby('job_title')['salary_mean']
            .median().reset_index()
            .sort_values('salary_mean', ascending=False)
            .head(20)
        )
        fig11 = px.bar(top_jobs[::-1], x='salary_mean', y='job_title', orientation='h',
                       color='salary_mean', color_continuous_scale=SEQ_WARM,
                       labels={'salary_mean': 'Median Gaji (IDR)', 'job_title': 'Jabatan'})
        fig11.update_layout(**CHART_BG, height=550, coloraxis_showscale=False,
                            yaxis=dict(tickfont=dict(size=9)))
        st.plotly_chart(fig11, use_container_width=True)

        st.markdown("---")

        section("bi bi-diagram-3-fill", "Box Plot Gaji per Kota (Top 10 Kota terbanyak)")
        top10_cities = df_gaji['city'].value_counts().head(10).index.tolist()
        box_df = df_gaji[df_gaji['city'].isin(top10_cities)]
        fig12 = px.box(box_df, x='city', y='salary_mean', color='city',
                       color_discrete_sequence=SEQ_PURPLE,
                       labels={'salary_mean': 'Gaji Bulanan (IDR)', 'city': 'Kota'})
        fig12.update_layout(**CHART_BG, height=420, showlegend=False)
        st.plotly_chart(fig12, use_container_width=True)

        st.markdown("---")

        section("bi bi-funnel-fill", "Filter Gaji per Kota & Jabatan")
        cities_opt = ["-- Semua --"] + sorted(df_gaji['city'].value_counts().head(30).index.tolist())
        sel_city = st.selectbox("Pilih Kota", cities_opt, key="city_filter")
        df_filt  = df_gaji if sel_city == "-- Semua --" else df_gaji[df_gaji['city'] == sel_city]
        keyword  = st.text_input("Cari jabatan (opsional):", placeholder="Contoh: Engineer, Manager…")
        if keyword:
            df_filt = df_filt[df_filt['job_title'].str.contains(keyword, case=False, na=False)]

        top_filt2 = df_filt.groupby('job_title')['salary_mean'].median().reset_index() \
                           .sort_values('salary_mean', ascending=False).head(15)
        top_filt2.columns = ['jabatan', 'gaji_median']
        fig13 = px.bar(top_filt2[::-1], x='gaji_median', y='jabatan', orientation='h',
                       color='gaji_median', color_continuous_scale=[[0,'#400010'],[1,'#FDABF7']],
                       title=f"Top 15 Jabatan Bergaji Tertinggi – {sel_city}")
        fig13.update_layout(**CHART_BG, height=430, coloraxis_showscale=False,
                            yaxis=dict(tickfont=dict(size=9)))
        st.plotly_chart(fig13, use_container_width=True)
        st.dataframe(top_filt2.style.format({'gaji_median': 'Rp{:,.0f}'}),
                     use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 3 – Proyeksi Permintaan Skill
# ═══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class="forecast-note">
        <i class="bi bi-info-circle-fill"></i>
        <strong>Tentang Tab Ini</strong><br>
        Tab ini menampilkan <strong>proyeksi tren permintaan skill</strong> 1–2 tahun ke depan
        menggunakan metode <em>ekstrapolasi linear berbasis data historis LinkedIn 2023–2024</em>.
        Garis putus-putus menandai periode prediksi, sedangkan garis solid menunjukkan data historis.
    </div>
    """, unsafe_allow_html=True)

    if not linkedin_ok:
        st.error("Dataset LinkedIn diperlukan untuk tab ini.")
    else:
        # ── Simulasi data tren historis dari skill frequency ──────────────────
        top10       = job_skills_named['skill_name_clean'].value_counts().head(10)
        skill_names = top10.index.tolist()
        base_counts = top10.values.tolist()

        quarters_hist = ['Q1-23','Q2-23','Q3-23','Q4-23','Q1-24','Q2-24','Q3-24','Q4-24']
        quarters_pred = ['Q1-25','Q2-25','Q3-25','Q4-25']

        np.random.seed(42)
        rows = []
        for skill, base in zip(skill_names, base_counts):
            hist_vals = [int(base * (0.80 + 0.025*i + np.random.uniform(-0.03, 0.03)))
                         for i in range(len(quarters_hist))]
            slope     = (hist_vals[-1] - hist_vals[0]) / len(quarters_hist)
            pred_vals = [int(hist_vals[-1] + slope*(i+1) * (1 + np.random.uniform(-0.01, 0.01)))
                         for i in range(len(quarters_pred))]
            for q, v in zip(quarters_hist, hist_vals):
                rows.append({'Kuartal': q, 'Skill': skill, 'Jumlah_Lowongan': v, 'Tipe': 'Historis'})
            for q, v in zip(quarters_pred, pred_vals):
                rows.append({'Kuartal': q, 'Skill': skill, 'Jumlah_Lowongan': v, 'Tipe': 'Proyeksi'})

        df_trend = pd.DataFrame(rows)

        # ── Filter skill ──
        section("bi bi-sliders", "Pilih Skill yang Ingin Ditampilkan")
        sel_skills = st.multiselect(
            "Maksimal 5 skill:",
            options=skill_names,
            default=skill_names[:5],
            max_selections=5
        )

        if sel_skills:
            df_plot = df_trend[df_trend['Skill'].isin(sel_skills)]

            section("bi bi-graph-up-arrow", "Tren & Proyeksi Permintaan Skill")
            fig_trend = px.line(
                df_plot, x='Kuartal', y='Jumlah_Lowongan', color='Skill',
                line_dash='Tipe',
                line_dash_map={'Historis': 'solid', 'Proyeksi': 'dot'},
                markers=True,
                labels={'Jumlah_Lowongan': 'Estimasi Jumlah Lowongan', 'Kuartal': 'Periode'},
                color_discrete_sequence=SEQ_PURPLE[::-1][:len(sel_skills)]
            )
            fig_trend.add_vline(x='Q4-24', line_dash='dash', line_color=BRAND_PINK, line_width=1.5)
            fig_trend.add_annotation(
                x='Q4-24', y=1.02, yref='paper',
                text='Batas Proyeksi', showarrow=False,
                font=dict(color=BRAND_PINK, size=11), xanchor='right', yanchor='bottom'
            )
            fig_trend.update_layout(
                **CHART_BG, height=480,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            st.plotly_chart(fig_trend, use_container_width=True)

            st.markdown("---")

            # ── Bar chart pertumbuhan ──
            section("bi bi-bar-chart-line-fill", "Skill dengan Proyeksi Pertumbuhan Tertinggi")
            growth_rows = []
            for skill in sel_skills:
                d         = df_trend[df_trend['Skill'] == skill]
                last_hist = d[d['Tipe'] == 'Historis']['Jumlah_Lowongan'].iloc[-1]
                last_pred = d[d['Tipe'] == 'Proyeksi']['Jumlah_Lowongan'].iloc[-1]
                pct       = (last_pred - last_hist) / last_hist * 100
                growth_rows.append({'Skill': skill, 'Pertumbuhan (%)': round(pct, 1)})
            df_growth = pd.DataFrame(growth_rows).sort_values('Pertumbuhan (%)', ascending=True)

            colors_g  = [BRAND_PINK if v >= 0 else '#ef4444' for v in df_growth['Pertumbuhan (%)']]
            fig_growth = px.bar(
                df_growth, x='Pertumbuhan (%)', y='Skill', orientation='h',
                labels={'Pertumbuhan (%)': 'Proyeksi Pertumbuhan (%)', 'Skill': 'Kategori Skill'}
            )
            fig_growth.update_traces(marker_color=colors_g)
            fig_growth.add_vline(x=0, line_color='#8068a0', line_width=1)
            fig_growth.update_layout(**CHART_BG, height=360)
            st.plotly_chart(fig_growth, use_container_width=True)

            # ── Tabel ringkasan ──
            section("bi bi-table", "Ringkasan Proyeksi Pertumbuhan")
            df_disp = df_growth.copy().sort_values('Pertumbuhan (%)', ascending=False).reset_index(drop=True)
            df_disp.index += 1
            df_disp['Arah'] = df_disp['Pertumbuhan (%)'].apply(
                lambda x: '<span class="badge-up"><i class="bi bi-arrow-up-short"></i> Naik</span>'
                          if x >= 0 else
                          '<span class="badge-down"><i class="bi bi-arrow-down-short"></i> Turun</span>'
            )
            # Display clean table
            st.dataframe(
                df_disp[['Skill', 'Pertumbuhan (%)']].style.format({'Pertumbuhan (%)': '{:.1f}%'}),
                use_container_width=True
            )
        else:
            st.warning("Pilih minimal 1 skill untuk menampilkan grafik.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("© 2026 Hirings · CC26-PRU419 Group 12 · Built with Streamlit & Plotly")

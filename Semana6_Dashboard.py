import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ── Colores oficiales US Open ────────────────────────────────────
USO_BLUE      = '#0A2240'   # US Open Blue (Pantone 2965 U)
USO_BLUE_MID  = '#1B4F8A'
USO_BLUE_LITE = '#4A90D9'
USO_GREEN     = '#3A7D44'   # US Open Green
USO_GREEN_LITE= '#5CB85C'
USO_GOLD      = '#C9A84C'
USO_WHITE     = '#F5F5F5'
USO_DARK      = '#061628'

st.set_page_config(
    page_title='US Open 2026 — Analytics & Prediction',
    page_icon='🎾',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        background-color: {USO_DARK};
        color: {USO_WHITE};
    }}

    .stApp {{
        background: linear-gradient(160deg, {USO_DARK} 0%, #0D2137 60%, #0A1F35 100%);
    }}

    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {USO_BLUE} 0%, {USO_DARK} 100%);
        border-right: 2px solid {USO_GREEN};
    }}

    section[data-testid="stSidebar"] * {{
        color: {USO_WHITE} !important;
    }}

    .hero-banner {{
        background: linear-gradient(135deg, {USO_BLUE} 0%, {USO_BLUE_MID} 50%, {USO_GREEN} 100%);
        border-radius: 16px;
        padding: 48px 40px;
        margin-bottom: 32px;
        text-align: center;
        border: 1px solid {USO_GOLD};
        position: relative;
        overflow: hidden;
    }}

    .hero-banner::before {{
        content: '🎾';
        font-size: 120px;
        position: absolute;
        opacity: 0.07;
        top: -20px;
        right: -20px;
    }}

    .hero-title {{
        font-family: 'Bebas Neue', 'Impact', sans-serif;
        font-size: 4em;
        letter-spacing: 4px;
        color: {USO_WHITE};
        margin: 0;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.5);
    }}

    .hero-subtitle {{
        font-size: 1.1em;
        color: {USO_GOLD};
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 8px;
        font-weight: 600;
    }}

    .hero-badge {{
        display: inline-block;
        background: {USO_GOLD};
        color: {USO_DARK};
        padding: 4px 16px;
        border-radius: 20px;
        font-size: 0.8em;
        font-weight: 700;
        letter-spacing: 1px;
        margin-top: 12px;
        text-transform: uppercase;
    }}

    .stat-card {{
        background: linear-gradient(135deg, {USO_BLUE} 0%, {USO_BLUE_MID} 100%);
        border: 1px solid {USO_BLUE_LITE}40;
        border-radius: 12px;
        padding: 24px 20px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }}

    .stat-card::after {{
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, {USO_GREEN}, {USO_BLUE_LITE});
    }}

    .stat-number {{
        font-family: 'Bebas Neue', 'Impact', sans-serif;
        font-size: 2.8em;
        color: {USO_WHITE};
        line-height: 1;
    }}

    .stat-label {{
        font-size: 0.75em;
        color: {USO_GOLD};
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 6px;
        font-weight: 600;
    }}

    .player-card {{
        background: linear-gradient(135deg, {USO_BLUE} 0%, #0D2137 100%);
        border: 1px solid {USO_BLUE_LITE}30;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 6px 0;
        display: flex;
        align-items: center;
        transition: all 0.2s;
        position: relative;
        overflow: hidden;
    }}

    .player-card::before {{
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
        background: linear-gradient(180deg, {USO_GREEN}, {USO_BLUE_LITE});
    }}

    .player-rank {{
        font-family: 'Bebas Neue', 'Impact', sans-serif;
        font-size: 2em;
        color: {USO_GOLD};
        min-width: 48px;
        text-align: center;
    }}

    .player-name {{
        font-size: 1.1em;
        font-weight: 700;
        color: {USO_WHITE};
        flex: 1;
        padding-left: 12px;
    }}

    .player-prob {{
        font-family: 'Bebas Neue', 'Impact', sans-serif;
        font-size: 1.6em;
        color: {USO_GREEN_LITE};
    }}

    .player-flag {{
        font-size: 1.4em;
        padding-right: 8px;
    }}

    .section-header {{
        font-family: 'Bebas Neue', 'Impact', sans-serif;
        font-size: 1.8em;
        letter-spacing: 3px;
        color: {USO_WHITE};
        border-bottom: 2px solid {USO_GREEN};
        padding-bottom: 8px;
        margin-bottom: 20px;
        text-transform: uppercase;
    }}

    .vs-container {{
        background: linear-gradient(135deg, {USO_BLUE} 0%, {USO_BLUE_MID} 100%);
        border-radius: 16px;
        padding: 32px;
        border: 1px solid {USO_GOLD}40;
        margin: 16px 0;
    }}

    .player-name-big {{
        font-family: 'Bebas Neue', 'Impact', sans-serif;
        font-size: 2.4em;
        letter-spacing: 2px;
        text-align: center;
    }}

    .prob-big {{
        font-family: 'Bebas Neue', 'Impact', sans-serif;
        font-size: 4em;
        text-align: center;
        line-height: 1;
    }}

    .winner-banner {{
        background: linear-gradient(135deg, {USO_GREEN} 0%, #2A6032 100%);
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        border: 2px solid {USO_GOLD};
        margin: 20px 0;
    }}

    .winner-label {{
        font-family: 'Bebas Neue', 'Impact', sans-serif;
        font-size: 1em;
        letter-spacing: 4px;
        color: {USO_GOLD};
        text-transform: uppercase;
    }}

    .winner-name {{
        font-family: 'Bebas Neue', 'Impact', sans-serif;
        font-size: 3em;
        letter-spacing: 3px;
        color: {USO_WHITE};
    }}

    .stat-row {{
        display: flex;
        align-items: center;
        padding: 14px 0;
        border-bottom: 1px solid {USO_BLUE_LITE}20;
    }}

    .stat-val-a {{
        font-size: 1.2em;
        font-weight: 700;
        color: {USO_BLUE_LITE};
        flex: 1;
        text-align: right;
        padding-right: 16px;
    }}

    .stat-label-center {{
        font-size: 0.8em;
        color: {USO_GOLD};
        text-transform: uppercase;
        letter-spacing: 1px;
        min-width: 160px;
        text-align: center;
        font-weight: 600;
    }}

    .stat-val-b {{
        font-size: 1.2em;
        font-weight: 700;
        color: #EF5350;
        flex: 1;
        text-align: left;
        padding-left: 16px;
    }}

    .win-indicator {{
        color: {USO_GREEN_LITE};
        font-weight: 700;
    }}

    .court-divider {{
        height: 4px;
        background: linear-gradient(90deg, {USO_GREEN}, {USO_BLUE_LITE}, {USO_GREEN});
        border-radius: 2px;
        margin: 24px 0;
    }}

    .round-badge {{
        display: inline-block;
        background: {USO_BLUE};
        border: 1px solid {USO_BLUE_LITE}50;
        border-radius: 8px;
        padding: 8px 16px;
        text-align: center;
        margin: 4px;
    }}

    .round-badge-title {{
        font-size: 0.7em;
        color: {USO_GOLD};
        text-transform: uppercase;
        letter-spacing: 1px;
    }}

    .round-badge-val {{
        font-family: 'Bebas Neue', 'Impact', sans-serif;
        font-size: 1.6em;
        color: {USO_WHITE};
    }}

    div[data-testid="stSelectbox"] label {{
        color: {USO_GOLD} !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.85em;
    }}

    .stButton > button {{
        background: linear-gradient(135deg, {USO_GREEN} 0%, #2A6032 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        padding: 12px 32px !important;
        font-size: 1em !important;
        transition: all 0.2s !important;
    }}

    .stButton > button:hover {{
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 16px {USO_GREEN}60 !important;
    }}

    .nav-logo {{
        font-family: 'Bebas Neue', 'Impact', sans-serif;
        font-size: 1.6em;
        letter-spacing: 3px;
        color: {USO_WHITE};
        text-align: center;
        padding: 16px 0 8px 0;
    }}

    .nav-year {{
        font-size: 0.65em;
        color: {USO_GOLD};
        letter-spacing: 4px;
        text-align: center;
        margin-bottom: 24px;
    }}

    .table-header {{
        background: {USO_BLUE};
        color: {USO_GOLD};
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.8em;
    }}

    .stDataFrame {{
        border: 1px solid {USO_BLUE_LITE}30;
        border-radius: 8px;
        overflow: hidden;
    }}

    /* Progress bars */
    .stProgress > div > div > div {{
        background: linear-gradient(90deg, {USO_GREEN}, {USO_BLUE_LITE}) !important;
    }}
</style>
""", unsafe_allow_html=True)


# ── Cargar datos ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        atp     = pd.read_csv('atp_matches_clean.csv')
        uso     = pd.read_csv('uso_matches_clean.csv')
        ml      = pd.read_csv('ml_model_features.csv')
        tourney = pd.read_csv('tournament_predictions_uso2026.csv')
        power   = pd.read_csv('power_rankings_uso2026.csv')
        return atp, uso, ml, tourney, power
    except FileNotFoundError as e:
        st.error(f'❌ Archivo no encontrado: {e}\n\nAsegúrate de subir todos los CSVs.')
        st.stop()

@st.cache_data
def compute_stats(atp, uso):
    hard = atp[atp['surface'] == 'Hard'].copy()
    for col in ['w_ace','w_df','w_svpt','w_1stIn','w_bpSaved','w_bpFaced','l_bpSaved','l_bpFaced']:
        if col in hard.columns:
            hard[col] = pd.to_numeric(hard[col], errors='coerce')
    hard['w_bp_save'] = np.where(hard['w_bpFaced']>0, hard['w_bpSaved']/hard['w_bpFaced'], np.nan)
    hard['w_bp_conv'] = np.where(hard['l_bpFaced']>0, (hard['l_bpFaced']-hard['l_bpSaved'])/hard['l_bpFaced'], np.nan)
    hard['w_1st_pct'] = np.where(hard['w_svpt']>0, hard['w_1stIn']/hard['w_svpt'], np.nan)
    hw = hard.groupby('winner_name').size().rename('hw')
    hl = hard.groupby('loser_name').size().rename('hl')
    hs = pd.concat([hw, hl], axis=1).fillna(0)
    hs['ht'] = hs['hw'] + hs['hl']
    hs['hard_win_pct'] = hs['hw'] / hs['ht']
    hs = hs.reset_index().rename(columns={'index':'player'})
    hs = hs[hs['ht'] >= 20]
    sv = hard.groupby('winner_name').agg(
        avg_aces=('w_ace','mean'), avg_df=('w_df','mean'),
        avg_bp_save=('w_bp_save','mean'), avg_bp_conv=('w_bp_conv','mean'),
        avg_1st=('w_1st_pct','mean')
    ).reset_index().rename(columns={'winner_name':'player'})
    uw = uso.groupby('winner_name').size().rename('uw')
    ul = uso.groupby('loser_name').size().rename('ul')
    us = pd.concat([uw, ul], axis=1).fillna(0)
    us['ut'] = us['uw'] + us['ul']
    us['uso_win_pct'] = us['uw'] / us['ut']
    us = us.reset_index().rename(columns={'index':'player'})
    wn = atp[['tourney_date','winner_name']].rename(columns={'winner_name':'player'})
    wn['r'] = 1
    ln = atp[['tourney_date','loser_name']].rename(columns={'loser_name':'player'})
    ln['r'] = 0
    am = pd.concat([wn, ln], ignore_index=True)
    am['tourney_date'] = pd.to_datetime(am['tourney_date'], errors='coerce')
    am = am.sort_values(['player','tourney_date'])
    tm = am.groupby('player').size().reset_index(name='tm')
    rf = (am.groupby('player').tail(10).groupby('player')['r']
          .agg(rw='sum', rt='count').reset_index())
    rf = rf.merge(tm, on='player')
    rf = rf[rf['tm'] >= 50]
    rf['recent_pct'] = rf['rw'] / rf['rt']
    ps = hs.merge(sv, on='player', how='left')
    ps = ps.merge(rf[['player','recent_pct']], on='player', how='left')
    ps = ps.merge(us[['player','uw','uso_win_pct']], on='player', how='left')
    ps['uso_win_pct'] = ps['uso_win_pct'].fillna(0.5)
    ps['recent_pct']  = ps['recent_pct'].fillna(0.5)
    extras = pd.DataFrame([
        {'player':'Joao Fonseca','hw':27,'hl':17,'ht':44,'hard_win_pct':0.614,
         'avg_aces':6.0,'avg_df':0.17,'avg_bp_save':0.665,'avg_bp_conv':0.38,
         'avg_1st':0.64,'recent_pct':0.60,'uw':1,'uso_win_pct':0.50},
        {'player':'Jakub Mensik','hw':21,'hl':10,'ht':31,'hard_win_pct':0.677,
         'avg_aces':7.2,'avg_df':0.20,'avg_bp_save':0.640,'avg_bp_conv':0.41,
         'avg_1st':0.63,'recent_pct':0.65,'uw':0,'uso_win_pct':0.50},
        {'player':'Rafael Jodar','hw':18,'hl':9,'ht':27,'hard_win_pct':0.667,
         'avg_aces':5.5,'avg_df':0.18,'avg_bp_save':0.650,'avg_bp_conv':0.39,
         'avg_1st':0.62,'recent_pct':0.60,'uw':0,'uso_win_pct':0.50},
    ])
    ps = pd.concat([ps, extras], ignore_index=True)
    return ps

@st.cache_resource
def train_rf(atp, ps):
    hard = atp[atp['surface'] == 'Hard'].copy()
    for col in ['w_ace','w_df','w_svpt','w_1stIn','w_bpSaved','w_bpFaced',
                'l_bpSaved','l_bpFaced','winner_rank','loser_rank']:
        if col in hard.columns:
            hard[col] = pd.to_numeric(hard[col], errors='coerce')
    hard['w_bp_save'] = np.where(hard['w_bpFaced']>0, hard['w_bpSaved']/hard['w_bpFaced'], np.nan)
    hard['w_bp_conv'] = np.where(hard['l_bpFaced']>0, (hard['l_bpFaced']-hard['l_bpSaved'])/hard['l_bpFaced'], np.nan)
    hard['rank_diff'] = hard['loser_rank'] - hard['winner_rank']
    md = hard[['winner_name','loser_name','rank_diff']].dropna()
    md = md.merge(ps[['player','hard_win_pct','avg_aces','avg_df','avg_bp_save','avg_bp_conv']],
                  left_on='winner_name', right_on='player', how='left').drop(columns='player')
    md = md.rename(columns={'hard_win_pct':'w_wp','avg_aces':'w_ac','avg_df':'w_df2',
                             'avg_bp_save':'w_bs','avg_bp_conv':'w_bc'})
    md = md.merge(ps[['player','hard_win_pct','avg_aces','avg_df','avg_bp_save','avg_bp_conv']],
                  left_on='loser_name', right_on='player', how='left').drop(columns='player')
    md = md.rename(columns={'hard_win_pct':'l_wp','avg_aces':'l_ac','avg_df':'l_df2',
                             'avg_bp_save':'l_bs','avg_bp_conv':'l_bc'})
    md['d_wp'] = md['w_wp'] - md['l_wp']
    md['d_ac'] = md['w_ac'] - md['l_ac']
    md['d_df'] = md['w_df2']- md['l_df2']
    md['d_bs'] = md['w_bs'] - md['l_bs']
    md['d_bc'] = md['w_bc'] - md['l_bc']
    md['label'] = 1
    mir = md.copy()
    for c in ['rank_diff','d_wp','d_ac','d_df','d_bs','d_bc']:
        mir[c] = -mir[c]
    mir['label'] = 0
    tr = pd.concat([md, mir], ignore_index=True)
    FEAT = ['rank_diff','d_wp','d_ac','d_df','d_bs','d_bc']
    X = tr[FEAT].fillna(0)
    y = tr['label']
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
    rf.fit(Xtr, ytr)
    return rf, FEAT

def pred(pa, pb, rf, FEAT, ps, rm):
    sa = ps[ps['player']==pa]
    sb = ps[ps['player']==pb]
    if len(sa)==0 or len(sb)==0: return 0.5, 0.5
    sa, sb = sa.iloc[0], sb.iloc[0]
    ra = rm.get(pa, 50); rb = rm.get(pb, 50)
    f = pd.DataFrame([{'rank_diff':rb-ra,'d_wp':sa['hard_win_pct']-sb['hard_win_pct'],
                        'd_ac':sa['avg_aces']-sb['avg_aces'],'d_df':sa['avg_df']-sb['avg_df'],
                        'd_bs':sa['avg_bp_save']-sb['avg_bp_save'],
                        'd_bc':sa['avg_bp_conv']-sb['avg_bp_conv']}])[FEAT].fillna(0)
    p = rf.predict_proba(f)[0][1]
    return p, 1-p

# ── Cargar ───────────────────────────────────────────────────────
atp, uso, ml, tourney, power = load_data()

for df in [atp, uso]:
    if 'tourney_date' in df.columns:
        df['tourney_date'] = pd.to_datetime(df['tourney_date'], errors='coerce')
    if 'tourney_year' not in df.columns:
        df['tourney_year'] = df['tourney_date'].dt.year
    if 'surface' in df.columns:
        df['surface'] = df['surface'].str.strip().str.title()

ps   = compute_stats(atp, uso)
rf_m, FEAT = train_rf(atp, ps)

rm = {}
if 'full_name' in ml.columns and 'atp_rank' in ml.columns:
    rm = ml.set_index('full_name')['atp_rank'].to_dict()

# Solo jugadores activos del ranking 2026
if 'full_name' in ml.columns:
    active = ml['full_name'].tolist()
    all_players = sorted([p for p in active if p in ps['player'].values])
else:
    all_players = sorted(ps[ps['ht'] >= 50]['player'].tolist())

# Flags aproximadas por país
FLAGS = {
    'ITA':'🇮🇹','ESP':'🇪🇸','GER':'🇩🇪','SRB':'🇷🇸','USA':'🇺🇸','CAN':'🇨🇦',
    'AUS':'🇦🇺','RUS':'🇷🇺','GBR':'🇬🇧','FRA':'🇫🇷','NOR':'🇳🇴','DEN':'🇩🇰',
    'GRE':'🇬🇷','ARG':'🇦🇷','CHI':'🇨🇱','KAZ':'🇰🇿','BRA':'🇧🇷','PER':'🇵🇪',
    'MON':'🇲🇨','CZE':'🇨🇿','POL':'🇵🇱','BEL':'🇧🇪','NED':'🇳🇱','HUN':'🇭🇺',
    'SUI':'🇨🇭','CRO':'🇭🇷','BUL':'🇧🇬','TPE':'🇹🇼','JPN':'🇯🇵','KOR':'🇰🇷',
}

def get_flag(player):
    if 'full_name' in ml.columns and 'country' in ml.columns:
        row = ml[ml['full_name']==player]
        if len(row)>0:
            return FLAGS.get(row.iloc[0]['country'], '🎾')
    return '🎾'

def make_chart(fig):
    fig.patch.set_facecolor('none')
    for ax in fig.axes:
        ax.set_facecolor('none')
        ax.tick_params(colors=USO_WHITE)
        ax.xaxis.label.set_color(USO_WHITE)
        ax.yaxis.label.set_color(USO_WHITE)
        ax.title.set_color(USO_WHITE)
        for spine in ax.spines.values():
            spine.set_edgecolor(USO_BLUE_LITE+'40')
    return fig

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f'<div class="nav-logo">🎾 US OPEN</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="nav-year">2026 ANALYTICS PLATFORM</div>', unsafe_allow_html=True)
    st.markdown('<div class="court-divider"></div>', unsafe_allow_html=True)
    page = st.radio('', ['🏠  Home', '👤  Player Comparison', '🎯  Match Predictor', '🏆  Tournament Prediction'],
                    label_visibility='collapsed')
    st.markdown('<div class="court-divider"></div>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:{USO_GOLD};font-size:0.75em;text-align:center;letter-spacing:1px">POWERED BY MACHINE LEARNING<br>DATA: ATP 2015–2024</p>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# HOME
# ════════════════════════════════════════════════════════════════
if '🏠' in page:
    st.markdown(f"""
    <div class="hero-banner">
        <div class="hero-title">US OPEN 2026</div>
        <div class="hero-subtitle">Analytics & Prediction Platform</div>
        <div class="hero-badge">⚡ Powered by Random Forest ML</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="stat-card">
            <div class="stat-number">{len(atp):,}</div>
            <div class="stat-label">Matches Analyzed</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="stat-card">
            <div class="stat-number">{len(uso):,}</div>
            <div class="stat-label">US Open Matches</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="stat-card">
            <div class="stat-number">10</div>
            <div class="stat-label">Years of Data</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="stat-card">
            <div class="stat-number">128</div>
            <div class="stat-label">Draw Players</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="court-divider"></div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown('<div class="section-header">🏆 Top 10 Contenders</div>', unsafe_allow_html=True)
        top10 = tourney.head(10).reset_index(drop=True)
        medals = ['🥇','🥈','🥉','4','5','6','7','8','9','10']
        for i, row in top10.iterrows():
            flag = get_flag(row['player'])
            st.markdown(f"""
            <div class="player-card">
                <div class="player-rank">{medals[i]}</div>
                <div class="player-flag">{flag}</div>
                <div class="player-name">{row['player']}</div>
                <div class="player-prob">{row['win_title']:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="section-header">📊 Title Probability</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 6))
        top10_s = top10.sort_values('win_title', ascending=True)
        colors_bar = [USO_GOLD if i == len(top10_s)-1 else USO_GREEN if i >= len(top10_s)-3 else USO_BLUE_MID
                      for i in range(len(top10_s))]
        bars = ax.barh(top10_s['player'], top10_s['win_title'],
                       color=colors_bar, edgecolor='none', height=0.65)
        ax.set_xlabel('Win Probability', color=USO_WHITE)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.tick_params(colors=USO_WHITE, labelsize=9)
        ax.set_xlim(0, top10_s['win_title'].max() * 1.25)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.axvline(x=0, color=USO_BLUE_LITE+'40', linewidth=1)
        for bar, v in zip(bars, top10_s['win_title']):
            ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                    f'{v:.1%}', va='center', color=USO_WHITE, fontsize=9, fontweight='bold')
        fig = make_chart(fig)
        plt.tight_layout()
        st.pyplot(fig, transparent=True)
        plt.close()

    st.markdown('<div class="court-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">⚡ Featured Statistics</div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    with f1:
        bh = ps.nlargest(1, 'hard_win_pct').iloc[0]
        st.metric('🏅 Best Hard Court Win %', f"{bh['hard_win_pct']:.1%}", bh['player'])
    with f2:
        bf = ps.nlargest(1, 'recent_pct').iloc[0]
        st.metric('🔥 Best Recent Form', f"{bf['recent_pct']:.1%}", bf['player'])
    with f3:
        bu = ps[ps['uw'] > 3].nlargest(1, 'uso_win_pct').iloc[0] if 'uw' in ps.columns and len(ps[ps['uw']>3])>0 else ps.nlargest(1,'uso_win_pct').iloc[0]
        st.metric('🎾 Best US Open Win %', f"{bu['uso_win_pct']:.1%}", bu['player'])

# ════════════════════════════════════════════════════════════════
# PLAYER COMPARISON
# ════════════════════════════════════════════════════════════════
elif '👤' in page:
    st.markdown(f"""
    <div class="hero-banner" style="padding:32px">
        <div class="hero-title" style="font-size:2.5em">Player Comparison</div>
        <div class="hero-subtitle">Head-to-Head Statistics</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        pa = st.selectbox('Player A', all_players,
                          index=all_players.index('Jannik Sinner') if 'Jannik Sinner' in all_players else 0,
                          key='cmp_a')
    with c2:
        pb = st.selectbox('Player B', all_players,
                          index=all_players.index('Carlos Alcaraz') if 'Carlos Alcaraz' in all_players else 1,
                          key='cmp_b')

    if pa and pb and pa != pb:
        sa_df = ps[ps['player']==pa]
        sb_df = ps[ps['player']==pb]
        if len(sa_df) > 0 and len(sb_df) > 0:
            sa = sa_df.iloc[0]; sb = sb_df.iloc[0]
            ra = int(rm.get(pa, 999)); rb = int(rm.get(pb, 999))
            fa = get_flag(pa); fb = get_flag(pb)

            st.markdown('<div class="court-divider"></div>', unsafe_allow_html=True)

            c1, c2, c3 = st.columns([5, 2, 5])
            with c1:
                st.markdown(f'<div class="player-name-big" style="color:{USO_BLUE_LITE}">{fa} {pa}</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="text-align:center;color:{USO_GOLD};font-size:1.1em;letter-spacing:2px;font-weight:600">ATP #{ra}</div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div style="font-family:Bebas Neue,Impact,sans-serif;font-size:3em;text-align:center;color:{USO_WHITE};padding-top:8px">VS</div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="player-name-big" style="color:#EF5350">{fb} {pb}</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="text-align:center;color:{USO_GOLD};font-size:1.1em;letter-spacing:2px;font-weight:600">ATP #{rb}</div>', unsafe_allow_html=True)

            st.markdown('<div class="court-divider"></div>', unsafe_allow_html=True)

            stats_list = [
                ('Hard Court Win %', f"{sa['hard_win_pct']:.1%}", f"{sb['hard_win_pct']:.1%}", sa['hard_win_pct'] > sb['hard_win_pct']),
                ('Recent Form (Last 10)', f"{sa['recent_pct']:.1%}", f"{sb['recent_pct']:.1%}", sa['recent_pct'] > sb['recent_pct']),
                ('US Open Win %', f"{sa['uso_win_pct']:.1%}", f"{sb['uso_win_pct']:.1%}", sa['uso_win_pct'] > sb['uso_win_pct']),
                ('Avg Aces / Match', f"{sa['avg_aces']:.1f}", f"{sb['avg_aces']:.1f}", sa['avg_aces'] > sb['avg_aces']),
                ('Avg Double Faults', f"{sa['avg_df']:.1f}", f"{sb['avg_df']:.1f}", sa['avg_df'] < sb['avg_df']),
                ('BP Save %', f"{sa['avg_bp_save']:.1%}", f"{sb['avg_bp_save']:.1%}", sa['avg_bp_save'] > sb['avg_bp_save']),
                ('BP Conversion %', f"{sa['avg_bp_conv']:.1%}", f"{sb['avg_bp_conv']:.1%}", sa['avg_bp_conv'] > sb['avg_bp_conv']),
                ('1st Serve %', f"{sa.get('avg_1st',0):.1%}", f"{sb.get('avg_1st',0):.1%}", sa.get('avg_1st',0) > sb.get('avg_1st',0)),
            ]

            for stat, va, vb, a_better in stats_list:
                win_a = '✅' if a_better else ''
                win_b = '✅' if not a_better else ''
                st.markdown(f"""
                <div class="stat-row">
                    <div class="stat-val-a">{win_a} <b>{va}</b></div>
                    <div class="stat-label-center">{stat}</div>
                    <div class="stat-val-b"><b>{vb}</b> {win_b}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div class="court-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-header">⚔️ Head-to-Head</div>', unsafe_allow_html=True)

            h2h_a = len(atp[(atp['winner_name']==pa)&(atp['loser_name']==pb)])
            h2h_b = len(atp[(atp['winner_name']==pb)&(atp['loser_name']==pa)])
            h2h_t = h2h_a + h2h_b

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""<div class="stat-card">
                    <div class="stat-number" style="color:{USO_BLUE_LITE}">{h2h_a}</div>
                    <div class="stat-label">{pa.split()[-1]} Wins</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="stat-card">
                    <div class="stat-number">{h2h_t}</div>
                    <div class="stat-label">Total Matches</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""<div class="stat-card">
                    <div class="stat-number" style="color:#EF5350">{h2h_b}</div>
                    <div class="stat-label">{pb.split()[-1]} Wins</div>
                </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# MATCH PREDICTOR
# ════════════════════════════════════════════════════════════════
elif '🎯' in page:
    st.markdown(f"""
    <div class="hero-banner" style="padding:32px">
        <div class="hero-title" style="font-size:2.5em">Match Predictor</div>
        <div class="hero-subtitle">AI-Powered Match Outcome Prediction</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        pa = st.selectbox('Player A', all_players,
                          index=all_players.index('Jannik Sinner') if 'Jannik Sinner' in all_players else 0,
                          key='p_a')
    with c2:
        pb = st.selectbox('Player B', all_players,
                          index=all_players.index('Carlos Alcaraz') if 'Carlos Alcaraz' in all_players else 1,
                          key='p_b')

    st.markdown('')
    predict_btn = st.button('🎾  PREDICT MATCH WINNER', use_container_width=True)

    if predict_btn:
        if pa == pb:
            st.warning('Please select two different players.')
        else:
            prob_a, prob_b = pred(pa, pb, rf_m, FEAT, ps, rm)
            winner = pa if prob_a > prob_b else pb
            win_pct = max(prob_a, prob_b)
            fa = get_flag(pa); fb = get_flag(pb)

            st.markdown('<div class="court-divider"></div>', unsafe_allow_html=True)

            st.markdown(f"""
            <div class="winner-banner">
                <div class="winner-label">🏆 Predicted Winner</div>
                <div class="winner-name">{get_flag(winner)} {winner}</div>
                <div style="color:{USO_GOLD};font-size:1.1em;font-weight:600;margin-top:8px">
                    Win Probability: {win_pct:.1%}
                </div>
            </div>
            """, unsafe_allow_html=True)

            c1, c2, c3 = st.columns([5, 2, 5])
            with c1:
                st.markdown(f'<div class="prob-big" style="color:{USO_BLUE_LITE}">{prob_a:.1%}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="player-name-big" style="color:{USO_BLUE_LITE};font-size:1.6em">{fa} {pa}</div>', unsafe_allow_html=True)
                st.progress(prob_a)
            with c2:
                st.markdown(f'<div style="font-family:Bebas Neue,Impact;font-size:2.5em;text-align:center;color:{USO_WHITE};padding-top:16px">VS</div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="prob-big" style="color:#EF5350">{prob_b:.1%}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="player-name-big" style="color:#EF5350;font-size:1.6em">{fb} {pb}</div>', unsafe_allow_html=True)
                st.progress(prob_b)

            st.markdown('<div class="court-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-header">🔑 Key Factors</div>', unsafe_allow_html=True)

            sa = ps[ps['player']==pa].iloc[0] if len(ps[ps['player']==pa])>0 else None
            sb = ps[ps['player']==pb].iloc[0] if len(ps[ps['player']==pb])>0 else None
            ra = rm.get(pa, 999); rb = rm.get(pb, 999)

            if sa is not None and sb is not None:
                factors = [
                    ('ATP Ranking', f'#{int(ra)}', f'#{int(rb)}', ra < rb),
                    ('Hard Court Win %', f"{sa['hard_win_pct']:.1%}", f"{sb['hard_win_pct']:.1%}", sa['hard_win_pct'] > sb['hard_win_pct']),
                    ('Recent Form', f"{sa['recent_pct']:.1%}", f"{sb['recent_pct']:.1%}", sa['recent_pct'] > sb['recent_pct']),
                    ('BP Save %', f"{sa['avg_bp_save']:.1%}", f"{sb['avg_bp_save']:.1%}", sa['avg_bp_save'] > sb['avg_bp_save']),
                ]
                for factor, va, vb, a_wins in factors:
                    wa = '✅' if a_wins else ''
                    wb = '✅' if not a_wins else ''
                    st.markdown(f"""
                    <div class="stat-row">
                        <div class="stat-val-a">{wa} <b>{va}</b></div>
                        <div class="stat-label-center">{factor}</div>
                        <div class="stat-val-b"><b>{vb}</b> {wb}</div>
                    </div>
                    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# TOURNAMENT PREDICTION
# ════════════════════════════════════════════════════════════════
elif '🏆' in page:
    st.markdown(f"""
    <div class="hero-banner" style="padding:32px">
        <div class="hero-title" style="font-size:2.5em">Tournament Prediction</div>
        <div class="hero-subtitle">US Open 2026 — Round-by-Round Probabilities</div>
    </div>
    """, unsafe_allow_html=True)

    top10 = tourney.head(10).reset_index(drop=True)
    medals = ['🥇','🥈','🥉','4️⃣','5️⃣','6️⃣','7️⃣','8️⃣','9️⃣','🔟']

    st.markdown('<div class="section-header">🏆 Top 10 Contenders</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        for i, row in top10.iterrows():
            flag = get_flag(row['player'])
            st.markdown(f"""
            <div class="player-card">
                <div class="player-rank">{medals[i]}</div>
                <div class="player-flag">{flag}</div>
                <div class="player-name">{row['player']}<br>
                    <span style="font-size:0.8em;color:{USO_GOLD};font-weight:400">
                    Final: {row['reach_final']:.1%} &nbsp;|&nbsp;
                    Semis: {row['reach_semis']:.1%} &nbsp;|&nbsp;
                    QF: {row['reach_qtrs']:.1%}
                    </span>
                </div>
                <div class="player-prob">{row['win_title']:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

    with c2:
        fig, ax = plt.subplots(figsize=(7, 7))
        rounds = ['reach_qtrs','reach_semis','reach_final','win_title']
        labels = ['Quarter-Finals','Semi-Finals','Final','Title']
        colors = [USO_BLUE_LITE+'80', USO_BLUE_LITE, USO_GREEN, USO_GOLD]
        x = np.arange(len(top10))
        w = 0.2
        for i, (r, label, color) in enumerate(zip(rounds, labels, colors)):
            ax.bar(x+i*w, top10[r], w, label=label, color=color, edgecolor='none')
        ax.set_xticks(x+w*1.5)
        ax.set_xticklabels([p.split()[-1] for p in top10['player']],
                           rotation=35, ha='right', color=USO_WHITE, fontsize=8)
        ax.set_ylabel('Probability', color=USO_WHITE)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.legend(fontsize=8, facecolor=USO_BLUE, labelcolor=USO_WHITE,
                  edgecolor=USO_BLUE_LITE+'40', loc='upper right')
        ax.tick_params(colors=USO_WHITE)
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig = make_chart(fig)
        plt.tight_layout()
        st.pyplot(fig, transparent=True)
        plt.close()

    st.markdown('<div class="court-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📋 Full Predictions Table</div>', unsafe_allow_html=True)

    disp = tourney.copy()
    disp['win_title']   = disp['win_title'].apply(lambda x: f'{x:.1%}')
    disp['reach_final'] = disp['reach_final'].apply(lambda x: f'{x:.1%}')
    disp['reach_semis'] = disp['reach_semis'].apply(lambda x: f'{x:.1%}')
    disp['reach_qtrs']  = disp['reach_qtrs'].apply(lambda x: f'{x:.1%}')
    disp.columns = ['Player','Win Title','Reach Final','Reach Semis','Reach QF']
    disp.index   = range(1, len(disp)+1)
    st.dataframe(disp, use_container_width=True, height=400)


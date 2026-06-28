import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title='US Open 2026',
    page_icon='🎾',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600;700&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: #061525 !important;
    color: #F0F4F8 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0A1F35 !important;
    border-right: 1px solid #1B4F8A30 !important;
    padding-top: 0 !important;
}
section[data-testid="stSidebar"] * { color: #F0F4F8 !important; }
section[data-testid="stSidebar"] .stRadio label {
    font-size: 0.9em !important;
    padding: 10px 0 !important;
    letter-spacing: 0.5px !important;
    cursor: pointer !important;
}
div[data-testid="stSidebarContent"] { padding: 0 16px !important; }

/* Remove streamlit branding */
#MainMenu, footer, header { visibility: hidden !important; }
.block-container { padding-top: 0 !important; padding-bottom: 40px !important; max-width: 1200px !important; }

/* Hero */
.hero {
    background: linear-gradient(135deg, #0A2240 0%, #0D3B6E 40%, #1A5C3A 100%);
    border-radius: 0 0 24px 24px;
    padding: 56px 48px 48px;
    margin-bottom: 40px;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.02'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
    pointer-events: none;
}
.hero-eyebrow {
    font-size: 0.75em;
    font-weight: 600;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #C9A84C;
    margin-bottom: 12px;
}
.hero-title {
    font-family: 'Bebas Neue', 'Impact', sans-serif;
    font-size: 5em;
    letter-spacing: 6px;
    color: #FFFFFF;
    line-height: 0.9;
    text-shadow: 0 4px 24px rgba(0,0,0,0.4);
}
.hero-sub {
    font-size: 1em;
    color: #94A3B8;
    margin-top: 16px;
    font-weight: 300;
    letter-spacing: 1px;
}
.hero-pill {
    display: inline-block;
    background: rgba(201,168,76,0.15);
    border: 1px solid #C9A84C60;
    color: #C9A84C;
    padding: 6px 20px;
    border-radius: 100px;
    font-size: 0.75em;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 20px;
}

/* Cards */
.kpi-card {
    background: #0D2137;
    border: 1px solid #1B4F8A25;
    border-radius: 16px;
    padding: 28px 24px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #1B4F8A, #3A7D44);
}
.kpi-num {
    font-family: 'Bebas Neue', 'Impact', sans-serif;
    font-size: 3em;
    color: #FFFFFF;
    line-height: 1;
}
.kpi-lbl {
    font-size: 0.7em;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 6px;
    font-weight: 600;
}

/* Section title */
.sec-title {
    font-family: 'Bebas Neue', 'Impact', sans-serif;
    font-size: 1.4em;
    letter-spacing: 3px;
    color: #FFFFFF;
    text-transform: uppercase;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.sec-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #1B4F8A40, transparent);
}

/* Player rows */
.p-row {
    display: flex;
    align-items: center;
    gap: 14px;
    background: #0D2137;
    border: 1px solid #1B4F8A20;
    border-radius: 12px;
    padding: 14px 20px;
    margin-bottom: 8px;
    transition: border-color 0.2s;
}
.p-row:hover { border-color: #3A7D4460; }
.p-pos {
    font-family: 'Bebas Neue', 'Impact', sans-serif;
    font-size: 1.8em;
    color: #C9A84C;
    min-width: 36px;
}
.p-flag { font-size: 1.3em; }
.p-name { font-size: 0.95em; font-weight: 600; color: #F0F4F8; flex: 1; }
.p-pct {
    font-family: 'Bebas Neue', 'Impact', sans-serif;
    font-size: 1.5em;
    color: #3A7D44;
}

/* Divider */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1B4F8A40, transparent);
    margin: 32px 0;
}

/* Compare */
.cmp-name {
    font-family: 'Bebas Neue', 'Impact', sans-serif;
    font-size: 2.2em;
    letter-spacing: 2px;
    text-align: center;
}
.cmp-rank {
    text-align: center;
    color: #C9A84C;
    font-weight: 600;
    font-size: 0.85em;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 4px;
}
.stat-line {
    display: flex;
    align-items: center;
    padding: 13px 0;
    border-bottom: 1px solid #1B4F8A15;
}
.sv-a { flex: 1; text-align: right; padding-right: 20px; font-weight: 700; font-size: 1.05em; color: #4A90D9; }
.sv-lbl { min-width: 180px; text-align: center; font-size: 0.72em; color: #64748B; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }
.sv-b { flex: 1; text-align: left; padding-left: 20px; font-weight: 700; font-size: 1.05em; color: #EF5350; }

/* Winner */
.winner-box {
    background: linear-gradient(135deg, #1A3D2B 0%, #0D2137 100%);
    border: 1px solid #3A7D4460;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    margin: 24px 0;
}
.winner-tag { font-size: 0.7em; letter-spacing: 4px; color: #C9A84C; text-transform: uppercase; font-weight: 600; }
.winner-nm {
    font-family: 'Bebas Neue', 'Impact', sans-serif;
    font-size: 3.2em;
    letter-spacing: 3px;
    color: #FFFFFF;
    margin: 8px 0 4px;
}
.winner-prob { font-size: 0.9em; color: #5CB85C; font-weight: 600; letter-spacing: 1px; }

/* Probability display */
.prob-display {
    font-family: 'Bebas Neue', 'Impact', sans-serif;
    font-size: 3.5em;
    text-align: center;
    line-height: 1;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #1A5C3A 0%, #0D3B26 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9em !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    padding: 14px 32px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(58,125,68,0.35) !important;
}

/* Selectbox */
div[data-testid="stSelectbox"] label {
    color: #94A3B8 !important;
    font-size: 0.75em !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
}

/* Table */
.stDataFrame { border-radius: 12px !important; overflow: hidden !important; }

/* Sidebar nav logo */
.sb-logo {
    padding: 32px 0 8px;
    text-align: center;
    font-family: 'Bebas Neue', 'Impact', sans-serif;
    font-size: 1.8em;
    letter-spacing: 4px;
    color: #FFFFFF;
}
.sb-year {
    text-align: center;
    font-size: 0.7em;
    color: #C9A84C;
    letter-spacing: 3px;
    margin-bottom: 28px;
}
.sb-line {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1B4F8A80, transparent);
    margin: 0 0 20px;
}
</style>
""", unsafe_allow_html=True)


# ── Data loading ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        atp     = pd.read_csv('atp_matches_clean.csv')
        uso     = pd.read_csv('uso_matches_clean.csv')
        ml      = pd.read_csv('ml_model_features.csv')
        tourney = pd.read_csv('tournament_predictions_uso2026.csv')
        return atp, uso, ml, tourney
    except FileNotFoundError as e:
        st.error(f'Missing file: {e}')
        st.stop()

@st.cache_data
def compute_stats(_atp, _uso):
    hard = _atp[_atp['surface'] == 'Hard'].copy()
    for col in ['w_ace','w_df','w_svpt','w_1stIn','w_bpSaved','w_bpFaced','l_bpSaved','l_bpFaced']:
        if col in hard.columns:
            hard[col] = pd.to_numeric(hard[col], errors='coerce')
    hard['w_bp_save'] = np.where(hard['w_bpFaced']>0, hard['w_bpSaved']/hard['w_bpFaced'], np.nan)
    hard['w_bp_conv'] = np.where(hard['l_bpFaced']>0, (hard['l_bpFaced']-hard['l_bpSaved'])/hard['l_bpFaced'], np.nan)
    hard['w_1st']     = np.where(hard['w_svpt']>0, hard['w_1stIn']/hard['w_svpt'], np.nan)
    hw = hard.groupby('winner_name').size().rename('hw')
    hl = hard.groupby('loser_name').size().rename('hl')
    hs = pd.concat([hw, hl], axis=1).fillna(0)
    hs['ht'] = hs['hw'] + hs['hl']
    hs['hwp'] = hs['hw'] / hs['ht']
    hs = hs.reset_index().rename(columns={'index':'player'})
    hs = hs[hs['ht'] >= 20]
    sv = hard.groupby('winner_name').agg(
        aces=('w_ace','mean'), df=('w_df','mean'),
        bps=('w_bp_save','mean'), bpc=('w_bp_conv','mean'), fsp=('w_1st','mean')
    ).reset_index().rename(columns={'winner_name':'player'})
    uw = _uso.groupby('winner_name').size().rename('uw')
    ul = _uso.groupby('loser_name').size().rename('ul')
    us = pd.concat([uw, ul], axis=1).fillna(0)
    us['ut']  = us['uw'] + us['ul']
    us['uwp'] = us['uw'] / us['ut']
    us = us.reset_index().rename(columns={'index':'player'})
    wn = _atp[['tourney_date','winner_name']].rename(columns={'winner_name':'player'}); wn['r'] = 1
    ln = _atp[['tourney_date','loser_name']].rename(columns={'loser_name':'player'});   ln['r'] = 0
    am = pd.concat([wn, ln], ignore_index=True)
    am['tourney_date'] = pd.to_datetime(am['tourney_date'], errors='coerce')
    am = am.sort_values(['player','tourney_date'])
    tm = am.groupby('player').size().reset_index(name='tm')
    rf = (am.groupby('player').tail(10).groupby('player')['r']
          .agg(rw='sum', rt='count').reset_index())
    rf = rf.merge(tm, on='player')
    rf = rf[rf['tm'] >= 50]
    rf['rfp'] = rf['rw'] / rf['rt']
    ps = hs.merge(sv, on='player', how='left')
    ps = ps.merge(rf[['player','rfp']], on='player', how='left')
    ps = ps.merge(us[['player','uw','uwp']], on='player', how='left')
    ps['uwp'] = ps['uwp'].fillna(0.5)
    ps['rfp'] = ps['rfp'].fillna(0.5)
    extras = pd.DataFrame([
        {'player':'Joao Fonseca','hw':27,'hl':17,'ht':44,'hwp':0.614,'aces':6.0,'df':0.17,'bps':0.665,'bpc':0.38,'fsp':0.64,'rfp':0.60,'uw':1,'uwp':0.50},
        {'player':'Jakub Mensik','hw':21,'hl':10,'ht':31,'hwp':0.677,'aces':7.2,'df':0.20,'bps':0.640,'bpc':0.41,'fsp':0.63,'rfp':0.65,'uw':0,'uwp':0.50},
        {'player':'Rafael Jodar','hw':18,'hl':9,'ht':27,'hwp':0.667,'aces':5.5,'df':0.18,'bps':0.650,'bpc':0.39,'fsp':0.62,'rfp':0.60,'uw':0,'uwp':0.50},
    ])
    return pd.concat([ps, extras], ignore_index=True)

@st.cache_resource
def train_model(_atp, _ps):
    hard = _atp[_atp['surface'] == 'Hard'].copy()
    for col in ['w_ace','w_df','w_svpt','w_1stIn','w_bpSaved','w_bpFaced',
                'l_bpSaved','l_bpFaced','winner_rank','loser_rank']:
        if col in hard.columns:
            hard[col] = pd.to_numeric(hard[col], errors='coerce')
    hard['w_bp_save'] = np.where(hard['w_bpFaced']>0, hard['w_bpSaved']/hard['w_bpFaced'], np.nan)
    hard['w_bp_conv'] = np.where(hard['l_bpFaced']>0, (hard['l_bpFaced']-hard['l_bpSaved'])/hard['l_bpFaced'], np.nan)
    hard['rank_diff'] = hard['loser_rank'] - hard['winner_rank']
    md = hard[['winner_name','loser_name','rank_diff']].dropna()
    for side, prefix in [('winner_name','w'), ('loser_name','l')]:
        md = md.merge(_ps[['player','hwp','aces','df','bps','bpc']],
                      left_on=side, right_on='player', how='left').drop(columns='player')
        md = md.rename(columns={'hwp':f'{prefix}wp','aces':f'{prefix}ac',
                                 'df':f'{prefix}df','bps':f'{prefix}bs','bpc':f'{prefix}bc'})
    for c in ['wp','ac','df','bs','bc']:
        md[f'd_{c}'] = md[f'w{c}'] - md[f'l{c}']
    md['label'] = 1
    mir = md.copy()
    for c in ['rank_diff','d_wp','d_ac','d_df','d_bs','d_bc']:
        mir[c] = -mir[c]
    mir['label'] = 0
    tr = pd.concat([md, mir], ignore_index=True)
    F = ['rank_diff','d_wp','d_ac','d_df','d_bs','d_bc']
    X = tr[F].fillna(0); y = tr['label']
    Xtr,_,ytr,_ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
    rf.fit(Xtr, ytr)
    return rf, F

def predict(pa, pb, rf, F, ps, rm):
    sa = ps[ps['player']==pa]; sb = ps[ps['player']==pb]
    if len(sa)==0 or len(sb)==0: return 0.5, 0.5
    sa, sb = sa.iloc[0], sb.iloc[0]
    ra = rm.get(pa, 50); rb = rm.get(pb, 50)
    f = pd.DataFrame([{'rank_diff':rb-ra,'d_wp':sa['hwp']-sb['hwp'],
                        'd_ac':sa['aces']-sb['aces'],'d_df':sa['df']-sb['df'],
                        'd_bs':sa['bps']-sb['bps'],'d_bc':sa['bpc']-sb['bpc']}])[F].fillna(0)
    p = rf.predict_proba(f)[0][1]
    return p, 1-p

# ── Init ─────────────────────────────────────────────────────────
atp, uso, ml, tourney = load_data()

for df in [atp, uso]:
    if 'tourney_date' in df.columns:
        df['tourney_date'] = pd.to_datetime(df['tourney_date'], errors='coerce')
    if 'tourney_year' not in df.columns:
        df['tourney_year'] = df['tourney_date'].dt.year
    if 'surface' in df.columns:
        df['surface'] = df['surface'].str.strip().str.title()

ps = compute_stats(atp, uso)
rf_m, F = train_model(atp, ps)

rm = ml.set_index('full_name')['atp_rank'].to_dict() if 'full_name' in ml.columns and 'atp_rank' in ml.columns else {}

if 'full_name' in ml.columns:
    all_players = sorted([p for p in ml['full_name'].tolist() if p in ps['player'].values])
else:
    all_players = sorted(ps[ps['ht'] >= 50]['player'].tolist())

FLAGS = {
    'ITA':'🇮🇹','ESP':'🇪🇸','GER':'🇩🇪','SRB':'🇷🇸','USA':'🇺🇸','CAN':'🇨🇦',
    'AUS':'🇦🇺','RUS':'🇷🇺','GBR':'🇬🇧','FRA':'🇫🇷','NOR':'🇳🇴','DEN':'🇩🇰',
    'GRE':'🇬🇷','ARG':'🇦🇷','CHI':'🇨🇱','KAZ':'🇰🇿','BRA':'🇧🇷','PER':'🇵🇪',
    'MON':'🇲🇨','CZE':'🇨🇿','POL':'🇵🇱','BEL':'🇧🇪','NED':'🇳🇱','HUN':'🇭🇺',
    'SUI':'🇨🇭','CRO':'🇭🇷','BUL':'🇧🇬','TPE':'🇹🇼','JPN':'🇯🇵','KOR':'🇰🇷',
}
def flag(p):
    if 'full_name' in ml.columns and 'country' in ml.columns:
        r = ml[ml['full_name']==p]
        if len(r)>0: return FLAGS.get(r.iloc[0]['country'], '🎾')
    return '🎾'

def mk_chart():
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')
    ax.tick_params(colors='#94A3B8', labelsize=9)
    ax.xaxis.label.set_color('#94A3B8')
    ax.yaxis.label.set_color('#94A3B8')
    for sp in ax.spines.values(): sp.set_visible(False)
    return fig, ax

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-logo">🎾 US OPEN</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-year">2026</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-line"></div>', unsafe_allow_html=True)
    page = st.radio('Navigation', ['Home', 'Player Comparison', 'Match Predictor', 'Tournament Prediction'],
                    label_visibility='collapsed')

# ════════════════════════════════════
# HOME
# ════════════════════════════════════
if page == 'Home':
    st.markdown("""
    <div class="hero">
        <div class="hero-eyebrow">Analytics & Prediction Platform</div>
        <div class="hero-title">US OPEN<br>2026</div>
        <div class="hero-sub">Machine learning predictions for the US Open draw</div>
        <div class="hero-pill">Flushing Meadows · New York</div>
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    for col, num, lbl in zip(
        [c1,c2,c3,c4],
        [f'{len(atp):,}', f'{len(uso):,}', '10', '128'],
        ['Matches Analyzed','US Open Matches','Years of Data','Draw Size']
    ):
        with col:
            st.markdown(f'<div class="kpi-card"><div class="kpi-num">{num}</div><div class="kpi-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([1,1])
    top10 = tourney.head(10).reset_index(drop=True)
    medals = ['1','2','3','4','5','6','7','8','9','10']

    with col_l:
        st.markdown('<div class="sec-title">Top 10 Contenders</div>', unsafe_allow_html=True)
        for i, row in top10.iterrows():
            st.markdown(f"""
            <div class="p-row">
                <div class="p-pos">{medals[i]}</div>
                <div class="p-flag">{flag(row['player'])}</div>
                <div class="p-name">{row['player']}</div>
                <div class="p-pct">{row['win_title']:.1%}</div>
            </div>""", unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="sec-title">Title Probability</div>', unsafe_allow_html=True)
        fig, ax = mk_chart()
        top10_s = top10.sort_values('win_title', ascending=True)
        bar_colors = ['#C9A84C' if i==len(top10_s)-1 else '#3A7D44' if i>=len(top10_s)-3 else '#1B4F8A'
                      for i in range(len(top10_s))]
        bars = ax.barh(top10_s['player'], top10_s['win_title'], color=bar_colors, edgecolor='none', height=0.6)
        ax.set_xlim(0, top10_s['win_title'].max() * 1.3)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        for bar, v in zip(bars, top10_s['win_title']):
            ax.text(bar.get_width()+0.004, bar.get_y()+bar.get_height()/2,
                    f'{v:.1%}', va='center', color='#F0F4F8', fontsize=9, fontweight='600')
        plt.tight_layout()
        st.pyplot(fig, transparent=True)
        plt.close()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Key Stats</div>', unsafe_allow_html=True)
    f1,f2,f3 = st.columns(3)
    bh = ps.nlargest(1,'hwp').iloc[0]
    bf = ps.nlargest(1,'rfp').iloc[0]
    bu = ps[ps['uw']>3].nlargest(1,'uwp').iloc[0] if 'uw' in ps.columns and len(ps[ps['uw']>3])>0 else ps.nlargest(1,'uwp').iloc[0]
    with f1: st.metric('Best Hard Court %', f"{bh['hwp']:.1%}", bh['player'])
    with f2: st.metric('Best Recent Form', f"{bf['rfp']:.1%}", bf['player'])
    with f3: st.metric('Best US Open %', f"{bu['uwp']:.1%}", bu['player'])

# ════════════════════════════════════
# PLAYER COMPARISON
# ════════════════════════════════════
elif page == 'Player Comparison':
    st.markdown("""
    <div class="hero" style="padding:40px 48px">
        <div class="hero-eyebrow">Head to Head</div>
        <div class="hero-title" style="font-size:3.5em">Player Comparison</div>
    </div>""", unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        pa = st.selectbox('Player A', all_players,
                          index=all_players.index('Jannik Sinner') if 'Jannik Sinner' in all_players else 0)
    with c2:
        pb = st.selectbox('Player B', all_players,
                          index=all_players.index('Carlos Alcaraz') if 'Carlos Alcaraz' in all_players else 1)

    if pa and pb and pa != pb:
        sa_df = ps[ps['player']==pa]; sb_df = ps[ps['player']==pb]
        if len(sa_df)>0 and len(sb_df)>0:
            sa = sa_df.iloc[0]; sb = sb_df.iloc[0]
            ra = int(rm.get(pa,999)); rb = int(rm.get(pb,999))
            fa_f = flag(pa); fb_f = flag(pb)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            c1,c2,c3 = st.columns([5,2,5])
            with c1:
                st.markdown(f'<div class="cmp-name" style="color:#4A90D9">{fa_f} {pa}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="cmp-rank">ATP #{ra}</div>', unsafe_allow_html=True)
            with c2:
                st.markdown('<div style="font-family:Bebas Neue,Impact,sans-serif;font-size:2.8em;text-align:center;color:#475569;padding-top:12px">VS</div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="cmp-name" style="color:#EF5350">{fb_f} {pb}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="cmp-rank">ATP #{rb}</div>', unsafe_allow_html=True)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            stats = [
                ('Hard Court Win %', f"{sa['hwp']:.1%}", f"{sb['hwp']:.1%}", sa['hwp']>sb['hwp']),
                ('Recent Form', f"{sa['rfp']:.1%}", f"{sb['rfp']:.1%}", sa['rfp']>sb['rfp']),
                ('US Open Win %', f"{sa['uwp']:.1%}", f"{sb['uwp']:.1%}", sa['uwp']>sb['uwp']),
                ('Avg Aces / Match', f"{sa['aces']:.1f}", f"{sb['aces']:.1f}", sa['aces']>sb['aces']),
                ('Avg Double Faults', f"{sa['df']:.1f}", f"{sb['df']:.1f}", sa['df']<sb['df']),
                ('BP Save %', f"{sa['bps']:.1%}", f"{sb['bps']:.1%}", sa['bps']>sb['bps']),
                ('BP Conversion %', f"{sa['bpc']:.1%}", f"{sb['bpc']:.1%}", sa['bpc']>sb['bpc']),
                ('1st Serve %', f"{sa.get('fsp',0):.1%}", f"{sb.get('fsp',0):.1%}", sa.get('fsp',0)>sb.get('fsp',0)),
            ]
            for stat, va, vb, a_better in stats:
                wa = '✓' if a_better else ''
                wb = '✓' if not a_better else ''
                st.markdown(f"""
                <div class="stat-line">
                    <div class="sv-a" style="{'color:#5CB85C' if a_better else ''}">{wa} {va}</div>
                    <div class="sv-lbl">{stat}</div>
                    <div class="sv-b" style="{'color:#5CB85C' if not a_better else ''}">{vb} {wb}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="sec-title">Head-to-Head Record</div>', unsafe_allow_html=True)
            h2h_a = len(atp[(atp['winner_name']==pa)&(atp['loser_name']==pb)])
            h2h_b = len(atp[(atp['winner_name']==pb)&(atp['loser_name']==pa)])
            c1,c2,c3 = st.columns(3)
            with c1: st.markdown(f'<div class="kpi-card"><div class="kpi-num" style="color:#4A90D9">{h2h_a}</div><div class="kpi-lbl">{pa.split()[-1]} wins</div></div>', unsafe_allow_html=True)
            with c2: st.markdown(f'<div class="kpi-card"><div class="kpi-num">{h2h_a+h2h_b}</div><div class="kpi-lbl">Total Matches</div></div>', unsafe_allow_html=True)
            with c3: st.markdown(f'<div class="kpi-card"><div class="kpi-num" style="color:#EF5350">{h2h_b}</div><div class="kpi-lbl">{pb.split()[-1]} wins</div></div>', unsafe_allow_html=True)

# ════════════════════════════════════
# MATCH PREDICTOR
# ════════════════════════════════════
elif page == 'Match Predictor':
    st.markdown("""
    <div class="hero" style="padding:40px 48px">
        <div class="hero-eyebrow">AI Prediction</div>
        <div class="hero-title" style="font-size:3.5em">Match Predictor</div>
        <div class="hero-sub">Select two players to predict the match outcome</div>
    </div>""", unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        pa = st.selectbox('Player A', all_players,
                          index=all_players.index('Jannik Sinner') if 'Jannik Sinner' in all_players else 0, key='mp_a')
    with c2:
        pb = st.selectbox('Player B', all_players,
                          index=all_players.index('Carlos Alcaraz') if 'Carlos Alcaraz' in all_players else 1, key='mp_b')
    st.markdown('')
    btn = st.button('Predict Match Winner', use_container_width=True)

    if btn:
        if pa == pb:
            st.warning('Please select two different players.')
        else:
            prob_a, prob_b = predict(pa, pb, rf_m, F, ps, rm)
            winner = pa if prob_a > prob_b else pb
            fa_f = flag(pa); fb_f = flag(pb)

            st.markdown(f"""
            <div class="winner-box">
                <div class="winner-tag">🏆 Predicted Winner</div>
                <div class="winner-nm">{flag(winner)} {winner}</div>
                <div class="winner-prob">Win probability: {max(prob_a,prob_b):.1%}</div>
            </div>""", unsafe_allow_html=True)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            c1,c2,c3 = st.columns([5,2,5])
            with c1:
                st.markdown(f'<div class="prob-display" style="color:#4A90D9">{prob_a:.1%}</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="text-align:center;font-weight:600;color:#94A3B8;margin-top:8px">{fa_f} {pa}</div>', unsafe_allow_html=True)
                st.progress(prob_a)
            with c2:
                st.markdown('<div style="font-family:Bebas Neue,Impact,sans-serif;font-size:2.5em;text-align:center;color:#334155;padding-top:16px">VS</div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="prob-display" style="color:#EF5350">{prob_b:.1%}</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="text-align:center;font-weight:600;color:#94A3B8;margin-top:8px">{fb_f} {pb}</div>', unsafe_allow_html=True)
                st.progress(prob_b)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="sec-title">Key Factors</div>', unsafe_allow_html=True)
            sa = ps[ps['player']==pa].iloc[0] if len(ps[ps['player']==pa])>0 else None
            sb = ps[ps['player']==pb].iloc[0] if len(ps[ps['player']==pb])>0 else None
            ra = rm.get(pa,999); rb = rm.get(pb,999)
            if sa is not None and sb is not None:
                factors = [
                    ('ATP Ranking', f'#{int(ra)}', f'#{int(rb)}', ra<rb),
                    ('Hard Court Win %', f"{sa['hwp']:.1%}", f"{sb['hwp']:.1%}", sa['hwp']>sb['hwp']),
                    ('Recent Form', f"{sa['rfp']:.1%}", f"{sb['rfp']:.1%}", sa['rfp']>sb['rfp']),
                    ('BP Save %', f"{sa['bps']:.1%}", f"{sb['bps']:.1%}", sa['bps']>sb['bps']),
                ]
                for factor, va, vb, a_wins in factors:
                    wa = '✓' if a_wins else ''
                    wb = '✓' if not a_wins else ''
                    st.markdown(f"""
                    <div class="stat-line">
                        <div class="sv-a" style="{'color:#5CB85C' if a_wins else ''}">{wa} {va}</div>
                        <div class="sv-lbl">{factor}</div>
                        <div class="sv-b" style="{'color:#5CB85C' if not a_wins else ''}">{vb} {wb}</div>
                    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════
# TOURNAMENT PREDICTION
# ════════════════════════════════════
elif page == 'Tournament Prediction':
    st.markdown("""
    <div class="hero" style="padding:40px 48px">
        <div class="hero-eyebrow">Simulation · 1,000 runs</div>
        <div class="hero-title" style="font-size:3em">Tournament Prediction</div>
        <div class="hero-sub">Round-by-round probabilities for every player in the draw</div>
    </div>""", unsafe_allow_html=True)

    top10 = tourney.head(10).reset_index(drop=True)
    medals = ['1','2','3','4','5','6','7','8','9','10']

    st.markdown('<div class="sec-title">Top 10 Contenders</div>', unsafe_allow_html=True)
    c1,c2 = st.columns([1,1])

    with c1:
        for i, row in top10.iterrows():
            st.markdown(f"""
            <div class="p-row">
                <div class="p-pos">{medals[i]}</div>
                <div class="p-flag">{flag(row['player'])}</div>
                <div class="p-name">
                    {row['player']}
                    <span style="display:block;font-size:0.75em;color:#64748B;font-weight:400;margin-top:2px">
                    Final {row['reach_final']:.1%} &nbsp;·&nbsp; Semis {row['reach_semis']:.1%} &nbsp;·&nbsp; QF {row['reach_qtrs']:.1%}
                    </span>
                </div>
                <div class="p-pct">{row['win_title']:.1%}</div>
            </div>""", unsafe_allow_html=True)

    with c2:
        fig, ax = mk_chart()
        fig.set_size_inches(7, 7)
        rounds  = ['reach_qtrs','reach_semis','reach_final','win_title']
        labels  = ['Quarter-Finals','Semi-Finals','Final','Title']
        colors  = ['#1B4F8A50','#1B4F8A','#3A7D44','#C9A84C']
        x = np.arange(len(top10)); w = 0.2
        for i,(r,lbl,col) in enumerate(zip(rounds,labels,colors)):
            ax.bar(x+i*w, top10[r], w, label=lbl, color=col, edgecolor='none')
        ax.set_xticks(x+w*1.5)
        ax.set_xticklabels([p.split()[-1] for p in top10['player']],
                           rotation=35, ha='right', color='#94A3B8', fontsize=8)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.tick_params(colors='#94A3B8')
        ax.legend(fontsize=8, facecolor='#0D2137', labelcolor='#94A3B8',
                  edgecolor='#1B4F8A30', loc='upper right')
        plt.tight_layout()
        st.pyplot(fig, transparent=True)
        plt.close()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Full Draw Predictions</div>', unsafe_allow_html=True)
    disp = tourney.copy()
    disp['win_title']   = disp['win_title'].apply(lambda x: f'{x:.1%}')
    disp['reach_final'] = disp['reach_final'].apply(lambda x: f'{x:.1%}')
    disp['reach_semis'] = disp['reach_semis'].apply(lambda x: f'{x:.1%}')
    disp['reach_qtrs']  = disp['reach_qtrs'].apply(lambda x: f'{x:.1%}')
    disp.columns = ['Player','Win Title','Reach Final','Reach Semis','Reach QF']
    disp.index = range(1, len(disp)+1)
    st.dataframe(disp, use_container_width=True, height=420)

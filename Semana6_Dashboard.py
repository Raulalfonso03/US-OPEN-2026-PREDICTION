import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title='US Open 2026',
    page_icon='🎾',
    layout='wide',
    initial_sidebar_state='collapsed'
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background: #050E1A !important;
    color: #E2E8F0 !important;
}

#MainMenu, footer, header { visibility: hidden !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { display: none !important; }

/* ── NAVBAR ── */
.navbar {
    position: sticky;
    top: 0;
    z-index: 999;
    background: rgba(5,14,26,0.96);
    backdrop-filter: blur(20px);
    border-bottom: 1px solid rgba(255,255,255,0.05);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 64px;
    height: 72px;
}
.nav-logo {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.6em;
    letter-spacing: 5px;
    color: #fff;
    display: flex;
    align-items: center;
    gap: 12px;
}
.nav-live {
    width: 8px; height: 8px;
    background: #22C55E;
    border-radius: 50%;
    box-shadow: 0 0 0 0 rgba(34,197,94,0.4);
    animation: ripple 2s infinite;
}
@keyframes ripple {
    0%   { box-shadow: 0 0 0 0 rgba(34,197,94,0.4); }
    70%  { box-shadow: 0 0 0 8px rgba(34,197,94,0); }
    100% { box-shadow: 0 0 0 0 rgba(34,197,94,0); }
}
.nav-city {
    font-size: 0.75em;
    color: #475569;
    font-weight: 500;
    letter-spacing: 2px;
}

/* ── HERO ── */
.hero {
    min-height: 520px;
    background:
        radial-gradient(ellipse at 80% 50%, rgba(34,197,94,0.07) 0%, transparent 60%),
        radial-gradient(ellipse at 20% 80%, rgba(14,116,144,0.07) 0%, transparent 60%),
        linear-gradient(160deg, #061525 0%, #050E1A 100%);
    padding: 96px 80px 80px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.18);
    color: #22C55E;
    padding: 7px 18px;
    border-radius: 100px;
    font-size: 0.72em;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 28px;
    width: fit-content;
}
.hero-h1 {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 8em;
    letter-spacing: 10px;
    color: #fff;
    line-height: 0.85;
    margin-bottom: 28px;
}
.hero-h1 em { color: #22C55E; font-style: normal; }
.hero-desc {
    font-size: 1.05em;
    color: #64748B;
    max-width: 440px;
    line-height: 1.8;
    font-weight: 400;
    margin-bottom: 56px;
}
.hero-numbers {
    display: flex;
    gap: 56px;
    padding-top: 32px;
    border-top: 1px solid rgba(255,255,255,0.05);
}
.hero-num { }
.hero-num-val {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.4em;
    color: #fff;
    line-height: 1;
}
.hero-num-lbl {
    font-size: 0.7em;
    color: #334155;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 700;
    margin-top: 6px;
}

/* ── PAGE CONTENT ── */
.page { padding: 72px 80px; }
.page-header {
    margin-bottom: 48px;
}
.page-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.8em;
    letter-spacing: 4px;
    color: #fff;
    line-height: 1;
    margin-bottom: 8px;
}
.page-desc { font-size: 0.9em; color: #475569; font-weight: 500; }

/* ── CONTENDER GRID ── */
.c-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 20px;
    margin-top: 8px;
}
.c-card {
    background: #0A1929;
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 20px;
    padding: 28px 24px;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, border-color 0.2s;
    cursor: default;
}
.c-card:hover {
    transform: translateY(-3px);
    border-color: rgba(34,197,94,0.15);
}
.c-accent {
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.c-rank {
    font-size: 0.68em;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #334155;
    margin-bottom: 16px;
}
.c-flag { font-size: 2em; display: block; margin-bottom: 10px; }
.c-name {
    font-size: 0.9em;
    font-weight: 700;
    color: #F1F5F9;
    line-height: 1.3;
    margin-bottom: 20px;
    min-height: 36px;
}
.c-pct {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.6em;
    color: #22C55E;
    line-height: 1;
}
.c-pct-lbl {
    font-size: 0.65em;
    color: #334155;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 700;
    margin-top: 3px;
}
.c-bar {
    height: 3px;
    background: rgba(255,255,255,0.04);
    border-radius: 2px;
    margin-top: 20px;
    overflow: hidden;
}
.c-bar-fill {
    height: 100%;
    border-radius: 2px;
    background: linear-gradient(90deg, #22C55E 0%, #16A34A 100%);
}

/* ── COMPARISON ── */
.cmp-box {
    background: #0A1929;
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 20px;
    overflow: hidden;
    margin-top: 32px;
}
.cmp-top {
    display: grid;
    grid-template-columns: 1fr 120px 1fr;
    align-items: center;
    padding: 48px;
    background: linear-gradient(135deg, #07192B, #0D2137);
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.cmp-p { text-align: center; }
.cmp-p-flag { font-size: 2.4em; margin-bottom: 12px; }
.cmp-p-name {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.2em;
    letter-spacing: 2px;
    line-height: 1;
    margin-bottom: 8px;
}
.cmp-p-rank {
    font-size: 0.72em;
    color: #C9A84C;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.cmp-vs {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2em;
    color: #1E293B;
    text-align: center;
}
.stat-row {
    display: grid;
    grid-template-columns: 1fr 200px 1fr;
    align-items: center;
    padding: 20px 48px;
    border-bottom: 1px solid rgba(255,255,255,0.025);
    transition: background 0.15s;
}
.stat-row:last-child { border-bottom: none; }
.stat-row:hover { background: rgba(255,255,255,0.015); }
.stat-v {
    font-size: 1.05em;
    font-weight: 700;
}
.stat-v-l { text-align: right; padding-right: 24px; }
.stat-v-r { text-align: left;  padding-left: 24px; }
.stat-n {
    font-size: 0.7em;
    color: #334155;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 700;
    text-align: center;
}
.s-win  { color: #22C55E !important; }
.s-lose { color: #1E293B !important; }

/* ── PREDICTOR ── */
.pred-winner-box {
    background: linear-gradient(135deg, #041910 0%, #071F2A 50%, #050E1A 100%);
    border: 1px solid rgba(34,197,94,0.12);
    border-radius: 20px;
    padding: 56px 48px;
    text-align: center;
    margin: 32px 0;
}
.pred-tag {
    font-size: 0.68em;
    color: #22C55E;
    letter-spacing: 4px;
    font-weight: 700;
    text-transform: uppercase;
    margin-bottom: 20px;
}
.pred-name {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 5em;
    letter-spacing: 5px;
    color: #fff;
    line-height: 1;
    margin-bottom: 14px;
}
.pred-prob-lbl {
    font-size: 0.85em;
    color: #22C55E;
    font-weight: 600;
    letter-spacing: 1px;
}
.prob-split {
    display: grid;
    grid-template-columns: 1fr 80px 1fr;
    align-items: center;
    gap: 0;
    margin: 40px 0;
}
.prob-side { text-align: center; padding: 0 32px; }
.prob-big {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 5em;
    line-height: 1;
    margin-bottom: 8px;
}
.prob-player-name {
    font-size: 0.85em;
    color: #64748B;
    font-weight: 600;
}
.prob-bar-wrap {
    height: 4px;
    background: rgba(255,255,255,0.04);
    border-radius: 2px;
    margin-top: 14px;
    overflow: hidden;
}
.prob-bar-fill { height: 100%; border-radius: 2px; }
.prob-sep {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.6em;
    color: #1E293B;
    text-align: center;
}
.factors-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-top: 32px;
}
.factor-card {
    background: #0A1929;
    border: 1px solid rgba(255,255,255,0.04);
    border-radius: 14px;
    padding: 22px 18px;
    text-align: center;
}
.factor-title {
    font-size: 0.65em;
    color: #334155;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 700;
    margin-bottom: 14px;
}
.factor-vals {
    display: flex;
    justify-content: space-around;
    align-items: center;
    gap: 8px;
}
.factor-val { font-size: 1em; font-weight: 700; }
.factor-dot { font-size: 0.6em; color: #1E293B; }

/* ── TOURNAMENT TABLE ── */
.t-table {
    background: #0A1929;
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 20px;
    overflow: hidden;
}
.t-head {
    display: grid;
    grid-template-columns: 60px 1fr 130px 130px 130px 130px;
    padding: 18px 32px;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    background: rgba(255,255,255,0.02);
}
.t-th {
    font-size: 0.65em;
    color: #334155;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 700;
    text-align: center;
}
.t-th:nth-child(2) { text-align: left; }
.t-row {
    display: grid;
    grid-template-columns: 60px 1fr 130px 130px 130px 130px;
    padding: 18px 32px;
    border-bottom: 1px solid rgba(255,255,255,0.025);
    align-items: center;
    transition: background 0.15s;
}
.t-row:last-child { border-bottom: none; }
.t-row:hover { background: rgba(255,255,255,0.015); }
.t-pos {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.2em;
    color: #1E293B;
    text-align: center;
}
.t-pos.hi { color: #C9A84C; }
.t-player { display: flex; align-items: center; gap: 12px; }
.t-flag { font-size: 1.2em; }
.t-name { font-size: 0.88em; font-weight: 600; color: #E2E8F0; }
.t-val {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.15em;
    text-align: center;
}
.t-hi  { color: #22C55E; }
.t-mid { color: #64748B; }
.t-lo  { color: #1E293B; }

/* ── INPUTS ── */
div[data-testid="stSelectbox"] label {
    font-size: 0.7em !important;
    font-weight: 700 !important;
    color: #334155 !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
}
div[data-testid="stSelectbox"] > div > div {
    background: #0A1929 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
    color: #E2E8F0 !important;
    font-size: 0.95em !important;
}

/* ── BUTTON ── */
.stButton > button {
    background: #22C55E !important;
    color: #050E1A !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 800 !important;
    font-size: 0.82em !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    padding: 16px 40px !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #16A34A !important;
    box-shadow: 0 8px 32px rgba(34,197,94,0.2) !important;
    transform: translateY(-1px) !important;
}

/* ── DIVIDER ── */
.div { height: 1px; background: rgba(255,255,255,0.04); margin: 0 80px; }

/* ── SECTION LABEL ── */
.sec-label {
    font-size: 0.68em;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #22C55E;
    margin-bottom: 10px;
}
.sec-h2 {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.2em;
    letter-spacing: 4px;
    color: #fff;
    margin-bottom: 6px;
}
.sec-sub { font-size: 0.85em; color: #475569; }
</style>
""", unsafe_allow_html=True)

# ── DATA ─────────────────────────────────────────────────────────
@st.cache_data
def load():
    try:
        atp = pd.read_csv('atp_matches_clean.csv')
        uso = pd.read_csv('uso_matches_clean.csv')
        ml  = pd.read_csv('ml_model_features.csv')
        t   = pd.read_csv('tournament_predictions_uso2026.csv')
        return atp, uso, ml, t
    except FileNotFoundError as e:
        st.error(f'⚠️  Missing file: {e}')
        st.stop()

@st.cache_data
def calc_stats(_atp, _uso):
    hard = _atp[_atp['surface']=='Hard'].copy()
    for c in ['w_ace','w_df','w_svpt','w_1stIn','w_bpSaved','w_bpFaced','l_bpSaved','l_bpFaced']:
        if c in hard.columns: hard[c]=pd.to_numeric(hard[c],errors='coerce')
    hard['bps']=np.where(hard['w_bpFaced']>0,hard['w_bpSaved']/hard['w_bpFaced'],np.nan)
    hard['bpc']=np.where(hard['l_bpFaced']>0,(hard['l_bpFaced']-hard['l_bpSaved'])/hard['l_bpFaced'],np.nan)
    hard['fsp']=np.where(hard['w_svpt']>0,hard['w_1stIn']/hard['w_svpt'],np.nan)
    hw=hard.groupby('winner_name').size().rename('hw')
    hl=hard.groupby('loser_name').size().rename('hl')
    hs=pd.concat([hw,hl],axis=1).fillna(0)
    hs['ht']=hs['hw']+hs['hl']; hs['hwp']=hs['hw']/hs['ht']
    hs=hs.reset_index().rename(columns={'index':'player'}); hs=hs[hs['ht']>=20]
    sv=hard.groupby('winner_name').agg(aces=('w_ace','mean'),df=('w_df','mean'),bps=('bps','mean'),bpc=('bpc','mean'),fsp=('fsp','mean')).reset_index().rename(columns={'winner_name':'player'})
    uw=_uso.groupby('winner_name').size().rename('uw'); ul=_uso.groupby('loser_name').size().rename('ul')
    us=pd.concat([uw,ul],axis=1).fillna(0); us['ut']=us['uw']+us['ul']; us['uwp']=us['uw']/us['ut']
    us=us.reset_index().rename(columns={'index':'player'})
    wn=_atp[['tourney_date','winner_name']].rename(columns={'winner_name':'player'}); wn['r']=1
    ln=_atp[['tourney_date','loser_name']].rename(columns={'loser_name':'player'}); ln['r']=0
    am=pd.concat([wn,ln],ignore_index=True)
    am['tourney_date']=pd.to_datetime(am['tourney_date'],errors='coerce')
    am=am.sort_values(['player','tourney_date'])
    tm=am.groupby('player').size().reset_index(name='tm')
    rf=am.groupby('player').tail(10).groupby('player')['r'].agg(rw='sum',rt='count').reset_index()
    rf=rf.merge(tm,on='player'); rf=rf[rf['tm']>=50]; rf['rfp']=rf['rw']/rf['rt']
    ps=hs.merge(sv,on='player',how='left').merge(rf[['player','rfp']],on='player',how='left').merge(us[['player','uw','uwp']],on='player',how='left')
    ps['uwp']=ps['uwp'].fillna(0.5); ps['rfp']=ps['rfp'].fillna(0.5)
    ex=pd.DataFrame([
        {'player':'Joao Fonseca','hw':27,'hl':17,'ht':44,'hwp':0.614,'aces':6.0,'df':0.17,'bps':0.665,'bpc':0.38,'fsp':0.64,'rfp':0.60,'uw':1,'uwp':0.50},
        {'player':'Jakub Mensik','hw':21,'hl':10,'ht':31,'hwp':0.677,'aces':7.2,'df':0.20,'bps':0.640,'bpc':0.41,'fsp':0.63,'rfp':0.65,'uw':0,'uwp':0.50},
        {'player':'Rafael Jodar','hw':18,'hl':9,'ht':27,'hwp':0.667,'aces':5.5,'df':0.18,'bps':0.650,'bpc':0.39,'fsp':0.62,'rfp':0.60,'uw':0,'uwp':0.50},
    ])
    return pd.concat([ps,ex],ignore_index=True)

@st.cache_resource
def train(_atp, _ps):
    hard=_atp[_atp['surface']=='Hard'].copy()
    for c in ['w_ace','w_df','w_svpt','w_1stIn','w_bpSaved','w_bpFaced','l_bpSaved','l_bpFaced','winner_rank','loser_rank']:
        if c in hard.columns: hard[c]=pd.to_numeric(hard[c],errors='coerce')
    hard['bps']=np.where(hard['w_bpFaced']>0,hard['w_bpSaved']/hard['w_bpFaced'],np.nan)
    hard['bpc']=np.where(hard['l_bpFaced']>0,(hard['l_bpFaced']-hard['l_bpSaved'])/hard['l_bpFaced'],np.nan)
    hard['rank_diff']=hard['loser_rank']-hard['winner_rank']
    md=hard[['winner_name','loser_name','rank_diff']].dropna()
    for side,pfx in[('winner_name','w'),('loser_name','l')]:
        md=md.merge(_ps[['player','hwp','aces','df','bps','bpc']],left_on=side,right_on='player',how='left').drop(columns='player')
        md=md.rename(columns={'hwp':f'{pfx}wp','aces':f'{pfx}ac','df':f'{pfx}df','bps':f'{pfx}bs','bpc':f'{pfx}bc'})
    for c in['wp','ac','df','bs','bc']: md[f'd_{c}']=md[f'w{c}']-md[f'l{c}']
    md['label']=1
    mir=md.copy()
    for c in['rank_diff','d_wp','d_ac','d_df','d_bs','d_bc']: mir[c]=-mir[c]
    mir['label']=0
    tr=pd.concat([md,mir],ignore_index=True)
    F=['rank_diff','d_wp','d_ac','d_df','d_bs','d_bc']
    X=tr[F].fillna(0); y=tr['label']
    Xtr,_,ytr,_=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    rf=RandomForestClassifier(n_estimators=100,max_depth=7,random_state=42)
    rf.fit(Xtr,ytr)
    return rf,F

def predict(pa,pb,rf,F,ps,rm):
    sa=ps[ps['player']==pa]; sb=ps[ps['player']==pb]
    if len(sa)==0 or len(sb)==0: return 0.5,0.5
    sa,sb=sa.iloc[0],sb.iloc[0]
    ra=rm.get(pa,50); rb=rm.get(pb,50)
    f=pd.DataFrame([{'rank_diff':rb-ra,'d_wp':sa['hwp']-sb['hwp'],'d_ac':sa['aces']-sb['aces'],
                      'd_df':sa['df']-sb['df'],'d_bs':sa['bps']-sb['bps'],'d_bc':sa['bpc']-sb['bpc']}])[F].fillna(0)
    p=rf.predict_proba(f)[0][1]
    return p,1-p

# ── INIT ─────────────────────────────────────────────────────────
atp,uso,ml,tourney=load()
for df in[atp,uso]:
    if 'tourney_date' in df.columns: df['tourney_date']=pd.to_datetime(df['tourney_date'],errors='coerce')
    if 'tourney_year' not in df.columns: df['tourney_year']=df['tourney_date'].dt.year
    if 'surface' in df.columns: df['surface']=df['surface'].str.strip().str.title()
ps=calc_stats(atp,uso)
rf_m,F=train(atp,ps)
rm=ml.set_index('full_name')['atp_rank'].to_dict() if 'full_name' in ml.columns and 'atp_rank' in ml.columns else {}
if 'full_name' in ml.columns:
    all_players=sorted([p for p in ml['full_name'].tolist() if p in ps['player'].values])
else:
    all_players=sorted(ps[ps['ht']>=50]['player'].tolist())

FL={'ITA':'🇮🇹','ESP':'🇪🇸','GER':'🇩🇪','SRB':'🇷🇸','USA':'🇺🇸','CAN':'🇨🇦',
    'AUS':'🇦🇺','RUS':'🇷🇺','GBR':'🇬🇧','FRA':'🇫🇷','NOR':'🇳🇴','DEN':'🇩🇰',
    'GRE':'🇬🇷','ARG':'🇦🇷','CHI':'🇨🇱','KAZ':'🇰🇿','BRA':'🇧🇷','PER':'🇵🇪',
    'MON':'🇲🇨','CZE':'🇨🇿','POL':'🇵🇱','BEL':'🇧🇪','NED':'🇳🇱','HUN':'🇭🇺',
    'SUI':'🇨🇭','CRO':'🇭🇷','BUL':'🇧🇬','TPE':'🇹🇼','JPN':'🇯🇵','KOR':'🇰🇷'}
def flg(p):
    return ""

top_max=tourney['win_title'].max()

# ── NAVBAR ───────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
    <div class="nav-logo">
        <div class="nav-live"></div>
        US OPEN 2026
    </div>
    <div class="nav-city">Flushing Meadows · New York</div>
</div>
""", unsafe_allow_html=True)

# ── NAV TABS ─────────────────────────────────────────────────────
if 'pg' not in st.session_state:
    st.session_state.pg = 'Home'

st.markdown('<div style="height:1px;background:rgba(255,255,255,0.04)"></div>', unsafe_allow_html=True)
st.markdown('<div style="padding:20px 80px 0">', unsafe_allow_html=True)

c1,c2,c3,c4,c5 = st.columns([2,2,2,2,10])
for col,pg,lbl in [(c1,'Home','Home'),(c2,'Players','Players'),(c3,'Predictor','Predictor'),(c4,'Tournament','Tournament')]:
    with col:
        if st.button(lbl, key=f'b_{pg}', use_container_width=True):
            st.session_state.pg = pg

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<div style="height:1px;background:rgba(255,255,255,0.04);margin-top:20px"></div>', unsafe_allow_html=True)

pg = st.session_state.pg

# ════════════════════════════════════
# HOME
# ════════════════════════════════════
if pg == 'Home':
    st.markdown(f"""
    <div class="hero">
        <div class="hero-eyebrow">
            <span style="width:6px;height:6px;background:#22C55E;border-radius:50%;display:inline-block"></span>
            Grand Slam · Hard Court · August 2026
        </div>
        <div class="hero-h1">US<br><em>OPEN</em><br>2026</div>
        <div class="hero-desc">
            Machine learning predictions built on 10 years of ATP match data.
            Discover who's most likely to lift the trophy at Flushing Meadows.
        </div>
        <div class="hero-numbers">
            <div class="hero-num">
                <div class="hero-num-val">{len(atp):,}</div>
                <div class="hero-num-lbl">Matches analyzed</div>
            </div>
            <div class="hero-num">
                <div class="hero-num-val">{len(uso):,}</div>
                <div class="hero-num-lbl">US Open matches</div>
            </div>
            <div class="hero-num">
                <div class="hero-num-val">128</div>
                <div class="hero-num-lbl">Players in draw</div>
            </div>
            <div class="hero-num">
                <div class="hero-num-val">1,000</div>
                <div class="hero-num-lbl">Simulations run</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="page">', unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Predictions</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-h2">Top Contenders</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub" style="margin-bottom:32px">Ranked by probability of winning the US Open 2026</div>', unsafe_allow_html=True)

    top10 = tourney.head(10).reset_index(drop=True)
    accents = [
        'background:linear-gradient(90deg,#C9A84C,#F0C060)',
        'background:linear-gradient(90deg,#94A3B8,#CBD5E1)',
        'background:linear-gradient(90deg,#B45309,#D97706)',
        'background:#22C55E22',
        'background:#22C55E22',
        'background:#22C55E22',
        'background:#22C55E22',
        'background:#22C55E22',
        'background:#22C55E22',
        'background:#22C55E22',
    ]
    cards = ""
    for i,row in top10.iterrows():
        bw = int(row['win_title']/top_max*100)
        cards += f"""
        <div class="c-card">
            <div class="c-accent" style="{accents[i]}"></div>
            <div class="c-rank">#{i+1} Contender</div>
            <div class="c-flag">{flg(row['player'])}</div>
            <div class="c-name">{row['player']}</div>
            <div class="c-pct">{row['win_title']:.1%}</div>
            <div class="c-pct-lbl">title probability</div>
            <div class="c-bar"><div class="c-bar-fill" style="width:{bw}%"></div></div>
        </div>"""
    st.markdown(f'<div class="c-grid">{cards}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════
# PLAYERS
# ════════════════════════════════════
elif pg == 'Players':
    st.markdown('<div class="page">', unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Head to Head</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-h2">Player Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub" style="margin-bottom:40px">Compare statistics between any two players in the 2026 draw</div>', unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        pa=st.selectbox('Player A',all_players,
                        index=all_players.index('Jannik Sinner') if 'Jannik Sinner' in all_players else 0)
    with c2:
        pb=st.selectbox('Player B',all_players,
                        index=all_players.index('Carlos Alcaraz') if 'Carlos Alcaraz' in all_players else 1)

    if pa and pb and pa!=pb:
        sa_df=ps[ps['player']==pa]; sb_df=ps[ps['player']==pb]
        if len(sa_df)>0 and len(sb_df)>0:
            sa=sa_df.iloc[0]; sb=sb_df.iloc[0]
            ra=int(rm.get(pa,999)); rb=int(rm.get(pb,999))
            h2h_a=len(atp[(atp['winner_name']==pa)&(atp['loser_name']==pb)])
            h2h_b=len(atp[(atp['winner_name']==pb)&(atp['loser_name']==pa)])

            def w(a,b,rev=False):
                if rev: return ('s-win','s-lose') if a<=b else ('s-lose','s-win')
                return ('s-win','s-lose') if a>=b else ('s-lose','s-win')

            rows=""
            stats_list=[
                ('Hard Court Win %',f"{sa['hwp']:.1%}",f"{sb['hwp']:.1%}",w(sa['hwp'],sb['hwp'])),
                ('Recent Form',f"{sa['rfp']:.1%}",f"{sb['rfp']:.1%}",w(sa['rfp'],sb['rfp'])),
                ('US Open Win %',f"{sa['uwp']:.1%}",f"{sb['uwp']:.1%}",w(sa['uwp'],sb['uwp'])),
                ('Aces per Match',f"{sa['aces']:.1f}",f"{sb['aces']:.1f}",w(sa['aces'],sb['aces'])),
                ('Double Faults',f"{sa['df']:.1f}",f"{sb['df']:.1f}",w(sa['df'],sb['df'],rev=True)),
                ('Break Point Save %',f"{sa['bps']:.1%}",f"{sb['bps']:.1%}",w(sa['bps'],sb['bps'])),
                ('Break Point Conv. %',f"{sa['bpc']:.1%}",f"{sb['bpc']:.1%}",w(sa['bpc'],sb['bpc'])),
                ('1st Serve %',f"{sa.get('fsp',0):.1%}",f"{sb.get('fsp',0):.1%}",w(sa.get('fsp',0),sb.get('fsp',0))),
                ('H2H Record',str(h2h_a),str(h2h_b),w(h2h_a,h2h_b)),
            ]
            for nm,va,vb,(ca,cb) in stats_list:
                rows+=f"""
                <div class="stat-row">
                    <div class="stat-v stat-v-l {ca}">{va}</div>
                    <div class="stat-n">{nm}</div>
                    <div class="stat-v stat-v-r {cb}">{vb}</div>
                </div>"""

            st.markdown(f"""
            <div class="cmp-box">
                <div class="cmp-top">
                    <div class="cmp-p">
                        <div class="cmp-p-flag">{flg(pa)}</div>
                        <div class="cmp-p-name" style="color:#60A5FA">{pa}</div>
                        <div class="cmp-p-rank">ATP #{ra}</div>
                    </div>
                    <div class="cmp-vs">VS</div>
                    <div class="cmp-p">
                        <div class="cmp-p-flag">{flg(pb)}</div>
                        <div class="cmp-p-name" style="color:#F87171">{pb}</div>
                        <div class="cmp-p-rank">ATP #{rb}</div>
                    </div>
                </div>
                {rows}
            </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════
# PREDICTOR
# ════════════════════════════════════
elif pg == 'Predictor':
    st.markdown('<div class="page">', unsafe_allow_html=True)
    st.markdown('<div class="sec-label">AI Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-h2">Match Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub" style="margin-bottom:40px">Select two players to predict who wins on hard court</div>', unsafe_allow_html=True)

    c1,c2=st.columns(2)
    with c1:
        pa=st.selectbox('Player A',all_players,
                        index=all_players.index('Jannik Sinner') if 'Jannik Sinner' in all_players else 0,key='pr_a')
    with c2:
        pb=st.selectbox('Player B',all_players,
                        index=all_players.index('Carlos Alcaraz') if 'Carlos Alcaraz' in all_players else 1,key='pr_b')

    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
    btn=st.button('Predict Match Winner', use_container_width=True)

    if btn:
        if pa==pb:
            st.warning('Please select two different players.')
        else:
            pa_p,pb_p=predict(pa,pb,rf_m,F,ps,rm)
            winner=pa if pa_p>pb_p else pb

            st.markdown(f"""
            <div class="pred-winner-box">
                <div class="pred-tag">🏆 &nbsp; Predicted Winner</div>
                <div class="pred-name">{flg(winner)}&nbsp;{winner}</div>
                <div class="pred-prob-lbl">Win probability &nbsp;·&nbsp; {max(pa_p,pb_p):.1%}</div>
            </div>""", unsafe_allow_html=True)

            ca='#60A5FA'; cb='#F87171'
            st.markdown(f"""
            <div class="prob-split">
                <div class="prob-side">
                    <div class="prob-big" style="color:{ca}">{pa_p:.1%}</div>
                    <div class="prob-player-name">{flg(pa)} {pa}</div>
                    <div class="prob-bar-wrap">
                        <div class="prob-bar-fill" style="width:{pa_p*100:.0f}%;background:{ca}"></div>
                    </div>
                </div>
                <div class="prob-sep">VS</div>
                <div class="prob-side">
                    <div class="prob-big" style="color:{cb}">{pb_p:.1%}</div>
                    <div class="prob-player-name">{flg(pb)} {pb}</div>
                    <div class="prob-bar-wrap">
                        <div class="prob-bar-fill" style="width:{pb_p*100:.0f}%;background:{cb}"></div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

            sa=ps[ps['player']==pa].iloc[0] if len(ps[ps['player']==pa])>0 else None
            sb=ps[ps['player']==pb].iloc[0] if len(ps[ps['player']==pb])>0 else None
            ra=rm.get(pa,999); rb=rm.get(pb,999)
            if sa is not None and sb is not None:
                def fstyle(a,b,rev=False):
                    better=(a<=b) if rev else (a>=b)
                    return 'color:#22C55E;font-weight:800' if better else 'color:#1E293B'
                factors=[
                    ('ATP Ranking',f'#{int(ra)}',f'#{int(rb)}',fstyle(ra,rb,rev=True),fstyle(rb,ra,rev=True)),
                    ('Hard Court Win %',f"{sa['hwp']:.1%}",f"{sb['hwp']:.1%}",fstyle(sa['hwp'],sb['hwp']),fstyle(sb['hwp'],sa['hwp'])),
                    ('Recent Form',f"{sa['rfp']:.1%}",f"{sb['rfp']:.1%}",fstyle(sa['rfp'],sb['rfp']),fstyle(sb['rfp'],sa['rfp'])),
                    ('BP Save %',f"{sa['bps']:.1%}",f"{sb['bps']:.1%}",fstyle(sa['bps'],sb['bps']),fstyle(sb['bps'],sa['bps'])),
                ]
                fc=""
                for nm,va,vb,fa_s,fb_s in factors:
                    fc+=f"""
                    <div class="factor-card">
                        <div class="factor-title">{nm}</div>
                        <div class="factor-vals">
                            <div class="factor-val" style="{fa_s}">{va}</div>
                            <div class="factor-dot">·</div>
                            <div class="factor-val" style="{fb_s}">{vb}</div>
                        </div>
                    </div>"""
                st.markdown(f'<div style="margin-top:8px;margin-bottom:4px;font-size:0.68em;font-weight:700;letter-spacing:3px;text-transform:uppercase;color:#334155">Key Factors</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="factors-grid">{fc}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════
# TOURNAMENT
# ════════════════════════════════════
elif pg == 'Tournament':
    st.markdown('<div class="page">', unsafe_allow_html=True)
    st.markdown('<div class="sec-label">1,000 Simulations</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-h2">Tournament Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub" style="margin-bottom:40px">Round-by-round probabilities for every player in the 2026 US Open draw</div>', unsafe_allow_html=True)

    rows_h=""
    for i,row in tourney.iterrows():
        pc=('t-hi' if i<3 else '')
        tc='t-hi' if row['win_title']>0.08 else 't-mid' if row['win_title']>0.02 else 't-lo'
        fc='t-hi' if row['reach_final']>0.25 else 't-mid' if row['reach_final']>0.08 else 't-lo'
        sc='t-hi' if row['reach_semis']>0.35 else 't-mid' if row['reach_semis']>0.15 else 't-lo'
        qc='t-hi' if row['reach_qtrs']>0.45 else 't-mid' if row['reach_qtrs']>0.25 else 't-lo'
        rows_h+=f"""
        <div class="t-row">
            <div class="t-pos {pc}">{i+1}</div>
            <div class="t-player">
                <div class="t-flag">{flg(row['player'])}</div>
                <div class="t-name">{row['player']}</div>
            </div>
            <div class="t-val {tc}">{row['win_title']:.1%}</div>
            <div class="t-val {fc}">{row['reach_final']:.1%}</div>
            <div class="t-val {sc}">{row['reach_semis']:.1%}</div>
            <div class="t-val {qc}">{row['reach_qtrs']:.1%}</div>
        </div>"""

    st.markdown(f"""
    <div class="t-table">
        <div class="t-head">
            <div class="t-th">#</div>
            <div class="t-th" style="text-align:left">Player</div>
            <div class="t-th">Title</div>
            <div class="t-th">Final</div>
            <div class="t-th">Semis</div>
            <div class="t-th">Quarter-F</div>
        </div>
        {rows_h}
    </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

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

* { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background: #04111F !important;
    color: #F1F5F9 !important;
}
#MainMenu, footer, header { visibility: hidden !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { display: none !important; }

/* NAV */
.navbar {
    background: rgba(4,17,31,0.95);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    padding: 0 48px;
    height: 64px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
}
.nav-brand {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.5em;
    letter-spacing: 4px;
    color: #FFFFFF;
    display: flex;
    align-items: center;
    gap: 10px;
}
.nav-dot {
    width: 8px; height: 8px;
    background: #22C55E;
    border-radius: 50%;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%,100% { opacity:1; transform:scale(1); }
    50% { opacity:0.5; transform:scale(1.3); }
}
.nav-tag {
    font-size: 0.65em;
    color: #C9A84C;
    letter-spacing: 3px;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
}

/* HERO */
.hero {
    background: linear-gradient(135deg, #061829 0%, #0A2E1A 40%, #061A30 80%, #04111F 100%);
    padding: 100px 80px 80px;
    position: relative;
    overflow: hidden;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.hero::before {
    content: '';
    position: absolute;
    top: -200px; right: -200px;
    width: 600px; height: 600px;
    background: radial-gradient(circle, rgba(34,197,94,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -100px; left: -100px;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(14,116,144,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-tag {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(34,197,94,0.1);
    border: 1px solid rgba(34,197,94,0.2);
    color: #22C55E;
    padding: 6px 16px;
    border-radius: 100px;
    font-size: 0.75em;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 24px;
}
.hero-h1 {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 7em;
    letter-spacing: 8px;
    color: #FFFFFF;
    line-height: 0.85;
    margin-bottom: 24px;
}
.hero-h1 span { color: #22C55E; }
.hero-p {
    font-size: 1.1em;
    color: #64748B;
    max-width: 480px;
    line-height: 1.7;
    font-weight: 400;
    margin-bottom: 40px;
}
.hero-stats {
    display: flex;
    gap: 48px;
    padding-top: 16px;
    border-top: 1px solid rgba(255,255,255,0.06);
}
.hero-stat-num {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.2em;
    color: #FFFFFF;
    line-height: 1;
}
.hero-stat-lbl {
    font-size: 0.72em;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
    margin-top: 4px;
}

/* SECTION */
.section { padding: 60px 80px; }
.section-sm { padding: 40px 80px; }
.s-header {
    display: flex;
    align-items: baseline;
    gap: 16px;
    margin-bottom: 32px;
}
.s-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2em;
    letter-spacing: 3px;
    color: #FFFFFF;
}
.s-sub { font-size: 0.85em; color: #475569; font-weight: 500; }

/* CONTENDER CARDS */
.contender-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 16px;
}
.c-card {
    background: #0A1929;
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 16px;
    padding: 24px 20px;
    position: relative;
    overflow: hidden;
    transition: all 0.2s;
}
.c-card:hover {
    border-color: rgba(34,197,94,0.2);
    transform: translateY(-2px);
}
.c-card.gold { border-color: rgba(201,168,76,0.3); }
.c-card.gold::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #C9A84C, #F0C060);
}
.c-card.silver::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #94A3B8, #CBD5E1);
}
.c-card.bronze::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #B45309, #D97706);
}
.c-pos {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 0.9em;
    letter-spacing: 2px;
    color: #334155;
    margin-bottom: 12px;
}
.c-flag { font-size: 1.8em; margin-bottom: 8px; display: block; }
.c-name {
    font-size: 0.9em;
    font-weight: 700;
    color: #F1F5F9;
    margin-bottom: 16px;
    line-height: 1.3;
}
.c-prob {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.2em;
    color: #22C55E;
    line-height: 1;
}
.c-prob-lbl {
    font-size: 0.65em;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
    margin-top: 2px;
}
.c-bar {
    height: 3px;
    background: rgba(255,255,255,0.05);
    border-radius: 2px;
    margin-top: 16px;
    overflow: hidden;
}
.c-bar-fill {
    height: 100%;
    border-radius: 2px;
    background: linear-gradient(90deg, #22C55E, #16A34A);
}

/* PAGE TABS */
.tab-bar {
    display: flex;
    gap: 4px;
    background: #0A1929;
    padding: 6px;
    border-radius: 12px;
    width: fit-content;
    margin: 0 80px 48px;
}
.tab-btn {
    padding: 10px 24px;
    border-radius: 8px;
    font-size: 0.85em;
    font-weight: 600;
    letter-spacing: 0.5px;
    cursor: pointer;
    transition: all 0.2s;
    color: #64748B;
    background: transparent;
    border: none;
}
.tab-btn.active {
    background: #0D2137;
    color: #FFFFFF;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}

/* COMPARISON */
.cmp-wrap {
    background: #0A1929;
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 20px;
    overflow: hidden;
}
.cmp-header {
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    align-items: center;
    padding: 40px 48px;
    background: linear-gradient(135deg, #061829, #0D2137);
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.cmp-player { text-align: center; }
.cmp-player-name {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.4em;
    letter-spacing: 2px;
    line-height: 1;
    margin-bottom: 8px;
}
.cmp-player-rank {
    font-size: 0.75em;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #C9A84C;
}
.cmp-vs {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.5em;
    color: #1E293B;
    padding: 0 32px;
}
.stat-block {
    display: grid;
    grid-template-columns: 1fr 200px 1fr;
    align-items: center;
    padding: 18px 48px;
    border-bottom: 1px solid rgba(255,255,255,0.03);
}
.stat-block:hover { background: rgba(255,255,255,0.01); }
.sv { font-size: 1.05em; font-weight: 700; }
.sv-left { text-align: right; padding-right: 24px; }
.sv-right { text-align: left; padding-left: 24px; }
.stat-name {
    font-size: 0.72em;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
    text-align: center;
}
.win { color: #22C55E !important; }
.lose { color: #334155 !important; }

/* PREDICTOR */
.pred-result {
    background: linear-gradient(135deg, #052010 0%, #0A1929 100%);
    border: 1px solid rgba(34,197,94,0.15);
    border-radius: 20px;
    padding: 48px;
    text-align: center;
    margin: 32px 0;
}
.pred-label {
    font-size: 0.72em;
    color: #22C55E;
    letter-spacing: 4px;
    text-transform: uppercase;
    font-weight: 700;
    margin-bottom: 16px;
}
.pred-winner {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 4em;
    letter-spacing: 4px;
    color: #FFFFFF;
    margin-bottom: 12px;
}
.pred-pct {
    font-size: 1em;
    color: #22C55E;
    font-weight: 600;
    letter-spacing: 1px;
}
.prob-row {
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    gap: 24px;
    align-items: center;
    margin-top: 40px;
}
.prob-side { text-align: center; }
.prob-pct-big {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 4.5em;
    line-height: 1;
}
.prob-name { font-size: 0.85em; color: #64748B; font-weight: 600; margin-top: 8px; }
.prob-bar-wrap {
    height: 6px;
    background: rgba(255,255,255,0.05);
    border-radius: 3px;
    margin-top: 12px;
    overflow: hidden;
}
.prob-bar { height: 100%; border-radius: 3px; }
.prob-vs {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.8em;
    color: #1E293B;
}

/* FACTOR CARDS */
.factor-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-top: 24px;
}
.factor-card {
    background: #0A1929;
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 20px 16px;
    text-align: center;
}
.factor-lbl {
    font-size: 0.68em;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
    margin-bottom: 12px;
}
.factor-vals {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
}
.factor-val { font-size: 1em; font-weight: 700; }
.factor-sep { font-size: 0.7em; color: #334155; }

/* TOURNAMENT TABLE */
.tourn-table {
    background: #0A1929;
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 16px;
    overflow: hidden;
}
.tourn-thead {
    display: grid;
    grid-template-columns: 48px 1fr 120px 120px 120px 120px;
    padding: 16px 24px;
    background: rgba(255,255,255,0.02);
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.tourn-th {
    font-size: 0.68em;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 700;
    text-align: center;
}
.tourn-th:nth-child(2) { text-align: left; }
.tourn-row {
    display: grid;
    grid-template-columns: 48px 1fr 120px 120px 120px 120px;
    padding: 16px 24px;
    border-bottom: 1px solid rgba(255,255,255,0.03);
    align-items: center;
    transition: background 0.15s;
}
.tourn-row:hover { background: rgba(255,255,255,0.02); }
.tourn-row:last-child { border-bottom: none; }
.tourn-pos {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.1em;
    color: #334155;
    text-align: center;
}
.tourn-pos.top3 { color: #C9A84C; }
.tourn-player { display: flex; align-items: center; gap: 10px; }
.tourn-flag { font-size: 1.1em; }
.tourn-name { font-size: 0.9em; font-weight: 600; color: #F1F5F9; }
.tourn-val {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.2em;
    text-align: center;
}
.tourn-val.hi { color: #22C55E; }
.tourn-val.mid { color: #94A3B8; }
.tourn-val.lo { color: #334155; }

/* SELECT styling */
div[data-testid="stSelectbox"] > div > div {
    background: #0A1929 !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    color: #F1F5F9 !important;
}
div[data-testid="stSelectbox"] label {
    color: #475569 !important;
    font-size: 0.72em !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
}

/* BUTTON */
.stButton > button {
    background: #22C55E !important;
    color: #04111F !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 800 !important;
    font-size: 0.85em !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    padding: 14px 40px !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: #16A34A !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(34,197,94,0.25) !important;
}

/* RADIO as tab */
div[data-testid="stHorizontalBlock"] { gap: 0 !important; }
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
        st.error(f'Missing file: {e}')
        st.stop()

@st.cache_data
def stats(_atp, _uso):
    hard = _atp[_atp['surface']=='Hard'].copy()
    for c in ['w_ace','w_df','w_svpt','w_1stIn','w_bpSaved','w_bpFaced','l_bpSaved','l_bpFaced']:
        if c in hard.columns: hard[c] = pd.to_numeric(hard[c], errors='coerce')
    hard['bps'] = np.where(hard['w_bpFaced']>0, hard['w_bpSaved']/hard['w_bpFaced'], np.nan)
    hard['bpc'] = np.where(hard['l_bpFaced']>0, (hard['l_bpFaced']-hard['l_bpSaved'])/hard['l_bpFaced'], np.nan)
    hard['fsp'] = np.where(hard['w_svpt']>0, hard['w_1stIn']/hard['w_svpt'], np.nan)
    hw = hard.groupby('winner_name').size().rename('hw')
    hl = hard.groupby('loser_name').size().rename('hl')
    hs = pd.concat([hw,hl],axis=1).fillna(0); hs['ht']=hs['hw']+hs['hl']; hs['hwp']=hs['hw']/hs['ht']
    hs = hs.reset_index().rename(columns={'index':'player'}); hs = hs[hs['ht']>=20]
    sv = hard.groupby('winner_name').agg(aces=('w_ace','mean'),df=('w_df','mean'),bps=('bps','mean'),bpc=('bpc','mean'),fsp=('fsp','mean')).reset_index().rename(columns={'winner_name':'player'})
    uw=_uso.groupby('winner_name').size().rename('uw'); ul=_uso.groupby('loser_name').size().rename('ul')
    us=pd.concat([uw,ul],axis=1).fillna(0); us['ut']=us['uw']+us['ul']; us['uwp']=us['uw']/us['ut']
    us=us.reset_index().rename(columns={'index':'player'})
    wn=_atp[['tourney_date','winner_name']].rename(columns={'winner_name':'player'}); wn['r']=1
    ln=_atp[['tourney_date','loser_name']].rename(columns={'loser_name':'player'}); ln['r']=0
    am=pd.concat([wn,ln],ignore_index=True)
    am['tourney_date']=pd.to_datetime(am['tourney_date'],errors='coerce')
    am=am.sort_values(['player','tourney_date'])
    tm=am.groupby('player').size().reset_index(name='tm')
    rf=(am.groupby('player').tail(10).groupby('player')['r'].agg(rw='sum',rt='count').reset_index())
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
def model(_atp, _ps):
    hard=_atp[_atp['surface']=='Hard'].copy()
    for c in ['w_ace','w_df','w_svpt','w_1stIn','w_bpSaved','w_bpFaced','l_bpSaved','l_bpFaced','winner_rank','loser_rank']:
        if c in hard.columns: hard[c]=pd.to_numeric(hard[c],errors='coerce')
    hard['bps']=np.where(hard['w_bpFaced']>0,hard['w_bpSaved']/hard['w_bpFaced'],np.nan)
    hard['bpc']=np.where(hard['l_bpFaced']>0,(hard['l_bpFaced']-hard['l_bpSaved'])/hard['l_bpFaced'],np.nan)
    hard['rank_diff']=hard['loser_rank']-hard['winner_rank']
    md=hard[['winner_name','loser_name','rank_diff']].dropna()
    for side,pfx in [('winner_name','w'),('loser_name','l')]:
        md=md.merge(_ps[['player','hwp','aces','df','bps','bpc']],left_on=side,right_on='player',how='left').drop(columns='player')
        md=md.rename(columns={'hwp':f'{pfx}wp','aces':f'{pfx}ac','df':f'{pfx}df','bps':f'{pfx}bs','bpc':f'{pfx}bc'})
    for c in ['wp','ac','df','bs','bc']: md[f'd_{c}']=md[f'w{c}']-md[f'l{c}']
    md['label']=1
    mir=md.copy()
    for c in ['rank_diff','d_wp','d_ac','d_df','d_bs','d_bc']: mir[c]=-mir[c]
    mir['label']=0
    tr=pd.concat([md,mir],ignore_index=True)
    F=['rank_diff','d_wp','d_ac','d_df','d_bs','d_bc']
    X=tr[F].fillna(0); y=tr['label']
    Xtr,_,ytr,_=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    rf=RandomForestClassifier(n_estimators=100,max_depth=7,random_state=42)
    rf.fit(Xtr,ytr)
    return rf,F

def pred(pa, pb, rf, F, ps, rm):
    sa=ps[ps['player']==pa]; sb=ps[ps['player']==pb]
    if len(sa)==0 or len(sb)==0: return 0.5,0.5
    sa,sb=sa.iloc[0],sb.iloc[0]
    ra=rm.get(pa,50); rb=rm.get(pb,50)
    f=pd.DataFrame([{'rank_diff':rb-ra,'d_wp':sa['hwp']-sb['hwp'],'d_ac':sa['aces']-sb['aces'],
                      'd_df':sa['df']-sb['df'],'d_bs':sa['bps']-sb['bps'],'d_bc':sa['bpc']-sb['bpc']}])[F].fillna(0)
    p=rf.predict_proba(f)[0][1]
    return p,1-p

# ── INIT ─────────────────────────────────────────────────────────
atp, uso, ml, tourney = load()
for df in [atp,uso]:
    if 'tourney_date' in df.columns: df['tourney_date']=pd.to_datetime(df['tourney_date'],errors='coerce')
    if 'tourney_year' not in df.columns: df['tourney_year']=df['tourney_date'].dt.year
    if 'surface' in df.columns: df['surface']=df['surface'].str.strip().str.title()

ps=stats(atp,uso)
rf_m,F=model(atp,ps)
rm=ml.set_index('full_name')['atp_rank'].to_dict() if 'full_name' in ml.columns and 'atp_rank' in ml.columns else {}
if 'full_name' in ml.columns:
    all_players=sorted([p for p in ml['full_name'].tolist() if p in ps['player'].values])
else:
    all_players=sorted(ps[ps['ht']>=50]['player'].tolist())

FLAGS={'ITA':'🇮🇹','ESP':'🇪🇸','GER':'🇩🇪','SRB':'🇷🇸','USA':'🇺🇸','CAN':'🇨🇦',
       'AUS':'🇦🇺','RUS':'🇷🇺','GBR':'🇬🇧','FRA':'🇫🇷','NOR':'🇳🇴','DEN':'🇩🇰',
       'GRE':'🇬🇷','ARG':'🇦🇷','CHI':'🇨🇱','KAZ':'🇰🇿','BRA':'🇧🇷','PER':'🇵🇪',
       'MON':'🇲🇨','CZE':'🇨🇿','POL':'🇵🇱','BEL':'🇧🇪','NED':'🇳🇱','HUN':'🇭🇺',
       'SUI':'🇨🇭','CRO':'🇭🇷','BUL':'🇧🇬','TPE':'🇹🇼','JPN':'🇯🇵','KOR':'🇰🇷'}
def flg(p):
    if 'full_name' in ml.columns and 'country' in ml.columns:
        r=ml[ml['full_name']==p]
        if len(r)>0: return FLAGS.get(r.iloc[0]['country'],'🎾')
    return '🎾'

top_max = tourney['win_title'].max()

# ── NAVBAR ───────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
    <div class="nav-brand">
        <div class="nav-dot"></div>
        US OPEN
        <span class="nav-tag">2026</span>
    </div>
    <div style="font-size:0.8em;color:#334155;font-weight:500;letter-spacing:1px">
        FLUSHING MEADOWS · NEW YORK
    </div>
</div>
""", unsafe_allow_html=True)

# ── PAGE NAV ─────────────────────────────────────────────────────
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

cols = st.columns([1,1,1,1,8])
pages = ['Home','Players','Predictor','Tournament']
labels = ['🏠  Home','👤  Players','🎯  Predictor','🏆  Tournament']
for i,(col,pg,lbl) in enumerate(zip(cols[:4],pages,labels)):
    with col:
        if st.button(lbl, key=f'nav_{pg}', use_container_width=True):
            st.session_state.page = pg

page = st.session_state.page

# ════════════════════════════════════
# HOME
# ════════════════════════════════════
if page == 'Home':
    top10 = tourney.head(10).reset_index(drop=True)

    st.markdown(f"""
    <div class="hero">
        <div class="hero-tag">
            <span style="width:6px;height:6px;background:#22C55E;border-radius:50%;display:inline-block"></span>
            Grand Slam · Hard Court · New York
        </div>
        <div class="hero-h1">US<br><span>OPEN</span><br>2026</div>
        <div class="hero-p">AI-powered predictions for the US Open draw. Built with 10 years of ATP match data and machine learning.</div>
        <div class="hero-stats">
            <div><div class="hero-stat-num">{len(atp):,}</div><div class="hero-stat-lbl">Matches analyzed</div></div>
            <div><div class="hero-stat-num">{len(uso):,}</div><div class="hero-stat-lbl">US Open matches</div></div>
            <div><div class="hero-stat-num">128</div><div class="hero-stat-lbl">Players in draw</div></div>
            <div><div class="hero-stat-num">1,000</div><div class="hero-stat-lbl">Simulations run</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="s-header"><div class="s-title">Top Contenders</div><div class="s-sub">Ranked by title probability</div></div>', unsafe_allow_html=True)

    cards = []
    styles = ['gold','silver','bronze','','','','','','','']
    for i, row in top10.iterrows():
        f = flg(row['player'])
        pct = row['win_title']
        bar_w = int(pct / top_max * 100)
        style = styles[i]
        cards.append(f"""
        <div class="c-card {style}">
            <div class="c-pos">#{i+1}</div>
            <div class="c-flag">{f}</div>
            <div class="c-name">{row['player']}</div>
            <div class="c-prob">{pct:.1%}</div>
            <div class="c-prob-lbl">title probability</div>
            <div class="c-bar"><div class="c-bar-fill" style="width:{bar_w}%"></div></div>
        </div>""")

    st.markdown(f'<div class="contender-grid">{"".join(cards)}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════
# PLAYERS
# ════════════════════════════════════
elif page == 'Players':
    st.markdown("""
    <div class="hero" style="padding:60px 80px">
        <div class="hero-tag">Head to Head</div>
        <div class="hero-h1" style="font-size:4em">Player<br><span>Comparison</span></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-sm">', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        pa = st.selectbox('Select Player A', all_players,
                          index=all_players.index('Jannik Sinner') if 'Jannik Sinner' in all_players else 0)
    with c2:
        pb = st.selectbox('Select Player B', all_players,
                          index=all_players.index('Carlos Alcaraz') if 'Carlos Alcaraz' in all_players else 1)

    if pa and pb and pa != pb:
        sa_df=ps[ps['player']==pa]; sb_df=ps[ps['player']==pb]
        if len(sa_df)>0 and len(sb_df)>0:
            sa=sa_df.iloc[0]; sb=sb_df.iloc[0]
            ra=int(rm.get(pa,999)); rb=int(rm.get(pb,999))

            h2h_a=len(atp[(atp['winner_name']==pa)&(atp['loser_name']==pb)])
            h2h_b=len(atp[(atp['winner_name']==pb)&(atp['loser_name']==pa)])

            def w(va,vb): return 'win' if va>=vb else 'lose'
            def wrev(va,vb): return 'win' if va<=vb else 'lose'

            stats_rows = ""
            stat_list = [
                ('Hard Court Win %', f"{sa['hwp']:.1%}", f"{sb['hwp']:.1%}", w(sa['hwp'],sb['hwp']), w(sb['hwp'],sa['hwp'])),
                ('Recent Form', f"{sa['rfp']:.1%}", f"{sb['rfp']:.1%}", w(sa['rfp'],sb['rfp']), w(sb['rfp'],sa['rfp'])),
                ('US Open Win %', f"{sa['uwp']:.1%}", f"{sb['uwp']:.1%}", w(sa['uwp'],sb['uwp']), w(sb['uwp'],sa['uwp'])),
                ('Avg Aces / Match', f"{sa['aces']:.1f}", f"{sb['aces']:.1f}", w(sa['aces'],sb['aces']), w(sb['aces'],sa['aces'])),
                ('Double Faults / Match', f"{sa['df']:.1f}", f"{sb['df']:.1f}", wrev(sa['df'],sb['df']), wrev(sb['df'],sa['df'])),
                ('BP Save Rate', f"{sa['bps']:.1%}", f"{sb['bps']:.1%}", w(sa['bps'],sb['bps']), w(sb['bps'],sa['bps'])),
                ('BP Conversion Rate', f"{sa['bpc']:.1%}", f"{sb['bpc']:.1%}", w(sa['bpc'],sb['bpc']), w(sb['bpc'],sa['bpc'])),
                ('1st Serve %', f"{sa.get('fsp',0):.1%}", f"{sb.get('fsp',0):.1%}", w(sa.get('fsp',0),sb.get('fsp',0)), w(sb.get('fsp',0),sa.get('fsp',0))),
                ('H2H Wins', str(h2h_a), str(h2h_b), w(h2h_a,h2h_b), w(h2h_b,h2h_a)),
            ]
            for nm, va, vb, ca, cb in stat_list:
                stats_rows += f"""
                <div class="stat-block">
                    <div class="sv sv-left {ca}">{va}</div>
                    <div class="stat-name">{nm}</div>
                    <div class="sv sv-right {cb}">{vb}</div>
                </div>"""

            st.markdown(f"""
            <div class="cmp-wrap" style="margin-top:24px">
                <div class="cmp-header">
                    <div class="cmp-player">
                        <div style="font-size:2em;margin-bottom:8px">{flg(pa)}</div>
                        <div class="cmp-player-name" style="color:#4A90D9">{pa}</div>
                        <div class="cmp-player-rank">ATP #{ra}</div>
                    </div>
                    <div class="cmp-vs">VS</div>
                    <div class="cmp-player">
                        <div style="font-size:2em;margin-bottom:8px">{flg(pb)}</div>
                        <div class="cmp-player-name" style="color:#EF5350">{pb}</div>
                        <div class="cmp-player-rank">ATP #{rb}</div>
                    </div>
                </div>
                {stats_rows}
            </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════
# PREDICTOR
# ════════════════════════════════════
elif page == 'Predictor':
    st.markdown("""
    <div class="hero" style="padding:60px 80px">
        <div class="hero-tag">Machine Learning</div>
        <div class="hero-h1" style="font-size:4em">Match<br><span>Predictor</span></div>
        <div class="hero-p">Select two players and our model will predict who wins on hard court.</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-sm">', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        pa = st.selectbox('Player A', all_players,
                          index=all_players.index('Jannik Sinner') if 'Jannik Sinner' in all_players else 0, key='pr_a')
    with c2:
        pb = st.selectbox('Player B', all_players,
                          index=all_players.index('Carlos Alcaraz') if 'Carlos Alcaraz' in all_players else 1, key='pr_b')
    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    btn = st.button('Predict Winner', use_container_width=True)

    if btn:
        if pa==pb:
            st.warning('Please select two different players.')
        else:
            pa_p, pb_p = pred(pa, pb, rf_m, F, ps, rm)
            winner = pa if pa_p > pb_p else pb

            sa=ps[ps['player']==pa].iloc[0] if len(ps[ps['player']==pa])>0 else None
            sb=ps[ps['player']==pb].iloc[0] if len(ps[ps['player']==pb])>0 else None
            ra=rm.get(pa,999); rb=rm.get(pb,999)

            st.markdown(f"""
            <div class="pred-result">
                <div class="pred-label">🏆 Predicted Winner</div>
                <div class="pred-winner">{flg(winner)} {winner}</div>
                <div class="pred-pct">Win probability · {max(pa_p,pb_p):.1%}</div>
            </div>""", unsafe_allow_html=True)

            col_a = '#4A90D9'
            col_b = '#EF5350'
            bar_a = f'background:linear-gradient(90deg,{col_a},{col_a}88)'
            bar_b = f'background:linear-gradient(90deg,{col_b},{col_b}88)'

            st.markdown(f"""
            <div class="prob-row">
                <div class="prob-side">
                    <div class="prob-pct-big" style="color:{col_a}">{pa_p:.1%}</div>
                    <div class="prob-name">{flg(pa)} {pa}</div>
                    <div class="prob-bar-wrap">
                        <div class="prob-bar" style="{bar_a};width:{pa_p*100:.0f}%"></div>
                    </div>
                </div>
                <div class="prob-vs">VS</div>
                <div class="prob-side">
                    <div class="prob-pct-big" style="color:{col_b}">{pb_p:.1%}</div>
                    <div class="prob-name">{flg(pb)} {pb}</div>
                    <div class="prob-bar-wrap">
                        <div class="prob-bar" style="{bar_b};width:{pb_p*100:.0f}%"></div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

            if sa is not None and sb is not None:
                def fw(va,vb,rev=False):
                    better = (va<=vb) if rev else (va>=vb)
                    return f'color:#22C55E;font-weight:800' if better else f'color:#334155'
                factors = [
                    ('ATP Ranking', f'#{int(ra)}', f'#{int(rb)}', fw(ra,rb,rev=True), fw(rb,ra,rev=True)),
                    ('Hard Court Win %', f"{sa['hwp']:.1%}", f"{sb['hwp']:.1%}", fw(sa['hwp'],sb['hwp']), fw(sb['hwp'],sa['hwp'])),
                    ('Recent Form', f"{sa['rfp']:.1%}", f"{sb['rfp']:.1%}", fw(sa['rfp'],sb['rfp']), fw(sb['rfp'],sa['rfp'])),
                    ('BP Save %', f"{sa['bps']:.1%}", f"{sb['bps']:.1%}", fw(sa['bps'],sb['bps']), fw(sb['bps'],sa['bps'])),
                ]
                cards_f = ""
                for nm, va, vb, fa_s, fb_s in factors:
                    cards_f += f"""
                    <div class="factor-card">
                        <div class="factor-lbl">{nm}</div>
                        <div class="factor-vals">
                            <div class="factor-val" style="{fa_s}">{va}</div>
                            <div class="factor-sep">·</div>
                            <div class="factor-val" style="{fb_s}">{vb}</div>
                        </div>
                    </div>"""
                st.markdown(f'<div style="margin-top:32px;font-family:Bebas Neue,Impact,sans-serif;font-size:1.1em;letter-spacing:3px;color:#334155;margin-bottom:12px">KEY FACTORS</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="factor-grid">{cards_f}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════
# TOURNAMENT
# ════════════════════════════════════
elif page == 'Tournament':
    st.markdown("""
    <div class="hero" style="padding:60px 80px">
        <div class="hero-tag">1,000 Simulations</div>
        <div class="hero-h1" style="font-size:4em">Tournament<br><span>Prediction</span></div>
        <div class="hero-p">Round-by-round win probabilities for the full US Open 2026 draw.</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-sm">', unsafe_allow_html=True)

    rows_html = ""
    for i, row in tourney.iterrows():
        pos_cls = 'top3' if i < 3 else ''
        t_cls = 'hi' if row['win_title'] > 0.1 else 'mid' if row['win_title'] > 0.03 else 'lo'
        f_cls = 'hi' if row['reach_final'] > 0.3 else 'mid' if row['reach_final'] > 0.1 else 'lo'
        s_cls = 'hi' if row['reach_semis'] > 0.4 else 'mid' if row['reach_semis'] > 0.2 else 'lo'
        q_cls = 'hi' if row['reach_qtrs'] > 0.5 else 'mid' if row['reach_qtrs'] > 0.3 else 'lo'
        rows_html += f"""
        <div class="tourn-row">
            <div class="tourn-pos {pos_cls}">{i+1}</div>
            <div class="tourn-player">
                <div class="tourn-flag">{flg(row['player'])}</div>
                <div class="tourn-name">{row['player']}</div>
            </div>
            <div class="tourn-val {t_cls}">{row['win_title']:.1%}</div>
            <div class="tourn-val {f_cls}">{row['reach_final']:.1%}</div>
            <div class="tourn-val {s_cls}">{row['reach_semis']:.1%}</div>
            <div class="tourn-val {q_cls}">{row['reach_qtrs']:.1%}</div>
        </div>"""

    st.markdown(f"""
    <div class="tourn-table">
        <div class="tourn-thead">
            <div class="tourn-th">#</div>
            <div class="tourn-th" style="text-align:left">Player</div>
            <div class="tourn-th">Title</div>
            <div class="tourn-th">Final</div>
            <div class="tourn-th">Semis</div>
            <div class="tourn-th">Quarter-F</div>
        </div>
        {rows_html}
    </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

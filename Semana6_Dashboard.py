import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title='US Open 2026 · Analytics',
    page_icon='🎾',
    layout='wide',
    initial_sidebar_state='expanded'
)

USO_BLUE  = '#0A2240'
USO_GREEN = '#3A7D44'
USO_GOLD  = '#C9A84C'
USO_WHITE = '#FFFFFF'

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif !important;
    background-color: {USO_BLUE} !important;
    color: {USO_WHITE} !important;
}}
.stApp {{ background-color: {USO_BLUE} !important; }}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding: 2rem 3rem 3rem; }}

section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #061629 0%, {USO_BLUE} 100%) !important;
    border-right: 2px solid {USO_GREEN} !important;
}}
section[data-testid="stSidebar"] * {{ color: {USO_WHITE} !important; }}

h1 {{
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 3.5rem !important;
    letter-spacing: 6px !important;
    color: {USO_WHITE} !important;
}}
h2 {{
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 2rem !important;
    letter-spacing: 3px !important;
    color: {USO_WHITE} !important;
    border-bottom: 2px solid {USO_GREEN} !important;
    padding-bottom: 8px !important;
    margin-bottom: 24px !important;
}}
h3 {{
    color: {USO_GOLD} !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
}}
p, li, span, div {{
    color: {USO_WHITE} !important;
}}
.stMetric {{
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    padding: 20px !important;
}}
.stMetric label {{
    color: {USO_GOLD} !important;
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
    font-weight: 700 !important;
}}
[data-testid="stMetricValue"] {{
    color: {USO_WHITE} !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
}}
[data-testid="stMetricDelta"] {{
    color: {USO_GREEN} !important;
}}
.stProgress > div > div > div {{
    background: linear-gradient(90deg, {USO_GREEN}, #22C55E) !important;
}}
.stProgress > div > div {{
    background: rgba(255,255,255,0.08) !important;
}}
.stButton > button {{
    background: {USO_GREEN} !important;
    color: {USO_WHITE} !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    padding: 14px 32px !important;
    width: 100% !important;
}}
.stButton > button:hover {{
    background: #2D6A35 !important;
    box-shadow: 0 4px 20px rgba(58,125,68,0.4) !important;
}}
div[data-testid="stSelectbox"] > div > div {{
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
    color: {USO_WHITE} !important;
}}
div[data-testid="stSelectbox"] label {{
    color: {USO_GOLD} !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
}}
.stDataFrame {{ border-radius: 12px !important; }}
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
def calc_stats(_atp, _uso):
    hard = _atp[_atp['surface'] == 'Hard'].copy()
    for c in ['w_ace','w_df','w_svpt','w_1stIn','w_bpSaved','w_bpFaced','l_bpSaved','l_bpFaced']:
        if c in hard.columns:
            hard[c] = pd.to_numeric(hard[c], errors='coerce')
    hard['bps'] = np.where(hard['w_bpFaced'] > 0, hard['w_bpSaved'] / hard['w_bpFaced'], np.nan)
    hard['bpc'] = np.where(hard['l_bpFaced'] > 0, (hard['l_bpFaced'] - hard['l_bpSaved']) / hard['l_bpFaced'], np.nan)
    hard['fsp'] = np.where(hard['w_svpt'] > 0, hard['w_1stIn'] / hard['w_svpt'], np.nan)
    hw = hard.groupby('winner_name').size().rename('hw')
    hl = hard.groupby('loser_name').size().rename('hl')
    hs = pd.concat([hw, hl], axis=1).fillna(0)
    hs['ht'] = hs['hw'] + hs['hl']
    hs['hwp'] = hs['hw'] / hs['ht']
    hs = hs.reset_index().rename(columns={'index': 'player'})
    hs = hs[hs['ht'] >= 20]
    sv = hard.groupby('winner_name').agg(
        aces=('w_ace', 'mean'), df=('w_df', 'mean'),
        bps=('bps', 'mean'), bpc=('bpc', 'mean'), fsp=('fsp', 'mean')
    ).reset_index().rename(columns={'winner_name': 'player'})
    uw = _uso.groupby('winner_name').size().rename('uw')
    ul = _uso.groupby('loser_name').size().rename('ul')
    us = pd.concat([uw, ul], axis=1).fillna(0)
    us['ut'] = us['uw'] + us['ul']
    us['uwp'] = us['uw'] / us['ut']
    us = us.reset_index().rename(columns={'index': 'player'})
    wn = _atp[['tourney_date', 'winner_name']].rename(columns={'winner_name': 'player'})
    wn['r'] = 1
    ln = _atp[['tourney_date', 'loser_name']].rename(columns={'loser_name': 'player'})
    ln['r'] = 0
    am = pd.concat([wn, ln], ignore_index=True)
    am['tourney_date'] = pd.to_datetime(am['tourney_date'], errors='coerce')
    am = am.sort_values(['player', 'tourney_date'])
    tm = am.groupby('player').size().reset_index(name='tm')
    rf = am.groupby('player').tail(10).groupby('player')['r'].agg(rw='sum', rt='count').reset_index()
    rf = rf.merge(tm, on='player')
    rf = rf[rf['tm'] >= 50]
    rf['rfp'] = rf['rw'] / rf['rt']
    ps = hs.merge(sv, on='player', how='left')
    ps = ps.merge(rf[['player', 'rfp']], on='player', how='left')
    ps = ps.merge(us[['player', 'uw', 'uwp']], on='player', how='left')
    ps['uwp'] = ps['uwp'].fillna(0.5)
    ps['rfp'] = ps['rfp'].fillna(0.5)
    ex = pd.DataFrame([
        {'player': 'Joao Fonseca', 'hw': 27, 'hl': 17, 'ht': 44, 'hwp': 0.614, 'aces': 6.0, 'df': 0.17, 'bps': 0.665, 'bpc': 0.38, 'fsp': 0.64, 'rfp': 0.60, 'uw': 1, 'uwp': 0.50},
        {'player': 'Jakub Mensik', 'hw': 21, 'hl': 10, 'ht': 31, 'hwp': 0.677, 'aces': 7.2, 'df': 0.20, 'bps': 0.640, 'bpc': 0.41, 'fsp': 0.63, 'rfp': 0.65, 'uw': 0, 'uwp': 0.50},
        {'player': 'Rafael Jodar', 'hw': 18, 'hl': 9,  'ht': 27, 'hwp': 0.667, 'aces': 5.5, 'df': 0.18, 'bps': 0.650, 'bpc': 0.39, 'fsp': 0.62, 'rfp': 0.60, 'uw': 0, 'uwp': 0.50},
    ])
    return pd.concat([ps, ex], ignore_index=True)

@st.cache_resource
def train(_atp, _ps):
    hard = _atp[_atp['surface'] == 'Hard'].copy()
    for c in ['w_ace','w_df','w_svpt','w_1stIn','w_bpSaved','w_bpFaced','l_bpSaved','l_bpFaced','winner_rank','loser_rank']:
        if c in hard.columns:
            hard[c] = pd.to_numeric(hard[c], errors='coerce')
    hard['bps'] = np.where(hard['w_bpFaced'] > 0, hard['w_bpSaved'] / hard['w_bpFaced'], np.nan)
    hard['bpc'] = np.where(hard['l_bpFaced'] > 0, (hard['l_bpFaced'] - hard['l_bpSaved']) / hard['l_bpFaced'], np.nan)
    hard['rank_diff'] = hard['loser_rank'] - hard['winner_rank']
    md = hard[['winner_name', 'loser_name', 'rank_diff']].dropna()
    for side, pfx in [('winner_name', 'w'), ('loser_name', 'l')]:
        md = md.merge(_ps[['player', 'hwp', 'aces', 'df', 'bps', 'bpc']], left_on=side, right_on='player', how='left').drop(columns='player')
        md = md.rename(columns={'hwp': f'{pfx}wp', 'aces': f'{pfx}ac', 'df': f'{pfx}df', 'bps': f'{pfx}bs', 'bpc': f'{pfx}bc'})
    for c in ['wp', 'ac', 'df', 'bs', 'bc']:
        md[f'd_{c}'] = md[f'w{c}'] - md[f'l{c}']
    md['label'] = 1
    mir = md.copy()
    for c in ['rank_diff', 'd_wp', 'd_ac', 'd_df', 'd_bs', 'd_bc']:
        mir[c] = -mir[c]
    mir['label'] = 0
    tr = pd.concat([md, mir], ignore_index=True)
    F = ['rank_diff', 'd_wp', 'd_ac', 'd_df', 'd_bs', 'd_bc']
    X = tr[F].fillna(0)
    y = tr['label']
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
    rf.fit(Xtr, ytr)
    return rf, F

def predict(pa, pb, rf, F, ps, rm):
    sa = ps[ps['player'] == pa]
    sb = ps[ps['player'] == pb]
    if len(sa) == 0 or len(sb) == 0:
        return 0.5, 0.5
    sa, sb = sa.iloc[0], sb.iloc[0]
    ra = rm.get(pa, 50)
    rb = rm.get(pb, 50)
    f = pd.DataFrame([{
        'rank_diff': rb - ra, 'd_wp': sa['hwp'] - sb['hwp'],
        'd_ac': sa['aces'] - sb['aces'], 'd_df': sa['df'] - sb['df'],
        'd_bs': sa['bps'] - sb['bps'], 'd_bc': sa['bpc'] - sb['bpc']
    }])[F].fillna(0)
    p = rf.predict_proba(f)[0][1]
    return p, 1 - p


# ── INIT ─────────────────────────────────────────────────────────
atp, uso, ml, tourney = load()
for df in [atp, uso]:
    if 'tourney_date' in df.columns:
        df['tourney_date'] = pd.to_datetime(df['tourney_date'], errors='coerce')
    if 'tourney_year' not in df.columns:
        df['tourney_year'] = df['tourney_date'].dt.year
    if 'surface' in df.columns:
        df['surface'] = df['surface'].str.strip().str.title()

ps = calc_stats(atp, uso)
rf_m, F = train(atp, ps)
rm = ml.set_index('full_name')['atp_rank'].to_dict() if 'full_name' in ml.columns and 'atp_rank' in ml.columns else {}
if 'full_name' in ml.columns:
    all_players = sorted([p for p in ml['full_name'].tolist() if p in ps['player'].values])
else:
    all_players = sorted(ps[ps['ht'] >= 50]['player'].tolist())


# ── SIDEBAR ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('# 🎾 US OPEN')
    st.markdown('### 2026')
    st.divider()
    page = st.radio(
        '',
        ['🏠  Home', '👤  Player Comparison', '🎯  Match Predictor', '🏆  Tournament'],
        label_visibility='collapsed'
    )
    st.divider()
    st.markdown(f'Matches: **{len(atp):,}**')
    st.markdown(f'US Open: **{len(uso):,}**')
    st.markdown('Years: **10**')


# ════════════════════════════════════
# HOME
# ════════════════════════════════════
if '🏠' in page:
    st.markdown('# 🎾 US OPEN 2026')
    st.markdown('### Analytics & Prediction Platform')
    st.markdown('Compare players, predict match winners, and view the Top 10 tournament contenders based on player statistics and recent performance.')
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Matches Analyzed', f'{len(atp):,}')
    c2.metric('US Open Matches', f'{len(uso):,}')
    c3.metric('Years of Data', '10')
    c4.metric('Players in Draw', '128')

    st.divider()
    st.markdown('## Top 10 Contenders')

    top10 = tourney.head(10).reset_index(drop=True)

    colors = [USO_GOLD if i == 0 else USO_GREEN if i < 3 else '#1B4F8A' for i in range(len(top10))]
    fig = go.Figure(go.Bar(
        x=top10['player'],
        y=top10['win_title'],
        marker=dict(color=colors, line=dict(width=0)),
        text=[f'{v:.1%}' for v in top10['win_title']],
        textposition='outside',
        textfont=dict(color=USO_WHITE, size=11, family='Inter'),
    ))
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=USO_WHITE, family='Inter'),
        xaxis=dict(showgrid=False, tickangle=-20, tickfont=dict(size=10, color=USO_WHITE)),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)',
                   tickformat='.0%', tickfont=dict(size=10, color=USO_WHITE)),
        margin=dict(l=0, r=0, t=30, b=0),
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    medals = ['🥇', '🥈', '🥉', '4', '5', '6', '7', '8', '9', '10']
    for i, row in top10.iterrows():
        c1, c2, c3, c4, c5 = st.columns([1, 5, 2, 2, 2])
        with c1: st.markdown(f'### {medals[i]}')
        with c2: st.markdown(f'**{row["player"]}**')
        with c3:
            st.caption('Title')
            st.markdown(f'**{row["win_title"]:.1%}**')
        with c4:
            st.caption('Final')
            st.markdown(f'**{row["reach_final"]:.1%}**')
        with c5:
            st.caption('Semis')
            st.markdown(f'**{row["reach_semis"]:.1%}**')


# ════════════════════════════════════
# PLAYER COMPARISON
# ════════════════════════════════════
elif '👤' in page:
    st.markdown('# 👤 PLAYER COMPARISON')
    st.markdown('### Compare any two players in the 2026 draw')
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        pa = st.selectbox('Player A', all_players,
                          index=all_players.index('Jannik Sinner') if 'Jannik Sinner' in all_players else 0)
    with c2:
        pb = st.selectbox('Player B', all_players,
                          index=all_players.index('Carlos Alcaraz') if 'Carlos Alcaraz' in all_players else 1)

    if pa and pb and pa != pb:
        sa_df = ps[ps['player'] == pa]
        sb_df = ps[ps['player'] == pb]
        if len(sa_df) > 0 and len(sb_df) > 0:
            sa = sa_df.iloc[0]
            sb = sb_df.iloc[0]
            ra = int(rm.get(pa, 999))
            rb = int(rm.get(pb, 999))
            h2h_a = len(atp[(atp['winner_name'] == pa) & (atp['loser_name'] == pb)])
            h2h_b = len(atp[(atp['winner_name'] == pb) & (atp['loser_name'] == pa)])

            st.divider()
            c1, c2, c3 = st.columns([3, 1, 3])
            with c1:
                st.markdown(f'## {pa}')
                st.markdown(f'**ATP #{ra}**')
            with c2:
                st.markdown('## VS')
            with c3:
                st.markdown(f'## {pb}')
                st.markdown(f'**ATP #{rb}**')

            st.divider()

            # Radar chart
            categories = ['Hard Court %', 'Recent Form', 'US Open %', 'BP Save %', 'BP Conv. %', '1st Serve %']
            vals_a = [sa['hwp'], sa['rfp'], sa['uwp'], sa['bps'], sa['bpc'], sa.get('fsp', 0.6)]
            vals_b = [sb['hwp'], sb['rfp'], sb['uwp'], sb['bps'], sb['bpc'], sb.get('fsp', 0.6)]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=vals_a, theta=categories, fill='toself', name=pa,
                line=dict(color='#60A5FA', width=2), fillcolor='rgba(96,165,250,0.15)'
            ))
            fig.add_trace(go.Scatterpolar(
                r=vals_b, theta=categories, fill='toself', name=pb,
                line=dict(color='#F87171', width=2), fillcolor='rgba(248,113,113,0.15)'
            ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1], tickformat='.0%',
                                   gridcolor='rgba(255,255,255,0.1)',
                                   tickfont=dict(size=8, color=USO_WHITE)),
                    angularaxis=dict(gridcolor='rgba(255,255,255,0.1)',
                                    tickfont=dict(size=10, color=USO_WHITE)),
                    bgcolor='rgba(0,0,0,0)'
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=USO_WHITE, family='Inter'),
                legend=dict(font=dict(color=USO_WHITE, size=12)),
                margin=dict(l=60, r=60, t=40, b=40),
                height=420,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.markdown('### Statistics')

            stats = [
                ('Hard Court Win %', f"{sa['hwp']:.1%}", f"{sb['hwp']:.1%}", sa['hwp'] > sb['hwp']),
                ('Recent Form (last 10)', f"{sa['rfp']:.1%}", f"{sb['rfp']:.1%}", sa['rfp'] > sb['rfp']),
                ('US Open Win %', f"{sa['uwp']:.1%}", f"{sb['uwp']:.1%}", sa['uwp'] > sb['uwp']),
                ('Aces per Match', f"{sa['aces']:.1f}", f"{sb['aces']:.1f}", sa['aces'] > sb['aces']),
                ('Double Faults', f"{sa['df']:.1f}", f"{sb['df']:.1f}", sa['df'] < sb['df']),
                ('BP Save %', f"{sa['bps']:.1%}", f"{sb['bps']:.1%}", sa['bps'] > sb['bps']),
                ('BP Conversion %', f"{sa['bpc']:.1%}", f"{sb['bpc']:.1%}", sa['bpc'] > sb['bpc']),
                ('H2H Record', str(h2h_a), str(h2h_b), h2h_a > h2h_b),
            ]
            for stat, va, vb, a_wins in stats:
                c1, c2, c3 = st.columns([3, 3, 3])
                with c1:
                    st.markdown(f'{"✅ " if a_wins else ""}**{va}**')
                with c2:
                    st.caption(stat)
                with c3:
                    st.markdown(f'{"✅ " if not a_wins else ""}**{vb}**')


# ════════════════════════════════════
# MATCH PREDICTOR
# ════════════════════════════════════
elif '🎯' in page:
    st.markdown('# 🎯 MATCH PREDICTOR')
    st.markdown('### AI-powered prediction on hard court')
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        pa = st.selectbox('Player A', all_players,
                          index=all_players.index('Jannik Sinner') if 'Jannik Sinner' in all_players else 0, key='pr_a')
    with c2:
        pb = st.selectbox('Player B', all_players,
                          index=all_players.index('Carlos Alcaraz') if 'Carlos Alcaraz' in all_players else 1, key='pr_b')

    st.write('')
    btn = st.button('🎾  Predict Match Winner', use_container_width=True)

    if btn:
        if pa == pb:
            st.warning('Please select two different players.')
        else:
            pa_p, pb_p = predict(pa, pb, rf_m, F, ps, rm)
            winner = pa if pa_p > pb_p else pb

            st.divider()
            st.success(f'🏆  **{winner}** wins with **{max(pa_p, pb_p):.1%}** probability')
            st.write('')

            # Gauge
            fig = go.Figure(go.Indicator(
                mode='gauge+number',
                value=max(pa_p, pb_p) * 100,
                number=dict(suffix='%', font=dict(color=USO_WHITE, size=40)),
                gauge=dict(
                    axis=dict(range=[50, 100], tickcolor=USO_WHITE,
                              tickfont=dict(color=USO_WHITE)),
                    bar=dict(color=USO_GREEN, thickness=0.3),
                    bgcolor='rgba(255,255,255,0.05)',
                    bordercolor='rgba(255,255,255,0.1)',
                    steps=[
                        dict(range=[50, 65], color='rgba(255,255,255,0.02)'),
                        dict(range=[65, 80], color='rgba(58,125,68,0.08)'),
                        dict(range=[80, 100], color='rgba(58,125,68,0.15)'),
                    ],
                    threshold=dict(line=dict(color=USO_GOLD, width=3),
                                   thickness=0.8, value=max(pa_p, pb_p) * 100)
                ),
                title=dict(text=f'{winner} — Win Probability',
                           font=dict(color=USO_GOLD, size=14))
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=USO_WHITE, family='Inter'),
                height=280,
                margin=dict(l=40, r=40, t=20, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

            c1, c2, c3 = st.columns([4, 1, 4])
            with c1:
                st.metric(pa, f'{pa_p:.1%}')
                st.progress(pa_p)
            with c2:
                st.markdown('### VS')
            with c3:
                st.metric(pb, f'{pb_p:.1%}')
                st.progress(pb_p)

            st.divider()
            st.markdown('### Key Factors')
            sa = ps[ps['player'] == pa].iloc[0] if len(ps[ps['player'] == pa]) > 0 else None
            sb = ps[ps['player'] == pb].iloc[0] if len(ps[ps['player'] == pb]) > 0 else None
            ra = rm.get(pa, 999)
            rb = rm.get(pb, 999)
            if sa is not None and sb is not None:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric('ATP Ranking', f'#{int(ra)}', f'vs #{int(rb)}')
                c2.metric('Hard Court Win %', f"{sa['hwp']:.1%}", f"{sa['hwp'] - sb['hwp']:+.1%}")
                c3.metric('Recent Form', f"{sa['rfp']:.1%}", f"{sa['rfp'] - sb['rfp']:+.1%}")
                c4.metric('BP Save %', f"{sa['bps']:.1%}", f"{sa['bps'] - sb['bps']:+.1%}")


# ════════════════════════════════════
# TOURNAMENT
# ════════════════════════════════════
elif '🏆' in page:
    st.markdown('# 🏆 TOURNAMENT PREDICTION')
    st.markdown('### Round-by-round probabilities · US Open 2026')
    st.divider()

    top10 = tourney.head(10).reset_index(drop=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Quarter-Final', x=top10['player'], y=top10['reach_qtrs'],
                         marker_color='#1B4F8A',
                         text=[f"{v:.0%}" for v in top10['reach_qtrs']],
                         textposition='inside', textfont=dict(size=9, color=USO_WHITE)))
    fig.add_trace(go.Bar(name='Semi-Final', x=top10['player'], y=top10['reach_semis'],
                         marker_color='#2D6FA6',
                         text=[f"{v:.0%}" for v in top10['reach_semis']],
                         textposition='inside', textfont=dict(size=9, color=USO_WHITE)))
    fig.add_trace(go.Bar(name='Final', x=top10['player'], y=top10['reach_final'],
                         marker_color=USO_GREEN,
                         text=[f"{v:.0%}" for v in top10['reach_final']],
                         textposition='inside', textfont=dict(size=9, color=USO_WHITE)))
    fig.add_trace(go.Bar(name='Win Title', x=top10['player'], y=top10['win_title'],
                         marker_color=USO_GOLD,
                         text=[f"{v:.0%}" for v in top10['win_title']],
                         textposition='inside', textfont=dict(size=9, color=USO_BLUE)))
    fig.update_layout(
        barmode='stack',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=USO_WHITE, family='Inter'),
        xaxis=dict(showgrid=False, tickangle=-20,
                   tickfont=dict(size=10, color=USO_WHITE)),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)',
                   tickformat='.0%', tickfont=dict(size=10, color=USO_WHITE)),
        legend=dict(font=dict(color=USO_WHITE), orientation='h', y=-0.2),
        margin=dict(l=0, r=0, t=20, b=60),
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown('### Full Draw Predictions')

    display = tourney.copy()
    display.index = range(1, len(display) + 1)
    display['win_title']   = display['win_title'].apply(lambda x: f'{x:.1%}')
    display['reach_final'] = display['reach_final'].apply(lambda x: f'{x:.1%}')
    display['reach_semis'] = display['reach_semis'].apply(lambda x: f'{x:.1%}')
    display['reach_qtrs']  = display['reach_qtrs'].apply(lambda x: f'{x:.1%}')
    display.columns = ['Player', 'Win Title', 'Reach Final', 'Reach Semis', 'Reach QF']
    st.dataframe(display, use_container_width=True, height=500)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ── Configuración de la página ───────────────────────────────────
st.set_page_config(
    page_title='US Open 2026 Analytics',
    page_icon='🎾',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ── Estilos CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0a1628; }
    .stApp { background-color: #0a1628; }
    h1, h2, h3 { color: #FFFFFF; }
    .metric-card {
        background: linear-gradient(135deg, #1565C0, #0D47A1);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: white;
        margin: 5px;
    }
    .winner-card {
        background: linear-gradient(135deg, #1B5E20, #2E7D32);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: white;
        font-size: 1.2em;
        font-weight: bold;
    }
    .rank-card {
        background: linear-gradient(135deg, #0D47A1, #1565C0);
        border-radius: 8px;
        padding: 10px 15px;
        color: white;
        margin: 3px 0;
    }
    .sidebar .sidebar-content { background-color: #0D47A1; }
</style>
""", unsafe_allow_html=True)

# ── Cargar datos ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        atp       = pd.read_csv('atp_matches_clean.csv')
        uso       = pd.read_csv('uso_matches_clean.csv')
        ml_model  = pd.read_csv('ml_model_features.csv')
        tourney   = pd.read_csv('tournament_predictions_uso2026.csv')
        power     = pd.read_csv('power_rankings_uso2026.csv')
        ranking   = pd.read_csv('final_ranking_uso2026.csv')
        return atp, uso, ml_model, tourney, power, ranking
    except FileNotFoundError as e:
        st.error(f'❌ Archivo no encontrado: {e}')
        st.stop()

@st.cache_data
def compute_player_stats(atp, uso):
    hard = atp[atp['surface'] == 'Hard'].copy()
    for col in ['w_ace','w_df','w_svpt','w_1stIn','w_bpSaved','w_bpFaced','l_bpSaved','l_bpFaced']:
        if col in hard.columns:
            hard[col] = pd.to_numeric(hard[col], errors='coerce')
    hard['w_1st_pct'] = np.where(hard['w_svpt']>0,   hard['w_1stIn']/hard['w_svpt'],    np.nan)
    hard['w_bp_save'] = np.where(hard['w_bpFaced']>0, hard['w_bpSaved']/hard['w_bpFaced'], np.nan)
    hard['w_bp_conv'] = np.where(hard['l_bpFaced']>0,
                                  (hard['l_bpFaced']-hard['l_bpSaved'])/hard['l_bpFaced'], np.nan)
    hard_wins   = hard.groupby('winner_name').size().rename('hard_wins')
    hard_losses = hard.groupby('loser_name').size().rename('hard_losses')
    hard_stats  = pd.concat([hard_wins, hard_losses], axis=1).fillna(0)
    hard_stats['hard_total']   = hard_stats['hard_wins'] + hard_stats['hard_losses']
    hard_stats['hard_win_pct'] = hard_stats['hard_wins'] / hard_stats['hard_total']
    hard_stats = hard_stats.reset_index().rename(columns={'index':'player'})
    hard_stats = hard_stats[hard_stats['hard_total'] >= 20]
    serve_stats = hard.groupby('winner_name').agg(
        avg_aces    = ('w_ace',     'mean'),
        avg_df      = ('w_df',      'mean'),
        avg_bp_conv = ('w_bp_conv', 'mean'),
        avg_bp_save = ('w_bp_save', 'mean'),
    ).reset_index().rename(columns={'winner_name':'player'})
    uso_wins   = uso.groupby('winner_name').size().rename('uso_wins')
    uso_losses = uso.groupby('loser_name').size().rename('uso_losses')
    uso_stats  = pd.concat([uso_wins, uso_losses], axis=1).fillna(0)
    uso_stats['uso_total']   = uso_stats['uso_wins'] + uso_stats['uso_losses']
    uso_stats['uso_win_pct'] = uso_stats['uso_wins'] / uso_stats['uso_total']
    uso_stats = uso_stats.reset_index().rename(columns={'index':'player'})
    winners = atp[['tourney_date','winner_name']].copy()
    winners['result'] = 1
    winners = winners.rename(columns={'winner_name':'player'})
    losers  = atp[['tourney_date','loser_name']].copy()
    losers['result'] = 0
    losers  = losers.rename(columns={'loser_name':'player'})
    all_m   = pd.concat([winners, losers], ignore_index=True)
    all_m['tourney_date'] = pd.to_datetime(all_m['tourney_date'], errors='coerce')
    all_m   = all_m.sort_values(['player','tourney_date'])
    total_m = all_m.groupby('player').size().reset_index(name='total_matches')
    recent  = (all_m.groupby('player').tail(10)
               .groupby('player')['result']
               .agg(recent_wins='sum', recent_total='count').reset_index())
    recent  = recent.merge(total_m, on='player', how='left')
    recent  = recent[recent['total_matches'] >= 50]
    recent['recent_form_pct'] = recent['recent_wins'] / recent['recent_total']
    ps = hard_stats.merge(serve_stats, on='player', how='left')
    ps = ps.merge(recent[['player','recent_form_pct']], on='player', how='left')
    ps = ps.merge(uso_stats[['player','uso_wins','uso_win_pct']], on='player', how='left')
    ps['uso_win_pct']     = ps['uso_win_pct'].fillna(0.5)
    ps['recent_form_pct'] = ps['recent_form_pct'].fillna(0.5)
    nuevos = pd.DataFrame([
        {'player':'Joao Fonseca', 'hard_wins':27,'hard_losses':17,'hard_total':44,
         'hard_win_pct':0.614,'avg_aces':6.0,'avg_df':0.17,'avg_bp_save':0.665,
         'avg_bp_conv':0.38,'recent_form_pct':0.60,'uso_wins':1,'uso_win_pct':0.50},
        {'player':'Jakub Mensik', 'hard_wins':21,'hard_losses':10,'hard_total':31,
         'hard_win_pct':0.677,'avg_aces':7.2,'avg_df':0.20,'avg_bp_save':0.640,
         'avg_bp_conv':0.41,'recent_form_pct':0.65,'uso_wins':0,'uso_win_pct':0.50},
        {'player':'Rafael Jodar', 'hard_wins':18,'hard_losses':9,'hard_total':27,
         'hard_win_pct':0.667,'avg_aces':5.5,'avg_df':0.18,'avg_bp_save':0.650,
         'avg_bp_conv':0.39,'recent_form_pct':0.60,'uso_wins':0,'uso_win_pct':0.50},
    ])
    ps = pd.concat([ps, nuevos], ignore_index=True)
    return ps

@st.cache_resource
def train_model(atp, player_stats):
    hard = atp[atp['surface'] == 'Hard'].copy()
    for col in ['w_ace','w_df','w_svpt','w_1stIn','w_bpSaved','w_bpFaced','l_bpSaved','l_bpFaced',
                'winner_rank','loser_rank']:
        if col in hard.columns:
            hard[col] = pd.to_numeric(hard[col], errors='coerce')
    hard['w_1st_pct'] = np.where(hard['w_svpt']>0,   hard['w_1stIn']/hard['w_svpt'],    np.nan)
    hard['w_bp_save'] = np.where(hard['w_bpFaced']>0, hard['w_bpSaved']/hard['w_bpFaced'], np.nan)
    hard['w_bp_conv'] = np.where(hard['l_bpFaced']>0,
                                  (hard['l_bpFaced']-hard['l_bpSaved'])/hard['l_bpFaced'], np.nan)
    hard['rank_diff'] = hard['loser_rank'] - hard['winner_rank']
    match_data = hard[['winner_name','loser_name','rank_diff']].copy().dropna()
    match_data = match_data.merge(
        player_stats[['player','hard_win_pct','avg_aces','avg_df','avg_bp_save','avg_bp_conv']],
        left_on='winner_name', right_on='player', how='left'
    ).rename(columns={'hard_win_pct':'w_win_pct','avg_aces':'w_avg_aces','avg_df':'w_avg_df',
                       'avg_bp_save':'w_avg_bpsave','avg_bp_conv':'w_avg_bpconv'}).drop(columns='player')
    match_data = match_data.merge(
        player_stats[['player','hard_win_pct','avg_aces','avg_df','avg_bp_save','avg_bp_conv']],
        left_on='loser_name', right_on='player', how='left'
    ).rename(columns={'hard_win_pct':'l_win_pct','avg_aces':'l_avg_aces','avg_df':'l_avg_df',
                       'avg_bp_save':'l_avg_bpsave','avg_bp_conv':'l_avg_bpconv'}).drop(columns='player')
    match_data['diff_win_pct'] = match_data['w_win_pct']    - match_data['l_win_pct']
    match_data['diff_aces']    = match_data['w_avg_aces']   - match_data['l_avg_aces']
    match_data['diff_df']      = match_data['w_avg_df']     - match_data['l_avg_df']
    match_data['diff_bpsave']  = match_data['w_avg_bpsave'] - match_data['l_avg_bpsave']
    match_data['diff_bpconv']  = match_data['w_avg_bpconv'] - match_data['l_avg_bpconv']
    match_data['label'] = 1
    mirror = match_data.copy()
    for col in ['rank_diff','diff_win_pct','diff_aces','diff_df','diff_bpsave','diff_bpconv']:
        mirror[col] = -mirror[col]
    mirror['label'] = 0
    train_df = pd.concat([match_data, mirror], ignore_index=True)
    FEATURES = ['rank_diff','diff_win_pct','diff_aces','diff_df','diff_bpsave','diff_bpconv']
    X = train_df[FEATURES].fillna(0)
    y = train_df['label']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
    rf.fit(X_train, y_train)
    return rf, FEATURES

def predict_match(player_a, player_b, rf, FEATURES, player_stats, rank_map):
    sa = player_stats[player_stats['player'] == player_a]
    sb = player_stats[player_stats['player'] == player_b]
    if len(sa)==0 or len(sb)==0:
        return 0.5, 0.5
    sa = sa.iloc[0]
    sb = sb.iloc[0]
    rank_a = rank_map.get(player_a, 50)
    rank_b = rank_map.get(player_b, 50)
    features = pd.DataFrame([{
        'rank_diff':    rank_b    - rank_a,
        'diff_win_pct': sa['hard_win_pct']  - sb['hard_win_pct'],
        'diff_aces':    sa['avg_aces']      - sb['avg_aces'],
        'diff_df':      sa['avg_df']        - sb['avg_df'],
        'diff_bpsave':  sa['avg_bp_save']   - sb['avg_bp_save'],
        'diff_bpconv':  sa['avg_bp_conv']   - sb['avg_bp_conv'],
    }])[FEATURES].fillna(0)
    prob_a = rf.predict_proba(features)[0][1]
    return prob_a, 1 - prob_a

# ── Cargar todo ──────────────────────────────────────────────────
atp, uso, ml_model, tourney, power, ranking = load_data()

for df in [atp, uso]:
    if 'tourney_date' in df.columns:
        df['tourney_date'] = pd.to_datetime(df['tourney_date'], errors='coerce')
    if 'tourney_year' not in df.columns:
        df['tourney_year'] = df['tourney_date'].dt.year
    df['surface'] = df['surface'].str.strip().str.title() if 'surface' in df.columns else df['surface']

player_stats = compute_player_stats(atp, uso)
rf, FEATURES  = train_model(atp, player_stats)

rank_map = {}
if 'full_name' in ml_model.columns and 'atp_rank' in ml_model.columns:
    rank_map = ml_model.set_index('full_name')['atp_rank'].to_dict()

all_players = sorted(player_stats['player'].tolist())

# ── Sidebar ──────────────────────────────────────────────────────
st.sidebar.markdown('# 🎾 US Open 2026')
st.sidebar.markdown('---')
page = st.sidebar.radio(
    'Navegación',
    ['🏠 Home', '👤 Player Comparison', '🎯 Match Predictor', '🏆 Tournament Prediction']
)

# ════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ════════════════════════════════════════════════════════════════
if page == '🏠 Home':
    st.markdown('# 🎾 US Open 2026 — Analytics Platform')
    st.markdown('### Predicción del ganador del US Open 2026 usando Machine Learning')
    st.markdown('---')

    # Métricas generales
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <h2>{len(atp):,}</h2><p>Partidos Analizados</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <h2>{len(uso):,}</h2><p>Partidos US Open</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <h2>{atp['tourney_year'].nunique()}</h2><p>Años de Datos</p></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <h2>128</h2><p>Jugadores en el Cuadro</p></div>""", unsafe_allow_html=True)

    st.markdown('---')

    # Top 10 contendientes
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown('### 🏆 Top 10 Contendientes US Open 2026')
        top10 = tourney.head(10).reset_index(drop=True)
        for i, row in top10.iterrows():
            medal = '🥇' if i==0 else '🥈' if i==1 else '🥉' if i==2 else f'{i+1}.'
            st.markdown(f"""<div class="rank-card">
                {medal} <b>{row['player']}</b>
                <span style="float:right">{row['win_title']:.1%} de ganar el título</span>
            </div>""", unsafe_allow_html=True)

    with col_right:
        st.markdown('### 📊 Probabilidades de Ganar el Título')
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('#0a1628')
        ax.set_facecolor('#0a1628')
        top10_s = top10.sort_values('win_title', ascending=True)
        bars = ax.barh(top10_s['player'], top10_s['win_title'],
                       color='#1565C0', edgecolor='#42A5F5', linewidth=0.5)
        ax.set_xlabel('Probabilidad', color='white')
        ax.tick_params(colors='white')
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        for spine in ax.spines.values():
            spine.set_edgecolor('#1565C0')
        for bar, v in zip(bars, top10_s['win_title']):
            ax.text(bar.get_width()+0.003, bar.get_y()+bar.get_height()/2,
                    f'{v:.1%}', va='center', color='white', fontsize=9, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown('---')
    st.markdown('### 📈 Featured Statistics')
    col1, col2, col3 = st.columns(3)

    with col1:
        best_hard = player_stats.nlargest(1, 'hard_win_pct').iloc[0]
        st.metric('🏅 Mejor Hard Court Win %',
                  f"{best_hard['hard_win_pct']:.1%}",
                  best_hard['player'])
    with col2:
        best_form = player_stats.nlargest(1, 'recent_form_pct').iloc[0]
        st.metric('🔥 Mejor Forma Reciente',
                  f"{best_form['recent_form_pct']:.1%}",
                  best_form['player'])
    with col3:
        best_uso = player_stats[player_stats['uso_wins']>3].nlargest(1, 'uso_win_pct').iloc[0]
        st.metric('🎾 Mejor US Open Win %',
                  f"{best_uso['uso_win_pct']:.1%}",
                  best_uso['player'])

# ════════════════════════════════════════════════════════════════
# PAGE 2 — PLAYER COMPARISON
# ════════════════════════════════════════════════════════════════
elif page == '👤 Player Comparison':
    st.markdown('# 👤 Player Comparison')
    st.markdown('Compara las estadísticas de dos jugadores cara a cara.')
    st.markdown('---')

    col1, col2 = st.columns(2)
    with col1:
        player_a = st.selectbox('Jugador A', all_players,
                                index=all_players.index('Jannik Sinner') if 'Jannik Sinner' in all_players else 0)
    with col2:
        player_b = st.selectbox('Jugador B', all_players,
                                index=all_players.index('Carlos Alcaraz') if 'Carlos Alcaraz' in all_players else 1)

    if player_a and player_b and player_a != player_b:
        sa = player_stats[player_stats['player'] == player_a].iloc[0] if len(player_stats[player_stats['player']==player_a])>0 else None
        sb = player_stats[player_stats['player'] == player_b].iloc[0] if len(player_stats[player_stats['player']==player_b])>0 else None

        if sa is not None and sb is not None:
            st.markdown('---')
            st.markdown(f'### {player_a} vs {player_b}')

            # Rankings
            rank_a = rank_map.get(player_a, 'N/A')
            rank_b = rank_map.get(player_b, 'N/A')

            col1, col2, col3 = st.columns([2,1,2])
            with col1:
                st.markdown(f'<h2 style="color:#42A5F5;text-align:center">{player_a}</h2>', unsafe_allow_html=True)
                st.markdown(f'<h3 style="color:white;text-align:center">ATP #{int(rank_a) if rank_a != "N/A" else "N/A"}</h3>', unsafe_allow_html=True)
            with col2:
                st.markdown('<h2 style="color:white;text-align:center">VS</h2>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<h2 style="color:#EF5350;text-align:center">{player_b}</h2>', unsafe_allow_html=True)
                st.markdown(f'<h3 style="color:white;text-align:center">ATP #{int(rank_b) if rank_b != "N/A" else "N/A"}</h3>', unsafe_allow_html=True)

            st.markdown('---')

            # Comparativa de estadísticas
            stats_compare = {
                'Hard Court Win %':    (f"{sa['hard_win_pct']:.1%}", f"{sb['hard_win_pct']:.1%}"),
                'Recent Form (last 10)':(f"{sa['recent_form_pct']:.1%}", f"{sb['recent_form_pct']:.1%}"),
                'US Open Win %':        (f"{sa['uso_win_pct']:.1%}", f"{sb['uso_win_pct']:.1%}"),
                'Avg Aces/Match':       (f"{sa['avg_aces']:.1f}", f"{sb['avg_aces']:.1f}"),
                'Avg Double Faults':    (f"{sa['avg_df']:.1f}", f"{sb['avg_df']:.1f}"),
                'BP Save %':            (f"{sa['avg_bp_save']:.1%}", f"{sb['avg_bp_save']:.1%}"),
                'BP Conversion %':      (f"{sa['avg_bp_conv']:.1%}", f"{sb['avg_bp_conv']:.1%}"),
            }

            for stat, (val_a, val_b) in stats_compare.items():
                col1, col2, col3 = st.columns([2,2,2])
                with col1:
                    st.markdown(f'<p style="color:#42A5F5;text-align:center;font-size:1.2em"><b>{val_a}</b></p>', unsafe_allow_html=True)
                with col2:
                    st.markdown(f'<p style="color:white;text-align:center">{stat}</p>', unsafe_allow_html=True)
                with col3:
                    st.markdown(f'<p style="color:#EF5350;text-align:center;font-size:1.2em"><b>{val_b}</b></p>', unsafe_allow_html=True)

            # H2H
            st.markdown('---')
            st.markdown('### Head-to-Head')
            h2h_a = len(atp[(atp['winner_name']==player_a) & (atp['loser_name']==player_b)])
            h2h_b = len(atp[(atp['winner_name']==player_b) & (atp['loser_name']==player_a)])
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f'{player_a} gana', h2h_a)
            with col2:
                st.metric('Total partidos', h2h_a + h2h_b)
            with col3:
                st.metric(f'{player_b} gana', h2h_b)

# ════════════════════════════════════════════════════════════════
# PAGE 3 — MATCH PREDICTOR
# ════════════════════════════════════════════════════════════════
elif page == '🎯 Match Predictor':
    st.markdown('# 🎯 Match Predictor')
    st.markdown('Predice el ganador de un partido entre dos jugadores.')
    st.markdown('---')

    col1, col2 = st.columns(2)
    with col1:
        player_a = st.selectbox('Jugador A', all_players,
                                index=all_players.index('Jannik Sinner') if 'Jannik Sinner' in all_players else 0,
                                key='pred_a')
    with col2:
        player_b = st.selectbox('Jugador B', all_players,
                                index=all_players.index('Carlos Alcaraz') if 'Carlos Alcaraz' in all_players else 1,
                                key='pred_b')

    if st.button('🎾 Predecir Ganador', type='primary', use_container_width=True):
        if player_a == player_b:
            st.warning('Selecciona dos jugadores diferentes.')
        else:
            prob_a, prob_b = predict_match(player_a, player_b, rf, FEATURES, player_stats, rank_map)
            winner  = player_a if prob_a > prob_b else player_b
            win_pct = max(prob_a, prob_b)

            st.markdown('---')

            # Resultado
            st.markdown(f"""<div class="winner-card">
                🏆 PREDICTED WINNER: {winner}<br>
                <span style="font-size:0.8em">Probabilidad: {win_pct:.1%}</span>
            </div>""", unsafe_allow_html=True)

            st.markdown('---')

            # Barras de probabilidad
            col1, col2 = st.columns(2)
            with col1:
                color_a = '#1565C0' if prob_a > prob_b else '#455A64'
                st.markdown(f'<h3 style="color:white;text-align:center">{player_a}</h3>', unsafe_allow_html=True)
                st.markdown(f'<h2 style="color:#42A5F5;text-align:center">{prob_a:.1%}</h2>', unsafe_allow_html=True)
                st.progress(prob_a)
            with col2:
                st.markdown(f'<h3 style="color:white;text-align:center">{player_b}</h3>', unsafe_allow_html=True)
                st.markdown(f'<h2 style="color:#EF5350;text-align:center">{prob_b:.1%}</h2>', unsafe_allow_html=True)
                st.progress(prob_b)

            st.markdown('---')
            st.markdown('### 🔑 Key Factors')

            sa = player_stats[player_stats['player']==player_a].iloc[0]
            sb = player_stats[player_stats['player']==player_b].iloc[0]
            rank_a = rank_map.get(player_a, 50)
            rank_b = rank_map.get(player_b, 50)

            factors = {
                'ATP Ranking':      (f"#{int(rank_a)}", f"#{int(rank_b)}", rank_b > rank_a),
                'Hard Court Win %': (f"{sa['hard_win_pct']:.1%}", f"{sb['hard_win_pct']:.1%}", sa['hard_win_pct'] > sb['hard_win_pct']),
                'Recent Form':      (f"{sa['recent_form_pct']:.1%}", f"{sb['recent_form_pct']:.1%}", sa['recent_form_pct'] > sb['recent_form_pct']),
                'BP Save %':        (f"{sa['avg_bp_save']:.1%}", f"{sb['avg_bp_save']:.1%}", sa['avg_bp_save'] > sb['avg_bp_save']),
            }

            for factor, (val_a, val_b, a_wins) in factors.items():
                col1, col2, col3 = st.columns([2,2,2])
                with col1:
                    icon = '✅' if a_wins else ''
                    st.markdown(f'<p style="color:#42A5F5;text-align:center">{icon} <b>{val_a}</b></p>', unsafe_allow_html=True)
                with col2:
                    st.markdown(f'<p style="color:white;text-align:center">{factor}</p>', unsafe_allow_html=True)
                with col3:
                    icon = '✅' if not a_wins else ''
                    st.markdown(f'<p style="color:#EF5350;text-align:center">{icon} <b>{val_b}</b></p>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# PAGE 4 — TOURNAMENT PREDICTION
# ════════════════════════════════════════════════════════════════
elif page == '🏆 Tournament Prediction':
    st.markdown('# 🏆 Tournament Prediction — US Open 2026')
    st.markdown('Probabilidades de cada jugador de llegar a cada ronda del torneo.')
    st.markdown('---')

    # Top 10
    st.markdown('### 🥇 Top 10 Contendientes')
    top10 = tourney.head(10).reset_index(drop=True)

    col1, col2 = st.columns([1,1])
    with col1:
        for i, row in top10.iterrows():
            medal = '🥇' if i==0 else '🥈' if i==1 else '🥉' if i==2 else f'{i+1}.'
            st.markdown(f"""<div class="rank-card">
                {medal} <b>{row['player']}</b><br>
                <small>
                Título: {row['win_title']:.1%} |
                Final: {row['reach_final']:.1%} |
                Semis: {row['reach_semis']:.1%} |
                Cuartos: {row['reach_qtrs']:.1%}
                </small>
            </div>""", unsafe_allow_html=True)

    with col2:
        fig, ax = plt.subplots(figsize=(8, 7))
        fig.patch.set_facecolor('#0a1628')
        ax.set_facecolor('#0a1628')
        rounds  = ['reach_qtrs','reach_semis','reach_final','win_title']
        labels  = ['Cuartos','Semis','Final','Título']
        colors  = ['#90CAF9','#42A5F5','#1565C0','#0D47A1']
        x = np.arange(len(top10))
        w = 0.2
        for i, (r, label, color) in enumerate(zip(rounds, labels, colors)):
            ax.bar(x+i*w, top10[r], w, label=label, color=color, edgecolor='white', linewidth=0.3)
        ax.set_xticks(x+w*1.5)
        ax.set_xticklabels([p.split()[-1] for p in top10['player']],
                           rotation=30, ha='right', color='white', fontsize=8)
        ax.set_ylabel('Probabilidad', color='white')
        ax.tick_params(colors='white')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.legend(fontsize=8, facecolor='#0D47A1', labelcolor='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#1565C0')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown('---')
    st.markdown('### 📋 Tabla Completa de Predicciones')
    display_df = tourney.copy()
    display_df['win_title']   = display_df['win_title'].apply(lambda x: f'{x:.1%}')
    display_df['reach_final'] = display_df['reach_final'].apply(lambda x: f'{x:.1%}')
    display_df['reach_semis'] = display_df['reach_semis'].apply(lambda x: f'{x:.1%}')
    display_df['reach_qtrs']  = display_df['reach_qtrs'].apply(lambda x: f'{x:.1%}')
    display_df.columns = ['Jugador','Ganar Título','Llegar Final','Llegar Semis','Llegar Cuartos']
    display_df.index = range(1, len(display_df)+1)
    st.dataframe(display_df, use_container_width=True)


"""
Day 3 - Streamlit Dashboard
Project: Defensive Packing & Pass Network Disruption Index
Author: Navjot
Date: May 27, 2026

Run with:
    streamlit run day3_dashboard.py

Requirements:
    pip install streamlit mplsoccer networkx pandas numpy matplotlib seaborn plotly
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import json
import pickle

from mplsoccer import Pitch
from statsbombpy import sb

warnings.filterwarnings("ignore")

import streamlit as st

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────

st.set_page_config(
    page_title="Defensive Packing & Network Disruption",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252a3d);
        border: 1px solid #3a3f5c;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #4ade80;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #9ca3af;
        margin-top: 4px;
    }
    .archetype-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e2e8f0;
        border-left: 3px solid #4ade80;
        padding-left: 10px;
        margin: 16px 0 12px 0;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# DATA LOADING — CACHED
# ─────────────────────────────────────────

@st.cache_data
def load_sequences():
    df = pd.read_csv("data/processed/pressing_sequences.csv")
    zone_map = {'defensive_third': 0, 'middle_third': 1, 'final_third': 2, 'unknown': 1}
    df['zone_encoded'] = df['field_zone'].map(zone_map).fillna(1).astype(int)
    df['is_winning'] = (df['scoreline_diff'] > 0).astype(int)
    df['is_losing']  = (df['scoreline_diff'] < 0).astype(int)
    df['minute_normalized'] = df['minute'] / 90.0
    df['is_late_game'] = (df['minute'] >= 75).astype(int)
    df['location_x'] = df['location_x'].fillna(60)
    df['location_y'] = df['location_y'].fillna(40)
    df['defensive_packing_norm'] = (df['defensive_packing'] / 5.0).clip(0, 1)
    df['lanes_cut_norm'] = (df['lanes_cut'] / 4.0).clip(0, 1)
    df['pressed_player_pagerank'] = df['pressed_player_pagerank'].clip(0, 1)
    df['pressing_key_node'] = (
        df['pressed_player_pagerank'] > df['pressed_player_pagerank'].quantile(0.75)
    ).astype(int)
    df['combined_press_value'] = (
        0.40 * df['defensive_packing_norm'] +
        0.35 * df['lanes_cut_norm'] +
        0.25 * (df['pressed_player_pagerank'] * 10).clip(0, 1)
    )
    return df

@st.cache_data
def load_team_archetypes():
    return pd.read_csv("outputs/reports/team_archetypes.csv")

@st.cache_data
def load_player_leaderboard():
    return pd.read_csv("outputs/reports/player_leaderboard.csv")

@st.cache_data
def load_matches():
    return pd.read_csv("data/raw/matches.csv")

@st.cache_data
def load_network_metrics():
    with open("data/processed/network_metrics.json", "r") as f:
        return json.load(f)

@st.cache_data
def build_team_stats(df):
    return df.groupby('pressing_team').agg(
        total_presses=('press_success', 'count'),
        press_success_rate=('press_success', 'mean'),
        avg_defensive_packing=('defensive_packing_norm', 'mean'),
        avg_lanes_cut=('lanes_cut_norm', 'mean'),
        avg_combined_value=('combined_press_value', 'mean'),
        avg_network_disruption=('pressed_player_pagerank', 'mean'),
        final_third_pct=('zone_encoded', lambda x: (x == 2).mean()),
        key_node_press_pct=('pressing_key_node', 'mean'),
    ).reset_index()

@st.cache_data
def get_match_events(match_id):
    return sb.events(match_id=int(match_id))


# ─────────────────────────────────────────
# LOAD ALL DATA
# ─────────────────────────────────────────

df          = load_sequences()
archetypes  = load_team_archetypes()
players     = load_player_leaderboard()
matches     = load_matches()
net_metrics = load_network_metrics()
team_stats  = build_team_stats(df)

ARCHETYPE_COLORS = {
    'High Press Hunters': '#ef4444',
    'Network Disruptors': '#3b82f6',
    'Efficient Pressers': '#22c55e',
    'Reactive Mid-Block': '#f59e0b'
}

ALL_TEAMS = sorted(df['pressing_team'].unique().tolist())


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚽ Defensive Packing")
    st.markdown("#### Pass Network Disruption Index")
    st.markdown("---")
    st.markdown("**FIFA World Cup 2022**")
    st.markdown(f"*{len(df):,} pressing sequences*")
    st.markdown(f"*{df['pressing_team'].nunique()} teams*")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Overview", "👥 Team Profile", "🗺️ Match View", "🕸️ Pass Network", "🏆 Leaderboard"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown(
        "<small>Built by Navjot · StatsBomb Open Data</small>",
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────
# PAGE 1: OVERVIEW
# ─────────────────────────────────────────

if page == "🏠 Overview":
    st.title("Defensive Packing & Pass Network Disruption")
    st.markdown("*Quantifying the spatial and network cost of pressing — FIFA World Cup 2022*")
    st.markdown("---")

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-label">Total Press Sequences</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{df['press_success'].mean():.1%}</div>
            <div class="metric-label">Overall Press Success Rate</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{df['combined_press_value'].mean():.3f}</div>
            <div class="metric-label">Avg Combined Press Value</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{df['has_freeze_frame'].sum():,}</div>
            <div class="metric-label">Presses with Spatial Data</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-header">Press Success Rate by Field Zone</div>', unsafe_allow_html=True)
        zone_data = df.groupby('field_zone').agg(
            success_rate=('press_success', 'mean'),
            count=('press_success', 'count')
        ).reindex(['defensive_third', 'middle_third', 'final_third']).reset_index()
        zone_data['field_zone'] = ['Defensive Third', 'Middle Third', 'Final Third']

        fig = px.bar(
            zone_data, x='field_zone', y='success_rate',
            text=zone_data['count'].apply(lambda x: f'n={x}'),
            color='success_rate',
            color_continuous_scale='RdYlGn',
            labels={'success_rate': 'Success Rate', 'field_zone': 'Zone'}
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#e2e8f0',
            showlegend=False,
            coloraxis_showscale=False,
            height=320
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">Combined Press Value Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(
            df, x='combined_press_value', nbins=40,
            color_discrete_sequence=['#4ade80'],
            labels={'combined_press_value': 'Combined Press Value'}
        )
        fig.add_vline(
            x=df['combined_press_value'].mean(),
            line_dash="dash", line_color="#ef4444",
            annotation_text=f"Mean: {df['combined_press_value'].mean():.3f}",
            annotation_font_color="#ef4444"
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#e2e8f0',
            height=320
        )
        st.plotly_chart(fig, use_container_width=True)

    # Archetype summary table
    st.markdown('<div class="section-header">Team Pressing Archetypes</div>', unsafe_allow_html=True)
    arch_summary = archetypes.groupby('archetype').agg(
        teams=('pressing_team', 'count'),
        avg_success=('press_success_rate', 'mean'),
        avg_combined=('avg_combined_value', 'mean'),
        avg_final_third=('final_third_pct', 'mean'),
    ).reset_index()

    for _, row in arch_summary.iterrows():
        color = ARCHETYPE_COLORS.get(row['archetype'], '#6b7280')
        col_a, col_b, col_c, col_d, col_e = st.columns([3, 1, 2, 2, 2])
        with col_a:
            st.markdown(
                f'<span style="color:{color};font-weight:600">● {row["archetype"]}</span>',
                unsafe_allow_html=True
            )
        with col_b:
            st.markdown(f"`{int(row['teams'])} teams`")
        with col_c:
            st.markdown(f"Success: **{row['avg_success']:.1%}**")
        with col_d:
            st.markdown(f"Press Value: **{row['avg_combined']:.3f}**")
        with col_e:
            st.markdown(f"Final Third: **{row['avg_final_third']:.1%}**")


# ─────────────────────────────────────────
# PAGE 2: TEAM PROFILE
# ─────────────────────────────────────────

elif page == "👥 Team Profile":
    st.title("Team Profile")
    st.markdown("---")

    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        team_a = st.selectbox("Select Team A", ALL_TEAMS, index=ALL_TEAMS.index("Argentina") if "Argentina" in ALL_TEAMS else 0)
    with col_sel2:
        team_b = st.selectbox("Select Team B (comparison)", ["None"] + ALL_TEAMS, index=0)

    teams_to_show = [team_a] if team_b == "None" else [team_a, team_b]

    for team in teams_to_show:
        t_data = archetypes[archetypes['pressing_team'] == team]
        t_seq = df[df['pressing_team'] == team]
        archetype = t_data['archetype'].values[0] if len(t_data) > 0 else "Unknown"
        arch_color = ARCHETYPE_COLORS.get(archetype, '#6b7280')

        st.markdown(f"### {team}")
        st.markdown(
            f'<span style="background:{arch_color};color:white;padding:3px 12px;border-radius:12px;font-size:0.85rem">{archetype}</span>',
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Total Presses", f"{len(t_seq):,}")
        with m2:
            st.metric("Success Rate", f"{t_seq['press_success'].mean():.1%}")
        with m3:
            st.metric("Avg Press Value", f"{t_seq['combined_press_value'].mean():.3f}")
        with m4:
            st.metric("Final Third %", f"{(t_seq['zone_encoded'] == 2).mean():.1%}")
        with m5:
            st.metric("Key Node Pressed %", f"{t_seq['pressing_key_node'].mean():.1%}")

        # Radar chart
        RADAR_COLS = ['press_success_rate', 'avg_defensive_packing', 'avg_lanes_cut',
                      'avg_network_disruption', 'final_third_pct', 'key_node_press_pct']
        RADAR_LABELS = ['Success Rate', 'Def Packing', 'Lanes Cut',
                        'Network Disruption', 'Final Third %', 'Key Node %']

        t_stats_row = team_stats[team_stats['pressing_team'] == team]
        if len(t_stats_row) > 0:
            row = t_stats_row.iloc[0]
            mins = team_stats[RADAR_COLS].min()
            maxs = team_stats[RADAR_COLS].max()
            vals = [(row[c] - mins[c]) / (maxs[c] - mins[c] + 1e-9) for c in RADAR_COLS]
            vals_closed = vals + [vals[0]]
            labels_closed = RADAR_LABELS + [RADAR_LABELS[0]]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=vals_closed,
                theta=labels_closed,
                fill='toself',
                fillcolor=f'rgba({int(arch_color[1:3],16)},{int(arch_color[3:5],16)},{int(arch_color[5:7],16)},0.2)',
                line=dict(color=arch_color, width=2),
                name=team
            ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(color='#9ca3af', size=9)),
                    angularaxis=dict(tickfont=dict(color='#e2e8f0', size=11)),
                    bgcolor='rgba(0,0,0,0)'
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e2e8f0',
                showlegend=False,
                height=380
            )
            st.plotly_chart(fig, use_container_width=True)

        # Press success by minute
        st.markdown('<div class="section-header">Press Success Rate by Minute</div>', unsafe_allow_html=True)
        minute_bins = pd.cut(t_seq['minute'], bins=range(0, 100, 10))
        minute_success = t_seq.groupby(minute_bins, observed=True)['press_success'].mean().reset_index()
        minute_success['minute'] = minute_success['minute'].astype(str)

        fig2 = px.line(
            minute_success, x='minute', y='press_success',
            markers=True, color_discrete_sequence=[arch_color],
            labels={'press_success': 'Success Rate', 'minute': 'Minute Range'}
        )
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#e2e8f0',
            height=280
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("---")


# ─────────────────────────────────────────
# PAGE 3: MATCH VIEW
# ─────────────────────────────────────────

elif page == "🗺️ Match View":
    st.title("Match Press Map")
    st.markdown("---")

    # Match selector
    matches['display'] = matches.apply(
        lambda r: f"{r['home_team']} vs {r['away_team']} ({r.get('match_date','')[:10]})",
        axis=1
    )
    match_display = st.selectbox("Select Match", matches['display'].tolist())
    selected_match = matches[matches['display'] == match_display].iloc[0]
    match_id = selected_match['match_id']

    match_seq = df[df['match_id'] == match_id]

    if len(match_seq) == 0:
        st.warning("No pressing data for this match.")
    else:
        home_team = selected_match['home_team']
        away_team = selected_match['away_team']

        team_filter = st.radio("Show pressing for:", [home_team, away_team, "Both"], horizontal=True)

        if team_filter != "Both":
            plot_seq = match_seq[match_seq['pressing_team'] == team_filter]
        else:
            plot_seq = match_seq

        col_stats1, col_stats2, col_stats3 = st.columns(3)
        with col_stats1:
            st.metric("Total Presses", len(plot_seq))
        with col_stats2:
            st.metric("Success Rate", f"{plot_seq['press_success'].mean():.1%}")
        with col_stats3:
            st.metric("Avg Press Value", f"{plot_seq['combined_press_value'].mean():.3f}")

        # Pitch plot
        fig, ax = plt.subplots(figsize=(14, 9))
        fig.patch.set_facecolor('#0e1117')

        pitch = Pitch(pitch_type='statsbomb', pitch_color='#1a1f35',
                      line_color='#4a5568', goal_type='box')
        pitch.draw(ax=ax)

        success = plot_seq[plot_seq['press_success'] == 1]
        fail    = plot_seq[plot_seq['press_success'] == 0]

        pitch.scatter(
            fail['location_x'], fail['location_y'], ax=ax,
            s=fail['combined_press_value'] * 400 + 30,
            color='#ef4444', alpha=0.6, edgecolors='white', linewidths=0.3, zorder=4,
            label='Press Failed'
        )
        pitch.scatter(
            success['location_x'], success['location_y'], ax=ax,
            s=success['combined_press_value'] * 400 + 30,
            color='#22c55e', alpha=0.7, edgecolors='white', linewidths=0.3, zorder=5,
            label='Press Won'
        )

        legend_elements = [
            mpatches.Patch(color='#22c55e', label=f'Press Won ({len(success)})'),
            mpatches.Patch(color='#ef4444', label=f'Press Failed ({len(fail)})'),
        ]
        ax.legend(handles=legend_elements, loc='upper left',
                  facecolor='#1a1f35', labelcolor='white', fontsize=11,
                  framealpha=0.8)
        ax.set_title(
            f"{home_team} vs {away_team} — Press Map\n(size = Combined Press Value)",
            color='white', fontsize=13, pad=10
        )

        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Zone breakdown
        st.markdown('<div class="section-header">Presses by Zone</div>', unsafe_allow_html=True)
        zone_breakdown = plot_seq.groupby('field_zone').agg(
            count=('press_success', 'count'),
            success_rate=('press_success', 'mean')
        ).reset_index()

        fig3 = px.bar(
            zone_breakdown, x='field_zone', y='count',
            color='success_rate', color_continuous_scale='RdYlGn',
            text='count',
            labels={'field_zone': 'Zone', 'count': 'Press Count', 'success_rate': 'Success Rate'}
        )
        fig3.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#e2e8f0',
            height=280
        )
        st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────
# PAGE 4: PASS NETWORK
# ─────────────────────────────────────────

elif page == "🕸️ Pass Network":
    st.title("Pass Network & Disruption")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        selected_team_net = st.selectbox("Select Team", ALL_TEAMS)
    with col2:
        team_matches = matches[
            (matches['home_team'] == selected_team_net) |
            (matches['away_team'] == selected_team_net)
        ]
        match_options = team_matches.apply(
            lambda r: f"{r['home_team']} vs {r['away_team']}", axis=1
        ).tolist()
        selected_match_net = st.selectbox("Select Match", match_options)

    selected_match_row = team_matches[
        team_matches.apply(lambda r: f"{r['home_team']} vs {r['away_team']}" == selected_match_net, axis=1)
    ].iloc[0]
    match_id_net = str(selected_match_row['match_id'])

    team_net_data = net_metrics.get(match_id_net, {}).get(selected_team_net, None)

    if team_net_data is None:
        st.warning(f"No network data found for {selected_team_net} in this match.")
    else:
        pagerank = team_net_data.get('pagerank', {})
        betweenness = team_net_data.get('betweenness', {})
        in_degree = team_net_data.get('in_degree', {})
        out_degree = team_net_data.get('out_degree', {})

        # Rebuild graph from metrics
        events_net = get_match_events(match_id_net)
        team_passes = events_net[
            (events_net['type'] == 'Pass') &
            (events_net['team'] == selected_team_net) &
            (events_net['pass_outcome'].isna())
        ]

        G = nx.DiGraph()
        for _, row in team_passes.iterrows():
            passer = row.get('player')
            recipient = row.get('pass_recipient')
            if pd.notna(passer) and pd.notna(recipient):
                if G.has_edge(passer, recipient):
                    G[passer][recipient]['weight'] += 1
                else:
                    G.add_edge(passer, recipient, weight=1)

        if len(G.nodes) == 0:
            st.warning("No pass network data available.")
        else:
            # Get pressed players in this match
            pressed_in_match = df[
                (df['match_id'] == int(match_id_net)) &
                (df['pressing_team'] != selected_team_net)
            ]['pressed_player'].dropna().unique().tolist()

            # Plotly network
            pos = nx.spring_layout(G, seed=42, k=3.0)

            edge_traces = []
            for u, v, data in G.edges(data=True):
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                weight = data.get('weight', 1)
                edge_traces.append(go.Scatter(
                    x=[x0, x1, None], y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=min(weight / 3, 4), color='rgba(100,150,255,0.4)'),
                    hoverinfo='none'
                ))

            node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
            for node in G.nodes:
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                pr = pagerank.get(node, 0)
                bt = betweenness.get(node, 0)
                short_name = str(node).split()[-1] if node else "?"
                is_pressed = node in pressed_in_match
                node_text.append(
                    f"<b>{node}</b><br>PageRank: {pr:.4f}<br>Betweenness: {bt:.4f}<br>"
                    + ("⚠️ Pressed" if is_pressed else "")
                )
                node_color.append('#ef4444' if is_pressed else '#3b82f6')
                node_size.append(max(pr * 800, 15))

            node_trace = go.Scatter(
                x=node_x, y=node_y, mode='markers+text',
                hoverinfo='text', hovertext=node_text,
                text=[str(n).split()[-1] for n in G.nodes],
                textposition='top center',
                textfont=dict(size=9, color='white'),
                marker=dict(
                    size=node_size, color=node_color,
                    line=dict(width=1, color='white')
                )
            )

            fig_net = go.Figure(data=edge_traces + [node_trace])
            fig_net.update_layout(
                showlegend=False,
                hovermode='closest',
                plot_bgcolor='rgba(15,20,40,1)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                height=550,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                title=dict(
                    text=f"{selected_team_net} Pass Network<br><sup>Node size = PageRank · Red = Pressed by opponent</sup>",
                    font=dict(color='white', size=13)
                )
            )
            st.plotly_chart(fig_net, use_container_width=True)

            # Top nodes table
            st.markdown('<div class="section-header">Key Network Players</div>', unsafe_allow_html=True)
            node_df = pd.DataFrame({
                'Player': list(pagerank.keys()),
                'PageRank': list(pagerank.values()),
                'Betweenness': [betweenness.get(p, 0) for p in pagerank.keys()],
                'Passes Received': [in_degree.get(p, 0) for p in pagerank.keys()],
                'Passes Made': [out_degree.get(p, 0) for p in pagerank.keys()],
                'Was Pressed': [p in pressed_in_match for p in pagerank.keys()],
            }).sort_values('PageRank', ascending=False).head(11)

            st.dataframe(
                node_df.style.format({'PageRank': '{:.4f}', 'Betweenness': '{:.4f}'}),
                use_container_width=True, hide_index=True
            )


# ─────────────────────────────────────────
# PAGE 5: LEADERBOARD
# ─────────────────────────────────────────

elif page == "🏆 Leaderboard":
    st.title("Player Leaderboard")
    st.markdown("*Players ranked by network importance — most valuable to press*")
    st.markdown("---")

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        min_presses = st.slider("Min times pressed", 1, 20, 5)
    with col_f2:
        team_filter_lb = st.selectbox("Filter by team", ["All"] + ALL_TEAMS)
    with col_f3:
        sort_by = st.selectbox("Sort by", ["avg_pagerank", "avg_betweenness", "times_pressed", "press_success_against"])

    filtered = players[players['times_pressed'] >= min_presses].copy()
    if team_filter_lb != "All":
        filtered = filtered[filtered['pressing_team'] == team_filter_lb]
    filtered = filtered.sort_values(sort_by, ascending=False).reset_index(drop=True)

    # Top 3 highlight
    if len(filtered) >= 3:
        st.markdown("#### Top 3 Most Valuable Press Targets")
        c1, c2, c3 = st.columns(3)
        for col, idx, medal in zip([c1, c2, c3], [0, 1, 2], ["🥇", "🥈", "🥉"]):
            row = filtered.iloc[idx]
            with col:
                st.markdown(f"""<div class="metric-card">
                    <div style="font-size:1.5rem">{medal}</div>
                    <div style="font-weight:600;color:#e2e8f0;margin:8px 0">{row['pressed_player']}</div>
                    <div style="color:#9ca3af;font-size:0.8rem">{row['pressing_team']}</div>
                    <div class="metric-value" style="font-size:1.4rem">{row['avg_pagerank']:.4f}</div>
                    <div class="metric-label">PageRank</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Full Leaderboard")

    display_cols = ['pressed_player', 'pressing_team', 'times_pressed',
                    'avg_pagerank', 'avg_betweenness', 'press_success_against',
                    'avg_combined_value_when_pressed']
    display_cols = [c for c in display_cols if c in filtered.columns]

    st.dataframe(
        filtered[display_cols].head(50).style.format({
            'avg_pagerank': '{:.4f}',
            'avg_betweenness': '{:.4f}',
            'press_success_against': '{:.2%}',
            'avg_combined_value_when_pressed': '{:.4f}'
        }).background_gradient(subset=['avg_pagerank'], cmap='YlOrRd'),
        use_container_width=True,
        hide_index=True
    )

    # Scatter: PageRank vs Press Success Against
    st.markdown('<div class="section-header">Network Importance vs Press Vulnerability</div>', unsafe_allow_html=True)
    st.markdown("*High PageRank + High success against = ideal press target*")

    scatter_data = filtered.head(100)
    fig_scatter = px.scatter(
        scatter_data,
        x='avg_pagerank', y='press_success_against',
        size='times_pressed', color='pressing_team',
        hover_name='pressed_player',
        labels={
            'avg_pagerank': 'Network Importance (PageRank)',
            'press_success_against': 'Press Success Rate Against',
            'pressing_team': 'Opponent Team'
        }
    )
    fig_scatter.add_hline(y=0.37, line_dash="dash", line_color="white", opacity=0.4,
                          annotation_text="Avg success rate")
    fig_scatter.update_layout(
        plot_bgcolor='rgba(15,20,40,1)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#e2e8f0',
        height=420
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
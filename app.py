"""
Defensive Packing & Pass Network Disruption Index
Run with: streamlit run app.py
"""
import streamlit as st
from data_loader import (load_sequences, load_team_archetypes, load_player_leaderboard,
                         load_matches, load_network_metrics, build_team_stats)
from views_overview import render_overview
from views_team import render_team_profile, render_match_view
from views_network_lb import render_pass_network, render_leaderboard

st.set_page_config(
    page_title="Defensive Packing & Network Disruption",
    page_icon="⚽", layout="wide", initial_sidebar_state="expanded"
)

st.markdown("""<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg,#1e2130,#252a3d);
        border: 1px solid #3a3f5c; border-radius: 10px;
        padding: 16px 20px; text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #4ade80; }
    .metric-label { font-size: 0.85rem; color: #9ca3af; margin-top: 4px; }
    .section-header {
        font-size: 1.1rem; font-weight: 600; color: #e2e8f0;
        border-left: 3px solid #4ade80; padding-left: 10px; margin: 16px 0 12px 0;
    }
</style>""", unsafe_allow_html=True)

df          = load_sequences()
archetypes  = load_team_archetypes()
players     = load_player_leaderboard()
matches     = load_matches()
net_metrics = load_network_metrics()
team_stats  = build_team_stats(df)
ALL_TEAMS   = sorted(df['pressing_team'].unique().tolist())

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
    st.markdown("<small>Built by Navjot · StatsBomb Open Data</small>", unsafe_allow_html=True)

if   page == "🏠 Overview":     render_overview(df, archetypes, team_stats)
elif page == "👥 Team Profile":  render_team_profile(df, archetypes, team_stats, ALL_TEAMS)
elif page == "🗺️ Match View":   render_match_view(df, matches)
elif page == "🕸️ Pass Network": render_pass_network(df, matches, net_metrics, ALL_TEAMS)
elif page == "🏆 Leaderboard":  render_leaderboard(df, players, ALL_TEAMS)
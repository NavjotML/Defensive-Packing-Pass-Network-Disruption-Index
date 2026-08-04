import json
import warnings
import pandas as pd
import streamlit as st
from statsbombpy import sb

warnings.filterwarnings("ignore")

ARCHETYPE_COLORS = {
    'High Press Hunters': '#ef4444',
    'Network Disruptors': '#3b82f6',
    'Efficient Pressers': '#22c55e',
    'Reactive Mid-Block': '#f59e0b',
}


@st.cache_data
def load_sequences():
    df = pd.read_csv("data/processed/pressing_sequences.csv")
    zone_map = {'defensive_third': 0, 'middle_third': 1, 'final_third': 2, 'unknown': 1}
    df['zone_encoded']            = df['field_zone'].map(zone_map).fillna(1).astype(int)
    df['is_winning']              = (df['scoreline_diff'] > 0).astype(int)
    df['is_losing']               = (df['scoreline_diff'] < 0).astype(int)
    df['minute_normalized']       = df['minute'] / 90.0
    df['is_late_game']            = (df['minute'] >= 75).astype(int)
    df['location_x']              = df['location_x'].fillna(60)
    df['location_y']              = df['location_y'].fillna(40)
    df['defensive_packing_norm']  = (df['defensive_packing'] / 5.0).clip(0, 1)
    df['lanes_cut_norm']          = (df['lanes_cut'] / 4.0).clip(0, 1)
    df['pressed_player_pagerank'] = df['pressed_player_pagerank'].clip(0, 1)
    df['pressing_key_node'] = (
        df['pressed_player_pagerank'] > df['pressed_player_pagerank'].quantile(0.75)
    ).astype(int)
    df['combined_press_value'] = (
        0.40 * df['defensive_packing_norm']
        + 0.35 * df['lanes_cut_norm']
        + 0.25 * (df['pressed_player_pagerank'] * 10).clip(0, 1)
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
        total_presses         = ('press_success', 'count'),
        press_success_rate    = ('press_success', 'mean'),
        avg_defensive_packing = ('defensive_packing_norm', 'mean'),
        avg_lanes_cut         = ('lanes_cut_norm', 'mean'),
        avg_combined_value    = ('combined_press_value', 'mean'),
        avg_network_disruption= ('pressed_player_pagerank', 'mean'),
        final_third_pct       = ('zone_encoded', lambda x: (x == 2).mean()),
        key_node_press_pct    = ('pressing_key_node', 'mean'),
    ).reset_index()


@st.cache_data
def get_match_events(match_id):
    return sb.events(match_id=int(match_id))

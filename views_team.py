import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from mplsoccer import Pitch
from data_loader import ARCHETYPE_COLORS

RADAR_COLS   = ['press_success_rate','avg_defensive_packing','avg_lanes_cut','avg_network_disruption','final_third_pct','key_node_press_pct']
RADAR_LABELS = ['Success Rate','Def Packing','Lanes Cut','Network Disruption','Final Third %','Key Node %']


def render_team_profile(df, archetypes, team_stats, ALL_TEAMS):
    st.title("Team Profile"); st.markdown("---")
    c1, c2 = st.columns(2)
    with c1: team_a = st.selectbox("Select Team A", ALL_TEAMS, index=ALL_TEAMS.index("Argentina") if "Argentina" in ALL_TEAMS else 0)
    with c2: team_b = st.selectbox("Select Team B (comparison)", ["None"] + ALL_TEAMS, index=0)

    for team in ([team_a] if team_b == "None" else [team_a, team_b]):
        t_data     = archetypes[archetypes['pressing_team'] == team]
        t_seq      = df[df['pressing_team'] == team]
        archetype  = t_data['archetype'].values[0] if len(t_data) else "Unknown"
        arch_color = ARCHETYPE_COLORS.get(archetype, '#6b7280')
        st.markdown(f"### {team}")
        st.markdown(f'<span style="background:{arch_color};color:white;padding:3px 12px;border-radius:12px;font-size:0.85rem">{archetype}</span>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Presses", f"{len(t_seq):,}")
        m2.metric("Success Rate", f"{t_seq['press_success'].mean():.1%}")
        m3.metric("Avg Press Value", f"{t_seq['combined_press_value'].mean():.3f}")
        m4.metric("Final Third %", f"{(t_seq['zone_encoded'] == 2).mean():.1%}")
        m5.metric("Key Node Pressed %", f"{t_seq['pressing_key_node'].mean():.1%}")
        t_row = team_stats[team_stats['pressing_team'] == team]
        if len(t_row):
            row = t_row.iloc[0]
            mins, maxs = team_stats[RADAR_COLS].min(), team_stats[RADAR_COLS].max()
            vals = [(row[c] - mins[c]) / (maxs[c] - mins[c] + 1e-9) for c in RADAR_COLS]
            r, g, b = int(arch_color[1:3], 16), int(arch_color[3:5], 16), int(arch_color[5:7], 16)
            fig = go.Figure(go.Scatterpolar(r=vals + [vals[0]], theta=RADAR_LABELS + [RADAR_LABELS[0]],
                fill='toself', fillcolor=f'rgba({r},{g},{b},0.2)', line=dict(color=arch_color, width=2), name=team))
            fig.update_layout(polar=dict(
                radialaxis=dict(visible=True, range=[0,1], tickfont=dict(color='#9ca3af', size=9)),
                angularaxis=dict(tickfont=dict(color='#e2e8f0', size=11)), bgcolor='rgba(0,0,0,0)'),
                paper_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0', showlegend=False, height=380)
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="section-header">Press Success Rate by Minute</div>', unsafe_allow_html=True)
        minute_data = t_seq.groupby(pd.cut(t_seq['minute'], bins=range(0,100,10)), observed=True)['press_success'].mean().reset_index()
        minute_data['minute'] = minute_data['minute'].astype(str)
        fig2 = px.line(minute_data, x='minute', y='press_success', markers=True,
                       color_discrete_sequence=[arch_color], labels={'press_success':'Success Rate','minute':'Minute Range'})
        fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0', height=280)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("---")


def render_match_view(df, matches):
    st.title("Match Press Map"); st.markdown("---")
    matches['display'] = matches.apply(lambda r: f"{r['home_team']} vs {r['away_team']} ({r.get('match_date','')[:10]})", axis=1)
    sel       = matches[matches['display'] == st.selectbox("Select Match", matches['display'].tolist())].iloc[0]
    match_seq = df[df['match_id'] == sel['match_id']]
    if len(match_seq) == 0:
        st.warning("No pressing data for this match."); return
    team_filter = st.radio("Show pressing for:", [sel['home_team'], sel['away_team'], "Both"], horizontal=True)
    plot_seq = match_seq if team_filter == "Both" else match_seq[match_seq['pressing_team'] == team_filter]
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Presses", len(plot_seq))
    c2.metric("Success Rate", f"{plot_seq['press_success'].mean():.1%}")
    c3.metric("Avg Press Value", f"{plot_seq['combined_press_value'].mean():.3f}")
    fig, ax = plt.subplots(figsize=(14, 9)); fig.patch.set_facecolor('#0e1117')
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#1a1f35', line_color='#4a5568', goal_type='box')
    pitch.draw(ax=ax)
    success, fail = plot_seq[plot_seq['press_success'] == 1], plot_seq[plot_seq['press_success'] == 0]
    pitch.scatter(fail['location_x'], fail['location_y'], ax=ax,
                  s=fail['combined_press_value']*400+30, color='#ef4444', alpha=0.6, edgecolors='white', linewidths=0.3, zorder=4)
    pitch.scatter(success['location_x'], success['location_y'], ax=ax,
                  s=success['combined_press_value']*400+30, color='#22c55e', alpha=0.7, edgecolors='white', linewidths=0.3, zorder=5)
    ax.legend(handles=[mpatches.Patch(color='#22c55e', label=f'Press Won ({len(success)})'),
                       mpatches.Patch(color='#ef4444', label=f'Press Failed ({len(fail)})')],
              loc='upper left', facecolor='#1a1f35', labelcolor='white', fontsize=11, framealpha=0.8)
    ax.set_title(f"{sel['home_team']} vs {sel['away_team']} — Press Map\n(size = Combined Press Value)", color='white', fontsize=13, pad=10)
    st.pyplot(fig, use_container_width=True); plt.close()
    st.markdown('<div class="section-header">Presses by Zone</div>', unsafe_allow_html=True)
    zone_data = plot_seq.groupby('field_zone').agg(count=('press_success','count'), success_rate=('press_success','mean')).reset_index()
    fig3 = px.bar(zone_data, x='field_zone', y='count', color='success_rate', color_continuous_scale='RdYlGn',
                  text='count', labels={'field_zone':'Zone','count':'Press Count','success_rate':'Success Rate'})
    fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0', height=280)
    st.plotly_chart(fig3, use_container_width=True)

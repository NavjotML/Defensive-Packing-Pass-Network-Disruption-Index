import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from data_loader import get_match_events


def render_pass_network(df, matches, net_metrics, ALL_TEAMS):
    st.title("Pass Network & Disruption"); st.markdown("---")
    c1, c2 = st.columns(2)
    with c1: selected_team = st.selectbox("Select Team", ALL_TEAMS)
    with c2:
        team_matches = matches[(matches['home_team'] == selected_team) | (matches['away_team'] == selected_team)]
        match_opts   = team_matches.apply(lambda r: f"{r['home_team']} vs {r['away_team']}", axis=1).tolist()
        sel_match    = st.selectbox("Select Match", match_opts)
    sel_row   = team_matches[team_matches.apply(lambda r: f"{r['home_team']} vs {r['away_team']}" == sel_match, axis=1)].iloc[0]
    match_id  = str(sel_row['match_id'])
    team_data = net_metrics.get(match_id, {}).get(selected_team, None)
    if team_data is None:
        st.warning(f"No network data for {selected_team}."); return
    pagerank, betweenness = team_data.get('pagerank', {}), team_data.get('betweenness', {})
    in_degree, out_degree = team_data.get('in_degree', {}), team_data.get('out_degree', {})
    events = get_match_events(match_id)
    passes = events[(events['type'] == 'Pass') & (events['team'] == selected_team) & (events['pass_outcome'].isna())]
    G = nx.DiGraph()
    for _, row in passes.iterrows():
        p, r = row.get('player'), row.get('pass_recipient')
        if pd.notna(p) and pd.notna(r):
            if G.has_edge(p, r): G[p][r]['weight'] += 1
            else: G.add_edge(p, r, weight=1)
    if not G.nodes:
        st.warning("No pass network data available."); return
    pressed = set(df[(df['match_id'] == int(match_id)) & (df['pressing_team'] != selected_team)]['pressed_player'].dropna())
    pos     = nx.spring_layout(G, seed=42, k=3.0)
    edge_traces = [go.Scatter(x=[pos[u][0],pos[v][0],None], y=[pos[u][1],pos[v][1],None], mode='lines',
                              hoverinfo='none', line=dict(width=min(d.get('weight',1)/3,4), color='rgba(100,150,255,0.4)'))
                   for u, v, d in G.edges(data=True)]
    nodes = list(G.nodes)
    node_trace = go.Scatter(
        x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode='markers+text',
        text=[str(n).split()[-1] for n in nodes], textposition='top center',
        textfont=dict(size=9, color='white'), hoverinfo='text',
        hovertext=[f"<b>{n}</b><br>PageRank:{pagerank.get(n,0):.4f}<br>Betweenness:{betweenness.get(n,0):.4f}"
                   f"<br>{'⚠️ Pressed' if n in pressed else ''}" for n in nodes],
        marker=dict(size=[max(pagerank.get(n,0)*800,15) for n in nodes],
                    color=['#ef4444' if n in pressed else '#3b82f6' for n in nodes], line=dict(width=1, color='white')))
    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(showlegend=False, hovermode='closest', height=550, plot_bgcolor='rgba(15,20,40,1)',
                      paper_bgcolor='rgba(0,0,0,0)', font_color='white',
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      title=dict(text=f"{selected_team} Pass Network<br><sup>Node size=PageRank · Red=Pressed</sup>", font=dict(color='white',size=13)))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="section-header">Key Network Players</div>', unsafe_allow_html=True)
    node_df = pd.DataFrame({'Player': list(pagerank), 'PageRank': list(pagerank.values()),
                            'Betweenness': [betweenness.get(p,0) for p in pagerank],
                            'Passes Received': [in_degree.get(p,0) for p in pagerank],
                            'Passes Made': [out_degree.get(p,0) for p in pagerank],
                            'Was Pressed': [p in pressed for p in pagerank],
                            }).sort_values('PageRank', ascending=False).head(11)
    st.dataframe(node_df.style.format({'PageRank':'{:.4f}','Betweenness':'{:.4f}'}), use_container_width=True, hide_index=True)


def render_leaderboard(df, players, ALL_TEAMS):
    st.title("Player Leaderboard")
    st.markdown("*Players ranked by network importance — most valuable to press*"); st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1: min_presses = st.slider("Min times pressed", 1, 20, 5)
    with c2: team_filter = st.selectbox("Filter by team", ["All"] + ALL_TEAMS)
    with c3: sort_by     = st.selectbox("Sort by", ["avg_pagerank","avg_betweenness","times_pressed","press_success_against"])
    filtered = players[players['times_pressed'] >= min_presses].copy()
    if team_filter != "All": filtered = filtered[filtered['pressing_team'] == team_filter]
    filtered = filtered.sort_values(sort_by, ascending=False).reset_index(drop=True)
    if len(filtered) >= 3:
        st.markdown("#### Top 3 Most Valuable Press Targets")
        for col, idx, medal in zip(st.columns(3), [0,1,2], ["🥇","🥈","🥉"]):
            row = filtered.iloc[idx]
            with col:
                st.markdown(f'<div class="metric-card"><div style="font-size:1.5rem">{medal}</div>'
                            f'<div style="font-weight:600;color:#e2e8f0;margin:8px 0">{row["pressed_player"]}</div>'
                            f'<div style="color:#9ca3af;font-size:0.8rem">{row["pressing_team"]}</div>'
                            f'<div class="metric-value" style="font-size:1.4rem">{row["avg_pagerank"]:.4f}</div>'
                            f'<div class="metric-label">PageRank</div></div>', unsafe_allow_html=True)
    st.markdown("<br>"); st.markdown("#### Full Leaderboard")
    display_cols = [c for c in ['pressed_player','pressing_team','times_pressed','avg_pagerank',
                                'avg_betweenness','press_success_against','avg_combined_value_when_pressed'] if c in filtered.columns]
    st.dataframe(filtered[display_cols].head(50).style.format({
        'avg_pagerank':'{:.4f}','avg_betweenness':'{:.4f}','press_success_against':'{:.2%}','avg_combined_value_when_pressed':'{:.4f}'
    }).background_gradient(subset=['avg_pagerank'], cmap='YlOrRd'), use_container_width=True, hide_index=True)
    st.markdown('<div class="section-header">Network Importance vs Press Vulnerability</div>', unsafe_allow_html=True)
    st.markdown("*High PageRank + High success against = ideal press target*")
    fig = px.scatter(filtered.head(100), x='avg_pagerank', y='press_success_against', size='times_pressed',
                     color='pressing_team', hover_name='pressed_player',
                     labels={'avg_pagerank':'Network Importance (PageRank)','press_success_against':'Press Success Rate Against'})
    fig.add_hline(y=0.37, line_dash="dash", line_color="white", opacity=0.4, annotation_text="Avg success rate")
    fig.update_layout(plot_bgcolor='rgba(15,20,40,1)', paper_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0', height=420)
    st.plotly_chart(fig, use_container_width=True)

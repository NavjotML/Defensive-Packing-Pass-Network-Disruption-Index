import streamlit as st
import plotly.express as px
from data_loader import ARCHETYPE_COLORS


def render_overview(df, archetypes, team_stats):
    st.title("Defensive Packing & Pass Network Disruption")
    st.markdown("*Quantifying the spatial and network cost of pressing — FIFA World Cup 2022*")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    for col, value, label in zip(
        [c1, c2, c3, c4],
        [f"{len(df):,}", f"{df['press_success'].mean():.1%}",
         f"{df['combined_press_value'].mean():.3f}", f"{df['has_freeze_frame'].sum():,}"],
        ["Total Press Sequences", "Overall Press Success Rate",
         "Avg Combined Press Value", "Presses with Spatial Data"],
    ):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{value}</div>'
                f'<div class="metric-label">{label}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-header">Press Success Rate by Field Zone</div>', unsafe_allow_html=True)
        zone_data = (
            df.groupby('field_zone')
            .agg(success_rate=('press_success', 'mean'), count=('press_success', 'count'))
            .reindex(['defensive_third', 'middle_third', 'final_third'])
            .reset_index()
        )
        zone_data['field_zone'] = ['Defensive Third', 'Middle Third', 'Final Third']
        fig = px.bar(
            zone_data, x='field_zone', y='success_rate',
            text=zone_data['count'].apply(lambda x: f'n={x}'),
            color='success_rate', color_continuous_scale='RdYlGn',
            labels={'success_rate': 'Success Rate', 'field_zone': 'Zone'},
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          font_color='#e2e8f0', showlegend=False,
                          coloraxis_showscale=False, height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">Combined Press Value Distribution</div>', unsafe_allow_html=True)
        mean_val = df['combined_press_value'].mean()
        fig = px.histogram(df, x='combined_press_value', nbins=40,
                           color_discrete_sequence=['#4ade80'],
                           labels={'combined_press_value': 'Combined Press Value'})
        fig.add_vline(x=mean_val, line_dash="dash", line_color="#ef4444",
                      annotation_text=f"Mean: {mean_val:.3f}",
                      annotation_font_color="#ef4444")
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          font_color='#e2e8f0', height=320)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Team Pressing Archetypes</div>', unsafe_allow_html=True)
    arch_summary = archetypes.groupby('archetype').agg(
        teams=('pressing_team', 'count'),
        avg_success=('press_success_rate', 'mean'),
        avg_combined=('avg_combined_value', 'mean'),
        avg_final_third=('final_third_pct', 'mean'),
    ).reset_index()

    for _, row in arch_summary.iterrows():
        color = ARCHETYPE_COLORS.get(row['archetype'], '#6b7280')
        ca, cb, cc, cd, ce = st.columns([3, 1, 2, 2, 2])
        with ca:
            st.markdown(f'<span style="color:{color};font-weight:600">● {row["archetype"]}</span>',
                        unsafe_allow_html=True)
        with cb: st.markdown(f'`{int(row["teams"])} teams`')
        with cc: st.markdown(f'Success: **{row["avg_success"]:.1%}**')
        with cd: st.markdown(f'Press Value: **{row["avg_combined"]:.3f}**')
        with ce: st.markdown(f'Final Third: **{row["avg_final_third"]:.1%}**')

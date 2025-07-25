# app.py - Complete fixed version with proper footer

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.set_page_config(
    page_title="Cricket Stats - ODI Data",
    page_icon="🏏",
    layout="wide"
)

@st.cache_data
def load_odi_data():
    """Load ODI cricket data from CSV file with robust data type conversion"""
    try:
        df = pd.read_csv("ODI Cricket Data new.csv")
        
        # Define numeric and string columns explicitly
        numeric_columns = [
            'total_runs', 'strike_rate', 'total_balls_faced', 'total_wickets_taken',
            'total_runs_conceded', 'total_overs_bowled', 'total_matches_played',
            'matches_played_as_batter', 'matches_played_as_bowler', 'matches_won',
            'matches_lost', 'player_of_match_awards', 'average'
        ]
        
        string_columns = ['player_name', 'team', 'role']
        
        # Clean and convert numeric columns
        for col in numeric_columns:
            if col in df.columns:
                # Convert to string first for cleaning
                df[col] = df[col].astype(str)
                # Remove unwanted characters
                df[col] = df[col].str.replace(',', '')
                df[col] = df[col].str.replace('$', '')
                df[col] = df[col].str.replace('%', '')
                df[col] = df[col].str.replace('*', '')
                df[col] = df[col].str.strip()
                
                # Handle special cases
                df[col] = df[col].replace(['', 'N/A', 'n/a', 'null', 'NULL', '-', 'nan'], '0')
                
                # Convert to numeric with error handling
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Fill NaN values with 0
                df[col] = df[col].fillna(0)
                # Ensure proper data type (float64 for consistency)
                df[col] = df[col].astype('float64')
                
                # Ensure no negative values for certain columns
                if col in ['total_runs', 'total_matches_played', 'strike_rate']:
                    df[col] = df[col].abs()
        
        # Clean and standardize string columns
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype(str)
                df[col] = df[col].str.strip()
                df[col] = df[col].replace(['nan', 'NaN', 'null', 'NULL'], 'Unknown')
        
        # Special handling for strike_rate
        if 'strike_rate' in df.columns:
            df['strike_rate'] = df['strike_rate'].clip(0, 300)
        
        # Ensure all remaining columns are properly typed
        for col in df.columns:
            if col not in numeric_columns and col not in string_columns:
                # Convert remaining columns to string to avoid mixed types
                df[col] = df[col].astype(str)
                df[col] = df[col].str.strip()
        
        # Remove any completely empty rows
        df = df.dropna(how='all')
        
        # Reset index to ensure clean integer index
        df = df.reset_index(drop=True)
        
        return df
        
    except FileNotFoundError:
        st.error("ODI-Cricket-Data-new.csv file not found. Please ensure the file is in the same directory as app.py")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def get_player_stats(df, player_name):
    """Get specific player statistics"""
    try:
        player_data = df[df['player_name'].str.contains(player_name, case=False, na=False)]
        if not player_data.empty:
            return player_data.iloc[0]
        return None
    except Exception as e:
        st.error(f"Error finding player: {str(e)}")
        return None

def create_performance_chart(player_stats):
    """Create performance visualization charts"""
    try:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Runs & Strike Rate', 'Matches Played', 'Bowling Performance', 'Win Rate'),
            specs=[[{"secondary_y": True}, {"type": "indicator"}],
                   [{"type": "bar"}, {"type": "pie"}]]
        )
        
        # Chart 1: Runs and Strike Rate
        fig.add_trace(
            go.Bar(name="Total Runs", x=["Performance"], y=[float(player_stats['total_runs'])], 
                   marker_color='lightblue'),
            row=1, col=1
        )
        
        # Chart 2: Matches indicator
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=float(player_stats['total_matches_played']),
                title={'text': "Total Matches"},
                gauge={'axis': {'range': [None, 600]},
                       'bar': {'color': "darkgreen"},
                       'steps': [{'range': [0, 200], 'color': "lightgray"},
                                {'range': [200, 400], 'color': "gray"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 500}}
            ),
            row=1, col=2
        )
        
        # Chart 3: Bowling stats
        if float(player_stats['total_wickets_taken']) > 0:
            fig.add_trace(
                go.Bar(name="Wickets", x=["Bowling"], y=[float(player_stats['total_wickets_taken'])], 
                       marker_color='orange'),
                row=2, col=1
            )
        
        # Chart 4: Win/Loss ratio
        wins = float(player_stats['matches_won'])
        losses = float(player_stats['matches_lost'])
        if wins + losses > 0:
            fig.add_trace(
                go.Pie(labels=['Wins', 'Losses'], values=[wins, losses], 
                       marker_colors=['green', 'red']),
                row=2, col=2
            )
        
        fig.update_layout(height=800, showlegend=True, title_text="Player Performance Dashboard")
        return fig
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return None

def add_clean_footer():
    """Add a clean footer without HTML complications"""
    st.markdown("---")
    
    # App title and info
    st.markdown("### 🏏 ODI Cricket Statistics Dashboard")
    
    # Copyright section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**© 2025 Aatif Raza**")
        st.caption("All Rights Reserved")
    
    with col2:
        st.markdown("**📊 Features**")
        st.caption("Professional cricket statistics analysis tool")
    
    with col3:
        st.markdown("**💻 Tech Stack**")
        st.caption("Built with Python • Streamlit • Plotly")
    
    # Links section
    st.markdown("#### 🔗 Links")
    
    link_col1, link_col2, link_col3, link_col4 = st.columns(4)
    
    with link_col1:
        st.markdown("📂 [Source Code]()")
    
    with link_col2:
        st.markdown("📧 [Contact](mailto:razaaatif25@gmail.com.com)")
    
    with link_col3:
        st.markdown("💼 [LinkedIn]()")
    
    with link_col4:
        if st.button("⬆️ Back to Top"):
            st.rerun()
    
    st.markdown("---")
    st.caption("Data sourced from official cricket statistics • Last updated: July 2025")

def safe_dataframe_display(df, use_container_width=True):
    """Safely display dataframe with proper type conversion"""
    try:
        # Create a copy for display
        display_df = df.copy()
        
        # Ensure all columns are properly typed
        for col in display_df.columns:
            if display_df[col].dtype == 'object':
                # Try to convert object columns to string
                display_df[col] = display_df[col].astype(str)
        
        return st.dataframe(display_df, use_container_width=use_container_width)
    except Exception as e:
        st.error(f"Error displaying dataframe: {str(e)}")
        # Fallback: display as text
        st.text(str(df))

def main():
    st.title("🏏 ODI Cricket Statistics Dashboard")
    
    # Load data
    df = load_odi_data()
    if df is None:
        st.stop()
    
    st.success(f"✅ Loaded {len(df)} ODI cricket players data")
    
    # Check if required columns exist
    required_columns = ['player_name', 'team', 'role', 'total_runs', 'strike_rate', 'total_matches_played']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
        st.write("Available columns:", df.columns.tolist())
        st.stop()
    
    # Sidebar for player search and filters
    with st.sidebar:
        st.header("🔍 Player Search & Filters")
        
        # Player search
        player_search = st.text_input("Search Player:", placeholder="e.g., Virat Kohli, MS Dhoni")
        
        # Role filter
        try:
            roles = ['All'] + sorted(df['role'].dropna().unique().tolist())
            selected_role = st.selectbox("Filter by Role:", roles)
        except:
            selected_role = 'All'
        
        # Team filter
        try:
            teams = ['All'] + sorted(df['team'].dropna().unique().tolist())
            selected_team = st.selectbox("Filter by Team:", teams)
        except:
            selected_team = 'All'
        
        # Minimum matches filter
        max_matches = int(df['total_matches_played'].max()) if df['total_matches_played'].max() > 0 else 600
        min_matches = st.slider("Minimum Matches Played:", 0, max_matches, min(50, max_matches))
    
    # Apply filters
    try:
        filtered_df = df.copy()
        if selected_role != 'All':
            filtered_df = filtered_df[filtered_df['role'] == selected_role]
        if selected_team != 'All':
            filtered_df = filtered_df[filtered_df['team'] == selected_team]
        filtered_df = filtered_df[filtered_df['total_matches_played'] >= min_matches]
    except Exception as e:
        st.error(f"Error applying filters: {str(e)}")
        filtered_df = df.copy()
    
    # Main content
    if player_search:
        # Individual player analysis
        player_stats = get_player_stats(df, player_search)
        if player_stats is not None:
            st.header(f"📊 {player_stats['player_name']} - Performance Analysis")
            
            # Key metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Total Runs", f"{int(float(player_stats['total_runs'])):,}")
            with col2:
                st.metric("Strike Rate", f"{float(player_stats['strike_rate']):.2f}")
            with col3:
                st.metric("Average", f"{float(player_stats.get('average', 0)):.2f}")
            with col4:
                st.metric("Matches", int(float(player_stats['total_matches_played'])))
            with col5:
                st.metric("Wickets", int(float(player_stats.get('total_wickets_taken', 0))))
            
            # Performance charts
            st.subheader("Performance Visualization")
            chart = create_performance_chart(player_stats)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            
        else:
            st.warning(f"No player found matching '{player_search}'. Please try a different name.")
    
    else:
        # Overview dashboard
        st.header("📈 ODI Cricket Overview Dashboard")
        
        # Top performers section
        st.subheader("🏆 Top Performers")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top Run Scorers**")
            try:
                top_runs = filtered_df.nlargest(10, 'total_runs')[['player_name', 'total_runs', 'team']].copy()
                top_runs.index = range(1, len(top_runs) + 1)
                top_runs['total_runs'] = top_runs['total_runs'].apply(lambda x: f"{int(float(x)):,}")
                safe_dataframe_display(top_runs)
            except Exception as e:
                st.error(f"Error displaying top run scorers: {str(e)}")
        
        with col2:
            st.write("**Highest Strike Rates** (Min 1000 runs)")
            try:
                # Ensure strike_rate is numeric before filtering and sorting
                sr_df = filtered_df.copy()
                sr_df = sr_df[sr_df['total_runs'] >= 1000]
                
                if len(sr_df) > 0:
                    high_sr = sr_df.nlargest(10, 'strike_rate')[['player_name', 'strike_rate', 'total_runs', 'team']].copy()
                    high_sr.index = range(1, len(high_sr) + 1)
                    high_sr['strike_rate'] = high_sr['strike_rate'].apply(lambda x: f"{float(x):.2f}")
                    high_sr['total_runs'] = high_sr['total_runs'].apply(lambda x: f"{int(float(x)):,}")
                    safe_dataframe_display(high_sr)
                else:
                    st.write("No players with 1000+ runs found in filtered data")
            except Exception as e:
                st.error(f"Error displaying highest strike rates: {str(e)}")
        
        # Team-wise analysis
        st.subheader("🌍 Team-wise Analysis")
        try:
            team_stats = filtered_df.groupby('team').agg({
                'total_runs': 'sum',
                'total_matches_played': 'sum',
                'player_name': 'count'
            }).rename(columns={'player_name': 'total_players'}).sort_values('total_runs', ascending=False)
            
            fig_team = px.bar(
                team_stats.reset_index(), 
                x='team', 
                y='total_runs',
                title='Total Runs by Team',
                color='total_runs',
                color_continuous_scale='viridis'
            )
            fig_team.update_xaxes(tickangle=45)
            st.plotly_chart(fig_team, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating team analysis: {str(e)}")
        
        # Summary statistics
        st.subheader("📊 Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Players", len(filtered_df))
        with col2:
            st.metric("Total Runs Scored", f"{int(filtered_df['total_runs'].sum()):,}")
        with col3:
            st.metric("Average Strike Rate", f"{filtered_df['strike_rate'].mean():.2f}")
        with col4:
            st.metric("Total Matches", f"{int(filtered_df['total_matches_played'].sum()):,}")

    # Add clean footer without HTML complications
    add_clean_footer()

if __name__ == "__main__":
    main()









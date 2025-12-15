"""
Interactive Streamlit Dashboard for COVID-19 Misinformation Analysis
Optimized for large datasets with disk caching, sampling, and efficient network building.
"""

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle
import os
import logging
from pathlib import Path
import random
import ast
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Suppress matplotlib output in Streamlit
import matplotlib
matplotlib.use('Agg')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DATA_PATH = "data/covid19_tweets.csv"
NETWORK_CACHE_PATH = "data/network_cache.pkl"
USER_HASHTAGS_CACHE_PATH = "data/user_hashtags_cache.pkl"
CACHE_DIR = Path("data")
CACHE_DIR.mkdir(exist_ok=True)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================
# OPTIMIZED DATA LOADING
# ==============================

@st.cache_data
def load_and_clean_data_optimized(data_path, sample_size=None, sample_ratio=0.05):
    """
    Load and clean data with optional sampling for faster development.
    Only loads necessary columns for memory efficiency.
    """
    logger.info(f"Loading data from {data_path}")
    
    # Essential columns only
    essential_cols = ['user_name', 'text', 'hashtags', 'user_followers', 'is_retweet']
    
    # Read only necessary columns
    try:
        df = pd.read_csv(data_path, usecols=essential_cols)
        logger.info(f"Loaded {len(df):,} rows with {len(df.columns)} columns")
    except ValueError:
        # If columns don't exist, read all and select
        df = pd.read_csv(data_path)
        available_cols = [col for col in essential_cols if col in df.columns]
        df = df[available_cols].copy()
        logger.info(f"Loaded {len(df):,} rows, selected {len(available_cols)} columns")
    
    # SMALL DATA MODE: sample immediately after load (default 5%)
    if sample_ratio is not None and 0 < sample_ratio < 1:
        original_size = len(df)
        df = df.sample(frac=sample_ratio, random_state=42).reset_index(drop=True)
        logger.info(f"Sampled {len(df):,} rows ({sample_ratio*100:.1f}% of {original_size:,})")
    elif sample_size is not None and sample_size > 0:
        original_size = len(df)
        sample_size = min(sample_size, len(df))
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        logger.info(f"Sampled {len(df):,} rows from {original_size:,}")
    
    # Remove rows with missing critical data
    initial_len = len(df)
    df.dropna(subset=['user_name', 'text'], inplace=True)
    removed = initial_len - len(df)
    if removed > 0:
        logger.info(f"Removed {removed:,} rows with missing data")
    
    # Clean follower counts
    if 'user_followers' in df.columns:
        df['user_followers'] = pd.to_numeric(df['user_followers'], errors='coerce').fillna(0)
    
    logger.info(f"Final dataset: {len(df):,} rows, {df['user_name'].nunique():,} unique users")
    return df


@st.cache_data
def detect_misinformation_optimized(df):
    """
    Detect misinformation tweets using keyword matching.
    Optimized for performance.
    """
    logger.info("Detecting misinformation...")
    
    misinfo_keywords = [
        'hoax', 'fake', 'plandemic', 'conspiracy', 'lie', 'scam', 
        'bioweapon', 'fake news', 'cover-up', 'not real', 'doesn\'t exist',
        'made up', 'false', 'deception', 'fraud'
    ]
    
    # Vectorized approach for better performance
    text_lower = df['text'].astype(str).str.lower()
    df['misinfo'] = text_lower.str.contains('|'.join(misinfo_keywords), case=False, na=False)
    
    misinfo_count = df['misinfo'].sum()
    logger.info(f"Detected {misinfo_count:,} misinformation tweets ({misinfo_count/len(df)*100:.2f}%)")
    
    return df


# ==============================
# OPTIMIZED NETWORK BUILDING
# ==============================

def build_user_network_optimized(df, use_cache=True, force_rebuild=False, max_users=500):
    """
    Build user-to-user network with disk caching.
    Uses add_edges_from for optimal performance.
    """
    # Check if cached network exists
    if use_cache and not force_rebuild and os.path.exists(NETWORK_CACHE_PATH):
        try:
            logger.info(f"Loading network from cache: {NETWORK_CACHE_PATH}")
            with open(NETWORK_CACHE_PATH, 'rb') as f:
                G = pickle.load(f)
            with open(USER_HASHTAGS_CACHE_PATH, 'rb') as f:
                user_hashtags = pickle.load(f)
            logger.info(f"✓ Loaded cached network: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
            return G, user_hashtags
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Rebuilding network...")
    
    # Build network from scratch
    logger.info("Building user network from data...")
    # PERFORMANCE OPTIMIZATION: cap network to top users to keep graph small
    top_users = df['user_name'].value_counts().head(max_users).index
    df = df[df['user_name'].isin(top_users)].copy()
    G = nx.Graph()
    
    # Dictionary to store hashtags per user
    user_hashtags = {}
    # Dictionary to store users per hashtag (for optimization)
    hashtag_users = {}
    
    # Parse hashtags and build bidirectional mapping
    logger.info("  Step 1/4: Parsing hashtags...")
    for _, row in df.iterrows():
        user = row['user_name']
        hashtags_str = str(row['hashtags'])
        
        # Try to parse as list if it's in string format
        try:
            if hashtags_str.startswith('['):
                hashtags = ast.literal_eval(hashtags_str)
            else:
                hashtags = [tag.strip() for tag in hashtags_str.split(',') if tag.strip()]
        except:
            hashtags = [tag.strip() for tag in hashtags_str.split(',') if tag.strip()]
        
        # Clean and normalize hashtags
        hashtags = [tag.strip().lower().replace('#', '') 
                   for tag in hashtags if tag and tag.strip() and tag != 'nan']
        
        if user not in user_hashtags:
            user_hashtags[user] = set()
        user_hashtags[user].update(hashtags)
        
        # Build reverse mapping: hashtag -> users
        for tag in hashtags:
            if tag not in hashtag_users:
                hashtag_users[tag] = set()
            hashtag_users[tag].add(user)
    
    # Add all users as nodes
    logger.info("  Step 2/4: Adding nodes...")
    G.add_nodes_from(user_hashtags.keys(), type='user')
    
    # Build edges efficiently using add_edges_from
    logger.info("  Step 3/4: Building edges (this may take a moment for large datasets)...")
    all_edges = []
    processed_pairs = set()  # Track edges to avoid duplicates
    
    for tag, users_with_tag in hashtag_users.items():
        users_list = list(users_with_tag)
        # For each hashtag, connect all pairs of users who used it
        for i, user1 in enumerate(users_list):
            for user2 in users_list[i+1:]:
                # Use sorted tuple to avoid duplicate edges
                edge_key = tuple(sorted([user1, user2]))
                if edge_key not in processed_pairs:
                    all_edges.append((user1, user2))
                    processed_pairs.add(edge_key)
    
    # Add all edges at once (much faster than adding one by one)
    logger.info(f"  Step 4/4: Adding {len(all_edges):,} edges...")
    G.add_edges_from(all_edges)
    
    logger.info(f"✓ Network built: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    if G.number_of_nodes() > 0:
        logger.info(f"  Average degree: {2*G.number_of_edges()/G.number_of_nodes():.2f}")
    
    # Save to disk cache
    if use_cache:
        try:
            logger.info(f"Saving network to cache: {NETWORK_CACHE_PATH}")
            with open(NETWORK_CACHE_PATH, 'wb') as f:
                pickle.dump(G, f)
            with open(USER_HASHTAGS_CACHE_PATH, 'wb') as f:
                pickle.dump(user_hashtags, f)
            logger.info("✓ Network cached successfully")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    return G, user_hashtags


# ==============================
# SIMULATION AND ML FUNCTIONS
# ==============================

def run_sir_simulation_dashboard(df, G, steps=20, beta=0.05, gamma=0.2):
    """Run SIR simulation for dashboard (returns data without plotting)."""
    users = list(G.nodes())
    status = {u: 'S' for u in users}
    
    # Infect initial users
    initial_infected = df[df['misinfo']]['user_name'].unique()
    initial_infected = [u for u in initial_infected if u in users][:min(10, len(initial_infected))]
    
    for u in initial_infected:
        status[u] = 'I'
    
    # Track history
    S_history = [list(status.values()).count('S')]
    I_history = [list(status.values()).count('I')]
    R_history = [list(status.values()).count('R')]
    
    # Run simulation
    for step in range(steps):
        new_status = status.copy()
        
        for u in users:
            if status[u] == 'I':
                if random.random() < gamma:
                    new_status[u] = 'R'
            elif status[u] == 'S':
                neighbors = list(G.neighbors(u))
                if neighbors:
                    infected_neighbors = sum(1 for n in neighbors if status[n] == 'I')
                    infection_prob = beta * (1 + 0.5 * infected_neighbors)
                    if random.random() < min(infection_prob, 1.0):
                        new_status[u] = 'I'
                else:
                    if random.random() < beta:
                        new_status[u] = 'I'
        
        status = new_status
        S_history.append(list(status.values()).count('S'))
        I_history.append(list(status.values()).count('I'))
        R_history.append(list(status.values()).count('R'))
    
    return S_history, I_history, R_history


@st.cache_data
def train_ml_model_dashboard(df):
    """Train ML model for dashboard (returns model and metrics without plotting)."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    except ImportError:
        return None, None, None, None, None
    
    logger.info("Training ML model...")
    
    # Prepare features (only necessary columns)
    texts = df['text'].fillna('').astype(str)
    
    user_features = pd.DataFrame()
    if 'user_followers' in df.columns:
        user_features['followers'] = df['user_followers'].fillna(0)
    else:
        user_features['followers'] = 0
    
    user_features['tweet_length'] = texts.str.len()
    
    def count_hashtags(hashtags_str):
        try:
            if pd.isna(hashtags_str):
                return 0
            if isinstance(hashtags_str, str) and hashtags_str.startswith('['):
                return len(ast.literal_eval(hashtags_str))
            return len([t for t in str(hashtags_str).split(',') if t.strip()])
        except:
            return 0
    
    user_features['hashtag_count'] = df['hashtags'].apply(count_hashtags)
    
    # TF-IDF vectorization
    logger.info("  Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english', ngram_range=(1, 2))
    text_features = vectorizer.fit_transform(texts)
    
    from scipy.sparse import hstack
    X = hstack([text_features, user_features.values])
    y = df['misinfo'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    logger.info("  Training Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"✓ Model trained. Accuracy: {accuracy:.4f}")
    
    # Get feature names for importance
    feature_names = [f'TF-IDF_{i}' for i in range(100)] + ['followers', 'tweet_length', 'hashtag_count']
    
    return model, vectorizer, accuracy, y_test, y_pred


# ==============================
# VISUALIZATION FUNCTIONS
# ==============================

def create_data_summary_plots(df):
    """Create interactive plots for data summary tab."""
    
    # 1. Misinformation Distribution Pie Chart
    misinfo_counts = df['misinfo'].value_counts()
    fig_pie = px.pie(
        values=misinfo_counts.values,
        names=['Regular Tweets', 'Misinformation'],
        title='Distribution of Misinformation vs Regular Tweets',
        color_discrete_sequence=['#66b3ff', '#ff9999'],
        hole=0.4
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    fig_pie.update_layout(height=400, showlegend=True)
    
    # 2. User Follower Distribution
    if 'user_followers' in df.columns:
        followers = df['user_followers'].replace(0, 1)
        fig_followers = px.histogram(
            x=np.log10(followers),
            nbins=50,
            title='Distribution of User Followers (Log Scale)',
            labels={'x': 'Log10(User Followers)', 'y': 'Frequency'},
            color_discrete_sequence=['skyblue']
        )
        fig_followers.update_layout(height=400, showlegend=False)
    else:
        fig_followers = None
    
    # 3. Top Users by Tweet Count
    top_users = df['user_name'].value_counts().head(10)
    fig_top_users = px.bar(
        x=top_users.values,
        y=top_users.index,
        orientation='h',
        title='Top 10 Users by Tweet Count',
        labels={'x': 'Number of Tweets', 'y': 'User Name'},
        color=top_users.values,
        color_continuous_scale='Blues'
    )
    fig_top_users.update_layout(height=400, showlegend=False, yaxis={'categoryorder': 'total ascending'})
    
    # 4. Top Misinformation Spreaders
    misinfo_users = df[df['misinfo']]['user_name'].value_counts().head(10)
    if len(misinfo_users) > 0:
        fig_misinfo_users = px.bar(
            x=misinfo_users.values,
            y=misinfo_users.index,
            orientation='h',
            title='Top 10 Misinformation Spreaders',
            labels={'x': 'Misinformation Tweets', 'y': 'User Name'},
            color=misinfo_users.values,
            color_continuous_scale='Reds'
        )
        fig_misinfo_users.update_layout(height=400, showlegend=False, yaxis={'categoryorder': 'total ascending'})
    else:
        fig_misinfo_users = None
    
    # 5. Tweet Length Distribution
    tweet_lengths = df['text'].str.len()
    fig_length = px.histogram(
        x=tweet_lengths,
        nbins=50,
        title='Distribution of Tweet Lengths',
        labels={'x': 'Tweet Length (characters)', 'y': 'Frequency'},
        color_discrete_sequence=['lightgreen']
    )
    fig_length.update_layout(height=400, showlegend=False)
    
    # 6. Follower Count Comparison
    if 'user_followers' in df.columns:
        regular_followers = df[~df['misinfo']]['user_followers'].replace(0, 1)
        misinfo_followers = df[df['misinfo']]['user_followers'].replace(0, 1)
        
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(
            y=np.log10(regular_followers),
            name='Regular Tweets',
            marker_color='lightblue'
        ))
        fig_box.add_trace(go.Box(
            y=np.log10(misinfo_followers),
            name='Misinformation',
            marker_color='lightcoral'
        ))
        fig_box.update_layout(
            title='Follower Count: Regular vs Misinformation',
            yaxis_title='Log10(User Followers)',
            height=400,
            showlegend=True
        )
    else:
        fig_box = None
    
    return {
        'pie': fig_pie,
        'followers': fig_followers,
        'top_users': fig_top_users,
        'misinfo_users': fig_misinfo_users,
        'length': fig_length,
        'box': fig_box
    }


def create_network_visualization(G, df, top_n=20):
    """Create interactive network visualization with hover information."""
    
    # Calculate centrality measures
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    
    # Get top N users by degree centrality
    top_users = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_user_names = [u for u, _ in top_users]
    
    # Create subgraph
    subG = G.subgraph(top_user_names)
    
    # Get positions using spring layout
    pos = nx.spring_layout(subG, seed=42, k=1, iterations=50)
    
    # Prepare edge traces
    edge_x = []
    edge_y = []
    for edge in subG.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Prepare node traces with hover information
    node_x = []
    node_y = []
    node_text = []
    node_info = []
    node_sizes = []
    node_colors = []
    
    # Pre-compute user stats for efficiency
    user_stats = {}
    for user in top_user_names:
        user_tweets = df[df['user_name'] == user]
        misinfo_count = len(user_tweets[user_tweets['misinfo']])
        total_tweets = len(user_tweets)
        followers = user_tweets['user_followers'].iloc[0] if len(user_tweets) > 0 and 'user_followers' in df.columns else 0
        user_stats[user] = {
            'misinfo_count': misinfo_count,
            'total_tweets': total_tweets,
            'followers': followers
        }
    
    for node in top_user_names:
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        stats = user_stats[node]
        degree = subG.degree(node)
        deg_cent = degree_centrality.get(node, 0)
        bet_cent = betweenness_centrality.get(node, 0)
        
        # Create hover text
        hover_text = (
            f"<b>{node}</b><br>"
            f"Total Tweets: {stats['total_tweets']}<br>"
            f"Misinformation: {stats['misinfo_count']} ({stats['misinfo_count']/stats['total_tweets']*100:.1f}%)<br>"
            f"Followers: {stats['followers']:,.0f}<br>"
            f"Connections: {degree}<br>"
            f"Degree Centrality: {deg_cent:.4f}<br>"
            f"Betweenness Centrality: {bet_cent:.4f}"
        )
        node_text.append(node[:30])
        node_info.append(hover_text)
        
        node_sizes.append(10 + degree * 2)
        
        if stats['misinfo_count'] > 0:
            node_colors.append('red')
        else:
            node_colors.append('lightblue')
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="middle center",
        hovertext=node_info,
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=node_colors,
            size=node_sizes,
            colorbar=dict(
                thickness=15,
                title="Node Connections",
                xanchor="left"
            ),
            line=dict(width=2, color='black')
        )
    )
    
    # Create figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text=f'Interactive Network Visualization (Top {top_n} Users)',
                font=dict(size=16)
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[dict(
                text="Hover over nodes to see user details. Red nodes spread misinformation.",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color='gray', size=10)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=700
        )
    )
    
    return fig, top_users


def create_sir_visualization(S_history, I_history, R_history, steps):
    """Create interactive SIR simulation visualization."""
    
    time_steps = list(range(len(S_history)))
    total = S_history[0] + I_history[0] + R_history[0]
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('SIR Model: Absolute Numbers', 'SIR Model: Percentage View'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Left plot: Absolute numbers
    fig.add_trace(
        go.Scatter(x=time_steps, y=S_history, name='Susceptible',
                  line=dict(color='blue', width=3), mode='lines+markers', marker=dict(size=6)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=time_steps, y=I_history, name='Infected',
                  line=dict(color='red', width=3), mode='lines+markers', marker=dict(size=6)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=time_steps, y=R_history, name='Recovered',
                  line=dict(color='green', width=3), mode='lines+markers', marker=dict(size=6)),
        row=1, col=1
    )
    
    # Right plot: Percentages
    S_pct = [s/total*100 for s in S_history]
    I_pct = [i/total*100 for i in I_history]
    R_pct = [r/total*100 for r in R_history]
    
    fig.add_trace(
        go.Scatter(x=time_steps, y=S_pct, name='Susceptible %',
                  line=dict(color='blue', width=3), mode='lines+markers', marker=dict(size=6),
                  showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=time_steps, y=I_pct, name='Infected %',
                  line=dict(color='red', width=3), mode='lines+markers', marker=dict(size=6),
                  showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=time_steps, y=R_pct, name='Recovered %',
                  line=dict(color='green', width=3), mode='lines+markers', marker=dict(size=6),
                  showlegend=False),
        row=1, col=2
    )
    
    # Update axes
    fig.update_xaxes(title_text="Time Steps", row=1, col=1)
    fig.update_xaxes(title_text="Time Steps", row=1, col=2)
    fig.update_yaxes(title_text="Number of Users", row=1, col=1)
    fig.update_yaxes(title_text="Percentage (%)", row=1, col=2, range=[0, 100])
    
    fig.update_layout(
        height=500,
        title_text="SIR Model: Misinformation Spread Simulation",
        hovermode='x unified'
    )
    
    return fig


def create_ml_metrics_visualization(y_test, y_pred, model, feature_names=None):
    """Create ML model metrics visualization."""
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    fig_cm = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=['Regular', 'Misinformation'],
        y=['Regular', 'Misinformation'],
        text_auto=True,
        aspect="auto",
        color_continuous_scale='Blues',
        title='Confusion Matrix'
    )
    fig_cm.update_layout(height=400)
    
    # Feature Importance (if available)
    fig_importance = None
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        if len(importances) > 0:
            # Get top 10
            top_indices = np.argsort(importances)[-10:][::-1]
            top_importances = importances[top_indices]
            
            if feature_names and len(feature_names) >= len(importances):
                names = [feature_names[i] for i in top_indices]
            else:
                names = [f'Feature {i+1}' for i in top_indices]
            
            fig_importance = px.bar(
                x=top_importances,
                y=names,
                orientation='h',
                title='Top 10 Feature Importances',
                labels={'x': 'Importance', 'y': 'Feature'},
                color=top_importances,
                color_continuous_scale='Blues'
            )
            fig_importance.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
    
    return fig_cm, fig_importance


# ==============================
# MAIN DASHBOARD
# ==============================

def main():
    """Main dashboard function."""
    
    # Check if running in Streamlit context
    try:
        st.set_page_config(
            page_title="COVID-19 Misinformation Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except:
        st.error("⚠️ Please run this dashboard using: `streamlit run dashboard.py`")
        st.stop()
    
    # Header
    st.markdown('<h1 class="main-header">📊 COVID-19 Misinformation Spread Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("⚙️ Dashboard Controls")
    
    # Sample mode for fast development
    st.sidebar.markdown("### 🚀 Performance Settings")
    use_sample = st.sidebar.checkbox("Use Sample Mode (Fast Testing)", value=False)
    
    if use_sample:
        sample_option = st.sidebar.radio(
            "Sample Size",
            ["10K tweets", "50K tweets", "100K tweets", "Custom"],
            index=0
        )
        
        if sample_option == "10K tweets":
            sample_size = 10000
        elif sample_option == "50K tweets":
            sample_size = 50000
        elif sample_option == "100K tweets":
            sample_size = 100000
        else:
            sample_size = st.sidebar.number_input("Custom Sample Size", min_value=1000, max_value=500000, value=10000, step=1000)
        
        sample_ratio = None
    else:
        sample_size = None
        # SMALL DATA MODE: default to 5% sample even when not in dev mode
        sample_ratio = 0.05
    
    # Network cache controls
    st.sidebar.markdown("### 💾 Cache Settings")
    use_cache = st.sidebar.checkbox("Use Network Cache", value=True)
    force_rebuild = st.sidebar.checkbox("Force Rebuild Network", value=False)
    
    if force_rebuild:
        st.sidebar.warning("⚠️ Network will be rebuilt from scratch")
    
    # Data loading with progress
    with st.spinner("Loading data..."):
        df = load_and_clean_data_optimized(DATA_PATH, sample_size=sample_size, sample_ratio=sample_ratio)
        df = detect_misinformation_optimized(df)
    
    st.sidebar.success(f"✓ Loaded {len(df):,} tweets")
    
    # Sidebar metrics
    st.sidebar.markdown("### 📈 Quick Stats")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Total Tweets", f"{len(df):,}")
        st.metric("Unique Users", f"{df['user_name'].nunique():,}")
    with col2:
        misinfo_count = df['misinfo'].sum()
        st.metric("Misinformation", f"{misinfo_count:,}")
        st.metric("Misinfo %", f"{(misinfo_count/len(df)*100):.2f}%")
    
    # Cache info
    if use_cache and os.path.exists(NETWORK_CACHE_PATH):
        cache_size = os.path.getsize(NETWORK_CACHE_PATH) / (1024 * 1024)  # MB
        st.sidebar.info(f"💾 Cache: {cache_size:.1f} MB")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Data Summary",
        "🕸️ Network Visualization",
        "🦠 SIR Simulation",
        "🤖 ML Model Metrics"
    ])
    
    # ==============================
    # TAB 1: DATA SUMMARY
    # ==============================
    with tab1:
        st.header("📊 Data Summary & Exploratory Analysis")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Tweets", f"{len(df):,}")
        with col2:
            st.metric("Unique Users", f"{df['user_name'].nunique():,}")
        with col3:
            st.metric("Misinformation Tweets", f"{df['misinfo'].sum():,}")
        with col4:
            avg_tweets = len(df) / df['user_name'].nunique()
            st.metric("Avg Tweets/User", f"{avg_tweets:.1f}")
        
        st.markdown("---")
        
        # Create and display plots
        plots = create_data_summary_plots(df)
        
        # Row 1: Pie chart and Follower distribution
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plots['pie'], use_container_width=True)
        with col2:
            if plots['followers']:
                st.plotly_chart(plots['followers'], use_container_width=True)
            else:
                st.info("Follower data not available")
        
        # Row 2: Top users
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plots['top_users'], use_container_width=True)
        with col2:
            if plots['misinfo_users']:
                st.plotly_chart(plots['misinfo_users'], use_container_width=True)
            else:
                st.info("No misinformation spreaders found")
        
        # Row 3: Tweet length and Box plot
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plots['length'], use_container_width=True)
        with col2:
            if plots['box']:
                st.plotly_chart(plots['box'], use_container_width=True)
            else:
                st.info("Follower comparison not available")
    
    # ==============================
    # TAB 2: NETWORK VISUALIZATION
    # ==============================
    with tab2:
        st.header("🕸️ User Network Visualization")
        
        # Network controls
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("### Interactive Network Graph")
            st.markdown("Hover over nodes to see detailed user information including follower count, misinformation spread, and network metrics.")
        with col2:
            top_n = st.slider("Number of Top Users", min_value=10, max_value=50, value=20, step=5)
        
        # Build network
        cache_status = "from cache" if (use_cache and os.path.exists(NETWORK_CACHE_PATH) and not force_rebuild) else "from scratch"
        st.info(f"💡 **Note**: Network will be loaded {cache_status}. First build may take 2-5 minutes for large datasets.")
        
        with st.spinner(f"Building/loading network ({cache_status})..."):
            G, user_hashtags = build_user_network_optimized(df, use_cache=use_cache, force_rebuild=force_rebuild)
        
        st.success(f"✅ Network ready! {G.number_of_nodes():,} users, {G.number_of_edges():,} connections")
        
        # Create visualization
        with st.spinner("Generating network visualization..."):
            fig_network, top_spreaders = create_network_visualization(G, df, top_n=top_n)
        
        st.plotly_chart(fig_network, use_container_width=True)
        
        # Top spreaders table
        st.markdown("### Top Super-Spreaders")
        spreader_data = []
        for user, score in top_spreaders:
            user_tweets = df[df['user_name'] == user]
            misinfo_count = len(user_tweets[user_tweets['misinfo']])
            total_tweets = len(user_tweets)
            followers = user_tweets['user_followers'].iloc[0] if len(user_tweets) > 0 and 'user_followers' in df.columns else 0
            spreader_data.append({
                'User': user,
                'Degree Centrality': f"{score:.4f}",
                'Total Tweets': total_tweets,
                'Misinformation': misinfo_count,
                'Followers': f"{followers:,.0f}"
            })
        
        spreader_df = pd.DataFrame(spreader_data)
        st.dataframe(spreader_df, use_container_width=True, hide_index=True)
    
    # ==============================
    # TAB 3: SIR SIMULATION
    # ==============================
    with tab3:
        st.header("🦠 SIR Model Simulation")
        
        # Simulation controls
        col1, col2, col3 = st.columns(3)
        with col1:
            steps = st.slider("Simulation Steps", min_value=10, max_value=50, value=20, step=5)
        with col2:
            beta = st.slider("Infection Rate (β)", min_value=0.01, max_value=0.20, value=0.05, step=0.01, format="%.2f")
        with col3:
            gamma = st.slider("Recovery Rate (γ)", min_value=0.05, max_value=0.50, value=0.20, step=0.05, format="%.2f")
        
        # Run simulation
        if st.button("🔄 Run Simulation", type="primary"):
            with st.spinner("Running SIR simulation..."):
                G, _ = build_user_network_optimized(df, use_cache=use_cache, force_rebuild=force_rebuild)
                S_history, I_history, R_history = run_sir_simulation_dashboard(
                    df, G, steps=steps, beta=beta, gamma=gamma
                )
                
                # Store in session state
                st.session_state['sir_results'] = {
                    'S': S_history,
                    'I': I_history,
                    'R': R_history,
                    'steps': steps
                }
        
        # Display results
        if 'sir_results' in st.session_state:
            results = st.session_state['sir_results']
            fig_sir = create_sir_visualization(
                results['S'], results['I'], results['R'], results['steps']
            )
            st.plotly_chart(fig_sir, use_container_width=True)
            
            # Summary statistics
            st.markdown("### Simulation Summary")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Initial Susceptible", f"{results['S'][0]:,}")
            with col2:
                st.metric("Final Infected", f"{results['I'][-1]:,}")
            with col3:
                st.metric("Final Recovered", f"{results['R'][-1]:,}")
            with col4:
                peak_infected = max(results['I'])
                st.metric("Peak Infected", f"{peak_infected:,}")
        else:
            st.info("👆 Click 'Run Simulation' to start the SIR model")
    
    # ==============================
    # TAB 4: ML MODEL METRICS
    # ==============================
    with tab4:
        st.header("🤖 Machine Learning Model Metrics")
        
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training ML model... This may take a few minutes."):
                try:
                    model, vectorizer, accuracy, y_test, y_pred = train_ml_model_dashboard(df)
                    if model is not None:
                        st.session_state['ml_model'] = model
                        st.session_state['ml_vectorizer'] = vectorizer
                        st.session_state['ml_accuracy'] = accuracy
                        st.session_state['ml_y_test'] = y_test
                        st.session_state['ml_y_pred'] = y_pred
                        st.success(f"✅ Model trained successfully! Accuracy: {accuracy:.4f}")
                    else:
                        st.error("❌ Model training failed. Please check if scikit-learn is installed.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        if 'ml_model' in st.session_state:
            st.markdown("### Model Performance")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", f"{st.session_state['ml_accuracy']:.4f}")
            with col2:
                st.metric("Accuracy %", f"{st.session_state['ml_accuracy']*100:.2f}%")
            
            # Display confusion matrix and feature importance
            if 'ml_y_test' in st.session_state and 'ml_y_pred' in st.session_state:
                y_test = st.session_state['ml_y_test']
                y_pred = st.session_state['ml_y_pred']
                model = st.session_state['ml_model']
                
                # Get feature names
                feature_names = [f'TF-IDF_{i}' for i in range(100)] + ['followers', 'tweet_length', 'hashtag_count']
                
                # Create visualizations
                fig_cm, fig_importance = create_ml_metrics_visualization(
                    y_test, y_pred, model, feature_names
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig_cm, use_container_width=True)
                with col2:
                    if fig_importance:
                        st.plotly_chart(fig_importance, use_container_width=True)
                
                # Classification report
                from sklearn.metrics import classification_report
                report = classification_report(y_test, y_pred, target_names=['Regular', 'Misinformation'], output_dict=True)
                st.markdown("### Classification Report")
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df, use_container_width=True)
        else:
            st.info("👆 Click 'Train Model' to train the misinformation prediction model")


if __name__ == "__main__":
    main()

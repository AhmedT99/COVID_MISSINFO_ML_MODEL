import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import random
from collections import Counter
import ast
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ==============================
# 1. DATA LOADING AND CLEANING
# ==============================

def load_and_clean_data(data_path="data/covid19_tweets.csv"):
    """
    Load and clean the COVID-19 tweets dataset.
    Returns cleaned DataFrame with essential columns.
    """
    print("=" * 60)
    print("STEP 1: LOADING AND CLEANING DATA")
    print("=" * 60)
    
    # Load data (SMALL DATA MODE: sample 5% for speed)
    df = pd.read_csv(data_path)
    df = df.sample(frac=0.05, random_state=42).reset_index(drop=True)
    print(f"✓ Dataset loaded successfully (sampled 5%)!")
    print(f"  Sampled shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    
    # Select and clean essential columns
    essential_cols = ['user_name', 'text', 'hashtags', 'user_followers', 'is_retweet']
    available_cols = [col for col in essential_cols if col in df.columns]
    df = df[available_cols].copy()
    
    # Remove rows with missing critical data
    df.dropna(subset=['user_name', 'text'], inplace=True)
    
    # Clean follower counts (handle any non-numeric values)
    if 'user_followers' in df.columns:
        df['user_followers'] = pd.to_numeric(df['user_followers'], errors='coerce').fillna(0)
    
    print(f"  Cleaned shape: {df.shape}")
    print(f"  Removed {len(pd.read_csv(data_path)) - len(df)} rows with missing data\n")
    
    return df


# ==============================
# 2. MISINFORMATION DETECTION
# ==============================

def detect_misinformation(df):
    """
    Detect misinformation tweets using keyword matching.
    Returns DataFrame with 'misinfo' column added.
    """
    print("=" * 60)
    print("STEP 2: DETECTING MISINFORMATION")
    print("=" * 60)
    
    # Expanded list of misinformation keywords
    misinfo_keywords = [
        'hoax', 'fake', 'plandemic', 'conspiracy', 'lie', 'scam', 
        'bioweapon', 'fake news', 'cover-up', 'not real', 'doesn\'t exist',
        'made up', 'false', 'deception', 'fraud'
    ]
    
    def is_misinfo(text):
        """Check if text contains misinformation keywords."""
        text = str(text).lower()
        return any(word in text for word in misinfo_keywords)
    
    # Apply detection
    df['misinfo'] = df['text'].apply(is_misinfo)
    
    misinfo_count = df['misinfo'].sum()
    misinfo_pct = (misinfo_count / len(df)) * 100
    
    print(f"✓ Misinformation detection complete!")
    print(f"  Total tweets: {len(df):,}")
    print(f"  Misinformation tweets: {misinfo_count:,} ({misinfo_pct:.2f}%)\n")
    
    return df


# ==============================
# 3. EXPLORATORY DATA ANALYSIS
# ==============================

def exploratory_analysis(df):
    """
    Perform comprehensive exploratory data analysis with visualizations.
    Creates histograms, pie charts, and statistical summaries.
    """
    print("=" * 60)
    print("STEP 3: EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 3.1 Tweet Statistics Summary
    print("3.1 Tweet Statistics:")
    print(f"  Total tweets: {len(df):,}")
    print(f"  Unique users: {df['user_name'].nunique():,}")
    print(f"  Average tweets per user: {len(df) / df['user_name'].nunique():.2f}")
    print(f"  Retweets: {df['is_retweet'].sum() if 'is_retweet' in df.columns else 'N/A'}")
    
    # 3.2 Misinformation Distribution (Pie Chart)
    ax1 = plt.subplot(2, 3, 1)
    misinfo_counts = df['misinfo'].value_counts()
    colors = ['#ff9999', '#66b3ff']
    labels = ['Regular Tweets', 'Misinformation']
    plt.pie(misinfo_counts.values, labels=labels, autopct='%1.1f%%', 
            colors=colors, startangle=90)
    plt.title('Distribution of Misinformation vs Regular Tweets', fontsize=12, fontweight='bold')
    
    # 3.3 User Follower Distribution (Histogram)
    ax2 = plt.subplot(2, 3, 2)
    if 'user_followers' in df.columns:
        # Log scale for better visualization
        followers = df['user_followers'].replace(0, 1)  # Replace 0 with 1 for log
        plt.hist(np.log10(followers), bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        plt.xlabel('Log10(User Followers)')
        plt.ylabel('Frequency')
        plt.title('Distribution of User Followers (Log Scale)', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
    
    # 3.4 Top Users by Tweet Count (Bar Chart)
    ax3 = plt.subplot(2, 3, 3)
    top_users = df['user_name'].value_counts().head(10)
    plt.barh(range(len(top_users)), top_users.values, color='coral')
    plt.yticks(range(len(top_users)), top_users.index)
    plt.xlabel('Number of Tweets')
    plt.title('Top 10 Users by Tweet Count', fontsize=12, fontweight='bold')
    plt.gca().invert_yaxis()
    
    # 3.5 Top Misinformation Spreaders (Bar Chart)
    ax4 = plt.subplot(2, 3, 4)
    misinfo_users = df[df['misinfo']]['user_name'].value_counts().head(10)
    if len(misinfo_users) > 0:
        plt.barh(range(len(misinfo_users)), misinfo_users.values, color='red', alpha=0.7)
        plt.yticks(range(len(misinfo_users)), misinfo_users.index)
        plt.xlabel('Misinformation Tweets')
        plt.title('Top 10 Misinformation Spreaders', fontsize=12, fontweight='bold')
        plt.gca().invert_yaxis()
    else:
        plt.text(0.5, 0.5, 'No misinformation spreaders found', 
                ha='center', va='center', transform=ax4.transAxes)
        plt.title('Top 10 Misinformation Spreaders', fontsize=12, fontweight='bold')
    
    # 3.6 Tweet Length Distribution (Histogram)
    ax5 = plt.subplot(2, 3, 5)
    tweet_lengths = df['text'].str.len()
    plt.hist(tweet_lengths, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
    plt.xlabel('Tweet Length (characters)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Tweet Lengths', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 3.7 Follower Count vs Misinformation (Box Plot)
    ax6 = plt.subplot(2, 3, 6)
    if 'user_followers' in df.columns:
        misinfo_followers = df[df['misinfo']]['user_followers'].replace(0, 1)
        regular_followers = df[~df['misinfo']]['user_followers'].replace(0, 1)
        
        data_to_plot = [
            np.log10(regular_followers),
            np.log10(misinfo_followers)
        ]
        
        bp = plt.boxplot(data_to_plot, labels=['Regular', 'Misinformation'], 
                        patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        plt.ylabel('Log10(User Followers)')
        plt.title('Follower Count: Regular vs Misinformation', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('exploratory_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ Exploratory analysis complete! Saved as 'exploratory_analysis.png'\n")
    plt.show()


# ==============================
# 4. BUILD USER-ONLY NETWORK
# ==============================

def build_user_network(df):
    """
    Build a user-to-user network based on shared hashtags.
    Two users are connected if they use at least one common hashtag.
    Returns NetworkX graph with only user nodes.
    
    Optimized version: Uses hashtag-to-users mapping for O(n*m) complexity
    instead of O(n²), where n=users and m=hashtags.
    """
    print("=" * 60)
    print("STEP 4: BUILDING USER-ONLY NETWORK")
    print("=" * 60)
    
    # PERFORMANCE OPTIMIZATION: limit network size to top users
    MAX_USERS = 500
    top_users = df['user_name'].value_counts().head(MAX_USERS).index
    df = df[df['user_name'].isin(top_users)].copy()

    G = nx.Graph()
    
    # Dictionary to store hashtags per user
    user_hashtags = {}
    # Dictionary to store users per hashtag (for optimization)
    hashtag_users = {}
    
    # Parse hashtags and build bidirectional mapping
    print("  Parsing hashtags...")
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
    print("  Adding nodes...")
    for user in user_hashtags.keys():
        G.add_node(user, type='user')
    
    # Build edges more efficiently: for each hashtag, connect all users who used it
    print("  Building edges (this may take a moment for large datasets)...")
    edges_added = 0
    processed_pairs = set()  # Track edges to avoid duplicates
    
    for tag, users_with_tag in hashtag_users.items():
        users_list = list(users_with_tag)
        # Connect all pairs of users who share this hashtag
        for i, user1 in enumerate(users_list):
            for user2 in users_list[i+1:]:
                # Use sorted tuple to avoid duplicate edges
                edge_key = tuple(sorted([user1, user2]))
                if edge_key not in processed_pairs:
                    G.add_edge(user1, user2)
                    processed_pairs.add(edge_key)
                    edges_added += 1
    
    print(f"✓ User-only network created!")
    print(f"  Nodes (users): {G.number_of_nodes():,}")
    print(f"  Edges (connections): {G.number_of_edges():,}")
    if G.number_of_nodes() > 0:
        print(f"  Average degree: {2*G.number_of_edges()/G.number_of_nodes():.2f}")
    print()
    
    return G, user_hashtags


# ==============================
# 5. SUPER-SPREADER ANALYSIS
# ==============================

def analyze_super_spreaders(G, df):
    """
    Identify and visualize top super-spreaders in the user network.
    Uses multiple centrality measures to find influential users.
    """
    print("=" * 60)
    print("STEP 5: SUPER-SPREADER ANALYSIS")
    print("=" * 60)
    
    # Calculate multiple centrality measures
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    
    # Get top 10 by degree centrality
    top_spreaders = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print("Top 10 Super-Spreaders (by Degree Centrality):")
    print("-" * 60)
    for i, (user, score) in enumerate(top_spreaders, 1):
        misinfo_count = len(df[(df['user_name'] == user) & (df['misinfo'])])
        total_tweets = len(df[df['user_name'] == user])
        print(f"{i:2d}. {user[:40]:40s} | Centrality: {score:.4f} | "
              f"Misinfo: {misinfo_count}/{total_tweets}")
    
    # Visualize super-spreader network
    top_users = [u for u, _ in top_spreaders]
    subG = G.subgraph(top_users)
    
    # Create a more informative visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left plot: Network visualization
    pos = nx.spring_layout(subG, seed=42, k=1, iterations=50)
    
    # Color nodes by misinformation spreading
    node_colors = []
    for user in top_users:
        misinfo_count = len(df[(df['user_name'] == user) & (df['misinfo'])])
        if misinfo_count > 0:
            node_colors.append('red')
        else:
            node_colors.append('lightblue')
    
    nx.draw(subG, pos, ax=ax1, with_labels=True, 
           node_color=node_colors, node_size=1500, 
           edge_color='gray', font_size=8, font_weight='bold',
           width=2, alpha=0.7)
    ax1.set_title('Top 10 Super-Spreaders Network\n(Red = Misinformation Spreader)', 
                  fontsize=12, fontweight='bold')
    
    # Right plot: Centrality comparison
    users_short = [u[:20] for u in top_users]
    deg_scores = [degree_centrality[u] for u in top_users]
    bet_scores = [betweenness_centrality[u] for u in top_users]
    
    x = np.arange(len(users_short))
    width = 0.35
    
    ax2.barh(x - width/2, deg_scores, width, label='Degree Centrality', color='skyblue')
    ax2.barh(x + width/2, bet_scores, width, label='Betweenness Centrality', color='coral')
    ax2.set_yticks(x)
    ax2.set_yticklabels(users_short)
    ax2.set_xlabel('Centrality Score')
    ax2.set_title('Centrality Measures Comparison', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('super_spreaders_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✓ Super-spreader analysis complete! Saved as 'super_spreaders_analysis.png'\n")
    plt.show()
    
    return top_spreaders


# ==============================
# 6. SIR SIMULATION
# ==============================

def run_sir_simulation(df, G, steps=20, beta=0.05, gamma=0.2):
    """
    Run SIR (Susceptible-Infected-Recovered) model simulation.
    Enhanced with network-based infection and better visualization.
    """
    print("=" * 60)
    print("STEP 6: SIR SIMULATION")
    print("=" * 60)
    
    users = list(G.nodes())
    status = {u: 'S' for u in users}  # All start as Susceptible
    
    # Infect initial users (those who spread misinformation)
    initial_infected = df[df['misinfo']]['user_name'].unique()
    initial_infected = [u for u in initial_infected if u in users][:min(10, len(initial_infected))]
    
    for u in initial_infected:
        status[u] = 'I'
    
    print(f"Initial conditions:")
    print(f"  Total users: {len(users):,}")
    print(f"  Initially infected: {len(initial_infected)}")
    print(f"  Infection rate (β): {beta}")
    print(f"  Recovery rate (γ): {gamma}")
    print(f"  Simulation steps: {steps}\n")
    
    # Track history
    S_history = [list(status.values()).count('S')]
    I_history = [list(status.values()).count('I')]
    R_history = [list(status.values()).count('R')]
    
    # Run simulation
    for step in range(steps):
        new_status = status.copy()
        
        for u in users:
            if status[u] == 'I':
                # Recovery: infected users can recover
                if random.random() < gamma:
                    new_status[u] = 'R'
            elif status[u] == 'S':
                # Infection: susceptible users can get infected
                # Check neighbors in network
                neighbors = list(G.neighbors(u))
                if neighbors:
                    # Higher infection probability if neighbors are infected
                    infected_neighbors = sum(1 for n in neighbors if status[n] == 'I')
                    infection_prob = beta * (1 + 0.5 * infected_neighbors)
                    if random.random() < min(infection_prob, 1.0):
                        new_status[u] = 'I'
                else:
                    # No neighbors: base infection rate
                    if random.random() < beta:
                        new_status[u] = 'I'
        
        status = new_status
        S = list(status.values()).count('S')
        I = list(status.values()).count('I')
        R = list(status.values()).count('R')
        
        S_history.append(S)
        I_history.append(I)
        R_history.append(R)
        
        if step % 5 == 0 or step == steps - 1:
            print(f"Step {step:2d}: S={S:6d}, I={I:6d}, R={R:6d}")
    
    # Enhanced visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: SIR curves
    time_steps = range(len(S_history))
    ax1.plot(time_steps, S_history, label='Susceptible', marker='o', linewidth=2, markersize=6, color='blue')
    ax1.plot(time_steps, I_history, label='Infected', marker='s', linewidth=2, markersize=6, color='red')
    ax1.plot(time_steps, R_history, label='Recovered', marker='^', linewidth=2, markersize=6, color='green')
    ax1.set_xlabel('Time Steps', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Number of Users', fontsize=11, fontweight='bold')
    ax1.set_title('SIR Model: Misinformation Spread Over Time', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Percentage view
    total = len(users)
    S_pct = [s/total*100 for s in S_history]
    I_pct = [i/total*100 for i in I_history]
    R_pct = [r/total*100 for r in R_history]
    
    ax2.plot(time_steps, S_pct, label='Susceptible %', marker='o', linewidth=2, markersize=6, color='blue')
    ax2.plot(time_steps, I_pct, label='Infected %', marker='s', linewidth=2, markersize=6, color='red')
    ax2.plot(time_steps, R_pct, label='Recovered %', marker='^', linewidth=2, markersize=6, color='green')
    ax2.set_xlabel('Time Steps', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Percentage of Users (%)', fontsize=11, fontweight='bold')
    ax2.set_title('SIR Model: Percentage View', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig('sir_simulation.png', dpi=150, bbox_inches='tight')
    print("\n✓ SIR simulation complete! Saved as 'sir_simulation.png'\n")
    plt.show()
    
    return S_history, I_history, R_history


# ==============================
# 7. MACHINE LEARNING MODEL
# ==============================

def train_misinformation_predictor(df):
    """
    Train a simple machine learning model to predict misinformation.
    Uses text features and user characteristics.
    """
    print("=" * 60)
    print("STEP 7: MACHINE LEARNING MODEL")
    print("=" * 60)
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        import seaborn as sns
    except ImportError:
        print("⚠ scikit-learn not installed. Skipping ML model.")
        print("  Install with: pip install scikit-learn\n")
        return None
    
    # Prepare features
    print("Preparing features...")
    
    # Text features
    texts = df['text'].fillna('').astype(str)
    
    # User features
    user_features = pd.DataFrame()
    if 'user_followers' in df.columns:
        user_features['followers'] = df['user_followers'].fillna(0)
    else:
        user_features['followers'] = 0
    
    # Tweet length
    user_features['tweet_length'] = texts.str.len()
    
    # Hashtag count
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
    
    # TF-IDF vectorization (limited features for speed)
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english', ngram_range=(1, 2))
    text_features = vectorizer.fit_transform(texts)
    
    # Combine features
    from scipy.sparse import hstack
    X = hstack([text_features, user_features.values])
    y = df['misinfo'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]:,} samples")
    print(f"Test set: {X_test.shape[0]:,} samples")
    
    # Train model
    print("Training Random Forest classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✓ Model training complete!")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Regular', 'Misinformation']))
    
    # Confusion matrix visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Regular', 'Misinformation'],
                yticklabels=['Regular', 'Misinformation'])
    ax1.set_xlabel('Predicted', fontweight='bold')
    ax1.set_ylabel('Actual', fontweight='bold')
    ax1.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    
    # Feature importance (top 10)
    feature_importance = model.feature_importances_[:10]
    feature_names = [f'Feature {i+1}' for i in range(10)]
    ax2.barh(range(len(feature_names)), feature_importance, color='steelblue')
    ax2.set_yticks(range(len(feature_names)))
    ax2.set_yticklabels(feature_names)
    ax2.set_xlabel('Importance', fontweight='bold')
    ax2.set_title('Top 10 Feature Importances', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('ml_model_results.png', dpi=150, bbox_inches='tight')
    print("\n✓ ML model results saved as 'ml_model_results.png'\n")
    plt.show()
    
    return model, vectorizer, accuracy


# ==============================
# MAIN EXECUTION
# ==============================

def main():
    """Main execution function that runs all analysis steps."""
    print("\n" + "="*60)
    print("COVID-19 MISINFORMATION SPREAD ANALYSIS")
    print("="*60 + "\n")
    
    # Step 1: Load and clean data
    df = load_and_clean_data()
    
    # Step 2: Detect misinformation
    df = detect_misinformation(df)
    
    # Step 3: Exploratory analysis
    exploratory_analysis(df)
    
    # Step 4: Build user-only network
    G, user_hashtags = build_user_network(df)
    
    # Step 5: Super-spreader analysis
    top_spreaders = analyze_super_spreaders(G, df)
    
    # Step 6: SIR simulation
    S_history, I_history, R_history = run_sir_simulation(df, G, steps=20, beta=0.05, gamma=0.2)
    
    # Step 7: Machine learning model
    ml_results = train_misinformation_predictor(df)
    
    print("=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - exploratory_analysis.png")
    print("  - super_spreaders_analysis.png")
    print("  - sir_simulation.png")
    if ml_results:
        print("  - ml_model_results.png")
    print()


if __name__ == "__main__":
    main()

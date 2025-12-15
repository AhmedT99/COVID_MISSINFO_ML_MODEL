## Project Cover Page

- **Project Title**: COVID-19 Misinformation Spread Analysis on Twitter  
- **Course Name**: [Add course name here]  
- **Student Name & ID**:  
  - [Your Name] – [Your ID]  
- **Instructor Name**: [Add instructor name here]  
- **Submission Date**: [Add date here]  

---

## Abstract

This project analyzes how COVID-19 misinformation spreads on Twitter.  
It loads a large dataset of tweets, detects misinformation using simple keyword rules, builds a user network, models the spread with an SIR model, and provides interactive visualizations through a Streamlit dashboard.  
The goal is to understand which users act as super-spreaders and how misinformation can spread through a social network over time.

---

## 1. Introduction

### 1.1 Overview
This project studies misinformation about COVID-19 on Twitter using data science and network analysis.  
It focuses on users, their tweets, the hashtags they use, and how they connect to each other.

### 1.2 Purpose and Objectives
- Detect tweets that likely contain misinformation.  
- Identify top users and super-spreaders.  
- Build a user network based on shared hashtags.  
- Simulate how misinformation spreads using an SIR (Susceptible–Infected–Recovered) model.  
- Provide an interactive dashboard to explore results.

### 1.3 Problem Statement
Misinformation about COVID-19 spreads quickly on social media and can affect public health decisions.  
We need a simple, data-driven way to see who is spreading it, how it spreads, and what patterns appear in the data.

---

## 2. Background and Literature Review

- Social networks like Twitter allow information to spread very fast.  
- Network analysis is a common way to model user connections and influence.  
- SIR models are often used in epidemiology but can also be applied to information and rumor spreading.  
- Previous work shows that a small group of influential users can drive a large portion of misinformation online.

---

## 3. Problem Statement

Given a large Twitter dataset about COVID-19:
- How can we detect potential misinformation in tweets using a simple method?  
- Which users appear to be super-spreaders?  
- How might misinformation spread through the user network over time?  
- How can we show all of this in a clear and interactive way?

---

## 4. System Requirements / Tools Used

### 4.1 Software
- Operating System: Any (tested on macOS / Mac M2)  
- Python 3.9+  
- Virtual environment (optional but recommended)

### 4.2 Programming Languages and Libraries
- **Language**: Python  
- **Data**: `pandas`, `numpy`  
- **Network analysis**: `networkx`  
- **Plots**: `matplotlib`, `plotly`, `seaborn`  
- **Dashboard**: `streamlit`  
- **Machine learning**: `scikit-learn`, `scipy`

### 4.3 Hardware
- Laptop with at least 8–16 GB RAM for comfortable use.  
- Project is optimized to run fast by using only a sample (5–10%) of the full dataset.

---

## 5. Methodology / Approach

1. **Data Loading and Sampling**  
   - Load the CSV file `data/covid19_tweets.csv`.  
   - Select only needed columns.  
   - Sample a small percentage of rows (e.g., 5%) for faster processing.

2. **Data Cleaning**  
   - Drop rows with missing `user_name` or `text`.  
   - Convert follower counts to numeric values.  

3. **Misinformation Detection**  
   - Use a list of keywords (e.g., “hoax”, “fake”, “plandemic”).  
   - Label a tweet as misinformation if its text contains any of these words.

4. **Exploratory Data Analysis (EDA)**  
   - Count tweets per user.  
   - Count misinformation tweets.  
   - Plot distributions (followers, tweet length, etc.).

5. **Network Construction**  
   - For each user, collect hashtags used.  
   - Connect two users if they share at least one hashtag.  
   - Limit to top users to keep the graph small and fast.

6. **SIR Model Simulation**  
   - Treat users as nodes in the network.  
   - Mark some misinformation users as initially infected.  
   - Run an SIR simulation over several time steps.

7. **Machine Learning Model (Optional)**  
   - Extract text features (TF‑IDF) and simple user features (followers, tweet length, hashtag count).  
   - Train a Random Forest classifier to predict misinformation.

8. **Dashboard (Streamlit)**  
   - Provide tabs for data summary, network visualization, SIR simulation, and ML metrics.  
   - Use caching and sampling for fast interaction.

---

## 6. System Design

### 6.1 Main Components
- **Data layer**: CSV file (`covid19_tweets.csv`).  
- **Processing layer**: Functions in `main.py` for cleaning, detection, network, SIR, and ML.  
- **Presentation layer**: Streamlit app in `dashboard.py`.

### 6.2 Example Diagrams (to include in final report)
You can draw and embed diagrams in the written report (outside this README):
- **Use Case Diagram**:  
  - Actor: Student / Analyst  
  - Use cases: Load data, view statistics, view network, run SIR, train model.

- **Flowchart (high level)**:  
  - Start → Load & sample data → Clean data → Detect misinformation  
  → Build network → Run EDA → SIR simulation → (Optional) train ML → Show dashboard → End

- **Simple Architecture Diagram**:  
  - Data → Processing (Python functions) → Visualizations / Dashboard.

---

## 7. Implementation

### 7.1 Files
- `main.py`: Runs the full analysis as a script.  
- `dashboard.py`: Streamlit dashboard for interactive exploration.  
- `requirements.txt`: Python dependencies.  
- `data/covid19_tweets.csv`: Tweet dataset.  
- `data/network_cache.pkl`, `data/user_hashtags_cache.pkl`: Network cache files for speed.

### 7.2 Key Features
- Loads and samples big data to keep performance high.  
- Detects misinformation using simple keyword rules.  
- Builds a user-only network based on shared hashtags.  
- Identifies top super-spreaders by centrality.  
- Simulates misinformation spread with an SIR model.  
- Trains an optional ML model to predict misinformation.  
- Provides an interactive web dashboard using Streamlit.

---

## 8. Results & Testing

### 8.1 Results
- Percentage of tweets labeled as misinformation.  
- Top users by tweet count and misinformation count.  
- Network view of top connected users.  
- SIR curves showing how misinformation could spread and then die out.  
- ML model accuracy and confusion matrix.

### 8.2 Testing
- Manual testing of each tab in the dashboard.  
- Checked that:
  - Data loads correctly.  
  - Plots render without errors.  
  - Network builds and is limited in size.  
  - SIR simulation runs and updates plots.  
  - ML model trains without crashing.

### 8.3 Problems Encountered
- Building the full network on all users was too slow and used a lot of memory.  
  - **Solution**: limit to top users and sample data.  
- Dashboard recomputation was slow.  
  - **Solution**: use Streamlit caching and save the network to disk.

---

## 9. Discussion

### 9.1 What Worked Well
- Sampling and caching made the project fast and smooth.  
- Simple keyword-based detection is easy to understand.  
- Network and SIR visualizations give an intuitive view of spread.

### 9.2 Challenges
- Handling large data and network size.  
- Balancing detail vs. performance.  
- Choosing simple models that still give useful insights.

### 9.3 Future Improvements
- Use more advanced NLP models (e.g., transformer-based) for better misinformation detection.  
- Add time-based analysis to see how misinformation changes over days/weeks.  
- Add more interactive controls in the dashboard (e.g., pick keywords, change thresholds).

---

## 10. Conclusion

This project shows a complete pipeline for analyzing Twitter COVID‑19 misinformation using Python.  
It covers data loading, cleaning, detection, network analysis, simulation, machine learning, and visualization.  
From this work, we learned how to handle relatively large datasets, design simple models, and present results in an interactive way that is easy to understand.

---

## 11. References

Add detailed references in your final report, for example:
- Twitter data collection source (if provided).  
- Articles or blog posts on SIR models for information spread.  
- Documentation for:
  - `pandas`, `networkx`, `matplotlib`, `seaborn`, `plotly`, `streamlit`, `scikit-learn`.  
- Any research papers or books you used about misinformation or social network analysis.

---

## 12. Appendix

### 12.1 Full Code

- Main analysis script: `main.py`  
- Dashboard application: `dashboard.py`  

The full code is already included in the project files and can be opened directly in any code editor or IDE.  
To run:

```bash
# activate venv (if used)
source venv/bin/activate

# install dependencies
pip install -r requirements.txt

# run analysis script
python main.py

# run dashboard
python -m streamlit run dashboard.py
```

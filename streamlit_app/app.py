# streamlit_app/app.py
"""
🎬 Movie Recommender System - Interactive Web App
Built with Streamlit for IBM ML Professional Certificate Capstone
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

project_root = Path(__file__).parent.parent

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Page configuration
st.set_page_config(
    page_title="🎬 Movie Recommender System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF6B6B;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #4ECDC4;
        padding-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        width: 100%;
        background-color: #FF6B6B;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.image("https://via.placeholder.com/300x100/FF6B6B/FFFFFF?text=MovieLens+25M", width=True)
st.sidebar.title("🎬 Navigation")

page = st.sidebar.radio(
    "Select a page:",
    ["🏠 Home", "🎯 Get Recommendations", "📊 Model Performance", "📈 Dataset Insights", "ℹ️ About"],
    index=0
)

# Add project stats in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Project Stats")
st.sidebar.metric("Total Ratings", "25M")
st.sidebar.metric("Users", "162K")
st.sidebar.metric("Movies", "62K")
st.sidebar.metric("Best Model MAE", "2.44")

st.sidebar.markdown("---")
st.sidebar.markdown("### 👨‍💻 Developer")
st.sidebar.info("**Kaustubh Mukdam**\n\nIBM ML Professional Certificate\n\nCapstone Project 2026")

# ============================================
# PAGE 1: HOME
# ============================================

if page == "🏠 Home":
    st.markdown('<p class="main-header">🎬 Movie Recommender System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Powered by Machine Learning & MovieLens 25M Dataset</p>', unsafe_allow_html=True)
    
    # Hero section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h2>🎯 Hybrid Model</h2>
            <p>Combining Content-Based & Collaborative Filtering</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h2>🧠 AI-Powered</h2>
            <p>SVD, NMF, KNN & Neural Networks</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h2>⚡ Real-Time</h2>
            <p>Instant personalized recommendations</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Project overview
    st.header("📖 Project Overview")
    
    st.markdown("""
    Welcome to the **Movie Recommender System** - a production-ready machine learning application built as part of 
    the **IBM Machine Learning Professional Certificate** capstone project.
    
    ### 🎯 What This System Does
    
    This intelligent recommender system analyzes **25 million ratings** from **162,000 users** on **62,000 movies** 
    to provide personalized movie recommendations. It combines multiple advanced algorithms:
    
    - **Content-Based Filtering**: Analyzes movie genres and titles using TF-IDF
    - **Collaborative Filtering**: Learns from user behavior patterns (SVD, NMF, KNN)
    - **Neural Collaborative Filtering**: Deep learning approach with embeddings
    - **Hybrid System**: Combines all approaches for optimal performance
    
    ### 🚀 Key Features
    
    - ✅ **Multi-Algorithm Approach**: 6 different recommendation algorithms
    - ✅ **Personalized Recommendations**: Tailored to individual user preferences
    - ✅ **Model Comparison**: See how different algorithms perform
    - ✅ **Interactive Visualizations**: Explore dataset insights
    - ✅ **Production-Ready**: Optimized for speed and accuracy
    
    ### 📊 Performance Highlights
    
    | Metric | Value | Notes |
    |--------|-------|-------|
    | **Best MAE** | 2.44 | Hybrid model (test set) |
    | **Best RMSE** | 2.84 | Hybrid model (test set) |
    | **Neural CF MAE** | 0.65 | Validation set |
    | **Training Time** | ~2 min | All models combined |
    
    ### 🎬 Get Started
    
    👈 Use the navigation menu on the left to:
    - Get personalized movie recommendations
    - Compare different recommendation algorithms
    - Explore dataset insights and visualizations
    - Learn more about the project
    """)
    
    # Technologies used
    st.markdown("---")
    st.header("🛠️ Technologies Used")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Data & ML**
        - Python 3.11
        - Pandas & NumPy
        - Scikit-learn
        - TensorFlow/Keras
        """)
    
    with col2:
        st.markdown("""
        **Algorithms**
        - SVD (Matrix Factorization)
        - NMF (Non-negative MF)
        - KNN (Nearest Neighbors)
        - Neural Networks
        """)
    
    with col3:
        st.markdown("""
        **Deployment**
        - Streamlit
        - GitHub
        - Plotly
        - SciPy
        """)

# ============================================
# PAGE 2: GET RECOMMENDATIONS
# ============================================

elif page == "🎯 Get Recommendations":
    st.markdown('<p class="main-header">🎯 Get Movie Recommendations</p>', unsafe_allow_html=True)
    
    # Load necessary data
    @st.cache_data
    def load_recommendation_data():
        try:
            movies = pd.read_csv(project_root / "data/processed/movies_with_genre_features.csv")
            train_ratings = pd.read_csv(project_root / "data/processed/train_ratings.csv")
            
            # Load mappings
            import pickle
            with open(project_root / "data/processed/interaction_matrix_mappings.pkl", 'rb') as f:
                mappings = pickle.load(f)
            
            return movies, train_ratings, mappings
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None, None, None
    
    movies, train_ratings, mappings = load_recommendation_data()
    
    if movies is not None:
        # User selection
        st.header("👤 Select User or Search Movies")
        
        tab1, tab2 = st.tabs(["Recommendations for User", "Similar Movies"])
        
        with tab1:
            st.subheader("Get Personalized Recommendations")
            
            # User ID input
            valid_users = list(mappings['user_to_idx'].keys())
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                user_input = st.selectbox(
                    "Select User ID (sample of active users):",
                    options=valid_users[:100],  # Show first 100 for speed
                    index=0
                )
            
            with col2:
                n_recommendations = st.slider("Number of recommendations:", 5, 20, 10)
            
            if st.button("🎬 Get Recommendations", key="get_recs"):
                with st.spinner("Generating recommendations..."):
                    try:
                        # Import hybrid recommender
                        from optimised_hybrid_recommender import HybridRecommender
                        
                        @st.cache_resource
                        def load_hybrid_model():
                            return HybridRecommender()
                        
                        hybrid = load_hybrid_model()
                        
                        # Get recommendations
                        recommendations = hybrid.recommend_hybrid(user_input, top_n=n_recommendations)
                        
                        if recommendations is not None and len(recommendations) > 0:
                            st.success(f"✅ Top {n_recommendations} recommendations for User {user_input}:")
                            
                            # Display recommendations
                            for idx, row in recommendations.iterrows():
                                with st.container():
                                    col1, col2 = st.columns([3, 1])
                                    
                                    with col1:
                                        st.markdown(f"**{idx + 1}. {row['title']}**")
                                        st.caption(f"Genres: {row['genres']}")
                                    
                                    with col2:
                                        score = row['predicted_rating']
                                        st.metric("Score", f"{score:.2f}/5.0")
                                
                                st.markdown("---")
                        else:
                            st.warning("No recommendations available for this user.")
                    
                    except Exception as e:
                        st.error(f"Error generating recommendations: {e}")
        
        with tab2:
            st.subheader("Find Similar Movies")
            
            # Movie search
            movie_titles = movies['title'].tolist()
            
            selected_movie = st.selectbox(
                "Search for a movie:",
                options=movie_titles[:500],  # First 500 for speed
                index=0
            )
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                similarity_method = st.radio(
                    "Similarity method:",
                    ["Genre-based", "Title-based (TF-IDF)"],
                    horizontal=True
                )
            
            with col2:
                n_similar = st.slider("Number of similar movies:", 5, 20, 10)
            
            if st.button("🔍 Find Similar Movies", key="similar"):
                with st.spinner("Finding similar movies..."):
                    try:
                        # FIXED: Load content-based recommender properly
                        from content_based_recommender import ContentBasedRecommender
                        
                        @st.cache_resource
                        def load_content_model():
                            cb = ContentBasedRecommender()
                            # Ensure all features are prepared
                            if not hasattr(cb, 'genre_features'):
                                cb.prepare_genre_features()
                            if not hasattr(cb, 'tfidf_features'):
                                cb.prepare_tfidf_features()
                            return cb
                        
                        cb = load_content_model()
                        
                        # Get movie ID
                        movie_id = movies[movies['title'] == selected_movie]['movieId'].values[0]
                        
                        # Check if movie exists in content-based index
                        if movie_id not in cb.movie_id_to_idx:
                            st.warning(f"Movie '{selected_movie}' not found in the recommendation system.")
                        else:
                            # Get similar movies
                            use_genre = similarity_method == "Genre-based"
                            similar = cb.get_similar_movies(movie_id, use_genre=use_genre, top_n=n_similar)
                            
                            if similar is not None and len(similar) > 0:
                                st.success(f"✅ Movies similar to **{selected_movie}**:")
                                
                                for idx, row in similar.iterrows():
                                    with st.container():
                                        col1, col2 = st.columns([3, 1])
                                        
                                        with col1:
                                            st.markdown(f"**{idx + 1}. {row['title']}**")
                                            st.caption(f"Genres: {row['genres']}")
                                        
                                        with col2:
                                            score = row['similarity_score']
                                            st.metric("Similarity", f"{score:.2%}")
                                    
                                    st.markdown("---")
                            else:
                                st.warning("No similar movies found.")
                    
                    except Exception as e:
                        st.error(f"Error finding similar movies: {e}")
                        import traceback
                        st.code(traceback.format_exc())

# ============================================
# PAGE 3: MODEL PERFORMANCE
# ============================================

elif page == "📊 Model Performance":
    st.markdown('<p class="main-header">📊 Model Performance Comparison</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Compare the performance of different recommendation algorithms implemented in this project.
    """)
    
    # Load performance data
    try:
        perf_data = pd.read_csv(project_root / "reports/final_model_comparison.csv")
        
        # Display metrics
        st.header("🎯 Overall Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Best MAE", "2.44", "Hybrid Model")
        
        with col2:
            st.metric("Best RMSE", "2.84", "Hybrid Model")
        
        with col3:
            st.metric("Models Trained", "6", "Content + CF")
        
        with col4:
            st.metric("Training Time", "~2 min", "All models")
        
        # Performance table
        st.markdown("---")
        st.header("📈 Detailed Results")
        
        st.dataframe(
            perf_data.style.highlight_min(subset=['MAE', 'RMSE'], color='lightgreen')
                          .highlight_max(subset=['Hit_Rate'], color='lightgreen'),
            use_container_width=True
        )
        
        # Display comparison chart
        st.markdown("---")
        st.header("📊 Visual Comparison")

        image_path = project_root / "reports/eda_figures/07_final_model_comparison.png"
        
        from PIL import Image
        if os.path.exists(image_path):
            try:
                img = Image.open(image_path)
                st.image(img, use_column_width=True)
            except Exception as e:
                st.error(f"Error loading image: {e}")
                st.info("📊 Comparison chart exists but couldn't be loaded. Check file format.")
        else:
            st.info("📊 Comparison chart not found. Run `python performance_comparison.py` to generate it.")
            
            # Show text-based comparison instead
            st.markdown("""
            ### Quick Comparison (without chart):
            
            **Best Models by Metric:**
            - **Lowest MAE**: Neural CF (0.65) → Hybrid (2.44) → SVD (3.29)
            - **Lowest RMSE**: Neural CF (0.71) → Hybrid (2.84) → SVD (3.46)
            - **Best Hit Rate**: TF-IDF (3.09%) → Genre (3.06%)
            
            **Winner**: 🏆 **Hybrid Model** (best test set performance)
            """)
        
        # Model explanations
        st.markdown("---")
        st.header("🧠 Model Explanations")
        
        with st.expander("📝 Content-Based Filtering"):
            st.markdown("""
            **How it works:**
            - Analyzes movie features (genres, titles)
            - Uses TF-IDF and cosine similarity
            - Recommends similar movies to what user liked
            
            **Pros:** Works for new users, explainable
            **Cons:** Limited to feature similarity, no collaborative patterns
            
            **Performance:**
            - Hit Rate: ~3%
            - Best for: Cold-start scenarios, explainable recommendations
            """)
        
        with st.expander("🔢 SVD (Singular Value Decomposition)"):
            st.markdown("""
            **How it works:**
            - Matrix factorization technique
            - Decomposes user-movie matrix into latent factors
            - Captures hidden patterns in user preferences
            
            **Pros:** Fast, scalable, good accuracy
            **Cons:** Requires many ratings per user
            
            **Performance:**
            - RMSE: 3.46
            - MAE: 3.29
            - Training time: 2.13s
            """)
        
        with st.expander("🔢 NMF (Non-negative Matrix Factorization)"):
            st.markdown("""
            **How it works:**
            - Similar to SVD but constrains factors to be non-negative
            - More interpretable latent features
            - Better for sparse data
            
            **Pros:** Interpretable, handles sparsity well
            **Cons:** Slower than SVD
            
            **Performance:**
            - Training time: 8.74s
            - Reconstruction error: 1833.03
            """)
        
        with st.expander("👥 KNN (K-Nearest Neighbors)"):
            st.markdown("""
            **How it works:**
            - Finds users with similar rating patterns
            - Recommends movies liked by similar users
            - User-based collaborative filtering
            
            **Pros:** Simple, intuitive, no training needed
            **Cons:** Slow for large datasets
            
            **Performance:**
            - Training time: <0.01s (lazy learning)
            - Good for diverse recommendations
            """)
        
        with st.expander("🧠 Neural Collaborative Filtering"):
            st.markdown("""
            **How it works:**
            - Deep learning with user/movie embeddings
            - Multi-layer perceptron combines embeddings
            - Learns complex non-linear patterns
            
            **Pros:** Best accuracy, captures complex patterns
            **Cons:** Requires more training time, less interpretable
            
            **Performance:**
            - Validation MAE: 0.65 (best!)
            - Validation loss: 0.71
            - Training time: 78.5s
            - Parameters: 714,097
            """)
        
        with st.expander("🎯 Hybrid System"):
            st.markdown("""
            **How it works:**
            - Combines SVD (60%) and NMF (40%)
            - Weighted ensemble approach
            - Leverages strengths of both algorithms
            
            **Pros:** Best overall performance, robust
            **Cons:** Slightly more complex
            
            **Performance:**
            - RMSE: 2.84 (18% improvement over SVD)
            - MAE: 2.44 (26% improvement over SVD)
            - Production-ready!
            """)
    
    except Exception as e:
        st.error(f"Error loading performance data: {e}")

# ============================================
# PAGE 4: DATASET INSIGHTS
# ============================================

elif page == "📈 Dataset Insights":
    st.markdown('<p class="main-header">📈 Dataset Insights & Visualizations</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Explore comprehensive exploratory data analysis of the MovieLens 25M dataset.
    """)
    
    # Dataset statistics
    st.header("📊 Dataset Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Ratings", "25,000,095")
    
    with col2:
        st.metric("Unique Users", "162,541")
    
    with col3:
        st.metric("Unique Movies", "59,047")
    
    with col4:
        st.metric("Sparsity", "99.74%")
    
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Avg Rating", "3.53 / 5.0")
    
    with col2:
        st.metric("Avg Ratings/User", "153.8")
    
    with col3:
        st.metric("Avg Ratings/Movie", "423.4")
    
    with col4:
        st.metric("Date Range", "1995-2019")
    
    # Visualizations
    st.markdown("---")
    st.header("📊 EDA Visualizations")
    
    from PIL import Image
    
    viz_files = [
        ("01_rating_distribution.png", "Rating Distribution"),
        ("02_genre_distribution.png", "Genre Distribution"),
        ("03_user_engagement.png", "User Engagement"),
        ("04_top_20_movies.png", "Top 20 Movies"),
        ("05_wordcloud.png", "Movie Titles Word Cloud"),
        ("06_temporal_analysis.png", "Temporal Analysis"),
        ("07_final_model_comparison.png", "Final Model Comparison")
    ]
    
    for viz_file, title in viz_files:
        with st.expander(f"📈 {title}"):
            try:
                img = Image.open(project_root / f"reports/eda_figures/{viz_file}")
                st.image(img, width=True)
            except Exception as e:
                st.warning(f"{viz_file} not found")
    
    # Key insights
    st.markdown("---")
    st.header("💡 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📊 Rating Patterns
        - Most common rating: **4.0** stars
        - Users tend to rate movies they like
        - Very few ratings below 2.0
        - Rating distribution is left-skewed
        """)
        
        st.markdown("""
        ### 🎭 Genre Distribution
        - **Drama** is the most common genre
        - **Comedy** and **Action** follow closely
        - **Documentary** and **IMAX** are least common
        - Many movies have multiple genres
        """)
    
    with col2:
        st.markdown("""
        ### 👥 User Behavior
        - Highly sparse matrix (99.74%)
        - Power users rate 1000+ movies
        - Average user rates ~154 movies
        - Rating activity increased over time
        """)
        
        st.markdown("""
        ### 🎬 Movie Popularity
        - Top movies have 50,000+ ratings
        - Long tail distribution
        - Blockbusters dominate popularity
        - Many movies have very few ratings
        """)

# ============================================
# PAGE 5: ABOUT
# ============================================

elif page == "ℹ️ About":
    st.markdown('<p class="main-header">ℹ️ About This Project</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🎓 IBM Machine Learning Professional Certificate
    
    This project is the **capstone** for the IBM Machine Learning Professional Certificate offered on Coursera.
    
    ### 👨‍💻 Developer
    
    **Kaustubh Mukdam**
    - GitHub: [KaustubhMukdam](https://github.com/KaustubhMukdam)
    - LinkedIn: [Kaustubh Mukdam](https://www.linkedin.com/in/kaustubh-mukdam-ab0170340/)
    - Email: [Kaustubh Mukdam](kaustubhmukdam7@gmail.com)
    
    ### 🎯 Project Objectives
    
    1. **Build a Production-Ready Recommender System**
       - Implement multiple recommendation algorithms
       - Compare and evaluate different approaches
       - Create an interactive user interface
    
    2. **Demonstrate Machine Learning Skills**
       - Data preprocessing and feature engineering
       - Model training and evaluation
       - Hyperparameter tuning
       - Deployment and productionization
    
    3. **Apply Best Practices**
       - Version control with Git
       - Modular, reusable code
       - Comprehensive documentation
       - Professional presentation
    
    ### 📚 Dataset
    
    **MovieLens 25M Dataset**
    - Source: [GroupLens Research](https://grouplens.org/datasets/movielens/)
    - Size: 25 million ratings, 162,000 users, 62,000 movies
    - Date range: 1995-2019
    - Includes: Ratings, tags, movie metadata
    
    ### 🛠️ Technical Stack
    
    **Languages & Frameworks:**
    - Python 3.11
    - Streamlit (Web App)
    - TensorFlow/Keras (Deep Learning)
    - Scikit-learn (ML Algorithms)
    
    **Key Libraries:**
    - Pandas & NumPy (Data processing)
    - SciPy (Sparse matrices)
    - Plotly (Visualizations)
    - Matplotlib & Seaborn (Charts)
    
    ### 📖 Project Structure
    
    ```
    recommender-system/
    ├── data/                    # Dataset storage
    ├── src/                     # Source code
    ├── models/                  # Trained models
    ├── streamlit_app/          # Web application
    ├── reports/                 # EDA visualizations
    ├── notebooks/               # Jupyter notebooks
    └── tests/                   # Unit tests
    ```
    
    ### 🚀 Implementation Timeline
    
    - **Week 1-2:** Data download, EDA, feature engineering
    - **Week 3:** Content-based filtering (BoW, TF-IDF)
    - **Week 4:** Collaborative filtering (SVD, NMF, KNN, Neural CF)
    - **Week 5:** Hybrid system and optimization
    - **Week 6:** Streamlit app development
    - **Week 7-8:** Documentation and presentation
    
    ### 📄 License
    
    This project is licensed under the MIT License.
    
    ### 🙏 Acknowledgments
    
    - GroupLens Research for the MovieLens dataset
    - IBM and Coursera for the ML Professional Certificate
    - The open-source community for amazing libraries
    
    ### 📞 Contact
    
    Feel free to reach out for questions, suggestions, or collaboration opportunities!
    
    ---
    
    **⭐ If you found this project helpful, please give it a star on GitHub!**
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with ❤️ using Streamlit | © 2026 Kaustubh Mukdam</p>
    <p>IBM Machine Learning Professional Certificate Capstone Project</p>
</div>
""", unsafe_allow_html=True)

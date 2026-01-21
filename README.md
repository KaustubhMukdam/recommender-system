# 🎬 Movie Recommender System - Hybrid Machine Learning Approach

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-FF4B4B.svg)](https://streamlit.io/)

> A production-ready hybrid recommendation system built with MovieLens 25M dataset, combining content-based filtering, collaborative filtering, and deep learning approaches.

---

## 📊 Project Overview

This capstone project builds a movie recommendation system using 25 million ratings from 162,000 users on 62,000 movies. The system employs multiple recommendation algorithms and provides an interactive Streamlit web application.

### 🎯 Key Features

- **Content-Based Filtering**: Genre, TF-IDF, and BoW features
- **Collaborative Filtering**: SVD, NMF, KNN, Neural Collaborative Filtering
- **Hybrid System**: Ensemble approach combining multiple models
- **Explainable AI**: SHAP/LIME integration for recommendation explanations
- **Interactive UI**: Real-time recommendations via Streamlit
- **Production-Ready**: Scalable architecture with Docker deployment

---

## 🏗️ Project Structure

```
recommender-system/
├── data/
│   ├── raw/                    # MovieLens 25M dataset
│   └── processed/              # Engineered features
├── src/
│   ├── data_processing/        # ETL pipelines
│   ├── models/                 # ML models
│   └── utils/                  # Helper functions
├── streamlit_app/              # Web application
├── notebooks/                  # Jupyter notebooks
├── reports/                    # Visualizations & reports
├── models/                     # Saved models
└── tests/                      # Unit tests
```

---

## 📈 Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Ratings** | 25,000,095 |
| **Unique Users** | 162,541 |
| **Unique Movies** | 59,047 |
| **Average Rating** | 3.53 / 5.0 |
| **Sparsity** | 99.74% |
| **Date Range** | 1995-01-09 to 2019-11-21 |

---

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/YourUsername/recommender-system.git
cd recommender-system
```

### 2. Setup Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Mac/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Download Dataset
```bash
python 01_setup_and_download.py
```

### 4. Run EDA
```bash
python 02_comprehensive_eda.py
```

### 5. Feature Engineering
```bash
python 03_feature_engineering.py
```

### 6. Launch Streamlit App (Coming Soon)
```bash
streamlit run streamlit_app/app.py
```

---

## 📊 Exploratory Data Analysis

### Rating Distribution
![Rating Distribution](reports/eda_figures/rating_distribution.png)

### Genre Analysis
![Genre Distribution](reports/eda_figures/genre_distribution.png)

### Top Movies
![Top 20 Movies](reports/eda_figures/top_20_movies.png)

---

## 🧠 Models Implemented

### Content-Based Filtering
- **Genre-Based**: Cosine similarity on genre vectors
- **TF-IDF**: Title-based similarity
- **User Profiles**: Aggregate user preferences

### Collaborative Filtering
- **SVD**: Matrix factorization
- **NMF**: Non-negative matrix factorization
- **KNN**: User/Item-based nearest neighbors
- **Neural CF**: Deep learning embeddings

### Hybrid Approach
- Weighted ensemble of all models
- Meta-learning for optimal weighting
- Cold-start handling

---

## 📐 Evaluation Metrics

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- Precision@K
- Recall@K
- NDCG (Normalized Discounted Cumulative Gain)
- Coverage
- Diversity

---

## 🛠️ Technologies Used

- **Python 3.11**
- **Pandas & NumPy**: Data processing
- **Scikit-learn**: ML algorithms
- **TensorFlow/PyTorch**: Deep learning
- **Streamlit**: Web application
- **SHAP/LIME**: Model explainability
- **Docker**: Containerization

---

## 📁 File Descriptions

### Scripts
- `01_setup_and_download.py`: Download and setup dataset
- `02_comprehensive_eda.py`: Exploratory data analysis
- `03_feature_engineering.py`: Feature extraction pipeline
- `04_content_based_recommender.py`: Content-based models (Week 3)
- `05_collaborative_filtering.py`: Collaborative models (Week 4)
- `06_hybrid_system.py`: Hybrid ensemble (Week 5)

### Data Files (Generated)
- `movies_with_genre_features.csv`: One-hot encoded genres
- `movies_with_text_features.csv`: TF-IDF & BoW features
- `user_features.csv`: User statistics
- `movie_features.csv`: Movie statistics
- `interaction_matrix_sparse.npz`: User-movie ratings (sparse)
- `train_ratings.csv`: Training set (80%)
- `test_ratings.csv`: Test set (20%)

---

## 📊 Results (To be updated)

| Model | RMSE | MAE | Precision@10 | Time (s) |
|-------|------|-----|--------------|----------|
| Content-Based | - | - | - | - |
| SVD | - | - | - | - |
| NMF | - | - | - | - |
| KNN | - | - | - | - |
| Neural CF | - | - | - | - |
| Hybrid | - | - | - | - |

---

## 🎯 Future Improvements

- [ ] Real-time collaborative filtering with online learning
- [ ] A/B testing framework
- [ ] Multi-modal features (movie posters, reviews)
- [ ] Reinforcement learning for personalization
- [ ] Graph neural networks
- [ ] Fairness and bias mitigation

---

## 👨‍💻 Author

**Your Name**
- GitHub: [Kaustubh Mukdam](https://github.com/KaustubhMukdam)
- LinkedIn: [Kaustubh Mukdam](https://www.linkedin.com/in/kaustubh-mukdam-ab0170340/)
- Email: kaustubhmukdam7@gmail.com

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [MovieLens](https://grouplens.org/datasets/movielens/) for the dataset
- [Streamlit](https://streamlit.io/) for the web application

---

## 📚 References

1. Koren, Y., Bell, R., & Volinsky, C. (2009). *Matrix factorization techniques for recommender systems.* IEEE Computer.
2. He, X., et al. (2017). *Neural collaborative filtering.* WWW Conference.
3. Ricci, F., et al. (2011). *Introduction to recommender systems handbook.* Springer.

---

⭐ **If you find this project helpful, please give it a star!**
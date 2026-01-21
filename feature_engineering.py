# 03_feature_engineering.py (MEMORY-OPTIMIZED VERSION)
"""
Step 3: Feature Engineering Pipeline
Purpose: Extract and engineer features for recommendation models
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import csr_matrix, save_npz
import pickle
from pathlib import Path

class FeatureEngineer:
    def __init__(self):
        """Initialize feature engineering"""
        self.data_path = "data/raw/ml-25m/"
        self.output_path = "data/processed/"
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        
        print("📥 Loading datasets...")
        self.ratings = pd.read_csv(self.data_path + "ratings.csv")
        self.movies = pd.read_csv(self.data_path + "movies.csv")
        self.tags = pd.read_csv(self.data_path + "tags.csv")
        print("✅ Datasets loaded!\n")
    
    def create_genre_features(self):
        """Create one-hot encoded genre features"""
        print("🎭 Creating genre features...")
        
        # Split genres
        self.movies['genres_list'] = self.movies['genres'].str.split('|')
        
        # One-hot encoding
        mlb = MultiLabelBinarizer()
        genre_encoded = mlb.fit_transform(self.movies['genres_list'])
        genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_)
        
        # Combine with movies
        movies_with_genres = pd.concat([
            self.movies[['movieId', 'title', 'genres']], 
            genre_df
        ], axis=1)
        
        # Save
        movies_with_genres.to_csv(self.output_path + 'movies_with_genre_features.csv', index=False)
        
        # Save encoder
        with open(self.output_path + 'genre_encoder.pkl', 'wb') as f:
            pickle.dump(mlb, f)
        
        print(f"   ✓ Genre features shape: {genre_df.shape}")
        print(f"   ✓ Unique genres: {len(mlb.classes_)}")
        print(f"   ✓ Saved to: {self.output_path}movies_with_genre_features.csv\n")
        
        return movies_with_genres
    
    def create_text_features(self):
        """Create TF-IDF and BoW features from titles"""
        print("📝 Creating text features from titles...")
        
        # Extract year from title
        self.movies['year'] = self.movies['title'].str.extract(r'\((\d{4})\)')
        title_pattern = r'\(\d{4}\)'
        self.movies['title_clean'] = self.movies['title'].str.replace(title_pattern, '', regex=True).str.strip()
        
        # TF-IDF features
        tfidf = TfidfVectorizer(max_features=100, stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = tfidf.fit_transform(self.movies['title_clean'])
        
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
        )
        
        # BoW features
        bow = CountVectorizer(max_features=50, stop_words='english')
        bow_matrix = bow.fit_transform(self.movies['title_clean'])
        
        bow_df = pd.DataFrame(
            bow_matrix.toarray(),
            columns=[f'bow_{i}' for i in range(bow_matrix.shape[1])]
        )
        
        # Combine
        text_features = pd.concat([
            self.movies[['movieId', 'title', 'title_clean', 'year']],
            tfidf_df,
            bow_df
        ], axis=1)
        
        # Save
        text_features.to_csv(self.output_path + 'movies_with_text_features.csv', index=False)
        
        # Save vectorizers
        with open(self.output_path + 'tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(tfidf, f)
        with open(self.output_path + 'bow_vectorizer.pkl', 'wb') as f:
            pickle.dump(bow, f)
        
        print(f"   ✓ TF-IDF features: {tfidf_matrix.shape}")
        print(f"   ✓ BoW features: {bow_matrix.shape}")
        print(f"   ✓ Saved to: {self.output_path}movies_with_text_features.csv\n")
        
        return text_features
    
    def create_user_features(self):
        """Create user-level aggregated features"""
        print("👤 Creating user features...")
        
        # Aggregate user statistics
        user_stats = self.ratings.groupby('userId').agg({
            'rating': ['count', 'mean', 'std', 'min', 'max'],
            'movieId': 'nunique',
            'timestamp': ['min', 'max']
        }).reset_index()
        
        user_stats.columns = ['userId', 'rating_count', 'rating_mean', 'rating_std', 
                               'rating_min', 'rating_max', 'unique_movies', 
                               'first_rating_time', 'last_rating_time']
        
        # Fill NaN std with 0
        user_stats['rating_std'] = user_stats['rating_std'].fillna(0)
        
        # User tenure (days between first and last rating)
        user_stats['user_tenure_seconds'] = user_stats['last_rating_time'] - user_stats['first_rating_time']
        
        # Save
        user_stats.to_csv(self.output_path + 'user_features.csv', index=False)
        
        print(f"   ✓ User features shape: {user_stats.shape}")
        print(f"   ✓ Saved to: {self.output_path}user_features.csv\n")
        
        return user_stats
    
    def create_movie_features(self):
        """Create movie-level aggregated features"""
        print("🎬 Creating movie features...")
        
        # Aggregate movie statistics
        movie_stats = self.ratings.groupby('movieId').agg({
            'rating': ['count', 'mean', 'std', 'min', 'max'],
            'userId': 'nunique',
            'timestamp': ['min', 'max']
        }).reset_index()
        
        movie_stats.columns = ['movieId', 'rating_count', 'rating_mean', 'rating_std',
                                'rating_min', 'rating_max', 'unique_users',
                                'first_rating_time', 'last_rating_time']
        
        # Fill NaN std with 0
        movie_stats['rating_std'] = movie_stats['rating_std'].fillna(0)
        
        # Popularity score (log of rating count)
        movie_stats['popularity_score'] = np.log1p(movie_stats['rating_count'])
        
        # Save
        movie_stats.to_csv(self.output_path + 'movie_features.csv', index=False)
        
        print(f"   ✓ Movie features shape: {movie_stats.shape}")
        print(f"   ✓ Saved to: {self.output_path}movie_features.csv\n")
        
        return movie_stats
    
    def create_interaction_matrix(self, sample_size=1000000, min_ratings_per_user=20, min_ratings_per_movie=50):
        """Create user-movie interaction matrix (MEMORY-OPTIMIZED)"""
        print("🔗 Creating interaction matrix...")
        
        # Sample if specified (for faster processing)
        if sample_size and sample_size < len(self.ratings):
            # Random sample
            ratings_sample = self.ratings.sample(n=sample_size, random_state=42)
            print(f"   ℹ️  Using sample of {sample_size:,} ratings")
        else:
            ratings_sample = self.ratings
            print(f"   ℹ️  Using all {len(ratings_sample):,} ratings")
        
        # Filter users and movies with minimum interactions (reduce sparsity)
        user_counts = ratings_sample['userId'].value_counts()
        movie_counts = ratings_sample['movieId'].value_counts()
        
        active_users = user_counts[user_counts >= min_ratings_per_user].index
        popular_movies = movie_counts[movie_counts >= min_ratings_per_movie].index
        
        ratings_filtered = ratings_sample[
            (ratings_sample['userId'].isin(active_users)) & 
            (ratings_sample['movieId'].isin(popular_movies))
        ]
        
        print(f"   ℹ️  After filtering: {len(ratings_filtered):,} ratings")
        print(f"   ℹ️  Active users: {ratings_filtered['userId'].nunique():,}")
        print(f"   ℹ️  Popular movies: {ratings_filtered['movieId'].nunique():,}")
        
        # Create mappings for continuous indices
        user_ids = ratings_filtered['userId'].unique()
        movie_ids = ratings_filtered['movieId'].unique()
        
        user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
        movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
        
        # Map to indices
        ratings_filtered['user_idx'] = ratings_filtered['userId'].map(user_to_idx)
        ratings_filtered['movie_idx'] = ratings_filtered['movieId'].map(movie_to_idx)
        
        # Create sparse matrix directly (MEMORY EFFICIENT!)
        n_users = len(user_ids)
        n_movies = len(movie_ids)
        
        sparse_matrix = csr_matrix(
            (ratings_filtered['rating'].values, 
             (ratings_filtered['user_idx'].values, ratings_filtered['movie_idx'].values)),
            shape=(n_users, n_movies)
        )
        
        print(f"   ✓ Sparse matrix shape: {sparse_matrix.shape}")
        print(f"   ✓ Non-zero elements: {sparse_matrix.nnz:,}")
        print(f"   ✓ Sparsity: {100 * (1 - sparse_matrix.nnz / (n_users * n_movies)):.4f}%")
        print(f"   ✓ Memory usage: {sparse_matrix.data.nbytes / 1e6:.2f} MB")
        
        # Save sparse matrix and mappings
        save_npz(self.output_path + 'interaction_matrix_sparse.npz', sparse_matrix)
        
        # Save mappings
        mappings = {
            'user_to_idx': user_to_idx,
            'movie_to_idx': movie_to_idx,
            'idx_to_user': {idx: user_id for user_id, idx in user_to_idx.items()},
            'idx_to_movie': {idx: movie_id for movie_id, idx in movie_to_idx.items()},
            'n_users': n_users,
            'n_movies': n_movies
        }
        
        with open(self.output_path + 'interaction_matrix_mappings.pkl', 'wb') as f:
            pickle.dump(mappings, f)
        
        print(f"   ✓ Saved to: {self.output_path}interaction_matrix_sparse.npz\n")
        
        return sparse_matrix, mappings
    
    def create_train_test_split(self, test_size=0.2):
        """Create train/test split for model evaluation"""
        print("✂️  Creating train/test split...")
        
        # Temporal split (more realistic)
        self.ratings_sorted = self.ratings.sort_values('timestamp')
        split_idx = int(len(self.ratings_sorted) * (1 - test_size))
        
        train = self.ratings_sorted.iloc[:split_idx]
        test = self.ratings_sorted.iloc[split_idx:]
        
        # Save
        train.to_csv(self.output_path + 'train_ratings.csv', index=False)
        test.to_csv(self.output_path + 'test_ratings.csv', index=False)
        
        print(f"   ✓ Train size: {len(train):,} ({(1-test_size)*100:.0f}%)")
        print(f"   ✓ Test size: {len(test):,} ({test_size*100:.0f}%)")
        print(f"   ✓ Saved to: {self.output_path}\n")
        
        return train, test
    
    def run_complete_pipeline(self):
        """Execute complete feature engineering pipeline"""
        print("\n🏗️  FEATURE ENGINEERING PIPELINE")
        print("="*70 + "\n")
        
        # Create all features
        movies_genres = self.create_genre_features()
        movies_text = self.create_text_features()
        user_features = self.create_user_features()
        movie_features = self.create_movie_features()
        
        # Create interaction matrix (smaller sample, filtered for active users/movies)
        # Adjust parameters based on your RAM:
        # - For 8GB RAM: sample_size=500000
        # - For 16GB RAM: sample_size=1000000
        # - For 32GB+ RAM: sample_size=2000000 or more
        interaction_matrix, mappings = self.create_interaction_matrix(
            sample_size=1000000,  # 1M ratings
            min_ratings_per_user=20,  # Users with at least 20 ratings
            min_ratings_per_movie=50   # Movies with at least 50 ratings
        )
        
        # Train/test split
        train, test = self.create_train_test_split(test_size=0.2)
        
        print("="*70)
        print("✅ FEATURE ENGINEERING COMPLETED!")
        print("="*70)
        print(f"\n📁 All features saved in: {self.output_path}")
        
        return {
            'movies_genres': movies_genres,
            'movies_text': movies_text,
            'user_features': user_features,
            'movie_features': movie_features,
            'interaction_matrix': interaction_matrix,
            'mappings': mappings,
            'train': train,
            'test': test
        }

# Main execution
if __name__ == "__main__":
    engineer = FeatureEngineer()
    features = engineer.run_complete_pipeline()

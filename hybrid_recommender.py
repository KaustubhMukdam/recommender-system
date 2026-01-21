# 06_hybrid_recommender.py
"""
Week 5: Hybrid Recommender System
Purpose: Combine content-based and collaborative filtering approaches
"""

import pandas as pd
import numpy as np
from scipy.sparse import load_npz
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class HybridRecommender:
    def __init__(self):
        """Initialize Hybrid Recommender"""
        self.data_path = "data/processed/"
        self.models_path = "models/saved_models/"
        Path(self.models_path).mkdir(parents=True, exist_ok=True)
        
        print("📥 Loading data and models...")
        self.load_all_models()
        print("✅ All models loaded successfully!\n")
    
    def load_all_models(self):
        """Load all trained models"""
        # Load data
        self.train_ratings = pd.read_csv(self.data_path + "train_ratings.csv")
        self.test_ratings = pd.read_csv(self.data_path + "test_ratings.csv")
        self.movies = pd.read_csv(self.data_path + "movies_with_genre_features.csv")
        
        # Load interaction matrix
        self.interaction_matrix = load_npz(self.data_path + "interaction_matrix_sparse.npz")
        
        with open(self.data_path + "interaction_matrix_mappings.pkl", 'rb') as f:
            self.mappings = pickle.load(f)
        
        self.user_to_idx = self.mappings['user_to_idx']
        self.movie_to_idx = self.mappings['movie_to_idx']
        self.idx_to_user = self.mappings['idx_to_user']
        self.idx_to_movie = self.mappings['idx_to_movie']
        
        # Load content-based features
        self.genre_features = np.load(self.models_path + 'genre_features.npy')
        self.tfidf_features = np.load(self.models_path + 'tfidf_features.npy')
        
        with open(self.models_path + 'movie_id_to_idx.pkl', 'rb') as f:
            self.content_movie_to_idx = pickle.load(f)
        
        # Load collaborative filtering models
        self.svd_user_factors = np.load(self.models_path + 'svd_user_factors.npy')
        self.svd_sigma = np.load(self.models_path + 'svd_sigma.npy')
        self.svd_movie_factors = np.load(self.models_path + 'svd_movie_factors.npy')
        self.svd_predictions = np.dot(np.dot(self.svd_user_factors, self.svd_sigma), 
                                      self.svd_movie_factors.T)
        
        self.nmf_user_factors = np.load(self.models_path + 'nmf_user_factors.npy')
        self.nmf_movie_factors = np.load(self.models_path + 'nmf_movie_factors.npy')
        self.nmf_predictions = np.dot(self.nmf_user_factors, self.nmf_movie_factors.T)
        
        with open(self.models_path + 'knn_model.pkl', 'rb') as f:
            self.knn_model = pickle.load(f)
        
        print("   ✓ Loaded: Ratings, movies, interaction matrix")
        print("   ✓ Loaded: Content-based features (genre, TF-IDF)")
        print("   ✓ Loaded: SVD, NMF, KNN models")
    
    def predict_content_based(self, user_id, movie_id, use_genre=True):
        """Get content-based score"""
        if user_id not in self.user_to_idx or movie_id not in self.content_movie_to_idx:
            return 0.0
        
        # Build user profile
        user_ratings = self.train_ratings[self.train_ratings['userId'] == user_id]
        if len(user_ratings) == 0:
            return 0.0
        
        rated_movie_ids = [mid for mid in user_ratings['movieId'].values 
                          if mid in self.content_movie_to_idx]
        
        if len(rated_movie_ids) == 0:
            return 0.0
        
        # Get features
        rated_indices = [self.content_movie_to_idx[mid] for mid in rated_movie_ids]
        movie_idx = self.content_movie_to_idx[movie_id]
        
        features = self.genre_features if use_genre else self.tfidf_features
        
        # Weighted average of rated movies
        rated_features = features[rated_indices]
        ratings = user_ratings[user_ratings['movieId'].isin(rated_movie_ids)]['rating'].values
        weights = ratings / ratings.sum()
        user_profile = np.average(rated_features, axis=0, weights=weights)
        
        # Cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        movie_vector = features[movie_idx].reshape(1, -1)
        user_vector = user_profile.reshape(1, -1)
        
        similarity = cosine_similarity(user_vector, movie_vector)[0][0]
        
        # Scale to rating range (0-5)
        return similarity * 5.0
    
    def predict_svd(self, user_id, movie_id):
        """Get SVD prediction"""
        if user_id not in self.user_to_idx or movie_id not in self.movie_to_idx:
            return 0.0
        
        user_idx = self.user_to_idx[user_id]
        movie_idx = self.movie_to_idx[movie_id]
        
        return self.svd_predictions[user_idx, movie_idx]
    
    def predict_nmf(self, user_id, movie_id):
        """Get NMF prediction"""
        if user_id not in self.user_to_idx or movie_id not in self.movie_to_idx:
            return 0.0
        
        user_idx = self.user_to_idx[user_id]
        movie_idx = self.movie_to_idx[movie_id]
        
        return self.nmf_predictions[user_idx, movie_idx]
    
    def predict_knn(self, user_id, movie_id):
        """Get KNN prediction"""
        if user_id not in self.user_to_idx or movie_id not in self.movie_to_idx:
            return 0.0
        
        user_idx = self.user_to_idx[user_id]
        movie_idx = self.movie_to_idx[movie_id]
        
        # Get similar users
        user_vector = self.interaction_matrix[user_idx, :]
        distances, indices = self.knn_model.kneighbors(user_vector, n_neighbors=21)
        
        # Exclude self
        similar_user_indices = indices[0][1:]
        
        # Get ratings from similar users for this movie
        similar_users_ratings = self.interaction_matrix[similar_user_indices, movie_idx].toarray().flatten()
        
        # Average of non-zero ratings
        non_zero_ratings = similar_users_ratings[similar_users_ratings > 0]
        if len(non_zero_ratings) == 0:
            return 0.0
        
        return np.mean(non_zero_ratings)
    
    def predict_hybrid(self, user_id, movie_id, weights=None):
        """
        Hybrid prediction combining all models
        
        weights: dict with keys ['content_genre', 'content_tfidf', 'svd', 'nmf', 'knn']
        """
        if weights is None:
            # Default equal weights
            weights = {
                'content_genre': 0.2,
                'content_tfidf': 0.1,
                'svd': 0.3,
                'nmf': 0.2,
                'knn': 0.2
            }
        
        predictions = {}
        predictions['content_genre'] = self.predict_content_based(user_id, movie_id, use_genre=True)
        predictions['content_tfidf'] = self.predict_content_based(user_id, movie_id, use_genre=False)
        predictions['svd'] = self.predict_svd(user_id, movie_id)
        predictions['nmf'] = self.predict_nmf(user_id, movie_id)
        predictions['knn'] = self.predict_knn(user_id, movie_id)
        
        # Weighted average (only non-zero predictions)
        total_weight = 0
        weighted_sum = 0
        
        for model, pred in predictions.items():
            if pred > 0:
                weighted_sum += weights[model] * pred
                total_weight += weights[model]
        
        if total_weight == 0:
            return 3.0  # Default rating
        
        return weighted_sum / total_weight
    
    def recommend_hybrid(self, user_id, top_n=10, weights=None):
        """Recommend movies using hybrid approach"""
        # Get all movies user hasn't rated
        user_rated_movies = self.train_ratings[self.train_ratings['userId'] == user_id]['movieId'].values
        
        # Get candidate movies (those in interaction matrix)
        candidate_movies = list(self.movie_to_idx.keys())
        unrated_movies = [mid for mid in candidate_movies if mid not in user_rated_movies]
        
        # Limit candidates for speed (sample if too many)
        if len(unrated_movies) > 500:
            unrated_movies = np.random.choice(unrated_movies, 500, replace=False)
        
        # Predict ratings for all unrated movies
        predictions = []
        for movie_id in unrated_movies:
            pred = self.predict_hybrid(user_id, movie_id, weights=weights)
            predictions.append((movie_id, pred))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Get top N
        top_predictions = predictions[:top_n]
        
        # Create recommendations dataframe
        recommendations = pd.DataFrame(top_predictions, columns=['movieId', 'predicted_rating'])
        
        # Merge with movie details
        recommendations = recommendations.merge(
            self.movies[['movieId', 'title', 'genres']], 
            on='movieId', 
            how='left'
        )
        
        return recommendations
    
    def evaluate_hybrid(self, n_users=100, weights=None):
        """Evaluate hybrid system on test set"""
        print(f"\n📊 Evaluating Hybrid System on {n_users} users...")
        
        # Sample test users
        test_user_ids = self.test_ratings['userId'].unique()
        valid_test_users = [uid for uid in test_user_ids if uid in self.user_to_idx]
        
        if len(valid_test_users) > n_users:
            sampled_users = np.random.choice(valid_test_users, n_users, replace=False)
        else:
            sampled_users = valid_test_users[:n_users]
        
        all_predictions = []
        all_actuals = []
        
        print("   Processing:", end=" ")
        for i, user_id in enumerate(sampled_users):
            if (i + 1) % 20 == 0:
                print(f"{i+1}...", end=" ", flush=True)
            
            user_test = self.test_ratings[self.test_ratings['userId'] == user_id]
            
            for _, row in user_test.iterrows():
                movie_id = row['movieId']
                actual_rating = row['rating']
                
                if movie_id in self.movie_to_idx:
                    pred = self.predict_hybrid(user_id, movie_id, weights=weights)
                    all_predictions.append(pred)
                    all_actuals.append(actual_rating)
        
        print("Done!")
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        rmse = np.sqrt(mean_squared_error(all_actuals, all_predictions))
        mae = mean_absolute_error(all_actuals, all_predictions)
        
        print(f"\n   ✓ Test samples: {len(all_predictions)}")
        print(f"   ✓ RMSE: {rmse:.4f}")
        print(f"   ✓ MAE: {mae:.4f}\n")
        
        return {'rmse': rmse, 'mae': mae, 'n_samples': len(all_predictions)}
    
    def run_complete_pipeline(self):
        """Run complete hybrid recommendation pipeline"""
        print("\n🎯 HYBRID RECOMMENDER SYSTEM")
        print("="*70)
        
        # Test with sample user
        sample_user = list(self.user_to_idx.keys())[0]
        
        print(f"\n👤 Example: Hybrid Recommendations for User {sample_user}")
        hybrid_recs = self.recommend_hybrid(sample_user, top_n=10)
        print(hybrid_recs[['title', 'genres', 'predicted_rating']].to_string(index=False))
        
        # Evaluate
        results = self.evaluate_hybrid(n_users=100)
        
        # Save results
        results_df = pd.DataFrame([results])
        results_df.to_csv('reports/hybrid_evaluation.csv', index=False)
        
        print("\n" + "="*70)
        print("✅ HYBRID SYSTEM COMPLETED!")
        print("="*70)
        print("\n📁 Results saved to reports/hybrid_evaluation.csv\n")
        
        return results

# Main execution
if __name__ == "__main__":
    hybrid = HybridRecommender()
    results = hybrid.run_complete_pipeline()
    
    print(f"\n📊 Final Hybrid Performance:")
    print(f"   RMSE: {results['rmse']:.4f}")
    print(f"   MAE: {results['mae']:.4f}")

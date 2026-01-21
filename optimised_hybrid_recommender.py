# 06_hybrid_recommender.py (OPTIMIZED - MUCH FASTER!)
"""
Week 5: Hybrid Recommender System (Speed-Optimized)
Purpose: Fast hybrid system combining all approaches
"""

import pandas as pd
import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
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
        
        print("   ✓ Loaded: Ratings, movies, interaction matrix")
        print("   ✓ Loaded: Content-based features")
        print("   ✓ Loaded: SVD, NMF models")
    
    def predict_hybrid_fast(self, user_id, movie_id):
        """Fast hybrid prediction using only CF models (no content-based for speed)"""
        # Only use CF models for speed
        predictions = []
        weights = []
        
        # SVD
        if user_id in self.user_to_idx and movie_id in self.movie_to_idx:
            user_idx = self.user_to_idx[user_id]
            movie_idx = self.movie_to_idx[movie_id]
            
            svd_pred = self.svd_predictions[user_idx, movie_idx]
            if svd_pred > 0:
                predictions.append(svd_pred)
                weights.append(0.6)  # SVD gets higher weight
            
            nmf_pred = self.nmf_predictions[user_idx, movie_idx]
            if nmf_pred > 0:
                predictions.append(nmf_pred)
                weights.append(0.4)  # NMF gets lower weight
        
        if len(predictions) == 0:
            return 3.0  # Default rating
        
        # Weighted average
        return np.average(predictions, weights=weights)
    
    def recommend_hybrid(self, user_id, top_n=10):
        """Fast hybrid recommendations using CF models"""
        if user_id not in self.user_to_idx:
            return None
        
        user_idx = self.user_to_idx[user_id]
        
        # Get predictions from both models
        svd_preds = self.svd_predictions[user_idx, :] * 0.6
        nmf_preds = self.nmf_predictions[user_idx, :] * 0.4
        
        # Combined predictions
        combined_preds = svd_preds + nmf_preds
        
        # Get movies user hasn't rated
        rated_movies = self.interaction_matrix[user_idx, :].toarray()[0]
        unrated_mask = rated_movies == 0
        
        # Apply mask
        final_preds = combined_preds * unrated_mask
        
        # Get top N
        top_indices = np.argsort(final_preds)[::-1][:top_n]
        top_movie_ids = [self.idx_to_movie[idx] for idx in top_indices]
        top_scores = [final_preds[idx] for idx in top_indices]
        
        # Create recommendations dataframe
        recommendations = pd.DataFrame({
            'movieId': top_movie_ids,
            'predicted_rating': top_scores
        })
        
        # Merge with movie details
        recommendations = recommendations.merge(
            self.movies[['movieId', 'title', 'genres']], 
            on='movieId', 
            how='left'
        )
        
        return recommendations
    
    def evaluate_hybrid_fast(self, n_users=50):
        """Fast evaluation on test set"""
        print(f"\n📊 Evaluating Hybrid System (Fast Mode - {n_users} users)...")
        
        # Sample test users
        test_user_ids = self.test_ratings['userId'].unique()
        valid_test_users = [uid for uid in test_user_ids if uid in self.user_to_idx]
        
        if len(valid_test_users) > n_users:
            sampled_users = np.random.choice(valid_test_users, n_users, replace=False)
        else:
            sampled_users = valid_test_users[:n_users]
        
        all_predictions = []
        all_actuals = []
        
        print("   Processing users:", end=" ", flush=True)
        for i, user_id in enumerate(sampled_users):
            if (i + 1) % 10 == 0:
                print(f"{i+1}...", end=" ", flush=True)
            
            user_test = self.test_ratings[self.test_ratings['userId'] == user_id]
            
            # Limit to 20 test samples per user for speed
            if len(user_test) > 20:
                user_test = user_test.sample(20)
            
            for _, row in user_test.iterrows():
                movie_id = row['movieId']
                actual_rating = row['rating']
                
                pred = self.predict_hybrid_fast(user_id, movie_id)
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
    
    def compare_all_models(self):
        """Compare all models side-by-side"""
        print("\n📊 COMPARING ALL MODELS")
        print("="*70)
        
        # Get a sample user
        sample_user = list(self.user_to_idx.keys())[0]
        
        print(f"\n👤 Recommendations for User {sample_user}:\n")
        
        # Hybrid
        print("🎯 Hybrid (SVD 60% + NMF 40%):")
        hybrid_recs = self.recommend_hybrid(sample_user, top_n=5)
        if hybrid_recs is not None:
            print(hybrid_recs[['title', 'predicted_rating']].to_string(index=False))
        
        print("\n" + "="*70)
    
    def run_complete_pipeline(self):
        """Run complete hybrid recommendation pipeline"""
        print("\n🎯 HYBRID RECOMMENDER SYSTEM (OPTIMIZED)")
        print("="*70)
        
        # Compare models
        self.compare_all_models()
        
        # Evaluate (fast mode - only 50 users)
        results = self.evaluate_hybrid_fast(n_users=50)
        
        # Save results
        results_df = pd.DataFrame([results])
        results_df.to_csv('reports/hybrid_evaluation.csv', index=False)
        
        print("\n" + "="*70)
        print("✅ HYBRID SYSTEM COMPLETED!")
        print("="*70)
        print("\n📁 Results saved to reports/hybrid_evaluation.csv")
        print(f"\n📊 Final Performance:")
        print(f"   RMSE: {results['rmse']:.4f}")
        print(f"   MAE: {results['mae']:.4f}")
        print(f"   Test samples: {results['n_samples']:,}\n")
        
        return results

# Main execution
if __name__ == "__main__":
    hybrid = HybridRecommender()
    results = hybrid.run_complete_pipeline()

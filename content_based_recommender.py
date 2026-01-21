# 04_content_based_recommender.py (OPTIMIZED VERSION)
"""
Week 3: Content-Based Recommendation System (Memory-Optimized)
Purpose: Build recommender using movie features (genres, TF-IDF, BoW)
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix, save_npz, load_npz
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ContentBasedRecommender:
    def __init__(self):
        """Initialize Content-Based Recommender"""
        self.data_path = "data/processed/"
        self.models_path = "models/saved_models/"
        Path(self.models_path).mkdir(parents=True, exist_ok=True)
        
        print("📥 Loading processed data...")
        self.load_data()
        print("✅ Data loaded successfully!\n")
    
    def load_data(self):
        """Load all processed features"""
        # Load movie features
        self.movies_genres = pd.read_csv(self.data_path + "movies_with_genre_features.csv")
        self.movies_text = pd.read_csv(self.data_path + "movies_with_text_features.csv")
        
        # Load ratings
        self.train_ratings = pd.read_csv(self.data_path + "train_ratings.csv")
        self.test_ratings = pd.read_csv(self.data_path + "test_ratings.csv")
        
        # Load movie stats
        self.movie_features = pd.read_csv(self.data_path + "movie_features.csv")
        
        # Get active movies (those with ratings)
        active_movie_ids = set(self.train_ratings['movieId'].unique()) | set(self.test_ratings['movieId'].unique())
        
        # Filter to only active movies to save memory
        self.movies_genres = self.movies_genres[self.movies_genres['movieId'].isin(active_movie_ids)].reset_index(drop=True)
        self.movies_text = self.movies_text[self.movies_text['movieId'].isin(active_movie_ids)].reset_index(drop=True)
        
        print(f"   ✓ Active movies (with ratings): {len(self.movies_genres)}")
        print(f"   ✓ Training ratings: {self.train_ratings.shape}")
        print(f"   ✓ Test ratings: {self.test_ratings.shape}")
    
    def prepare_genre_features(self):
        """Prepare and save genre feature matrix"""
        print("\n🎭 Preparing genre features...")
        
        # Extract genre columns
        genre_cols = [col for col in self.movies_genres.columns 
                     if col not in ['movieId', 'title', 'genres', 'genres_list']]
        
        # Get genre features
        self.genre_features = self.movies_genres[genre_cols].values
        
        # Create movie ID to index mapping
        self.movie_id_to_idx = {
            movie_id: idx for idx, movie_id in enumerate(self.movies_genres['movieId'])
        }
        
        print(f"   ✓ Genre feature matrix: {self.genre_features.shape}")
        print(f"   ✓ Features per movie: {len(genre_cols)}")
        
        # Save features (much smaller than similarity matrix)
        np.save(self.models_path + 'genre_features.npy', self.genre_features)
        with open(self.models_path + 'movie_id_to_idx.pkl', 'wb') as f:
            pickle.dump(self.movie_id_to_idx, f)
        
        print(f"   ✓ Saved genre features\n")
        
        return self.genre_features
    
    def prepare_tfidf_features(self):
        """Prepare and save TF-IDF feature matrix"""
        print("📝 Preparing TF-IDF features...")
        
        # Extract TF-IDF columns
        tfidf_cols = [col for col in self.movies_text.columns if col.startswith('tfidf_')]
        
        # Get TF-IDF features
        self.tfidf_features = self.movies_text[tfidf_cols].values
        
        # Normalize features
        self.tfidf_features = normalize(self.tfidf_features)
        
        print(f"   ✓ TF-IDF feature matrix: {self.tfidf_features.shape}")
        
        # Save features
        np.save(self.models_path + 'tfidf_features.npy', self.tfidf_features)
        
        print(f"   ✓ Saved TF-IDF features\n")
        
        return self.tfidf_features
    
    def compute_similarity_for_movie(self, movie_id, use_genre=True, top_n=20):
        """Compute similarity for a specific movie (on-demand)"""
        if movie_id not in self.movie_id_to_idx:
            return None
        
        movie_idx = self.movie_id_to_idx[movie_id]
        
        # Get features
        if use_genre:
            if not hasattr(self, 'genre_features'):
                self.prepare_genre_features()
            movie_vector = self.genre_features[movie_idx].reshape(1, -1)
            all_features = self.genre_features
        else:
            if not hasattr(self, 'tfidf_features'):
                self.prepare_tfidf_features()
            movie_vector = self.tfidf_features[movie_idx].reshape(1, -1)
            all_features = self.tfidf_features
        
        # Compute similarity with all movies
        similarities = cosine_similarity(movie_vector, all_features)[0]
        
        # Get top N (excluding self)
        top_indices = np.argsort(similarities)[::-1][1:top_n+1]
        top_scores = similarities[top_indices]
        
        return top_indices, top_scores
    
    def get_similar_movies(self, movie_id, use_genre=True, top_n=10):
        """Get top N similar movies for a given movie"""
        result = self.compute_similarity_for_movie(movie_id, use_genre=use_genre, top_n=top_n)
        
        if result is None:
            return None
        
        top_indices, top_scores = result
        
        # Get movie details
        similar_movies = self.movies_genres.iloc[top_indices][['movieId', 'title', 'genres']].copy()
        similar_movies['similarity_score'] = top_scores
        
        return similar_movies
    
    def build_user_profile(self, user_id, use_genre=True):
        """Build user profile based on rated movies"""
        # Get user's ratings
        user_ratings = self.train_ratings[self.train_ratings['userId'] == user_id]
        
        if len(user_ratings) == 0:
            return None
        
        # Get rated movie IDs that exist in our filtered dataset
        rated_movie_ids = user_ratings['movieId'].values
        rated_movie_ids = [mid for mid in rated_movie_ids if mid in self.movie_id_to_idx]
        
        if len(rated_movie_ids) == 0:
            return None
        
        # Get indices
        rated_indices = [self.movie_id_to_idx[mid] for mid in rated_movie_ids]
        
        # Get features
        if use_genre:
            if not hasattr(self, 'genre_features'):
                self.prepare_genre_features()
            features = self.genre_features[rated_indices]
        else:
            if not hasattr(self, 'tfidf_features'):
                self.prepare_tfidf_features()
            features = self.tfidf_features[rated_indices]
        
        # Get ratings for weighted average
        ratings_df = user_ratings[user_ratings['movieId'].isin(rated_movie_ids)]
        ratings_df = ratings_df.set_index('movieId').loc[rated_movie_ids].reset_index()
        ratings = ratings_df['rating'].values
        
        # Normalize ratings to weights
        weights = ratings / ratings.sum()
        
        # Compute weighted profile
        user_profile = np.average(features, axis=0, weights=weights)
        
        return user_profile
    
    def recommend_for_user(self, user_id, top_n=10, use_genre=True, similarity_threshold=0.0):
        """Recommend movies for a user based on their profile"""
        # Build user profile
        user_profile = self.build_user_profile(user_id, use_genre=use_genre)
        
        if user_profile is None:
            return None
        
        # Get all movie features
        if use_genre:
            if not hasattr(self, 'genre_features'):
                self.prepare_genre_features()
            all_features = self.genre_features
        else:
            if not hasattr(self, 'tfidf_features'):
                self.prepare_tfidf_features()
            all_features = self.tfidf_features
        
        # Compute similarity between user profile and all movies
        user_profile_reshaped = user_profile.reshape(1, -1)
        similarities = cosine_similarity(user_profile_reshaped, all_features)[0]
        
        # Create recommendations dataframe
        recommendations = self.movies_genres[['movieId', 'title', 'genres']].copy()
        recommendations['similarity_score'] = similarities
        
        # Get movies user hasn't rated
        user_rated_movies = self.train_ratings[self.train_ratings['userId'] == user_id]['movieId'].values
        recommendations = recommendations[~recommendations['movieId'].isin(user_rated_movies)]
        
        # Filter by similarity threshold
        recommendations = recommendations[recommendations['similarity_score'] >= similarity_threshold]
        
        # Sort by similarity
        recommendations = recommendations.sort_values('similarity_score', ascending=False)
        
        # Return top N
        return recommendations[['movieId', 'title', 'similarity_score']].head(top_n)
    
    def evaluate_recommendations(self, n_users=100, top_n=10, use_genre=True):
        """Evaluate recommendation quality on test set"""
        print(f"\n📊 Evaluating {'Genre' if use_genre else 'TF-IDF'}-based recommendations...")
        print(f"   Testing on {n_users} users, Top-{top_n} recommendations\n")
        
        # Sample users who have ratings in both train and test
        test_users = self.test_ratings['userId'].unique()
        train_users = self.train_ratings['userId'].unique()
        common_users = list(set(test_users) & set(train_users))
        
        # Sample n_users
        if len(common_users) > n_users:
            sampled_users = np.random.choice(common_users, n_users, replace=False)
        else:
            sampled_users = common_users[:n_users]
        
        precision_scores = []
        recall_scores = []
        hit_rates = []
        
        print("   Processing users:", end=" ")
        for i, user_id in enumerate(sampled_users):
            if (i + 1) % 20 == 0:
                print(f"{i+1}...", end=" ", flush=True)
            
            # Get recommendations
            recs = self.recommend_for_user(user_id, top_n=top_n, use_genre=use_genre)
            
            if recs is None or len(recs) == 0:
                continue
            
            recommended_movies = set(recs['movieId'].values)
            
            # Get actual movies user liked in test set (rating >= 4.0)
            actual_liked = set(
                self.test_ratings[
                    (self.test_ratings['userId'] == user_id) & 
                    (self.test_ratings['rating'] >= 4.0)
                ]['movieId'].values
            )
            
            if len(actual_liked) == 0:
                continue
            
            # Calculate metrics
            hits = len(recommended_movies & actual_liked)
            
            precision = hits / len(recommended_movies) if len(recommended_movies) > 0 else 0
            recall = hits / len(actual_liked) if len(actual_liked) > 0 else 0
            hit_rate = 1 if hits > 0 else 0
            
            precision_scores.append(precision)
            recall_scores.append(recall)
            hit_rates.append(hit_rate)
        
        print("Done!")
        
        # Calculate average metrics
        avg_precision = np.mean(precision_scores) if precision_scores else 0
        avg_recall = np.mean(recall_scores) if recall_scores else 0
        avg_hit_rate = np.mean(hit_rates) if hit_rates else 0
        
        print(f"\n   ✓ Evaluated {len(precision_scores)} users")
        print(f"   ✓ Precision@{top_n}: {avg_precision:.4f}")
        print(f"   ✓ Recall@{top_n}: {avg_recall:.4f}")
        print(f"   ✓ Hit Rate: {avg_hit_rate:.4f}\n")
        
        results = {
            'method': 'Genre-based' if use_genre else 'TF-IDF-based',
            'n_users': len(precision_scores),
            'top_n': top_n,
            'precision': avg_precision,
            'recall': avg_recall,
            'hit_rate': avg_hit_rate
        }
        
        return results
    
    def run_complete_pipeline(self):
        """Run complete content-based recommendation pipeline"""
        print("\n🎬 CONTENT-BASED RECOMMENDATION SYSTEM")
        print("="*70)
        
        # Prepare feature matrices (instead of full similarity matrices)
        self.prepare_genre_features()
        self.prepare_tfidf_features()
        
        # Example: Find similar movies
        print("\n🔍 Example: Movies similar to 'Toy Story (1995)' (movieId=1)")
        
        if 1 in self.movie_id_to_idx:
            toy_story_similar_genre = self.get_similar_movies(1, use_genre=True, top_n=5)
            if toy_story_similar_genre is not None:
                print("\nGenre-based similar movies:")
                print(toy_story_similar_genre.to_string(index=False))
            
            toy_story_similar_tfidf = self.get_similar_movies(1, use_genre=False, top_n=5)
            if toy_story_similar_tfidf is not None:
                print("\nTF-IDF-based similar movies:")
                print(toy_story_similar_tfidf.to_string(index=False))
        else:
            print("   (Toy Story not in active movies)")
        
        # Example: User recommendations
        print("\n\n👤 Example: Recommendations for User ID 1")
        user_recs_genre = self.recommend_for_user(1, top_n=10, use_genre=True)
        if user_recs_genre is not None:
            print("\nGenre-based recommendations:")
            print(user_recs_genre.to_string(index=False))
        else:
            print("   (No recommendations for this user)")
        
        user_recs_tfidf = self.recommend_for_user(1, top_n=10, use_genre=False)
        if user_recs_tfidf is not None:
            print("\nTF-IDF-based recommendations:")
            print(user_recs_tfidf.to_string(index=False))
        
        # Evaluate both approaches
        results_genre = self.evaluate_recommendations(n_users=100, top_n=10, use_genre=True)
        results_tfidf = self.evaluate_recommendations(n_users=100, top_n=10, use_genre=False)
        
        # Save evaluation results
        results_df = pd.DataFrame([results_genre, results_tfidf])
        results_df.to_csv('reports/content_based_evaluation.csv', index=False)
        
        print("\n" + "="*70)
        print("✅ CONTENT-BASED SYSTEM COMPLETED!")
        print("="*70)
        print("\n📁 Saved:")
        print("   - Genre feature matrix (not full similarity)")
        print("   - TF-IDF feature matrix (not full similarity)")
        print("   - Movie ID mappings")
        print("   - Evaluation results\n")
        
        return results_df

# Main execution
if __name__ == "__main__":
    recommender = ContentBasedRecommender()
    results = recommender.run_complete_pipeline()
    print("\n📊 Final Results:")
    print(results.to_string(index=False))

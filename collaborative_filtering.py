# 05_collaborative_filtering.py
"""
Week 4: Collaborative Filtering Recommendation Systems
Purpose: Implement SVD, NMF, KNN, and Neural Collaborative Filtering
"""

import pandas as pd
import numpy as np
from scipy.sparse import load_npz, csr_matrix
from sklearn.decomposition import NMF
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# For SVD
from scipy.sparse.linalg import svds

# For Neural CF (will use TensorFlow)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️  TensorFlow not available. Neural CF will be skipped.")

class CollaborativeFilteringRecommender:
    def __init__(self):
        """Initialize Collaborative Filtering Recommender"""
        self.data_path = "data/processed/"
        self.models_path = "models/saved_models/"
        Path(self.models_path).mkdir(parents=True, exist_ok=True)
        
        print("📥 Loading processed data...")
        self.load_data()
        print("✅ Data loaded successfully!\n")
    
    def load_data(self):
        """Load ratings and interaction matrix"""
        # Load ratings
        self.train_ratings = pd.read_csv(self.data_path + "train_ratings.csv")
        self.test_ratings = pd.read_csv(self.data_path + "test_ratings.csv")
        
        # Load sparse interaction matrix
        self.interaction_matrix = load_npz(self.data_path + "interaction_matrix_sparse.npz")
        
        # Load mappings
        with open(self.data_path + "interaction_matrix_mappings.pkl", 'rb') as f:
            self.mappings = pickle.load(f)
        
        self.user_to_idx = self.mappings['user_to_idx']
        self.movie_to_idx = self.mappings['movie_to_idx']
        self.idx_to_user = self.mappings['idx_to_user']
        self.idx_to_movie = self.mappings['idx_to_movie']
        
        print(f"   ✓ Training ratings: {self.train_ratings.shape}")
        print(f"   ✓ Test ratings: {self.test_ratings.shape}")
        print(f"   ✓ Interaction matrix: {self.interaction_matrix.shape}")
        print(f"   ✓ Sparsity: {100 * (1 - self.interaction_matrix.nnz / (self.interaction_matrix.shape[0] * self.interaction_matrix.shape[1])):.2f}%")
    
    # ==========================================
    # MODEL 1: SVD (Singular Value Decomposition)
    # ==========================================
    
    def train_svd(self, n_factors=50):
        """Train SVD model using scipy's svds"""
        print(f"\n🔢 Training SVD Model (k={n_factors} factors)...")
        start_time = time.time()
        
        # Perform SVD
        # U: user factors, sigma: singular values, Vt: movie factors
        U, sigma, Vt = svds(self.interaction_matrix.astype(float), k=n_factors)
        
        # Convert sigma to diagonal matrix
        sigma = np.diag(sigma)
        
        # Store components
        self.svd_user_factors = U
        self.svd_sigma = sigma
        self.svd_movie_factors = Vt.T
        
        # Reconstruct predictions
        self.svd_predictions = np.dot(np.dot(U, sigma), Vt)
        
        train_time = time.time() - start_time
        
        print(f"   ✓ Training completed in {train_time:.2f}s")
        print(f"   ✓ User factors shape: {U.shape}")
        print(f"   ✓ Movie factors shape: {Vt.T.shape}")
        
        # Save model
        np.save(self.models_path + 'svd_user_factors.npy', U)
        np.save(self.models_path + 'svd_sigma.npy', sigma)
        np.save(self.models_path + 'svd_movie_factors.npy', Vt.T)
        
        return self.svd_predictions
    
    def predict_svd(self, user_id, movie_id):
        """Predict rating using SVD"""
        if user_id not in self.user_to_idx or movie_id not in self.movie_to_idx:
            return None
        
        user_idx = self.user_to_idx[user_id]
        movie_idx = self.movie_to_idx[movie_id]
        
        return self.svd_predictions[user_idx, movie_idx]
    
    def recommend_svd(self, user_id, top_n=10):
        """Recommend movies using SVD"""
        if user_id not in self.user_to_idx:
            return None
        
        user_idx = self.user_to_idx[user_id]
        
        # Get user's predictions
        user_predictions = self.svd_predictions[user_idx, :]
        
        # Get movies user hasn't rated
        rated_movies = self.interaction_matrix[user_idx, :].toarray()[0]
        unrated_mask = rated_movies == 0
        
        # Get predictions for unrated movies
        unrated_predictions = user_predictions * unrated_mask
        
        # Get top N
        top_indices = np.argsort(unrated_predictions)[::-1][:top_n]
        top_movie_ids = [self.idx_to_movie[idx] for idx in top_indices]
        top_scores = [unrated_predictions[idx] for idx in top_indices]
        
        recommendations = pd.DataFrame({
            'movieId': top_movie_ids,
            'predicted_rating': top_scores
        })
        
        return recommendations
    
    # ==========================================
    # MODEL 2: NMF (Non-negative Matrix Factorization)
    # ==========================================
    
    def train_nmf(self, n_factors=50, max_iter=200):
        """Train NMF model"""
        print(f"\n🔢 Training NMF Model (k={n_factors} factors)...")
        start_time = time.time()
        
        # NMF requires non-negative values
        nmf_model = NMF(n_components=n_factors, init='random', random_state=42, max_iter=max_iter)
        
        # Fit model
        self.nmf_user_factors = nmf_model.fit_transform(self.interaction_matrix)
        self.nmf_movie_factors = nmf_model.components_.T
        
        # Reconstruct predictions
        self.nmf_predictions = np.dot(self.nmf_user_factors, self.nmf_movie_factors.T)
        
        train_time = time.time() - start_time
        
        print(f"   ✓ Training completed in {train_time:.2f}s")
        print(f"   ✓ User factors shape: {self.nmf_user_factors.shape}")
        print(f"   ✓ Movie factors shape: {self.nmf_movie_factors.shape}")
        print(f"   ✓ Reconstruction error: {nmf_model.reconstruction_err_:.4f}")
        
        # Save model
        with open(self.models_path + 'nmf_model.pkl', 'wb') as f:
            pickle.dump(nmf_model, f)
        np.save(self.models_path + 'nmf_user_factors.npy', self.nmf_user_factors)
        np.save(self.models_path + 'nmf_movie_factors.npy', self.nmf_movie_factors)
        
        return self.nmf_predictions
    
    def recommend_nmf(self, user_id, top_n=10):
        """Recommend movies using NMF"""
        if user_id not in self.user_to_idx:
            return None
        
        user_idx = self.user_to_idx[user_id]
        
        # Get user's predictions
        user_predictions = self.nmf_predictions[user_idx, :]
        
        # Get movies user hasn't rated
        rated_movies = self.interaction_matrix[user_idx, :].toarray()[0]
        unrated_mask = rated_movies == 0
        
        # Get predictions for unrated movies
        unrated_predictions = user_predictions * unrated_mask
        
        # Get top N
        top_indices = np.argsort(unrated_predictions)[::-1][:top_n]
        top_movie_ids = [self.idx_to_movie[idx] for idx in top_indices]
        top_scores = [unrated_predictions[idx] for idx in top_indices]
        
        recommendations = pd.DataFrame({
            'movieId': top_movie_ids,
            'predicted_rating': top_scores
        })
        
        return recommendations
    
    # ==========================================
    # MODEL 3: KNN (K-Nearest Neighbors)
    # ==========================================
    
    def train_knn(self, n_neighbors=50, metric='cosine'):
        """Train KNN model (User-based collaborative filtering)"""
        print(f"\n👥 Training KNN Model (k={n_neighbors} neighbors, metric={metric})...")
        start_time = time.time()
        
        # User-based KNN
        self.knn_model = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, algorithm='brute')
        self.knn_model.fit(self.interaction_matrix)
        
        train_time = time.time() - start_time
        
        print(f"   ✓ Training completed in {train_time:.2f}s")
        print(f"   ✓ Model: User-based collaborative filtering")
        
        # Save model
        with open(self.models_path + 'knn_model.pkl', 'wb') as f:
            pickle.dump(self.knn_model, f)
        
        return self.knn_model
    
    def recommend_knn(self, user_id, top_n=10):
        """Recommend movies using KNN"""
        if user_id not in self.user_to_idx:
            return None
        
        user_idx = self.user_to_idx[user_id]
        
        # Get user vector
        user_vector = self.interaction_matrix[user_idx, :]
        
        # Find similar users
        distances, indices = self.knn_model.kneighbors(user_vector, n_neighbors=51)
        
        # Exclude self
        similar_user_indices = indices[0][1:]
        
        # Get ratings from similar users
        similar_users_ratings = self.interaction_matrix[similar_user_indices, :].toarray()
        
        # Average ratings (simple approach)
        avg_ratings = np.mean(similar_users_ratings, axis=0)
        
        # Get movies user hasn't rated
        rated_movies = user_vector.toarray()[0]
        unrated_mask = rated_movies == 0
        
        # Get predictions for unrated movies
        unrated_predictions = avg_ratings * unrated_mask
        
        # Get top N
        top_indices = np.argsort(unrated_predictions)[::-1][:top_n]
        top_movie_ids = [self.idx_to_movie[idx] for idx in top_indices]
        top_scores = [unrated_predictions[idx] for idx in top_indices]
        
        recommendations = pd.DataFrame({
            'movieId': top_movie_ids,
            'predicted_rating': top_scores
        })
        
        return recommendations
    
    # ==========================================
    # MODEL 4: Neural Collaborative Filtering
    # ==========================================
    
    def build_neural_cf_model(self, n_factors=50):
        """Build Neural Collaborative Filtering model"""
        if not TENSORFLOW_AVAILABLE:
            print("\n⚠️  Skipping Neural CF - TensorFlow not available")
            return None
        
        print(f"\n🧠 Building Neural CF Model (embedding_dim={n_factors})...")
        
        n_users = self.interaction_matrix.shape[0]
        n_movies = self.interaction_matrix.shape[1]
        
        # User input
        user_input = layers.Input(shape=(1,), name='user_input')
        user_embedding = layers.Embedding(n_users, n_factors, name='user_embedding')(user_input)
        user_vec = layers.Flatten(name='user_flatten')(user_embedding)
        
        # Movie input
        movie_input = layers.Input(shape=(1,), name='movie_input')
        movie_embedding = layers.Embedding(n_movies, n_factors, name='movie_embedding')(movie_input)
        movie_vec = layers.Flatten(name='movie_flatten')(movie_embedding)
        
        # Concatenate
        concat = layers.Concatenate()([user_vec, movie_vec])
        
        # Dense layers
        dense1 = layers.Dense(128, activation='relu')(concat)
        dropout1 = layers.Dropout(0.2)(dense1)
        dense2 = layers.Dense(64, activation='relu')(dropout1)
        dropout2 = layers.Dropout(0.2)(dense2)
        dense3 = layers.Dense(32, activation='relu')(dropout2)
        
        # Output
        output = layers.Dense(1, activation='linear')(dense3)
        
        # Model
        model = keras.Model(inputs=[user_input, movie_input], outputs=output)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        print(f"   ✓ Model built successfully")
        print(f"   ✓ Total parameters: {model.count_params():,}")
        
        return model
    
    def train_neural_cf(self, epochs=5, batch_size=256):
        """Train Neural CF model"""
        if not TENSORFLOW_AVAILABLE:
            return None
        
        print(f"\n🧠 Training Neural CF Model...")
        start_time = time.time()
        
        # Build model
        self.neural_cf_model = self.build_neural_cf_model(n_factors=50)
        
        if self.neural_cf_model is None:
            return None
        
        # Prepare training data from interaction matrix
        print("   Preparing training data...")
        rows, cols = self.interaction_matrix.nonzero()
        ratings = self.interaction_matrix.data
        
        # Shuffle
        idx = np.random.permutation(len(rows))
        user_indices = rows[idx]
        movie_indices = cols[idx]
        rating_values = ratings[idx]
        
        print(f"   ✓ Training samples: {len(user_indices):,}")
        
        # Train model
        history = self.neural_cf_model.fit(
            [user_indices, movie_indices],
            rating_values,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.1,
            verbose=1
        )
        
        train_time = time.time() - start_time
        
        print(f"   ✓ Training completed in {train_time:.2f}s")
        
        # Save model
        self.neural_cf_model.save(self.models_path + 'neural_cf_model.h5')
        
        return self.neural_cf_model
    
    def recommend_neural_cf(self, user_id, top_n=10):
        """Recommend movies using Neural CF"""
        if not TENSORFLOW_AVAILABLE or not hasattr(self, 'neural_cf_model'):
            return None
        
        if user_id not in self.user_to_idx:
            return None
        
        user_idx = self.user_to_idx[user_id]
        
        # Get all movie indices
        n_movies = self.interaction_matrix.shape[1]
        movie_indices = np.arange(n_movies)
        user_indices = np.full(n_movies, user_idx)
        
        # Predict ratings
        predictions = self.neural_cf_model.predict([user_indices, movie_indices], verbose=0).flatten()
        
        # Get movies user hasn't rated
        rated_movies = self.interaction_matrix[user_idx, :].toarray()[0]
        unrated_mask = rated_movies == 0
        
        # Get predictions for unrated movies
        unrated_predictions = predictions * unrated_mask
        
        # Get top N
        top_indices = np.argsort(unrated_predictions)[::-1][:top_n]
        top_movie_ids = [self.idx_to_movie[idx] for idx in top_indices]
        top_scores = [unrated_predictions[idx] for idx in top_indices]
        
        recommendations = pd.DataFrame({
            'movieId': top_movie_ids,
            'predicted_rating': top_scores
        })
        
        return recommendations
    
    # ==========================================
    # EVALUATION
    # ==========================================
    
    def evaluate_model(self, model_name, predict_func, n_users=100):
        """Evaluate a model on test set"""
        print(f"\n📊 Evaluating {model_name}...")
        
        # Sample test users
        test_user_ids = self.test_ratings['userId'].unique()
        valid_test_users = [uid for uid in test_user_ids if uid in self.user_to_idx]
        
        if len(valid_test_users) > n_users:
            sampled_users = np.random.choice(valid_test_users, n_users, replace=False)
        else:
            sampled_users = valid_test_users[:n_users]
        
        all_predictions = []
        all_actuals = []
        
        for user_id in sampled_users:
            user_test = self.test_ratings[self.test_ratings['userId'] == user_id]
            
            for _, row in user_test.iterrows():
                movie_id = row['movieId']
                actual_rating = row['rating']
                
                if movie_id in self.movie_to_idx:
                    pred = predict_func(user_id, movie_id)
                    if pred is not None:
                        all_predictions.append(pred)
                        all_actuals.append(actual_rating)
        
        if len(all_predictions) == 0:
            print(f"   ⚠️  No predictions available")
            return None
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(all_actuals, all_predictions))
        mae = mean_absolute_error(all_actuals, all_predictions)
        
        print(f"   ✓ Test samples: {len(all_predictions)}")
        print(f"   ✓ RMSE: {rmse:.4f}")
        print(f"   ✓ MAE: {mae:.4f}")
        
        return {'model': model_name, 'rmse': rmse, 'mae': mae, 'n_samples': len(all_predictions)}
    
    def run_complete_pipeline(self):
        """Run complete collaborative filtering pipeline"""
        print("\n🤝 COLLABORATIVE FILTERING SYSTEMS")
        print("="*70)
        
        results = []
        
        # Train and evaluate SVD
        self.train_svd(n_factors=50)
        svd_results = self.evaluate_model('SVD', self.predict_svd, n_users=100)
        if svd_results:
            results.append(svd_results)
        
        # Train and evaluate NMF
        self.train_nmf(n_factors=50, max_iter=200)
        # Note: NMF predict similar to SVD
        results.append({'model': 'NMF', 'note': 'Trained successfully'})
        
        # Train and evaluate KNN
        self.train_knn(n_neighbors=50)
        results.append({'model': 'KNN', 'note': 'Trained successfully'})
        
        # Train Neural CF (optional - takes longer)
        if TENSORFLOW_AVAILABLE:
            self.train_neural_cf(epochs=3, batch_size=256)
        
        # Example recommendations
        print("\n\n👤 Example: Recommendations for User ID 1")
        
        print("\n📊 SVD Recommendations:")
        svd_recs = self.recommend_svd(1, top_n=5)
        if svd_recs is not None:
            print(svd_recs.to_string(index=False))
        
        print("\n📊 NMF Recommendations:")
        nmf_recs = self.recommend_nmf(1, top_n=5)
        if nmf_recs is not None:
            print(nmf_recs.to_string(index=False))
        
        print("\n📊 KNN Recommendations:")
        knn_recs = self.recommend_knn(1, top_n=5)
        if knn_recs is not None:
            print(knn_recs.to_string(index=False))
        
        if TENSORFLOW_AVAILABLE and hasattr(self, 'neural_cf_model'):
            print("\n📊 Neural CF Recommendations:")
            ncf_recs = self.recommend_neural_cf(1, top_n=5)
            if ncf_recs is not None:
                print(ncf_recs.to_string(index=False))
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv('reports/collaborative_filtering_evaluation.csv', index=False)
        
        print("\n" + "="*70)
        print("✅ COLLABORATIVE FILTERING COMPLETED!")
        print("="*70)
        print("\n📁 All models and results saved!")
        
        return results_df

# Main execution
if __name__ == "__main__":
    cf_recommender = CollaborativeFilteringRecommender()
    results = cf_recommender.run_complete_pipeline()

# 02_comprehensive_eda.py (FIXED VERSION)
"""
Step 2: Comprehensive Exploratory Data Analysis
Purpose: Deep dive into the dataset with visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class MovieLensEDA:
    def __init__(self):
        """Initialize EDA with data loading"""
        self.data_path = "data/raw/ml-25m/"
        self.output_path = "reports/eda_figures/"
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        
        print("📥 Loading datasets...")
        self.ratings = pd.read_csv(self.data_path + "ratings.csv")
        self.movies = pd.read_csv(self.data_path + "movies.csv")
        self.tags = pd.read_csv(self.data_path + "tags.csv")
        
        # Convert timestamp to datetime
        self.ratings['timestamp'] = pd.to_datetime(self.ratings['timestamp'], unit='s')
        self.tags['timestamp'] = pd.to_datetime(self.tags['timestamp'], unit='s')
        
        print("✅ Datasets loaded!\n")
    
    def basic_statistics(self):
        """Display basic statistics"""
        print("="*70)
        print("📊 BASIC STATISTICS")
        print("="*70)
        
        # Extract years from movie titles (FIXED)
        year_pattern = r'\((\d{4})\)'
        years = self.movies['title'].str.extract(year_pattern)[0]
        # Convert to numeric and drop NaN
        years = pd.to_numeric(years, errors='coerce').dropna()
        
        if len(years) > 0:
            min_year = int(years.min())
            max_year = int(years.max())
        else:
            min_year = "N/A"
            max_year = "N/A"
        
        print(f"\n🎬 MOVIES:")
        print(f"   Total movies: {self.movies['movieId'].nunique():,}")
        print(f"   Date range: {min_year} - {max_year}")
        
        print(f"\n⭐ RATINGS:")
        print(f"   Total ratings: {len(self.ratings):,}")
        print(f"   Unique users: {self.ratings['userId'].nunique():,}")
        print(f"   Unique movies rated: {self.ratings['movieId'].nunique():,}")
        print(f"   Rating range: {self.ratings['rating'].min()} - {self.ratings['rating'].max()}")
        print(f"   Average rating: {self.ratings['rating'].mean():.2f}")
        print(f"   Date range: {self.ratings['timestamp'].min().date()} to {self.ratings['timestamp'].max().date()}")
        
        print(f"\n🏷️ TAGS:")
        print(f"   Total tags: {len(self.tags):,}")
        print(f"   Unique tags: {self.tags['tag'].nunique():,}")
        print(f"   Users who tagged: {self.tags['userId'].nunique():,}")
        
        # Sparsity
        n_users = self.ratings['userId'].nunique()
        n_movies = self.ratings['movieId'].nunique()
        n_ratings = len(self.ratings)
        sparsity = 1 - (n_ratings / (n_users * n_movies))
        print(f"\n📉 SPARSITY:")
        print(f"   Matrix sparsity: {sparsity*100:.4f}%")
        print(f"   Density: {(1-sparsity)*100:.6f}%")
        
        print("\n" + "="*70 + "\n")
    
    def plot_rating_distribution(self):
        """Plot rating distribution - REQUIRED for presentation"""
        print("📊 Creating rating distribution plot...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Rating counts
        rating_counts = self.ratings['rating'].value_counts().sort_index()
        axes[0].bar(rating_counts.index, rating_counts.values, color='steelblue', edgecolor='black')
        axes[0].set_xlabel('Rating', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Count (millions)', fontsize=12, fontweight='bold')
        axes[0].set_title('Distribution of Ratings', fontsize=14, fontweight='bold')
        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
        axes[0].grid(axis='y', alpha=0.3)
        
        # Rating percentage
        rating_pct = (rating_counts / rating_counts.sum() * 100).sort_index()
        axes[1].bar(rating_pct.index, rating_pct.values, color='coral', edgecolor='black')
        axes[1].set_xlabel('Rating', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        axes[1].set_title('Percentage Distribution of Ratings', fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_path + '01_rating_distribution.png', dpi=300, bbox_inches='tight')
        print("   ✅ Saved: 01_rating_distribution.png\n")
        plt.close()
    
    def plot_genre_distribution(self):
        """Plot genre distribution - REQUIRED for presentation"""
        print("🎭 Creating genre distribution plot...")
        
        # Split genres
        genres_split = self.movies['genres'].str.split('|', expand=True)
        all_genres = pd.Series([g for sublist in genres_split.values for g in sublist if g is not None and g != '(no genres listed)'])
        genre_counts = all_genres.value_counts()
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        genre_counts.plot(kind='barh', color='teal', edgecolor='black', ax=ax)
        ax.set_xlabel('Number of Movies', fontsize=12, fontweight='bold')
        ax.set_ylabel('Genre', fontsize=12, fontweight='bold')
        ax.set_title('Movie Count per Genre', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add values on bars
        for i, v in enumerate(genre_counts.values):
            ax.text(v + 50, i, str(v), va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_path + '02_genre_distribution.png', dpi=300, bbox_inches='tight')
        print("   ✅ Saved: 02_genre_distribution.png\n")
        plt.close()
        
        return genre_counts
    
    def plot_user_engagement(self):
        """Plot user engagement metrics - REQUIRED for presentation"""
        print("👥 Creating user engagement plots...")
        
        # Ratings per user
        ratings_per_user = self.ratings.groupby('userId').size()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Histogram of ratings per user
        axes[0, 0].hist(ratings_per_user, bins=100, color='purple', edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Number of Ratings', fontsize=11, fontweight='bold')
        axes[0, 0].set_ylabel('Number of Users', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('User Engagement Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlim(0, 500)
        axes[0, 0].grid(alpha=0.3)
        
        # 2. Top 20 most active users
        top_users = ratings_per_user.nlargest(20)
        axes[0, 1].barh(range(20), top_users.values, color='orange', edgecolor='black')
        axes[0, 1].set_xlabel('Number of Ratings', fontsize=11, fontweight='bold')
        axes[0, 1].set_ylabel('User Rank', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('Top 20 Most Active Users', fontsize=12, fontweight='bold')
        axes[0, 1].invert_yaxis()
        axes[0, 1].grid(axis='x', alpha=0.3)
        
        # 3. Ratings per movie
        ratings_per_movie = self.ratings.groupby('movieId').size()
        axes[1, 0].hist(ratings_per_movie, bins=100, color='green', edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Number of Ratings', fontsize=11, fontweight='bold')
        axes[1, 0].set_ylabel('Number of Movies', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('Movie Popularity Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlim(0, 1000)
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(alpha=0.3)
        
        # 4. Cumulative ratings over time
        ratings_timeline = self.ratings.set_index('timestamp').resample('M').size().cumsum()
        axes[1, 1].plot(ratings_timeline.index, ratings_timeline.values / 1e6, color='red', linewidth=2)
        axes[1, 1].set_xlabel('Date', fontsize=11, fontweight='bold')
        axes[1, 1].set_ylabel('Cumulative Ratings (Millions)', fontsize=11, fontweight='bold')
        axes[1, 1].set_title('Rating Growth Over Time', fontsize=12, fontweight='bold')
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_path + '03_user_engagement.png', dpi=300, bbox_inches='tight')
        print("   ✅ Saved: 03_user_engagement.png\n")
        plt.close()
    
    def plot_top_20_movies(self):
        """Plot top 20 most popular movies - REQUIRED for presentation"""
        print("🎬 Creating top 20 movies plot...")
        
        # Get top rated movies
        movie_stats = self.ratings.groupby('movieId').agg({
            'rating': ['count', 'mean']
        }).reset_index()
        movie_stats.columns = ['movieId', 'rating_count', 'rating_mean']
        
        # Filter movies with at least 1000 ratings
        popular_movies = movie_stats[movie_stats['rating_count'] >= 1000]
        top_20 = popular_movies.nlargest(20, 'rating_count')
        
        # Merge with movie titles
        top_20 = top_20.merge(self.movies[['movieId', 'title']], on='movieId')
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        bars = ax.barh(range(20), top_20['rating_count'].values, color='royalblue', edgecolor='black')
        
        # Color bars by rating
        for i, (bar, rating) in enumerate(zip(bars, top_20['rating_mean'].values)):
            bar.set_color(plt.cm.RdYlGn(rating / 5))
        
        ax.set_yticks(range(20))
        ax.set_yticklabels(top_20['title'].str[:40], fontsize=10)
        ax.set_xlabel('Number of Ratings', fontsize=12, fontweight='bold')
        ax.set_title('Top 20 Most Popular Movies', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(vmin=0, vmax=5))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Average Rating', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_path + '04_top_20_movies.png', dpi=300, bbox_inches='tight')
        print("   ✅ Saved: 04_top_20_movies.png\n")
        plt.close()
        
        return top_20
    
    def create_wordcloud(self):
        """Create word cloud from movie titles - REQUIRED for presentation"""
        print("☁️  Creating word cloud...")
        
        # Extract all words from titles
        title_pattern = r'\(\d{4}\)'
        titles_text = ' '.join(self.movies['title'].str.replace(title_pattern, '', regex=True))
        
        # Create wordcloud
        wordcloud = WordCloud(width=1600, height=800, 
                              background_color='white',
                              colormap='viridis',
                              max_words=100,
                              relative_scaling=0.5,
                              min_font_size=10).generate(titles_text)
        
        # Plot
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Word Cloud of Movie Titles', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(self.output_path + '05_wordcloud.png', dpi=300, bbox_inches='tight')
        print("   ✅ Saved: 05_wordcloud.png\n")
        plt.close()
    
    def temporal_analysis(self):
        """Analyze temporal patterns"""
        print("📅 Creating temporal analysis plots...")
        
        # Extract year and month
        self.ratings['year'] = self.ratings['timestamp'].dt.year
        self.ratings['month'] = self.ratings['timestamp'].dt.month
        self.ratings['hour'] = self.ratings['timestamp'].dt.hour
        self.ratings['dayofweek'] = self.ratings['timestamp'].dt.dayofweek
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Ratings by year
        yearly_ratings = self.ratings['year'].value_counts().sort_index()
        axes[0, 0].plot(yearly_ratings.index, yearly_ratings.values / 1e6, marker='o', linewidth=2, color='darkblue')
        axes[0, 0].set_xlabel('Year', fontsize=11, fontweight='bold')
        axes[0, 0].set_ylabel('Ratings (Millions)', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('Ratings by Year', fontsize=12, fontweight='bold')
        axes[0, 0].grid(alpha=0.3)
        
        # Ratings by month
        monthly_ratings = self.ratings['month'].value_counts().sort_index()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        axes[0, 1].bar(monthly_ratings.index, monthly_ratings.values / 1e6, color='orange', edgecolor='black')
        axes[0, 1].set_xticks(range(1, 13))
        axes[0, 1].set_xticklabels(month_names, rotation=45)
        axes[0, 1].set_xlabel('Month', fontsize=11, fontweight='bold')
        axes[0, 1].set_ylabel('Ratings (Millions)', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('Ratings by Month', fontsize=12, fontweight='bold')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Ratings by hour
        hourly_ratings = self.ratings['hour'].value_counts().sort_index()
        axes[1, 0].plot(hourly_ratings.index, hourly_ratings.values / 1e6, marker='o', linewidth=2, color='green')
        axes[1, 0].set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
        axes[1, 0].set_ylabel('Ratings (Millions)', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('Ratings by Hour of Day', fontsize=12, fontweight='bold')
        axes[1, 0].set_xticks(range(0, 24, 2))
        axes[1, 0].grid(alpha=0.3)
        
        # Ratings by day of week
        dow_ratings = self.ratings['dayofweek'].value_counts().sort_index()
        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        axes[1, 1].bar(dow_ratings.index, dow_ratings.values / 1e6, color='purple', edgecolor='black')
        axes[1, 1].set_xticks(range(7))
        axes[1, 1].set_xticklabels(dow_names)
        axes[1, 1].set_xlabel('Day of Week', fontsize=11, fontweight='bold')
        axes[1, 1].set_ylabel('Ratings (Millions)', fontsize=11, fontweight='bold')
        axes[1, 1].set_title('Ratings by Day of Week', fontsize=12, fontweight='bold')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_path + '06_temporal_analysis.png', dpi=300, bbox_inches='tight')
        print("   ✅ Saved: 06_temporal_analysis.png\n")
        plt.close()
    
    def generate_eda_summary_report(self):
        """Generate a comprehensive EDA summary"""
        summary = {
            'total_ratings': len(self.ratings),
            'unique_users': self.ratings['userId'].nunique(),
            'unique_movies': self.ratings['movieId'].nunique(),
            'avg_rating': self.ratings['rating'].mean(),
            'std_rating': self.ratings['rating'].std(),
            'median_rating': self.ratings['rating'].median(),
            'sparsity': 1 - (len(self.ratings) / (self.ratings['userId'].nunique() * self.ratings['movieId'].nunique())),
            'avg_ratings_per_user': self.ratings.groupby('userId').size().mean(),
            'avg_ratings_per_movie': self.ratings.groupby('movieId').size().mean()
        }
        
        # Save to CSV
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv('reports/eda_summary.csv', index=False)
        print("✅ EDA Summary saved to: reports/eda_summary.csv\n")
        
        return summary
    
    def run_complete_eda(self):
        """Run all EDA analyses"""
        print("\n🔍 Starting Comprehensive EDA...\n")
        
        self.basic_statistics()
        self.plot_rating_distribution()
        self.plot_genre_distribution()
        self.plot_user_engagement()
        top_movies = self.plot_top_20_movies()
        self.create_wordcloud()
        self.temporal_analysis()
        summary = self.generate_eda_summary_report()
        
        print("="*70)
        print("✅ EDA COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\n📁 All visualizations saved in: {self.output_path}")
        print("\n📊 Key Insights:")
        print(f"   - Average rating: {summary['avg_rating']:.2f}")
        print(f"   - Matrix sparsity: {summary['sparsity']*100:.4f}%")
        print(f"   - Avg ratings per user: {summary['avg_ratings_per_user']:.1f}")
        print(f"   - Avg ratings per movie: {summary['avg_ratings_per_movie']:.1f}")
        
        return summary

# Main execution
if __name__ == "__main__":
    eda = MovieLensEDA()
    summary = eda.run_complete_eda()

# 01_setup_and_download.py
"""
Step 1: Project Setup and Dataset Download
Purpose: Initialize project structure and download MovieLens 25M dataset
"""

import os
import zipfile
import requests
from pathlib import Path
import pandas as pd
import numpy as np

# Create project directory structure
def create_project_structure():
    """Create organized project folders"""
    directories = [
        'data/raw',
        'data/processed',
        'notebooks',
        'src/data_processing',
        'src/models',
        'src/utils',
        'streamlit_app/pages',
        'streamlit_app/components',
        'models/saved_models',
        'reports',
        'tests'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {directory}")
    
    print("\n✅ Project structure created successfully!")

# Download MovieLens 25M dataset
def download_movielens_25m():
    """Download and extract MovieLens 25M dataset"""
    url = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
    zip_path = "data/raw/ml-25m.zip"
    extract_path = "data/raw/"
    
    print("📥 Downloading MovieLens 25M dataset (250 MB)...")
    print("This may take a few minutes...\n")
    
    # Download the dataset
    if not os.path.exists(zip_path):
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_path, 'wb') as file:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    downloaded += len(chunk)
                    progress = (downloaded / total_size) * 100
                    print(f"\rProgress: {progress:.1f}%", end='')
        
        print("\n✅ Download completed!")
    else:
        print("✓ Dataset already downloaded!")
    
    # Extract the zip file
    print("\n📂 Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    print("✅ Extraction completed!")
    
    # Display dataset info
    print("\n📊 Dataset Contents:")
    dataset_path = "data/raw/ml-25m"
    for file in os.listdir(dataset_path):
        file_path = os.path.join(dataset_path, file)
        if os.path.isfile(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  - {file}: {size_mb:.2f} MB")

# Load and inspect datasets
def load_and_inspect_data():
    """Load all dataset files and display basic information"""
    data_path = "data/raw/ml-25m/"
    
    print("\n" + "="*60)
    print("📊 LOADING AND INSPECTING DATASETS")
    print("="*60)
    
    # Load ratings
    print("\n1️⃣ RATINGS Dataset:")
    ratings = pd.read_csv(data_path + "ratings.csv")
    print(f"   Shape: {ratings.shape}")
    print(f"   Columns: {list(ratings.columns)}")
    print(f"   Memory: {ratings.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    print(f"\n   First 3 rows:")
    print(ratings.head(3))
    
    # Load movies
    print("\n2️⃣ MOVIES Dataset:")
    movies = pd.read_csv(data_path + "movies.csv")
    print(f"   Shape: {movies.shape}")
    print(f"   Columns: {list(movies.columns)}")
    print(f"\n   First 3 rows:")
    print(movies.head(3))
    
    # Load tags
    print("\n3️⃣ TAGS Dataset:")
    tags = pd.read_csv(data_path + "tags.csv")
    print(f"   Shape: {tags.shape}")
    print(f"   Columns: {list(tags.columns)}")
    print(f"\n   First 3 rows:")
    print(tags.head(3))
    
    # Load genome scores
    print("\n4️⃣ GENOME SCORES Dataset:")
    genome_scores = pd.read_csv(data_path + "genome-scores.csv")
    print(f"   Shape: {genome_scores.shape}")
    print(f"   Columns: {list(genome_scores.columns)}")
    
    # Load genome tags
    print("\n5️⃣ GENOME TAGS Dataset:")
    genome_tags = pd.read_csv(data_path + "genome-tags.csv")
    print(f"   Shape: {genome_tags.shape}")
    print(f"   Columns: {list(genome_tags.columns)}")
    
    print("\n" + "="*60)
    print("✅ All datasets loaded successfully!")
    print("="*60)
    
    return ratings, movies, tags, genome_scores, genome_tags

# Main execution
if __name__ == "__main__":
    print("🎬 MOVIELENS 25M - RECOMMENDER SYSTEM PROJECT")
    print("="*60)
    
    # Step 1: Create project structure
    create_project_structure()
    
    # Step 2: Download dataset
    download_movielens_25m()
    
    # Step 3: Load and inspect
    ratings, movies, tags, genome_scores, genome_tags = load_and_inspect_data()
    
    print("\n🎉 Setup completed! Ready for EDA.")

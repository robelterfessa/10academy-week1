# scripts/data_loader.py

import pandas as pd
import os

def get_data_paths():
    """Return paths to local data files"""
    base_paths = ['.', 'data', '../data']
    
    for base in base_paths:
        news_path = os.path.join(base, 'news.csv')
        if os.path.exists(news_path):
            return {
                'news': news_path,
                'base': base
            }
    
    print("⚠️  Data files not found. Please ensure your data files are in the 'data/' directory.")
    return None

def load_news_sample(n_rows=10000):
    """Load a sample of news data for development"""
    paths = get_data_paths()
    if paths and os.path.exists(paths['news']):
        try:
            print(f"Loading {n_rows} news articles from {paths['news']}")
            return pd.read_csv(paths['news'], nrows=n_rows)
        except Exception as e:
            print(f"Error loading news data: {e}")
    return None

def check_data_availability():
    """Check what data is available"""
    paths = get_data_paths()
    if paths:
        print("✅ Data files found:")
        for name, path in paths.items():
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                print(f"   {name}: {path} ({size_mb:.1f} MB)")
    else:
        print("❌ No data files found")
    return paths

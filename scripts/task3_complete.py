# scripts/task3_complete.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
from textblob import TextBlob
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Try to import data_loader, but continue if not available
try:
    from scripts.data_loader import load_news_sample
    DATA_LOADER_AVAILABLE = True
except ImportError:
    DATA_LOADER_AVAILABLE = False
    print("⚠️  Data loader not available, using sample data")

print("🚀 Task 3: Correlation Analysis - News Sentiment vs Stock Movement")

class CorrelationAnalyzer:
    def __init__(self):
        self.sentiment_results = {}
        self.correlation_results = {}
        
    def analyze_sentiment(self, text):
        """Analyze sentiment of news headlines"""
        try:
            return TextBlob(str(text)).sentiment.polarity
        except:
            return 0.0
            
    def create_sample_news_data(self):
        """Create sample news data for demonstration"""
        print("📰 Creating sample news data for demonstration...")
        
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        n_articles = 1000
        
        # Create sample news data
        headlines = [
            "Stock Market Reaches New High Amid Positive Earnings",
            "Company Reports Strong Quarterly Results", 
            "Economic Indicators Show Growth Potential",
            "Market Volatility Concerns Investors",
            "Tech Sector Leads Market Gains",
            "Federal Reserve Announces Rate Decision",
            "Global Markets Show Mixed Performance",
            "Company Announces Major Acquisition",
            "Economic Data Exceeds Expectations",
            "Market Reacts to Geopolitical News"
        ]
        
        news_data = pd.DataFrame({
            'date': np.random.choice(dates, n_articles),
            'headline': np.random.choice(headlines, n_articles),
            'publisher': np.random.choice(['Reuters', 'Bloomberg', 'CNBC', 'WSJ'], n_articles)
        })
        
        # Add sentiment based on headline content
        def infer_sentiment(headline):
            positive_words = ['high', 'strong', 'growth', 'gains', 'exceeds', 'positive']
            negative_words = ['volatility', 'concerns', 'mixed', 'reacts']
            
            if any(word in headline.lower() for word in positive_words):
                return np.random.normal(0.3, 0.2)
            elif any(word in headline.lower() for word in negative_words):
                return np.random.normal(-0.2, 0.2)
            else:
                return np.random.normal(0.0, 0.1)
        
        news_data['sentiment'] = news_data['headline'].apply(infer_sentiment)
        news_data['date'] = pd.to_datetime(news_data['date'])
        
        return news_data
        
    def load_and_process_news(self, sample_size=5000):
        """Load and process news data with sentiment analysis"""
        print("📰 Loading and processing news data...")
        
        if DATA_LOADER_AVAILABLE:
            news_data = load_news_sample(sample_size)
        else:
            news_data = None
            
        if news_data is None:
            print("⚠️  Using sample news data for demonstration")
            news_data = self.create_sample_news_data()
            
        # Display basic info
        print(f"   Loaded {len(news_data)} news articles")
        print(f"   Columns: {list(news_data.columns)}")
        
        # Find date and headline columns
        date_col = 'date'
        headline_col = 'headline'
        
        # Clean data
        news_clean = news_data[[date_col, headline_col]].copy()
        news_clean = news_clean.dropna()
        
        # Convert date
        news_clean['date'] = pd.to_datetime(news_clean[date_col], errors='coerce')
        news_clean = news_clean.dropna(subset=['date'])
        
        # Analyze sentiment if not already present
        if 'sentiment' not in news_clean.columns:
            print("😊 Analyzing sentiment...")
            news_clean['sentiment'] = news_clean[headline_col].apply(self.analyze_sentiment)
        
        # Categorize sentiment
        news_clean['sentiment_category'] = news_clean['sentiment'].apply(
            lambda x: 'positive' if x > 0.1 else 'negative' if x < -0.1 else 'neutral'
        )
        
        # Daily aggregation
        news_clean['date_only'] = news_clean['date'].dt.date
        daily_sentiment = news_clean.groupby('date_only').agg({
            'sentiment': 'mean',
            headline_col: 'count'
        }).reset_index()
        daily_sentiment.rename(columns={headline_col: 'article_count'}, inplace=True)
        daily_sentiment['date_only'] = pd.to_datetime(daily_sentiment['date_only'])
        
        self.sentiment_results = {
            'raw_data': news_clean,
            'daily_data': daily_sentiment
        }
        
        print(f"✅ Processed {len(news_clean)} articles into {len(daily_sentiment)} daily records")
        return daily_sentiment
        
    def load_stock_data(self):
        """Load stock data from Task 2 results"""
        print("📈 Loading stock data...")
        
        stock_data = {}
        data_dir = 'notebooks/data'
        
        if not os.path.exists(data_dir):
            print("❌ Task 2 data not found. Running sample analysis...")
            return self.create_sample_stock_data()
            
        # Load processed stock data
        stock_files = [f for f in os.listdir(data_dir) if f.endswith('_with_indicators.csv')]
        
        for file in stock_files:
            try:
                symbol = file.replace('_with_indicators.csv', '')
                file_path = os.path.join(data_dir, file)
                stock_df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                
                # Ensure we have daily returns
                if 'Daily_Return' not in stock_df.columns:
                    stock_df['Daily_Return'] = stock_df['Close'].pct_change() * 100
                    
                stock_data[symbol] = stock_df
                print(f"   ✅ {symbol}: {len(stock_df)} records")
                
            except Exception as e:
                print(f"   ❌ Error loading {file}: {e}")
                
        if not stock_data:
            print("⚠️ No stock data found, using sample data")
            return self.create_sample_stock_data()
            
        return stock_data
        
    def create_sample_stock_data(self):
        """Create sample stock data for demonstration"""
        print("📊 Creating sample stock data for demonstration...")
        
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        
        stock_data = {}
        for symbol in symbols:
            np.random.seed(42)  # For reproducibility
            returns = np.random.normal(0.1, 2, len(dates))  # Random returns
            
            df = pd.DataFrame({
                'Close': 100 + np.cumsum(returns),
                'Daily_Return': returns
            }, index=dates)
            
            stock_data[symbol] = df
            
        return stock_data
        
    def calculate_correlations(self, daily_sentiment, stock_data):
        """Calculate correlations between sentiment and stock returns"""
        print("\n🔗 Calculating correlations...")
        
        results = []
        
        for symbol, stock_df in stock_data.items():
            print(f"\nAnalyzing {symbol}...")
            
            # Prepare stock data
            stock_daily = stock_df[['Daily_Return']].copy()
            stock_daily['date_only'] = stock_daily.index.normalize()
            stock_daily = stock_daily.groupby('date_only')['Daily_Return'].mean().reset_index()
            
            # Merge with sentiment data
            merged = pd.merge(stock_daily, daily_sentiment, on='date_only', how='inner')
            merged = merged.dropna()
            
            if len(merged) < 5:
                print(f"   ⚠️ Insufficient overlapping data: {len(merged)} points")
                continue
                
            # Calculate correlations
            pearson_corr, pearson_p = pearsonr(merged['sentiment'], merged['Daily_Return'])
            spearman_corr, spearman_p = spearmanr(merged['sentiment'], merged['Daily_Return'])
            
            result = {
                'symbol': symbol,
                'pearson_correlation': pearson_corr,
                'pearson_p_value': pearson_p,
                'spearman_correlation': spearman_corr,
                'spearman_p_value': spearman_p,
                'data_points': len(merged),
                'avg_sentiment': merged['sentiment'].mean(),
                'avg_return': merged['Daily_Return'].mean(),
                'sentiment_std': merged['sentiment'].std(),
                'return_std': merged['Daily_Return'].std()
            }
            
            results.append(result)
            
            print(f"   📊 Pearson: {pearson_corr:.3f} (p={pearson_p:.3f})")
            print(f"   📈 Spearman: {spearman_corr:.3f} (p={spearman_p:.3f})")
            print(f"   📅 Data points: {len(merged)}")
            
            if pearson_p < 0.05:
                print("   ✅ Statistically significant!")
                
        return pd.DataFrame(results)
        
    def create_visualizations(self, correlation_results, daily_sentiment):
        """Create comprehensive visualizations"""
        print("\n📊 Creating visualizations...")
        
        os.makedirs('notebooks/plots/task3', exist_ok=True)
        
        # Plot 1: Sentiment Distribution
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        sentiment_counts = self.sentiment_results['raw_data']['sentiment_category'].value_counts()
        colors = ['green', 'gray', 'red']
        plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        plt.title('News Sentiment Distribution')
        
        # Plot 2: Sentiment Score Histogram
        plt.subplot(1, 3, 2)
        plt.hist(self.sentiment_results['raw_data']['sentiment'], bins=50, 
                alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('Sentiment Score')
        plt.ylabel('Frequency')
        plt.title('Sentiment Scores Distribution')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Daily Sentiment Trend
        plt.subplot(1, 3, 3)
        plt.plot(daily_sentiment['date_only'], daily_sentiment['sentiment'], 
                color='purple', linewidth=1, alpha=0.7)
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        plt.xlabel('Date')
        plt.ylabel('Avg Daily Sentiment')
        plt.title('Daily Sentiment Trend')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('notebooks/plots/task3/sentiment_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 4: Correlation Results
        if not correlation_results.empty:
            plt.figure(figsize=(10, 6))
            
            # Use Pearson correlation
            symbols = correlation_results['symbol']
            correlations = correlation_results['pearson_correlation']
            p_values = correlation_results['pearson_p_value']
            
            # Color based on significance and direction
            colors = []
            for corr, p_val in zip(correlations, p_values):
                if p_val < 0.05:
                    colors.append('green' if corr > 0 else 'red')
                else:
                    colors.append('gray')
                    
            bars = plt.bar(symbols, correlations, color=colors, alpha=0.7, edgecolor='black')
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            plt.xlabel('Stock Symbol')
            plt.ylabel('Correlation Coefficient')
            plt.title('Correlation: News Sentiment vs Stock Returns\n(Pearson Correlation)')
            plt.grid(True, alpha=0.3)
            
            # Add value labels
            for i, (bar, corr, p_val) in enumerate(zip(bars, correlations, p_values)):
                height = bar.get_height()
                va = 'bottom' if height >= 0 else 'top'
                y_pos = height + 0.01 if height >= 0 else height - 0.01
                plt.text(bar.get_x() + bar.get_width()/2, y_pos,
                        f'{corr:.3f}\n(p={p_val:.3f})', 
                        ha='center', va=va, fontsize=8)
                        
            plt.tight_layout()
            plt.savefig('notebooks/plots/task3/correlation_results.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        print("✅ Visualizations saved to notebooks/plots/task3/")
        
    def generate_report(self, correlation_results):
        """Generate a summary report"""
        print("\n" + "="*60)
        print("📋 TASK 3 CORRELATION ANALYSIS REPORT")
        print("="*60)
        
        if not correlation_results.empty:
            print("\nCorrelation Results (Pearson):")
            print("-" * 50)
            for _, row in correlation_results.iterrows():
                significance = "✅ SIGNIFICANT" if row['pearson_p_value'] < 0.05 else "⚠️ Not significant"
                direction = "positive" if row['pearson_correlation'] > 0 else "negative"
                print(f"{row['symbol']}: {row['pearson_correlation']:.3f} ({direction}) - {significance}")
                
            print(f"\nSummary:")
            print(f"Stocks analyzed: {len(correlation_results)}")
            print(f"Significant correlations: {len(correlation_results[correlation_results['pearson_p_value'] < 0.05])}")
            print(f"Average correlation: {correlation_results['pearson_correlation'].mean():.3f}")
            
        else:
            print("No correlation results to report.")
            
        print("\n" + "="*60)
        
    def run_analysis(self, news_sample_size=5000):
        """Run the complete analysis"""
        print("🚀 Starting complete correlation analysis...")
        
        # 1. Load and process news data
        daily_sentiment = self.load_and_process_news(news_sample_size)
        if daily_sentiment is None:
            print("❌ Cannot proceed without news data")
            return
            
        # 2. Load stock data
        stock_data = self.load_stock_data()
        
        # 3. Calculate correlations
        correlation_results = self.calculate_correlations(daily_sentiment, stock_data)
        
        # 4. Create visualizations
        self.create_visualizations(correlation_results, daily_sentiment)
        
        # 5. Generate report
        self.generate_report(correlation_results)
        
        # 6. Save results
        if not correlation_results.empty:
            os.makedirs('notebooks/data', exist_ok=True)
            correlation_results.to_csv('notebooks/data/task3_correlation_results.csv', index=False)
            print(f"\n💾 Saved results to notebooks/data/task3_correlation_results.csv")
            
        print(f"\n🎉 Task 3 Correlation Analysis Complete!")
        
def main():
    analyzer = CorrelationAnalyzer()
    analyzer.run_analysis(news_sample_size=1000)  # Start with small sample
    
if __name__ == "__main__":
    main()

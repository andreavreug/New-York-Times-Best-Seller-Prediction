"""
Main pipeline for NYT Best Seller prediction.

This script orchestrates the entire pipeline:
1. Data collection from Hardcover and NYT API
2. Text preprocessing
3. Sentiment analysis
4. Feature engineering
5. Model training
6. Model evaluation
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_collection import (
    HardcoverScraper, NYTBestSellerScraper,
    create_labeled_dataset, save_data, load_data
)
from preprocessing import TextPreprocessor
from sentiment_analysis import extract_review_sentiment_features
from feature_engineering import (
    FeatureEngineering, create_book_level_features,
    prepare_features_for_modeling
)
from model_training import ModelTrainer
from model_evaluation import ModelEvaluator


def setup_directories():
    """Create necessary directories."""
    dirs = ['data/raw', 'data/processed', 'models', 'results']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def collect_data(use_sample: bool = False):
    """
    Collect data from APIs or use sample data.
    
    Args:
        use_sample: Whether to use sample data instead of API calls
    """
    print("\n" + "="*60)
    print("STEP 1: Data Collection")
    print("="*60)
    
    if use_sample:
        print("Using sample data for demonstration...")
        # Create sample data for demonstration
        sample_data = []
        
        # Sample best sellers
        best_sellers = [
            "The Great Novel", "Amazing Story", "Bestseller Book",
            "Popular Fiction", "Top Rated"
        ]
        
        # Sample non-best sellers
        non_best_sellers = [
            "Unknown Book", "Obscure Title", "Lesser Known",
            "New Author", "Hidden Gem"
        ]
        
        # Generate sample reviews
        positive_reviews = [
            "This book is absolutely amazing! I couldn't put it down.",
            "Wonderful story with great characters. Highly recommend!",
            "Best book I've read this year. Simply brilliant.",
            "Engaging plot and beautiful writing. A masterpiece.",
            "Loved every page. The author is incredibly talented."
        ]
        
        negative_reviews = [
            "Disappointing and poorly written. Not worth the time.",
            "The plot was confusing and the characters were flat.",
            "I couldn't get into this book at all. Very boring.",
            "Expected much more. The writing felt rushed.",
            "Not what I hoped for. Wouldn't recommend."
        ]
        
        neutral_reviews = [
            "It was okay. Nothing special but not terrible.",
            "Decent read but forgettable.",
            "Some parts were good, others not so much.",
            "Average book. Probably won't read it again.",
            "Neither loved it nor hated it. Just meh."
        ]
        
        # Create sample data for best sellers
        for book in best_sellers:
            for i in range(10):
                review = np.random.choice(positive_reviews + neutral_reviews)
                rating = np.random.choice([4, 5]) if any(p in review for p in positive_reviews) else 3
                sample_data.append({
                    'book_title': book,
                    'review_text': review,
                    'rating': rating,
                    'is_best_seller': 1,
                    'created_at': '2024-01-01'
                })
        
        # Create sample data for non-best sellers
        for book in non_best_sellers:
            for i in range(8):
                review = np.random.choice(negative_reviews + neutral_reviews)
                rating = np.random.choice([2, 3]) if any(n in review for n in negative_reviews) else 3
                sample_data.append({
                    'book_title': book,
                    'review_text': review,
                    'rating': rating,
                    'is_best_seller': 0,
                    'created_at': '2024-01-01'
                })
        
        save_data(sample_data, 'data/raw/reviews_data.json')
        print(f"Created {len(sample_data)} sample reviews")
        
    else:
        print("Collecting data from APIs...")
        print("Note: Requires API keys in .env file")
        
        # Initialize scrapers
        hardcover = HardcoverScraper()
        nyt = NYTBestSellerScraper()
        
        # Fetch NYT best sellers
        print("Fetching NYT Best Seller lists...")
        best_sellers = nyt.fetch_historical_lists(weeks=4)
        
        if not best_sellers:
            print("Warning: No best sellers fetched. Check API key.")
            return
        
        # Get book titles
        book_titles = [book['title'] for book in best_sellers[:20]]  # Limit to 20
        
        # Fetch reviews
        print("Fetching book reviews...")
        reviews_data = hardcover.fetch_multiple_books(book_titles)
        
        # Create labeled dataset
        labeled_data = create_labeled_dataset(best_sellers, reviews_data)
        
        # Save data
        save_data(labeled_data, 'data/raw/reviews_data.json')
        save_data(best_sellers, 'data/raw/best_sellers.json')
        
        print(f"Collected {len(labeled_data)} reviews for {len(reviews_data)} books")


def preprocess_data():
    """Preprocess review texts."""
    print("\n" + "="*60)
    print("STEP 2: Text Preprocessing")
    print("="*60)
    
    # Load data
    data = load_data('data/raw/reviews_data.json')
    df = pd.DataFrame(data)
    
    print(f"Loaded {len(df)} reviews")
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True)
    
    # Preprocess texts
    print("Preprocessing review texts...")
    df['preprocessed_text'] = preprocessor.preprocess_batch(
        df['review_text'].fillna('').tolist()
    )
    
    # Save preprocessed data
    df.to_csv('data/processed/preprocessed_reviews.csv', index=False)
    print(f"Saved preprocessed data to data/processed/preprocessed_reviews.csv")
    
    return df


def analyze_sentiment(df):
    """Perform sentiment analysis on reviews."""
    print("\n" + "="*60)
    print("STEP 3: Sentiment Analysis")
    print("="*60)
    
    # Extract sentiment features
    print("Analyzing sentiment...")
    df = extract_review_sentiment_features(df, text_column='review_text')
    
    # Save data with sentiment
    df.to_csv('data/processed/reviews_with_sentiment.csv', index=False)
    print(f"Saved sentiment data to data/processed/reviews_with_sentiment.csv")
    
    # Print statistics
    print("\nSentiment Statistics:")
    print(df.groupby('is_best_seller')['sentiment_polarity'].describe())
    
    return df


def engineer_features(df):
    """Engineer features for machine learning."""
    print("\n" + "="*60)
    print("STEP 4: Feature Engineering")
    print("="*60)
    
    # Create book-level features
    print("Creating book-level features...")
    book_features = create_book_level_features(
        df,
        text_column='preprocessed_text',
        book_column='book_title'
    )
    
    # Get target labels (majority vote per book)
    book_labels = df.groupby('book_title')['is_best_seller'].agg(
        lambda x: 1 if x.mean() > 0.5 else 0
    ).reset_index()
    
    # Merge features and labels
    book_features = book_features.merge(book_labels, on='book_title', how='left')
    
    # Save features
    book_features.to_csv('data/processed/book_features.csv', index=False)
    print(f"Saved book features to data/processed/book_features.csv")
    print(f"Total books: {len(book_features)}")
    print(f"Best sellers: {book_features['is_best_seller'].sum()}")
    
    return book_features


def train_models(X_train, X_test, y_train, y_test):
    """Train classification models."""
    print("\n" + "="*60)
    print("STEP 5: Model Training")
    print("="*60)
    
    # Initialize trainer
    trainer = ModelTrainer(random_state=42)
    
    # Handle class imbalance
    print("Handling class imbalance with SMOTE...")
    X_train_balanced, y_train_balanced = trainer.handle_class_imbalance(X_train, y_train)
    print(f"Training samples after SMOTE: {len(X_train_balanced)}")
    
    # Train models
    print("\nTraining models...")
    models = trainer.train_all_models(X_train_balanced, y_train_balanced, hyperparameter_tuning=False)
    
    # Select best model
    print("\nSelecting best model...")
    best_name, best_model = trainer.select_best_model(X_test, y_test)
    
    # Save best model
    trainer.save_model(best_model, f'models/{best_name}_best.joblib')
    
    return trainer, best_model, best_name


def evaluate_models(trainer, X_test, y_test):
    """Evaluate trained models."""
    print("\n" + "="*60)
    print("STEP 6: Model Evaluation")
    print("="*60)
    
    evaluator = ModelEvaluator()
    
    # Evaluate each model
    results = {}
    for name, model in trainer.models.items():
        print(f"\nEvaluating {name}...")
        result = evaluator.evaluate_model(model, X_test, y_test, model_name=name)
        evaluator.print_evaluation_results(result)
        results[name] = result['metrics']
        
        # Plot confusion matrix
        evaluator.plot_confusion_matrix(
            y_test,
            model.predict(X_test),
            labels=['Not Best Seller', 'Best Seller'],
            save_path=f'results/{name}_confusion_matrix.png'
        )
        
        # Plot ROC curve if probabilities available
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
            evaluator.plot_roc_curve(
                y_test,
                y_prob,
                model_name=name,
                save_path=f'results/{name}_roc_curve.png'
            )
    
    # Compare models
    print("\n" + "="*60)
    print("Model Comparison")
    print("="*60)
    comparison = evaluator.compare_models(results)
    print(comparison)
    comparison.to_csv('results/model_comparison.csv')
    
    return results


def main(use_sample=False, skip_collection=False):
    """
    Run the complete pipeline.
    
    Args:
        use_sample: Use sample data instead of API calls
        skip_collection: Skip data collection step
    """
    print("\n" + "="*60)
    print("NYT BEST SELLER PREDICTION PIPELINE")
    print("="*60)
    
    # Setup
    setup_directories()
    
    # Step 1: Data Collection
    if not skip_collection:
        collect_data(use_sample=use_sample)
    
    # Step 2: Preprocessing
    df = preprocess_data()
    
    # Step 3: Sentiment Analysis
    df = analyze_sentiment(df)
    
    # Step 4: Feature Engineering
    book_features = engineer_features(df)
    
    # Prepare features for modeling
    X, y = prepare_features_for_modeling(
        book_features,
        target_column='is_best_seller',
        drop_columns=['book_title']
    )
    
    # Fill NaN values
    X = np.nan_to_num(X, nan=0.0)
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target distribution: {np.bincount(y.astype(int))}")
    
    # Split data
    trainer = ModelTrainer(random_state=42)
    X_train, X_test, y_train, y_test = trainer.split_data(X, y, test_size=0.2)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Step 5: Model Training
    trainer, best_model, best_name = train_models(X_train, X_test, y_train, y_test)
    
    # Step 6: Model Evaluation
    evaluate_models(trainer, X_test, y_test)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print(f"\nBest model: {best_name}")
    print(f"Results saved in: results/")
    print(f"Model saved in: models/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NYT Best Seller Prediction Pipeline')
    parser.add_argument('--sample', action='store_true',
                       help='Use sample data instead of API calls')
    parser.add_argument('--skip-collection', action='store_true',
                       help='Skip data collection step')
    
    args = parser.parse_args()
    
    main(use_sample=args.sample, skip_collection=args.skip_collection)

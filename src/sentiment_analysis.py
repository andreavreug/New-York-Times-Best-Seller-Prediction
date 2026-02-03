"""
Sentiment analysis module for extracting sentiment features from book reviews.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
from textblob import TextBlob


class SentimentAnalyzer:
    """Sentiment analysis for book reviews."""
    
    def __init__(self):
        """Initialize the sentiment analyzer."""
        pass
    
    def get_sentiment_polarity(self, text: str) -> float:
        """
        Get sentiment polarity score using TextBlob.
        
        Args:
            text: Input text
            
        Returns:
            Polarity score between -1 (negative) and 1 (positive)
        """
        if not text or not isinstance(text, str):
            return 0.0
        
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return 0.0
    
    def get_sentiment_subjectivity(self, text: str) -> float:
        """
        Get sentiment subjectivity score using TextBlob.
        
        Args:
            text: Input text
            
        Returns:
            Subjectivity score between 0 (objective) and 1 (subjective)
        """
        if not text or not isinstance(text, str):
            return 0.0
        
        try:
            blob = TextBlob(text)
            return blob.sentiment.subjectivity
        except Exception as e:
            print(f"Error analyzing subjectivity: {e}")
            return 0.0
    
    def get_sentiment_category(self, polarity: float) -> str:
        """
        Categorize sentiment based on polarity score.
        
        Args:
            polarity: Sentiment polarity score
            
        Returns:
            Sentiment category: 'positive', 'negative', or 'neutral'
        """
        if polarity > 0.1:
            return 'positive'
        elif polarity < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Perform complete sentiment analysis on text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment features
        """
        polarity = self.get_sentiment_polarity(text)
        subjectivity = self.get_sentiment_subjectivity(text)
        category = self.get_sentiment_category(polarity)
        
        return {
            'polarity': polarity,
            'subjectivity': subjectivity,
            'category': category
        }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of sentiment analysis results
        """
        return [self.analyze_text(text) for text in texts]


def extract_review_sentiment_features(reviews_df: pd.DataFrame, 
                                      text_column: str = 'review_text') -> pd.DataFrame:
    """
    Extract sentiment features from a DataFrame of reviews.
    
    Args:
        reviews_df: DataFrame containing reviews
        text_column: Name of the column containing review text
        
    Returns:
        DataFrame with added sentiment features
    """
    analyzer = SentimentAnalyzer()
    
    # Analyze each review
    sentiments = analyzer.analyze_batch(reviews_df[text_column].fillna('').tolist())
    
    # Add sentiment features to DataFrame
    reviews_df['sentiment_polarity'] = [s['polarity'] for s in sentiments]
    reviews_df['sentiment_subjectivity'] = [s['subjectivity'] for s in sentiments]
    reviews_df['sentiment_category'] = [s['category'] for s in sentiments]
    
    return reviews_df


def aggregate_book_sentiments(reviews_df: pd.DataFrame,
                              book_column: str = 'book_title') -> pd.DataFrame:
    """
    Aggregate sentiment features by book.
    
    Args:
        reviews_df: DataFrame containing reviews with sentiment features
        book_column: Name of the column containing book titles
        
    Returns:
        DataFrame with aggregated sentiment features per book
    """
    agg_features = reviews_df.groupby(book_column).agg({
        'sentiment_polarity': ['mean', 'std', 'min', 'max'],
        'sentiment_subjectivity': ['mean', 'std'],
        'rating': ['mean', 'std', 'count']
    }).reset_index()
    
    # Flatten column names
    agg_features.columns = ['_'.join(col).strip('_') for col in agg_features.columns.values]
    
    # Rename book column
    agg_features.rename(columns={book_column: 'book_title'}, inplace=True)
    
    return agg_features


def calculate_sentiment_statistics(sentiments: List[float]) -> Dict[str, float]:
    """
    Calculate statistical measures for sentiment scores.
    
    Args:
        sentiments: List of sentiment polarity scores
        
    Returns:
        Dictionary with statistical measures
    """
    if not sentiments:
        return {
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'range': 0.0
        }
    
    sentiments_array = np.array(sentiments)
    
    return {
        'mean': float(np.mean(sentiments_array)),
        'median': float(np.median(sentiments_array)),
        'std': float(np.std(sentiments_array)),
        'min': float(np.min(sentiments_array)),
        'max': float(np.max(sentiments_array)),
        'range': float(np.max(sentiments_array) - np.min(sentiments_array))
    }

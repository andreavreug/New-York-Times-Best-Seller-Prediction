"""
Feature engineering module for creating ML features from book reviews.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler


class FeatureEngineering:
    """Feature engineering for book review prediction."""
    
    def __init__(self, max_features: int = 500, ngram_range: Tuple[int, int] = (1, 2)):
        """
        Initialize feature engineering.
        
        Args:
            max_features: Maximum number of TF-IDF features
            ngram_range: Range of n-grams to extract
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.tfidf_vectorizer = None
        self.scaler = StandardScaler()
    
    def create_tfidf_features(self, texts: List[str], fit: bool = True) -> np.ndarray:
        """
        Create TF-IDF features from texts.
        
        Args:
            texts: List of text documents
            fit: Whether to fit the vectorizer (True for training, False for testing)
            
        Returns:
            TF-IDF feature matrix
        """
        if fit:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=2,
                max_df=0.95,
                strip_accents='unicode',
                lowercase=True
            )
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        else:
            if self.tfidf_vectorizer is None:
                raise ValueError("Vectorizer must be fitted before transform")
            tfidf_matrix = self.tfidf_vectorizer.transform(texts)
        
        return tfidf_matrix.toarray()
    
    def create_count_features(self, texts: List[str]) -> np.ndarray:
        """
        Create count-based features from texts.
        
        Args:
            texts: List of text documents
            
        Returns:
            Count feature matrix
        """
        vectorizer = CountVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=2,
            max_df=0.95
        )
        count_matrix = vectorizer.fit_transform(texts)
        return count_matrix.toarray()
    
    def create_text_statistics(self, texts: List[str]) -> np.ndarray:
        """
        Create statistical features from text.
        
        Args:
            texts: List of text documents
            
        Returns:
            Array of statistical features
        """
        features = []
        
        for text in texts:
            if not text or not isinstance(text, str):
                features.append([0, 0, 0, 0, 0])
                continue
            
            # Character count
            char_count = len(text)
            
            # Word count
            word_count = len(text.split())
            
            # Average word length
            avg_word_len = np.mean([len(word) for word in text.split()]) if word_count > 0 else 0
            
            # Sentence count (approximation)
            sentence_count = text.count('.') + text.count('!') + text.count('?')
            sentence_count = max(1, sentence_count)
            
            # Average sentence length
            avg_sentence_len = word_count / sentence_count if sentence_count > 0 else 0
            
            features.append([
                char_count,
                word_count,
                avg_word_len,
                sentence_count,
                avg_sentence_len
            ])
        
        return np.array(features)
    
    def create_aggregated_features(self, df: pd.DataFrame,
                                   group_column: str = 'book_title') -> pd.DataFrame:
        """
        Create aggregated features by grouping reviews.
        
        Args:
            df: DataFrame with review-level features
            group_column: Column to group by (e.g., book_title)
            
        Returns:
            DataFrame with aggregated features
        """
        # Identify numeric columns for aggregation
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove the target variable if present
        if 'is_best_seller' in numeric_cols:
            numeric_cols.remove('is_best_seller')
        
        # Aggregate features
        agg_dict = {col: ['mean', 'std', 'min', 'max'] for col in numeric_cols}
        
        # Add count of reviews
        agg_dict[df.columns[0]] = 'count'
        
        aggregated = df.groupby(group_column).agg(agg_dict).reset_index()
        
        # Flatten column names
        aggregated.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                             for col in aggregated.columns.values]
        
        return aggregated
    
    def combine_features(self, *feature_arrays: np.ndarray) -> np.ndarray:
        """
        Combine multiple feature arrays.
        
        Args:
            *feature_arrays: Variable number of feature arrays
            
        Returns:
            Combined feature array
        """
        return np.hstack(feature_arrays)
    
    def scale_features(self, features: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Scale features using StandardScaler.
        
        Args:
            features: Feature array
            fit: Whether to fit the scaler
            
        Returns:
            Scaled features
        """
        if fit:
            return self.scaler.fit_transform(features)
        else:
            return self.scaler.transform(features)


def create_book_level_features(reviews_df: pd.DataFrame,
                               text_column: str = 'preprocessed_text',
                               book_column: str = 'book_title') -> pd.DataFrame:
    """
    Create book-level features from review-level data.
    
    Args:
        reviews_df: DataFrame with review-level data
        text_column: Name of the text column
        book_column: Name of the book identifier column
        
    Returns:
        DataFrame with book-level features
    """
    # Aggregate sentiment and rating features
    sentiment_cols = [col for col in reviews_df.columns if 'sentiment' in col or col == 'rating']
    
    agg_dict = {}
    for col in sentiment_cols:
        if pd.api.types.is_numeric_dtype(reviews_df[col]):
            agg_dict[col] = ['mean', 'std', 'min', 'max']
    
    # Add review count
    agg_dict[text_column] = 'count'
    
    book_features = reviews_df.groupby(book_column).agg(agg_dict).reset_index()
    
    # Flatten columns
    book_features.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col
                            for col in book_features.columns.values]
    
    # Rename book column
    if book_column + '_' in book_features.columns[0]:
        book_features.rename(columns={book_features.columns[0]: 'book_title'}, inplace=True)
    
    return book_features


def prepare_features_for_modeling(df: pd.DataFrame, 
                                  target_column: str = 'is_best_seller',
                                  drop_columns: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare features and target for modeling.
    
    Args:
        df: DataFrame with features
        target_column: Name of the target column
        drop_columns: List of columns to drop
        
    Returns:
        Tuple of (features, target)
    """
    if drop_columns is None:
        drop_columns = []
    
    # Get target
    y = df[target_column].values if target_column in df.columns else None
    
    # Drop non-feature columns
    feature_cols = df.columns.tolist()
    cols_to_drop = [target_column] + drop_columns
    
    for col in cols_to_drop:
        if col in feature_cols:
            feature_cols.remove(col)
    
    # Get features
    X = df[feature_cols].select_dtypes(include=[np.number]).values
    
    return X, y

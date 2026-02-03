"""
Text preprocessing module for cleaning and normalizing book reviews.
"""

import re
import string
from typing import List, Optional, Union
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)


class TextPreprocessor:
    """Text preprocessing pipeline for book reviews."""
    
    def __init__(self, remove_stopwords: bool = True, lemmatize: bool = True):
        """
        Initialize the text preprocessor.
        
        Args:
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to lemmatize tokens
        """
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing special characters and normalizing.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        try:
            tokens = word_tokenize(text)
            return tokens
        except Exception as e:
            print(f"Error tokenizing text: {e}")
            return text.split()
    
    def remove_punctuation(self, tokens: List[str]) -> List[str]:
        """
        Remove punctuation from tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Tokens without punctuation
        """
        return [token for token in tokens if token not in string.punctuation]
    
    def remove_stop_words(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Tokens without stopwords
        """
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens to their base form.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess(self, text: str, return_string: bool = True) -> Union[str, List[str]]:
        """
        Complete preprocessing pipeline.
        
        Args:
            text: Input text
            return_string: Whether to return a string (True) or list of tokens (False)
            
        Returns:
            Preprocessed text or tokens
        """
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove punctuation
        tokens = self.remove_punctuation(tokens)
        
        # Remove stopwords if enabled
        if self.remove_stopwords:
            tokens = self.remove_stop_words(tokens)
        
        # Lemmatize if enabled
        if self.lemmatize:
            tokens = self.lemmatize_tokens(tokens)
        
        # Filter out empty tokens
        tokens = [token for token in tokens if token.strip()]
        
        if return_string:
            return ' '.join(tokens)
        return tokens
    
    def preprocess_batch(self, texts: List[str], return_string: bool = True) -> List[Union[str, List[str]]]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of input texts
            return_string: Whether to return strings or token lists
            
        Returns:
            List of preprocessed texts or token lists
        """
        return [self.preprocess(text, return_string) for text in texts]


def preprocess_reviews(reviews: List[str], **kwargs) -> List[str]:
    """
    Convenience function to preprocess a list of reviews.
    
    Args:
        reviews: List of review texts
        **kwargs: Additional arguments for TextPreprocessor
        
    Returns:
        List of preprocessed reviews
    """
    preprocessor = TextPreprocessor(**kwargs)
    return preprocessor.preprocess_batch(reviews)

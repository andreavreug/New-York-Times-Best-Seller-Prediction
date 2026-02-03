"""
Example script demonstrating individual module usage.
This shows how to use each module independently.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import TextPreprocessor
from sentiment_analysis import SentimentAnalyzer

# Example 1: Text Preprocessing
print("="*60)
print("Example 1: Text Preprocessing")
print("="*60)

preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True)
sample_text = "This is a great book! I absolutely loved it and couldn't put it down."

clean_text = preprocessor.preprocess(sample_text)
print(f"Original: {sample_text}")
print(f"Preprocessed: {clean_text}")

# Example 2: Sentiment Analysis
print("\n" + "="*60)
print("Example 2: Sentiment Analysis")
print("="*60)

analyzer = SentimentAnalyzer()

texts = [
    "This book is absolutely amazing! I loved it!",
    "Terrible book, waste of time.",
    "It was okay, nothing special."
]

for text in texts:
    sentiment = analyzer.analyze_text(text)
    print(f"\nText: {text}")
    print(f"Polarity: {sentiment['polarity']:.3f}")
    print(f"Subjectivity: {sentiment['subjectivity']:.3f}")
    print(f"Category: {sentiment['category']}")

print("\n" + "="*60)
print("Examples completed successfully!")
print("="*60)

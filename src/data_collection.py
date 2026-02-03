"""
Data collection module for scraping book reviews from Hardcover
and NYT Best Seller lists.
"""

import os
import time
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import requests
from dotenv import load_dotenv

load_dotenv()


class HardcoverScraper:
    """Scraper for Hardcover GraphQL API to fetch book reviews."""
    
    def __init__(self):
        self.api_url = os.getenv('HARDCOVER_API_URL', 'https://hardcover.app/graphql')
        self.headers = {
            'Content-Type': 'application/json',
        }
    
    def fetch_book_reviews(self, book_title: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Fetch reviews for a specific book from Hardcover.
        
        Args:
            book_title: Title of the book
            limit: Maximum number of reviews to fetch
            
        Returns:
            List of review dictionaries
        """
        query = """
        query SearchBooks($title: String!) {
            books(where: {title: {_ilike: $title}}, limit: 1) {
                id
                title
                reviews(limit: %d) {
                    id
                    body
                    rating
                    created_at
                    user {
                        name
                    }
                }
            }
        }
        """ % limit
        
        variables = {"title": f"%{book_title}%"}
        
        try:
            response = requests.post(
                self.api_url,
                json={'query': query, 'variables': variables},
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            if 'data' in data and data['data'].get('books'):
                books = data['data']['books']
                if books:
                    return books[0].get('reviews', [])
            return []
        except Exception as e:
            print(f"Error fetching reviews for {book_title}: {e}")
            return []
    
    def fetch_multiple_books(self, book_titles: List[str], delay: float = 1.0) -> Dict[str, List[Dict]]:
        """
        Fetch reviews for multiple books.
        
        Args:
            book_titles: List of book titles
            delay: Delay between requests in seconds
            
        Returns:
            Dictionary mapping book titles to their reviews
        """
        results = {}
        for title in book_titles:
            reviews = self.fetch_book_reviews(title)
            results[title] = reviews
            time.sleep(delay)  # Rate limiting
        return results


class NYTBestSellerScraper:
    """Scraper for NYT Best Seller lists API."""
    
    def __init__(self):
        self.api_key = os.getenv('NYT_API_KEY')
        if not self.api_key:
            print("Warning: NYT_API_KEY not found in environment variables")
        self.base_url = 'https://api.nytimes.com/svc/books/v3'
    
    def fetch_best_sellers(self, list_name: str = 'combined-print-and-e-book-fiction',
                          date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Fetch NYT Best Seller list for a specific date.
        
        Args:
            list_name: Name of the best seller list
            date: Date in YYYY-MM-DD format (default: current date)
            
        Returns:
            List of best seller books
        """
        if not self.api_key:
            print("Cannot fetch data without NYT API key")
            return []
        
        endpoint = f"{self.base_url}/lists/{date or 'current'}/{list_name}.json"
        params = {'api-key': self.api_key}
        
        try:
            response = requests.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'results' in data and 'books' in data['results']:
                return data['results']['books']
            return []
        except Exception as e:
            print(f"Error fetching NYT best sellers: {e}")
            return []
    
    def fetch_historical_lists(self, list_name: str = 'combined-print-and-e-book-fiction',
                              weeks: int = 12) -> List[Dict[str, Any]]:
        """
        Fetch historical best seller lists.
        
        Args:
            list_name: Name of the best seller list
            weeks: Number of weeks to go back
            
        Returns:
            List of all books from historical lists
        """
        all_books = []
        current_date = datetime.now()
        
        for i in range(weeks):
            date = current_date - timedelta(weeks=i)
            date_str = date.strftime('%Y-%m-%d')
            books = self.fetch_best_sellers(list_name, date_str)
            
            for book in books:
                book['list_date'] = date_str
                all_books.append(book)
            
            time.sleep(6)  # NYT API rate limit: 10 requests per minute
        
        return all_books


def create_labeled_dataset(best_sellers: List[Dict[str, Any]],
                          reviews_data: Dict[str, List[Dict]]) -> List[Dict[str, Any]]:
    """
    Create a labeled dataset by matching books with their reviews.
    
    Args:
        best_sellers: List of NYT best seller books
        reviews_data: Dictionary of book titles to reviews
        
    Returns:
        List of labeled examples with reviews and best seller status
    """
    labeled_data = []
    best_seller_titles = {book['title'].lower() for book in best_sellers}
    
    for title, reviews in reviews_data.items():
        is_best_seller = title.lower() in best_seller_titles
        
        for review in reviews:
            labeled_data.append({
                'book_title': title,
                'review_text': review.get('body', ''),
                'rating': review.get('rating', 0),
                'is_best_seller': 1 if is_best_seller else 0,
                'created_at': review.get('created_at', '')
            })
    
    return labeled_data


def save_data(data: List[Dict[str, Any]], filepath: str):
    """Save data to JSON file."""
    import numpy as np
    
    # Convert numpy types to Python types
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Convert all values in the data
    converted_data = []
    for item in data:
        converted_item = {k: convert_types(v) for k, v in item.items()}
        converted_data.append(converted_item)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(data)} records to {filepath}")


def load_data(filepath: str) -> List[Dict[str, Any]]:
    """Load data from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

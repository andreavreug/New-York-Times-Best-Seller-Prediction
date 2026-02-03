# New York Times Best Seller Prediction

A text analytics project to predict whether a book becomes a New York Times Best Seller using sentiment analysis from early reader reviews. Reviews are scraped from Hardcover (GraphQL API) and matched with NYT Best Seller lists to label books as best sellers or not.

## Project Overview

This project implements a complete machine learning pipeline for predicting NYT Best Sellers:

1. **Data Collection**: Scrapes book reviews from Hardcover API and NYT Best Seller lists
2. **Text Preprocessing**: Cleans and normalizes review text (tokenization, lemmatization, stopword removal)
3. **Sentiment Analysis**: Extracts sentiment features using TextBlob (polarity, subjectivity)
4. **Feature Engineering**: Creates TF-IDF features, sentiment statistics, and aggregated review metrics
5. **Supervised Classification**: Trains multiple models (Logistic Regression, Random Forest, Gradient Boosting, SVM)
6. **Model Evaluation**: Comprehensive evaluation with accuracy, precision, recall, F1-score, ROC-AUC curves

## Project Structure

```
New-York-Times-Best-Seller-Prediction/
├── src/
│   ├── __init__.py
│   ├── data_collection.py      # API scrapers for Hardcover and NYT
│   ├── preprocessing.py         # Text preprocessing pipeline
│   ├── sentiment_analysis.py    # Sentiment feature extraction
│   ├── feature_engineering.py   # Feature creation and aggregation
│   ├── model_training.py        # Model training and selection
│   └── model_evaluation.py      # Model evaluation and metrics
├── data/
│   ├── raw/                     # Raw scraped data
│   └── processed/               # Preprocessed data
├── models/                      # Saved trained models
├── results/                     # Evaluation results and plots
├── notebooks/                   # Jupyter notebooks (optional)
├── main.py                      # Main pipeline script
├── requirements.txt             # Python dependencies
├── .env.example                 # Example environment variables
├── .gitignore                   # Git ignore file
└── README.md                    # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/andreavreug/New-York-Times-Best-Seller-Prediction.git
cd New-York-Times-Best-Seller-Prediction
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required NLTK data (will be downloaded automatically on first run):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

## Configuration

1. Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

2. Add your API keys to `.env`:
```
NYT_API_KEY=your_nyt_api_key_here
HARDCOVER_API_URL=https://hardcover.app/graphql
```

To get an NYT API key:
- Visit https://developer.nytimes.com/
- Create an account and register for the Books API

## Usage

### Quick Start with Sample Data

Run the pipeline with sample data (no API keys required):

```bash
python main.py --sample
```

This will:
- Generate sample book reviews
- Preprocess the text
- Extract sentiment features
- Train and evaluate multiple models
- Save results to `results/` directory

### Full Pipeline with Real Data

Once you have API keys configured:

```bash
python main.py
```

This will collect real data from Hardcover and NYT APIs.

### Command-line Options

- `--sample`: Use sample data instead of API calls (for demonstration)
- `--skip-collection`: Skip data collection and use existing data files

### Using Individual Modules

You can also use individual modules in your own scripts:

```python
from src.preprocessing import TextPreprocessor
from src.sentiment_analysis import SentimentAnalyzer
from src.feature_engineering import FeatureEngineering
from src.model_training import ModelTrainer
from src.model_evaluation import ModelEvaluator

# Preprocess text
preprocessor = TextPreprocessor()
clean_text = preprocessor.preprocess("This is a great book!")

# Analyze sentiment
analyzer = SentimentAnalyzer()
sentiment = analyzer.analyze_text("This is a great book!")
print(sentiment)  # {'polarity': 0.8, 'subjectivity': 0.75, 'category': 'positive'}

# Train a model
trainer = ModelTrainer()
X_train, X_test, y_train, y_test = trainer.split_data(X, y)
model = trainer.train_random_forest(X_train, y_train)

# Evaluate model
evaluator = ModelEvaluator()
results = evaluator.evaluate_model(model, X_test, y_test)
```

## Features

### Text Preprocessing
- Lowercase conversion
- URL and email removal
- HTML tag removal
- Special character cleaning
- Tokenization
- Stopword removal
- Lemmatization

### Sentiment Analysis
- Polarity score (-1 to 1)
- Subjectivity score (0 to 1)
- Sentiment categorization (positive/negative/neutral)
- Aggregated sentiment statistics per book

### Feature Engineering
- TF-IDF vectorization (uni-grams and bi-grams)
- Text statistics (word count, character count, etc.)
- Sentiment features (mean, std, min, max)
- Rating statistics
- Review count per book

### Models
- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)
- With SMOTE for class imbalance handling
- Optional hyperparameter tuning

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- Classification Report
- ROC Curve plots
- Precision-Recall Curve plots

## Results

After running the pipeline, you'll find:

- **models/**: Trained model files (`.joblib` format)
- **results/**: 
  - `model_comparison.csv`: Comparison of all models
  - `*_confusion_matrix.png`: Confusion matrix plots
  - `*_roc_curve.png`: ROC curve plots
- **data/processed/**: 
  - `preprocessed_reviews.csv`: Preprocessed review texts
  - `reviews_with_sentiment.csv`: Reviews with sentiment scores
  - `book_features.csv`: Aggregated book-level features

## Example Output

```
======================================================================
NYT BEST SELLER PREDICTION PIPELINE
======================================================================

STEP 1: Data Collection
Created 90 sample reviews

STEP 2: Text Preprocessing
Loaded 90 reviews
Preprocessing review texts...
Saved preprocessed data to data/processed/preprocessed_reviews.csv

STEP 3: Sentiment Analysis
Analyzing sentiment...
Saved sentiment data to data/processed/reviews_with_sentiment.csv

STEP 4: Feature Engineering
Creating book-level features...
Total books: 10
Best sellers: 5

STEP 5: Model Training
Training models...
logistic_regression: F1-Score = 0.8571
random_forest: F1-Score = 0.8000
gradient_boosting: F1-Score = 0.8571
svm: F1-Score = 0.6667

Best model: logistic_regression (F1-Score: 0.8571)

STEP 6: Model Evaluation
Model comparison saved to results/model_comparison.csv

PIPELINE COMPLETE!
Best model: logistic_regression
Results saved in: results/
Model saved in: models/
```

## Dependencies

- Python 3.8+
- pandas: Data manipulation
- numpy: Numerical operations
- nltk: Natural language processing
- textblob: Sentiment analysis
- scikit-learn: Machine learning
- imbalanced-learn: SMOTE for class imbalance
- matplotlib: Visualization
- seaborn: Statistical visualization
- requests: API calls
- python-dotenv: Environment variable management

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is available for educational and research purposes.

## Acknowledgments

- New York Times API for best seller data
- Hardcover for book review data
- TextBlob for sentiment analysis
- scikit-learn for machine learning tools
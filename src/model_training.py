"""
Model training module for supervised classification of NYT Best Sellers.
"""

import numpy as np
from typing import Dict, Any, Tuple, List
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from imblearn.over_sampling import SMOTE
import joblib


class ModelTrainer:
    """Train and manage classification models."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the model trainer.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   test_size: float = 0.2,
                   stratify: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of data for testing
            stratify: Whether to stratify the split
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y if stratify else None
        )
    
    def handle_class_imbalance(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Handle class imbalance using SMOTE.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Balanced X_train, y_train
        """
        smote = SMOTE(random_state=self.random_state)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        return X_resampled, y_resampled
    
    def train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray,
                                  hyperparameter_tuning: bool = False) -> LogisticRegression:
        """
        Train a Logistic Regression model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            
        Returns:
            Trained model
        """
        if hyperparameter_tuning:
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
            model = GridSearchCV(
                LogisticRegression(random_state=self.random_state, max_iter=1000),
                param_grid,
                cv=5,
                scoring='f1',
                n_jobs=-1
            )
        else:
            model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        
        model.fit(X_train, y_train)
        self.models['logistic_regression'] = model
        return model
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                           hyperparameter_tuning: bool = False) -> RandomForestClassifier:
        """
        Train a Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            
        Returns:
            Trained model
        """
        if hyperparameter_tuning:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = GridSearchCV(
                RandomForestClassifier(random_state=self.random_state),
                param_grid,
                cv=5,
                scoring='f1',
                n_jobs=-1
            )
        else:
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            )
        
        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        return model
    
    def train_gradient_boosting(self, X_train: np.ndarray, y_train: np.ndarray,
                               hyperparameter_tuning: bool = False) -> GradientBoostingClassifier:
        """
        Train a Gradient Boosting model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            
        Returns:
            Trained model
        """
        if hyperparameter_tuning:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            }
            model = GridSearchCV(
                GradientBoostingClassifier(random_state=self.random_state),
                param_grid,
                cv=5,
                scoring='f1',
                n_jobs=-1
            )
        else:
            model = GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.random_state
            )
        
        model.fit(X_train, y_train)
        self.models['gradient_boosting'] = model
        return model
    
    def train_svm(self, X_train: np.ndarray, y_train: np.ndarray,
                  hyperparameter_tuning: bool = False) -> SVC:
        """
        Train an SVM model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            
        Returns:
            Trained model
        """
        if hyperparameter_tuning:
            param_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
            model = GridSearchCV(
                SVC(random_state=self.random_state, probability=True),
                param_grid,
                cv=5,
                scoring='f1',
                n_jobs=-1
            )
        else:
            model = SVC(random_state=self.random_state, probability=True)
        
        model.fit(X_train, y_train)
        self.models['svm'] = model
        return model
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                        hyperparameter_tuning: bool = False) -> Dict[str, Any]:
        """
        Train all available models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary of trained models
        """
        print("Training Logistic Regression...")
        self.train_logistic_regression(X_train, y_train, hyperparameter_tuning)
        
        print("Training Random Forest...")
        self.train_random_forest(X_train, y_train, hyperparameter_tuning)
        
        print("Training Gradient Boosting...")
        self.train_gradient_boosting(X_train, y_train, hyperparameter_tuning)
        
        print("Training SVM...")
        self.train_svm(X_train, y_train, hyperparameter_tuning)
        
        return self.models
    
    def cross_validate_model(self, model: Any, X: np.ndarray, y: np.ndarray,
                            cv: int = 5, scoring: str = 'f1') -> Dict[str, float]:
        """
        Perform cross-validation on a model.
        
        Args:
            model: Model to validate
            X: Features
            y: Labels
            cv: Number of cross-validation folds
            scoring: Scoring metric
            
        Returns:
            Dictionary with mean and std of scores
        """
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        return {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'scores': scores
        }
    
    def select_best_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[str, Any]:
        """
        Select the best model based on test performance.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Tuple of (best_model_name, best_model)
        """
        from model_evaluation import ModelEvaluator
        
        evaluator = ModelEvaluator()
        best_score = -1
        best_name = None
        best_model = None
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            metrics = evaluator.calculate_metrics(y_test, y_pred)
            f1_score = metrics['f1']
            
            print(f"{name}: F1-Score = {f1_score:.4f}")
            
            if f1_score > best_score:
                best_score = f1_score
                best_name = name
                best_model = model
        
        self.best_model_name = best_name
        self.best_model = best_model
        
        print(f"\nBest model: {best_name} (F1-Score: {best_score:.4f})")
        return best_name, best_model
    
    def save_model(self, model: Any, filepath: str):
        """
        Save a trained model to disk.
        
        Args:
            model: Model to save
            filepath: Path to save the model
        """
        joblib.dump(model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> Any:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model
        """
        model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model

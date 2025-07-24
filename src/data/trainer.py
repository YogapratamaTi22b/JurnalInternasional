import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import yaml
import logging
from pathlib import Path
import joblib
import time

class MalwareTrainer:
    """
    Train machine learning models for malware classification
    """
    
    def __init__(self, config_path="config/config.yaml"):
        """Initialize the trainer with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Setup paths
        self.processed_data_path = Path(self.config['data']['processed_data_path'])
        self.models_path = Path(self.config['output']['models_path'])
        
        # Create model directories
        self.models_path.mkdir(parents=True, exist_ok=True)
        (self.models_path / "random_forest").mkdir(parents=True, exist_ok=True)
        (self.models_path / "decision_tree").mkdir(parents=True, exist_ok=True)
        (self.models_path / "svm").mkdir(parents=True, exist_ok=True)
        (self.models_path / "ensemble").mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize models with config parameters
        self.models = {
            'random_forest': RandomForestClassifier(**self.config['models']['random_forest']),
            'decision_tree': DecisionTreeClassifier(**self.config['models']['decision_tree']),
            'svm': SVC(**self.config['models']['svm'], probability=True)  # probability=True for ensemble
        }
        
        # Store trained models
        self.trained_models = {}
        
    def load_training_data(self):
        """Load prepared training data"""
        try:
            # Load numpy arrays
            data = np.load(self.processed_data_path / "scaled_features.npz")
            X_train = data['X_train']
            X_test = data['X_test']
            y_train = data['y_train']
            y_test = data['y_test']
            feature_names = data['feature_names']
            
            # Load preprocessors
            with open(self.processed_data_path / "label_encoder.pkl", 'rb') as f:
                self.label_encoder = pickle.load(f)
                
            with open(self.processed_data_path / "scaler.pkl", 'rb') as f:
                self.scaler = pickle.load(f)
            
            self.logger.info(f"Training data loaded: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
            self.logger.info(f"Features: {X_train.shape[1]}, Classes: {len(self.label_encoder.classes_)}")
            
            return X_train, X_test, y_train, y_test, feature_names
            
        except FileNotFoundError as e:
            self.logger.error(f"Training data not found: {e}")
            self.logger.info("Run data processing first")
            return None
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train Random Forest model"""
        self.logger.info("Training Random Forest model...")
        
        start_time = time.time()
        
        # Train the model
        rf_model = self.models['random_forest']
        rf_model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # Make predictions
        train_predictions = rf_model.predict(X_train)
        test_predictions = rf_model.predict(X_test)
        
        # Calculate accuracies
        train_accuracy = accuracy_score(y_train, train_predictions)
        test_accuracy = accuracy_score(y_test, test_predictions)
        
        # Cross-validation
        cv_scores = cross_val_score(rf_model, X_train, y_train, 
                                  cv=self.config['models']['cross_validation_folds'])
        
        self.logger.info(f"Random Forest - Train Accuracy: {train_accuracy:.4f}")
        self.logger.info(f"Random Forest - Test Accuracy: {test_accuracy:.4f}")
        self.logger.info(f"Random Forest - CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        self.logger.info(f"Random Forest - Training Time: {training_time:.2f} seconds")
        
        # Save model
        rf_path = self.models_path / "random_forest"
        joblib.dump(rf_model, rf_path / "rf_model.pkl")
        joblib.dump(self.scaler, rf_path / "rf_scaler.pkl")
        
        # Store results
        self.trained_models['random_forest'] = {
            'model': rf_model,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'training_time': training_time,
            'predictions': test_predictions
        }
        
        return rf_model
    
    def train_decision_tree(self, X_train, y_train, X_test, y_test):
        """Train Decision Tree model"""
        self.logger.info("Training Decision Tree model...")
        
        start_time = time.time()
        
        # Train the model
        dt_model = self.models['decision_tree']
        dt_model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # Make predictions
        train_predictions = dt_model.predict(X_train)
        test_predictions = dt_model.predict(X_test)
        
        # Calculate accuracies
        train_accuracy = accuracy_score(y_train, train_predictions)
        test_accuracy = accuracy_score(y_test, test_predictions)
        
        # Cross-validation
        cv_scores = cross_val_score(dt_model, X_train, y_train, 
                                  cv=self.config['models']['cross_validation_folds'])
        
        self.logger.info(f"Decision Tree - Train Accuracy: {train_accuracy:.4f}")
        self.logger.info(f"Decision Tree - Test Accuracy: {test_accuracy:.4f}")
        self.logger.info(f"Decision Tree - CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        self.logger.info(f"Decision Tree - Training Time: {training_time:.2f} seconds")
        
        # Save model
        dt_path = self.models_path / "decision_tree"
        joblib.dump(dt_model, dt_path / "dt_model.pkl")
        joblib.dump(self.scaler, dt_path / "dt_scaler.pkl")
        
        # Store results
        self.trained_models['decision_tree'] = {
            'model': dt_model,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'training_time': training_time,
            'predictions': test_predictions
        }
        
        return dt_model
    
    def train_svm(self, X_train, y_train, X_test, y_test):
        """Train SVM model"""
        self.logger.info("Training SVM model...")
        
        start_time = time.time()
        
        # Train the model
        svm_model = self.models['svm']
        svm_model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # Make predictions
        train_predictions = svm_model.predict(X_train)
        test_predictions = svm_model.predict(X_test)
        
        # Calculate accuracies
        train_accuracy = accuracy_score(y_train, train_predictions)
        test_accuracy = accuracy_score(y_test, test_predictions)
        
        # Cross-validation
        cv_scores = cross_val_score(svm_model, X_train, y_train, 
                                  cv=self.config['models']['cross_validation_folds'])
        
        self.logger.info(f"SVM - Train Accuracy: {train_accuracy:.4f}")
        self.logger.info(f"SVM - Test Accuracy: {test_accuracy:.4f}")
        self.logger.info(f"SVM - CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        self.logger.info(f"SVM - Training Time: {training_time:.2f} seconds")
        
        # Save model
        svm_path = self.models_path / "svm"
        joblib.dump(svm_model, svm_path / "svm_model.pkl")
        joblib.dump(self.scaler, svm_path / "svm_scaler.pkl") 
        
        # Store results
        self.trained_models['svm'] = {
            'model': svm_model,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'training_time': training_time,
            'predictions': test_predictions
        }
        
        return svm_model
    
    def train_ensemble_models(self, X_train, y_train, X_test, y_test):
        """Train ensemble models (Voting and Stacking)"""
        self.logger.info("Training ensemble models...")
        
        from sklearn.ensemble import VotingClassifier, StackingClassifier
        from sklearn.linear_model import LogisticRegression
        
        # Ensure all base models are trained
        if len(self.trained_models) < 3:
            self.logger.warning("Not all base models are trained. Training them first...")
            self.train_all_models(X_train, y_train, X_test, y_test)
        
        # Voting Classifier
        voting_clf = VotingClassifier(
            estimators=[
                ('rf', self.trained_models['random_forest']['model']),
                ('dt', self.trained_models['decision_tree']['model']),
                ('svm', self.trained_models['svm']['model'])
            ],
            voting='soft'  # Use probability predictions
        )
        
        start_time = time.time()
        voting_clf.fit(X_train, y_train)
        voting_training_time = time.time() - start_time
        
        # Voting predictions
        voting_test_pred = voting_clf.predict(X_test)
        voting_accuracy = accuracy_score(y_test, voting_test_pred)
        
        # Stacking Classifier
        stacking_clf = StackingClassifier(
            estimators=[
                ('rf', self.trained_models['random_forest']['model']),
                ('dt', self.trained_models['decision_tree']['model']),
                ('svm', self.trained_models['svm']['model'])
            ],
            final_estimator=LogisticRegression(random_state=42),
            cv=3
        )
        
        start_time = time.time()
        stacking_clf.fit(X_train, y_train)
        stacking_training_time = time.time() - start_time
        
        # Stacking predictions
        stacking_test_pred = stacking_clf.predict(X_test)
        stacking_accuracy = accuracy_score(y_test, stacking_test_pred)
        
        self.logger.info(f"Voting Classifier - Test Accuracy: {voting_accuracy:.4f}")
        self.logger.info(f"Stacking Classifier - Test Accuracy: {stacking_accuracy:.4f}")
        
        # Save ensemble models
        ensemble_path = self.models_path / "ensemble"
        joblib.dump(voting_clf, ensemble_path / "voting_classifier.pkl")
        joblib.dump(stacking_clf, ensemble_path / "stacking_classifier.pkl")
        
        # Store results
        self.trained_models['voting'] = {
            'model': voting_clf,
            'test_accuracy': voting_accuracy,
            'training_time': voting_training_time,
            'predictions': voting_test_pred
        }
        
        self.trained_models['stacking'] = {
            'model': stacking_clf,
            'test_accuracy': stacking_accuracy,
            'training_time': stacking_training_time,
            'predictions': stacking_test_pred
        }
        
        return voting_clf, stacking_clf
    
    def hyperparameter_tuning(self, X_train, y_train, model_name='random_forest'):
        """Perform hyperparameter tuning for specified model"""
        self.logger.info(f"Performing hyperparameter tuning for {model_name}...")
        
        if model_name == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestClassifier(random_state=42)
            
        elif model_name == 'decision_tree':
            param_grid = {
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy']
            }
            model = DecisionTreeClassifier(random_state=42)
            
        elif model_name == 'svm':
            param_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
            model = SVC(random_state=42, probability=True)
        
        else:
            self.logger.error(f"Unknown model: {model_name}")
            return None
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            model, param_grid, 
            cv=self.config['models']['cross_validation_folds'], 
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        tuning_time = time.time() - start_time
        
        self.logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
        self.logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        self.logger.info(f"Hyperparameter tuning time: {tuning_time:.2f} seconds")
        
        # Update model with best parameters
        self.models[model_name] = grid_search.best_estimator_
        
        return grid_search.best_estimator_
    
    def train_all_models(self, X_train=None, y_train=None, X_test=None, y_test=None):
        """Train all models"""
        if X_train is None:
            # Load training data
            data = self.load_training_data()
            if data is None:
                return None
            X_train, X_test, y_train, y_test, feature_names = data
        
        self.logger.info("Starting training of all models...")
        
        # Train individual models
        self.train_random_forest(X_train, y_train, X_test, y_test)
        self.train_decision_tree(X_train, y_train, X_test, y_test)
        self.train_svm(X_train, y_train, X_test, y_test)
        
        # Train ensemble models
        self.train_ensemble_models(X_train, y_train, X_test, y_test)
        
        self.logger.info("All models trained successfully!")
        
        return self.trained_models
    
    def get_model_comparison(self):
        """Get comparison of all trained models"""
        if not self.trained_models:
            self.logger.warning("No models trained yet")
            return None
        
        comparison_data = []
        
        for name, model_info in self.trained_models.items():
            comparison_data.append({
                'Model': name.replace('_', ' ').title(),
                'Test Accuracy': model_info.get('test_accuracy', 0),
                'CV Mean': model_info.get('cv_mean', 0),
                'CV Std': model_info.get('cv_std', 0),
                'Training Time (s)': model_info.get('training_time', 0)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Test Accuracy', ascending=False)
        
        # Save comparison
        results_path = Path(self.config['output']['results_path'])
        results_path.mkdir(parents=True, exist_ok=True)
        benchmarks_path = results_path / "benchmarks"
        benchmarks_path.mkdir(parents=True, exist_ok=True)
        
        comparison_df.to_csv(benchmarks_path / "model_comparison.csv", index=False)
        
        return comparison_df
    
    def save_detailed_results(self, y_test):
        """Save detailed results for each model"""
        if not self.trained_models:
            return
        
        results_path = Path(self.config['output']['results_path'])
        benchmarks_path = results_path / "benchmarks"
        
        detailed_results = {}
        
        for name, model_info in self.trained_models.items():
            if 'predictions' in model_info:
                predictions = model_info['predictions']
                
                # Classification report
                class_report = classification_report(
                    y_test, predictions, 
                    target_names=self.label_encoder.classes_,
                    output_dict=True
                )
                
                detailed_results[name] = {
                    'accuracy': model_info.get('test_accuracy', 0),
                    'classification_report': class_report,
                    'confusion_matrix': confusion_matrix(y_test, predictions).tolist()
                }
        
        # Save as JSON
        import json
        with open(benchmarks_path / "performance_metrics.json", 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        self.logger.info(f"Detailed results saved to {benchmarks_path}")

def main():
    """Main function for testing the trainer"""
    trainer = MalwareTrainer()
    
    # Load training data
    data = trainer.load_training_data()
    if data is None:
        print("Could not load training data")
        return
    
    X_train, X_test, y_train, y_test, feature_names = data
    
    # Train all models
    trained_models = trainer.train_all_models(X_train, y_train, X_test, y_test)
    
    # Get model comparison
    comparison = trainer.get_model_comparison()
    print("\nModel Comparison:")
    print(comparison)
    
    # Save detailed results
    trainer.save_detailed_results(y_test)
    
    print("\nTraining completed! Check the results directory for detailed outputs.")

if __name__ == "__main__":
    main()
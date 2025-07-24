# src/models/evaluator.py
import pandas as pd
import numpy as np
import json
import os
import logging
from typing import Dict, List, Any, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve
)
from sklearn.preprocessing import LabelBinarizer
import joblib

class ModelEvaluator:
    """Module untuk evaluasi model machine learning"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.results_dir = config.get('results_dir', 'results/')
        self.models_dir = config.get('models_dir', 'models/')
        self.logger = logging.getLogger(__name__)
        
    def load_trained_models(self) -> Dict[str, Any]:
        """Load all trained models"""
        models = {}
        model_names = ['random_forest', 'decision_tree', 'svm']
        
        for model_name in model_names:
            model_path = os.path.join(self.models_dir, model_name, f'{model_name}_model.pkl')
            if os.path.exists(model_path):
                try:
                    models[model_name] = joblib.load(model_path)
                    self.logger.info(f"Loaded model: {model_name}")
                except Exception as e:
                    self.logger.error(f"Error loading {model_name}: {str(e)}")
            else:
                self.logger.warning(f"Model not found: {model_path}")
        
        # Load ensemble model
        ensemble_path = os.path.join(self.models_dir, 'ensemble', 'voting_classifier.pkl')
        if os.path.exists(ensemble_path):
            try:
                models['voting_classifier'] = joblib.load(ensemble_path)
                self.logger.info("Loaded ensemble model: voting_classifier")
            except Exception as e:
                self.logger.error(f"Error loading ensemble model: {str(e)}")
        
        return models
    
    def evaluate_single_model(self, model: Any, model_name: str,
                             X_test: np.ndarray, y_test: np.ndarray,
                             class_names: List[str] = None) -> Dict[str, Any]:
        """Evaluate a single model"""
        self.logger.info(f"Evaluating model: {model_name}")
        
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = None
            
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
            
            # Calculate basic metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Classification report
            if class_names:
                target_names = class_names
            else:
                target_names = [f'Class_{i}' for i in range(len(np.unique(y_test)))]
            
            class_report = classification_report(
                y_test, y_pred, 
                target_names=target_names,
                output_dict=True,
                zero_division=0
            )
            
            # Confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # ROC AUC (for multiclass)
            roc_auc = None
            if y_pred_proba is not None:
                try:
                    lb = LabelBinarizer()
                    y_test_bin = lb.fit_transform(y_test)
                    if y_test_bin.shape[1] == 1:  # Binary case
                        y_test_bin = np.hstack([1 - y_test_bin, y_test_bin])
                    roc_auc = roc_auc_score(y_test_bin, y_pred_proba, average='weighted')
                except Exception as e:
                    self.logger.warning(f"Could not calculate ROC AUC for {model_name}: {str(e)}")
            
            # Per-class metrics
            per_class_metrics = {}
            for i, class_name in enumerate(target_names):
                if class_name in class_report and isinstance(class_report[class_name], dict):
                    per_class_metrics[class_name] = {
                        'precision': class_report[class_name]['precision'],
                        'recall': class_report[class_name]['recall'],
                        'f1-score': class_report[class_name]['f1-score'],
                        'support': class_report[class_name]['support']
                    }
            
            evaluation_results = {
                'model_name': model_name,
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'roc_auc': float(roc_auc) if roc_auc is not None else None,
                'confusion_matrix': conf_matrix.tolist(),
                'classification_report': class_report,
                'per_class_metrics': per_class_metrics,
                'predictions': y_pred.tolist(),
                'prediction_probabilities': y_pred_proba.tolist() if y_pred_proba is not None else None
            }
            
            self.logger.info(f"{model_name} evaluation completed - Accuracy: {accuracy:.4f}")
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Error evaluating {model_name}: {str(e)}")
            return {'model_name': model_name, 'error': str(e)}
    
    def evaluate_all_models(self, X_test: np.ndarray, y_test: np.ndarray,
                           class_names: List[str] = None) -> Dict[str, Any]:
        """Evaluate all trained models"""
        self.logger.info("Starting evaluation of all models...")
        
        # Load trained models
        models = self.load_trained_models()
        
        if not models:
            self.logger.error("No trained models found")
            return {}
        
        evaluation_results = {}
        
        for model_name, model in models.items():
            results = self.evaluate_single_model(
                model, model_name, X_test, y_test, class_names
            )
            evaluation_results[model_name] = results
        
        self.logger.info("All models evaluation completed")
        return evaluation_results
    
    def create_benchmark_comparison(self, evaluation_results: Dict[str, Any]) -> pd.DataFrame:
        """Create benchmark comparison table"""
        self.logger.info("Creating benchmark comparison...")
        
        benchmark_data = []
        
        for model_name, results in evaluation_results.items():
            if 'error' not in results:
                benchmark_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'Accuracy': results.get('accuracy', 0),
                    'Precision': results.get('precision', 0),
                    'Recall': results.get('recall', 0),
                    'F1-Score': results.get('f1_score', 0),
                    'ROC-AUC': results.get('roc_auc', 0) or 0
                })
        
        benchmark_df = pd.DataFrame(benchmark_data)
        
        if not benchmark_df.empty:
            # Sort by F1-Score
            benchmark_df = benchmark_df.sort_values('F1-Score', ascending=False)
            benchmark_df = benchmark_df.round(4)
        
        return benchmark_df
    
    def analyze_malware_families(self, evaluation_results: Dict[str, Any],
                                class_names: List[str]) -> Dict[str, Any]:
        """Analyze performance per malware family"""
        self.logger.info("Analyzing malware families performance...")
        
        family_analysis = {}
        
        for model_name, results in evaluation_results.items():
            if 'error' not in results and 'per_class_metrics' in results:
                family_metrics = {}
                
                for family_name, metrics in results['per_class_metrics'].items():
                    family_metrics[family_name] = {
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                        'f1_score': metrics['f1-score'],
                        'support': metrics['support']
                    }
                
                family_analysis[model_name] = family_metrics
        
        return family_analysis
    
    def find_misclassified_samples(self, evaluation_results: Dict[str, Any],
                                  X_test: np.ndarray, y_test: np.ndarray,
                                  class_names: List[str]) -> Dict[str, Any]:
        """Find and analyze misclassified samples"""
        self.logger.info("Analyzing misclassified samples...")
        
        misclassified_analysis = {}
        
        for model_name, results in evaluation_results.items():
            if 'error' not in results and 'predictions' in results:
                y_pred = np.array(results['predictions'])
                misclassified_indices = np.where(y_test != y_pred)[0]
                
                misclassified_data = []
                for idx in misclassified_indices[:100]:  # Limit to first 100
                    misclassified_data.append({
                        'sample_index': int(idx),
                        'true_label': class_names[y_test[idx]] if class_names else int(y_test[idx]),
                        'predicted_label': class_names[y_pred[idx]] if class_names else int(y_pred[idx]),
                        'confidence': float(max(results['prediction_probabilities'][idx])) if results['prediction_probabilities'] else None
                    })
                
                misclassified_analysis[model_name] = {
                    'total_misclassified': len(misclassified_indices),
                    'misclassification_rate': len(misclassified_indices) / len(y_test),
                    'sample_misclassifications': misclassified_data
                }
        
        return misclassified_analysis
    
    def save_evaluation_results(self, evaluation_results: Dict[str, Any],
                               benchmark_df: pd.DataFrame,
                               family_analysis: Dict[str, Any],
                               misclassified_analysis: Dict[str, Any]) -> Dict[str, str]:
        """Save all evaluation results"""
        self.logger.info("Saving evaluation results...")
        
        # Create directories
        os.makedirs(f"{self.results_dir}/benchmarks", exist_ok=True)
        os.makedirs(f"{self.results_dir}/reports", exist_ok=True)
        os.makedirs(f"{self.results_dir}/predictions", exist_ok=True)
        
        file_paths = {}
        
        # Save benchmark comparison
        benchmark_path = f"{self.results_dir}/benchmarks/model_comparison.csv"
        benchmark_df.to_csv(benchmark_path, index=False)
        file_paths['benchmark'] = benchmark_path
        
        # Save detailed evaluation results
        evaluation_path = f"{self.results_dir}/benchmarks/performance_metrics.json"
        with open(evaluation_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        file_paths['detailed_evaluation'] = evaluation_path
        
        # Save family analysis
        family_path = f"{self.results_dir}/reports/family_analysis.json"
        with open(family_path, 'w') as f:
            json.dump(family_analysis, f, indent=2)
        file_paths['family_analysis'] = family_path
        
        # Save misclassified analysis
        misclassified_path = f"{self.results_dir}/predictions/misclassified_samples.json"
        with open(misclassified_path, 'w') as f:
            json.dump(misclassified_analysis, f, indent=2)
        file_paths['misclassified'] = misclassified_path
        
        self.logger.info("All evaluation results saved successfully")
        return file_paths
    
    def full_evaluation_pipeline(self, X_test: np.ndarray, y_test: np.ndarray,
                                class_names: List[str] = None) -> Dict[str, Any]:
        """Execute complete evaluation pipeline"""
        self.logger.info("Starting full evaluation pipeline...")
        
        # Evaluate all models
        evaluation_results = self.evaluate_all_models(X_test, y_test, class_names)
        
        if not evaluation_results:
            self.logger.error("No evaluation results available")
            return {}
        
        # Create benchmark comparison
        benchmark_df = self.create_benchmark_comparison(evaluation_results)
        
        # Analyze malware families
        family_analysis = self.analyze_malware_families(evaluation_results, class_names)
        
        # Find misclassified samples
        misclassified_analysis = self.find_misclassified_samples(
            evaluation_results, X_test, y_test, class_names
        )
        
        # Save results
        file_paths = self.save_evaluation_results(
            evaluation_results, benchmark_df, family_analysis, misclassified_analysis
        )
        
        final_results = {
            'evaluation_results': evaluation_results,
            'benchmark_comparison': benchmark_df.to_dict('records'),
            'family_analysis': family_analysis,
            'misclassified_analysis': misclassified_analysis,
            'file_paths': file_paths,
            'best_model': benchmark_df.iloc[0]['Model'].lower().replace(' ', '_') if not benchmark_df.empty else None
        }
        
        self.logger.info("Full evaluation pipeline completed")
        return final_results
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import json
import yaml
import logging
from pathlib import Path
import joblib
from itertools import cycle

class MalwareVisualizer:
    """
    Create visualizations for malware classification results
    """
    
    def __init__(self, config_path="config/config.yaml"):
        """Initialize the visualizer with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Setup paths
        self.results_path = Path(self.config['output']['results_path'])
        self.models_path = Path(self.config['output']['models_path'])
        self.processed_data_path = Path(self.config['data']['processed_data_path'])
        
        # Create visualization directories
        self.viz_path = self.results_path / "visualizations"
        self.plots_path = self.viz_path / "plots"
        self.plots_path.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Load data and models if available
        self.load_data()
        
    def load_data(self):
        """Load training data and models"""
        try:
            # Load training data
            data = np.load(self.processed_data_path / "scaled_features.npz")
            self.X_train = data['X_train']
            self.X_test = data['X_test']
            self.y_train = data['y_train']
            self.y_test = data['y_test']
            self.feature_names = data['feature_names']
            
            # Load label encoder
            with open(self.processed_data_path / "label_encoder.pkl", 'rb') as f:
                self.label_encoder = joblib.load(f)
            
            self.class_names = list(self.label_encoder.classes_)
            self.logger.info("Data loaded successfully")
            
        except FileNotFoundError:
            self.logger.warning("Training data not found. Run data processing first.")
            self.X_train = None
            self.X_test = None
            self.y_train = None
            self.y_test = None
            self.feature_names = None
            self.class_names = None
    
    def load_models(self):
        """Load trained models"""
        self.models = {}
        
        try:
            # Load individual models
            self.models['random_forest'] = joblib.load(self.models_path / "random_forest" / "rf_model.pkl")
            self.models['decision_tree'] = joblib.load(self.models_path / "decision_tree" / "dt_model.pkl")
            self.models['svm'] = joblib.load(self.models_path / "svm" / "svm_model.pkl")
            
            # Load ensemble models
            try:
                self.models['voting'] = joblib.load(self.models_path / "ensemble" / "voting_classifier.pkl")
                self.models['stacking'] = joblib.load(self.models_path / "ensemble" / "stacking_classifier.pkl")
            except FileNotFoundError:
                self.logger.warning("Ensemble models not found")
            
            self.logger.info(f"Loaded {len(self.models)} models")
            
        except FileNotFoundError as e:
            self.logger.warning(f"Models not found: {e}. Train models first.")
            self.models = {}
    
    def plot_confusion_matrices(self):
        """Create confusion matrix plots for all models"""
        if self.X_test is None or self.y_test is None:
            self.logger.error("Test data not available")
            return
        
        self.load_models()
        if not self.models:
            self.logger.error("No models available")
            return
        
        self.logger.info("Creating confusion matrix plots...")
        
        # Create subplots
        n_models = len(self.models)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (name, model) in enumerate(self.models.items()):
            if idx >= 6:  # Maximum 6 subplots
                break
                
            # Make predictions
            y_pred = model.predict(self.X_test)
            
            # Create confusion matrix
            cm = confusion_matrix(self.y_test, y_pred)
            
            # Plot
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=self.class_names, 
                       yticklabels=self.class_names,
                       ax=axes[idx])
            
            axes[idx].set_title(f'{name.replace("_", " ").title()} Confusion Matrix')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        # Hide unused subplots
        for idx in range(len(self.models), 6):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.plots_path / "confusion_matrices.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create individual confusion matrices
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            cm = confusion_matrix(self.y_test, y_pred)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.class_names,
                       yticklabels=self.class_names)
            plt.title(f'{name.replace("_", " ").title()} Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            plt.savefig(self.plots_path / f"confusion_matrix_{name}.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        self.logger.info("Confusion matrix plots saved")
    
    def plot_model_comparison(self):
        """Create model comparison plots"""
        try:
            # Load comparison data
            comparison_df = pd.read_csv(self.results_path / "benchmarks" / "model_comparison.csv")
        except FileNotFoundError:
            self.logger.error("Model comparison data not found. Train models first.")
            return
        
        self.logger.info("Creating model comparison plots...")
        
        # Accuracy comparison
        plt.figure(figsize=(12, 8))
        bars = plt.bar(comparison_df['Model'], comparison_df['Test Accuracy'], 
                      color=sns.color_palette("husl", len(comparison_df)))
        
        # Add value labels on bars
        for bar, accuracy in zip(bars, comparison_df['Test Accuracy']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{accuracy:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.title('Model Accuracy Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Test Accuracy', fontsize=12)
        plt.xticks(rotation=45)
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plots_path / "accuracy_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Training time comparison
        if 'Training Time (s)' in comparison_df.columns:
            plt.figure(figsize=(12, 8))
            bars = plt.bar(comparison_df['Model'], comparison_df['Training Time (s)'],
                          color=sns.color_palette("viridis", len(comparison_df)))
            
            for bar, time_val in zip(bars, comparison_df['Training Time (s)']):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{time_val:.2f}s', ha='center', va='bottom', fontweight='bold')
            
            plt.title('Model Training Time Comparison', fontsize=16, fontweight='bold')
            plt.xlabel('Model', fontsize=12)
            plt.ylabel('Training Time (seconds)', fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.plots_path / "training_time_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        self.logger.info("Model comparison plots saved")
    
    def plot_feature_importance(self):
        """Plot feature importance for tree-based models"""
        if self.feature_names is None:
            self.logger.error("Feature names not available")
            return
        
        self.load_models()
        if not self.models:
            self.logger.error("No models available")
            return
        
        self.logger.info("Creating feature importance plots...")
        
        # Plot for Random Forest and Decision Tree
        tree_models = ['random_forest', 'decision_tree']
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        for idx, model_name in enumerate(tree_models):
            if model_name not in self.models:
                continue
                
            model = self.models[model_name]
            
            # Get feature importance
            importance = model.feature_importances_
            
            # Sort features by importance
            indices = np.argsort(importance)[::-1]
            
            # Select top 20 features
            top_n = min(20, len(importance))
            top_indices = indices[:top_n]
            top_importance = importance[top_indices]
            top_features = [self.feature_names[i] for i in top_indices]
            
            # Plot
            axes[idx].barh(range(top_n), top_importance[::-1])
            axes[idx].set_yticks(range(top_n))
            axes[idx].set_yticklabels(top_features[::-1])
            axes[idx].set_xlabel('Feature Importance')
            axes[idx].set_title(f'{model_name.replace("_", " ").title()} - Top {top_n} Features')
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_path / "feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Feature importance plots saved")
    
    def plot_roc_curves(self):
        """Plot ROC curves for all models"""
        if self.X_test is None or self.y_test is None:
            self.logger.error("Test data not available")
            return
        
        self.load_models()
        if not self.models:
            self.logger.error("No models available")
            return
        
        self.logger.info("Creating ROC curves...")
        
        # Binarize the output for multiclass ROC
        y_test_bin = label_binarize(self.y_test, classes=range(len(self.class_names)))
        n_classes = y_test_bin.shape[1]
        
        plt.figure(figsize=(12, 10))
        
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple'])
        
        for name, model in self.models.items():
            color = next(colors)
            
            try:
                # Get prediction probabilities
                y_score = model.predict_proba(self.X_test)
                
                # Compute ROC curve and ROC area for each class
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                
                for i in range(n_classes):
                    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                
                # Compute micro-average ROC curve and ROC area
                fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
                roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                
                # Plot micro-average ROC curve
                plt.plot(fpr["micro"], tpr["micro"],
                        color=color, linestyle='-', linewidth=2,
                        label=f'{name.replace("_", " ").title()} (AUC = {roc_auc["micro"]:.3f})')
                
            except AttributeError:
                # Model doesn't support predict_proba
                self.logger.warning(f"Model {name} doesn't support probability predictions")
                continue
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Multiclass (Micro-average)', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plots_path / "roc_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("ROC curves saved")
    
    def plot_class_distribution(self):
        """Plot class distribution in the dataset"""
        if self.y_train is None or self.y_test is None:
            self.logger.error("Training data not available")
            return
        
        self.logger.info("Creating class distribution plots...")
        
        # Combine train and test labels
        y_all = np.concatenate([self.y_train, self.y_test])
        train_labels = [self.class_names[i] for i in self.y_train]
        test_labels = [self.class_names[i] for i in self.y_test]
        all_labels = [self.class_names[i] for i in y_all]
        
        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Training set distribution
        train_counts = pd.Series(train_labels).value_counts()
        axes[0].pie(train_counts.values, labels=train_counts.index, autopct='%1.1f%%')
        axes[0].set_title('Training Set Distribution')
        
        # Test set distribution
        test_counts = pd.Series(test_labels).value_counts()
        axes[1].pie(test_counts.values, labels=test_counts.index, autopct='%1.1f%%')
        axes[1].set_title('Test Set Distribution')
        
        # Overall distribution
        all_counts = pd.Series(all_labels).value_counts()
        bars = axes[2].bar(all_counts.index, all_counts.values)
        axes[2].set_title('Overall Class Distribution')
        axes[2].set_xlabel('Malware Family')
        axes[2].set_ylabel('Number of Samples')
        axes[2].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, all_counts.values):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.plots_path / "class_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Class distribution plots saved")
    
    def plot_learning_curves(self):
        """Plot learning curves for models"""
        if self.X_train is None or self.y_train is None:
            self.logger.error("Training data not available")
            return
        
        from sklearn.model_selection import learning_curve
        
        self.logger.info("Creating learning curves...")
        
        # Define models for learning curves
        models_for_curves = {
            'Random Forest': self.models.get('random_forest'),
            'Decision Tree': self.models.get('decision_tree'),
            'SVM': self.models.get('svm')
        }
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, (name, model) in enumerate(models_for_curves.items()):
            if model is None:
                continue
            
            # Calculate learning curve
            train_sizes, train_scores, val_scores = learning_curve(
                model, self.X_train, self.y_train, cv=3,
                train_sizes=np.linspace(0.1, 1.0, 10),
                n_jobs=-1, random_state=42
            )
            
            # Calculate mean and std
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            # Plot
            axes[idx].plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
            axes[idx].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
            
            axes[idx].plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
            axes[idx].fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
            
            axes[idx].set_title(f'{name} Learning Curve')
            axes[idx].set_xlabel('Training Set Size')
            axes[idx].set_ylabel('Accuracy Score')
            axes[idx].legend(loc='best')
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_path / "learning_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Learning curves saved")
    
    def create_family_classification_report(self):
        """Create detailed classification report showing which families are classified correctly"""
        if self.X_test is None or self.y_test is None:
            self.logger.error("Test data not available")
            return
        
        self.load_models()
        if not self.models:
            self.logger.error("No models available")
            return
        
        self.logger.info("Creating family classification report...")
        
        # Load performance metrics
        try:
            with open(self.results_path / "benchmarks" / "performance_metrics.json", 'r') as f:
                performance_data = json.load(f)
        except FileNotFoundError:
            self.logger.error("Performance metrics not found. Train models first.")
            return
        
        # Create family-wise accuracy comparison
        family_results = []
        
        for model_name, metrics in performance_data.items():
            if 'classification_report' in metrics:
                class_report = metrics['classification_report']
                
                for family, family_metrics in class_report.items():
                    if family in self.class_names:
                        family_results.append({
                            'Model': model_name.replace('_', ' ').title(),
                            'Family': family,
                            'Precision': family_metrics['precision'],
                            'Recall': family_metrics['recall'],
                            'F1-Score': family_metrics['f1-score'],
                            'Support': family_metrics['support']
                        })
        
        # Convert to DataFrame
        family_df = pd.DataFrame(family_results)
        
        if family_df.empty:
            self.logger.error("No family classification data available")
            return
        
        # Create heatmap for F1-scores
        pivot_df = family_df.pivot(index='Family', columns='Model', values='F1-Score')
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_df, annot=True, cmap='YlOrRd', fmt='.3f', cbar_kws={'label': 'F1-Score'})
        plt.title('F1-Score by Malware Family and Model', fontsize=16, fontweight='bold')
        plt.xlabel('Model')
        plt.ylabel('Malware Family')
        plt.tight_layout()
        plt.savefig(self.plots_path / "family_f1_scores.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed family report
        family_df.to_csv(self.results_path / "reports" / "family_classification_report.csv", index=False)
        
        self.logger.info("Family classification report saved")
        
        return family_df
    
    def generate_all_plots(self):
        """Generate all visualization plots"""
        self.logger.info("Generating all visualization plots...")
        
        try:
            self.plot_class_distribution()
            self.plot_confusion_matrices()
            self.plot_model_comparison()
            self.plot_feature_importance()
            self.plot_roc_curves()
            self.plot_learning_curves()
            self.create_family_classification_report()
            
            self.logger.info("All plots generated successfully!")
            self.logger.info(f"Plots saved to: {self.plots_path}")
            
        except Exception as e:
            self.logger.error(f"Error generating plots: {e}")

def main():
    """Main function for testing the visualizer"""
    visualizer = MalwareVisualizer()
    
    # Generate all plots
    visualizer.generate_all_plots()
    
    print(f"Visualization complete! Check the plots in: {visualizer.plots_path}")

if __name__ == "__main__":
    main()
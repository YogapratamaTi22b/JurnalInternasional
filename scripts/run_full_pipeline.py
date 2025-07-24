"""
Main execution script for the Malware Classification Pipeline
This script runs the complete pipeline: download -> process -> train -> visualize
"""

import sys
import os
import argparse
import logging
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.downloader import MalwareBazaarDownloader
from data.processor import MalwareProcessor
from data.trainer import MalwareTrainer
from visualization.plotter import MalwareVisualizer

def setup_logging():
    """Setup logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "pipeline.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def run_data_download(args, logger):
    """Run data download step"""
    logger.info("="*60)
    logger.info("STEP 1: DATA DOWNLOAD")
    logger.info("="*60)
    
    downloader = MalwareBazaarDownloader()
    
    if args.use_synthetic:
        logger.info("Creating synthetic dataset...")
        samples_df = downloader.create_synthetic_dataset(args.max_samples)
        return len(samples_df)
    else:
        logger.info("Downloading malware samples from MalwareBazaar...")
        # Get recent samples
        samples_df = downloader.get_recent_samples(limit=args.max_samples)
        
        if samples_df.empty:
            logger.warning("No samples retrieved from API, creating synthetic dataset")
            samples_df = downloader.create_synthetic_dataset(args.max_samples)
            return len(samples_df)
        else:
            # Download samples
            downloaded = downloader.download_samples_batch(samples_df, max_samples=args.max_samples)
            return len(downloaded)

def run_data_processing(args, logger):
    """Run data processing step"""
    logger.info("="*60)
    logger.info("STEP 2: DATA PROCESSING")
    logger.info("="*60)
    
    processor = MalwareProcessor()
    
    # Process samples and extract features
    features_df = processor.process_downloaded_samples()
    logger.info(f"Extracted features from {len(features_df)} samples")
    
    # Prepare training data
    training_data = processor.prepare_training_data()
    
    if training_data:
        X_train, X_test, y_train, y_test = training_data
        logger.info(f"Training data prepared: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
        logger.info(f"Number of features: {X_train.shape[1]}")
        logger.info(f"Number of classes: {len(processor.get_class_names())}")
        return True
    else:
        logger.error("Failed to prepare training data")
        return False

def run_model_training(args, logger):
    """Run model training step"""
    logger.info("="*60)
    logger.info("STEP 3: MODEL TRAINING")
    logger.info("="*60)
    
    trainer = MalwareTrainer()
    
    # Load training data
    data = trainer.load_training_data()
    if data is None:
        logger.error("Could not load training data")
        return False
    
    X_train, X_test, y_train, y_test, feature_names = data
    
    # Hyperparameter tuning if requested
    if args.tune_hyperparameters:
        logger.info("Performing hyperparameter tuning...")
        for model_name in ['random_forest', 'decision_tree', 'svm']:
            trainer.hyperparameter_tuning(X_train, y_train, model_name)
    
    # Train all models
    trained_models = trainer.train_all_models(X_train, y_train, X_test, y_test)
    
    if trained_models:
        # Get model comparison
        comparison = trainer.get_model_comparison()
        logger.info("\nModel Comparison:")
        logger.info("\n" + comparison.to_string(index=False))
        
        # Save detailed results
        trainer.save_detailed_results(y_test)
        
        # Find best model
        best_model = comparison.iloc[0]
        logger.info(f"\nBest performing model: {best_model['Model']} with accuracy: {best_model['Test Accuracy']:.4f}")
        
        return True
    else:
        logger.error("Model training failed")
        return False

def run_visualization(args, logger):
    """Run visualization step"""
    logger.info("="*60)
    logger.info("STEP 4: VISUALIZATION")
    logger.info("="*60)
    
    visualizer = MalwareVisualizer()
    
    # Generate all plots
    visualizer.generate_all_plots()
    
    logger.info(f"All visualizations saved to: {visualizer.plots_path}")
    return True

def generate_final_report(logger):
    """Generate final report"""
    logger.info("="*60)
    logger.info("GENERATING FINAL REPORT")
    logger.info("="*60)
    
    try:
        import pandas as pd
        import json
        
        # Load results
        results_path = Path("results")
        
        # Model comparison
        comparison_df = pd.read_csv(results_path / "benchmarks" / "model_comparison.csv")
        
        # Performance metrics
        with open(results_path / "benchmarks" / "performance_metrics.json", 'r') as f:
            performance_data = json.load(f)
        
        # Generate report
        report_lines = []
        report_lines.append("MALWARE CLASSIFICATION PROJECT - FINAL REPORT")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # Dataset summary
        report_lines.append("DATASET SUMMARY:")
        report_lines.append("-" * 20)
        
        # Try to get dataset info
        try:
            data = np.load("data/processed/scaled_features.npz")
            total_samples = len(data['X_train']) + len(data['X_test'])
            n_features = data['X_train'].shape[1]
            
            with open("data/processed/label_encoder.pkl", 'rb') as f:
                import pickle
                label_encoder = pickle.load(f)
            n_classes = len(label_encoder.classes_)
            class_names = list(label_encoder.classes_)
            
            report_lines.append(f"Total samples: {total_samples}")
            report_lines.append(f"Features: {n_features}")
            report_lines.append(f"Classes: {n_classes}")
            report_lines.append(f"Malware families: {', '.join(class_names)}")
            
        except:
            report_lines.append("Dataset information not available")
        
        report_lines.append("")
        
        # Model performance
        report_lines.append("MODEL PERFORMANCE COMPARISON:")
        report_lines.append("-" * 35)
        report_lines.append("")
        report_lines.append(comparison_df.to_string(index=False))
        report_lines.append("")
        
        # Best model details
        best_model = comparison_df.iloc[0]
        report_lines.append("BEST PERFORMING MODEL:")
        report_lines.append("-" * 25)
        report_lines.append(f"Model: {best_model['Model']}")
        report_lines.append(f"Test Accuracy: {best_model['Test Accuracy']:.4f}")
        if 'CV Mean' in best_model:
            report_lines.append(f"Cross-validation Score: {best_model['CV Mean']:.4f} ± {best_model['CV Std']:.4f}")
        report_lines.append(f"Training Time: {best_model['Training Time (s)']:.2f} seconds")
        report_lines.append("")
        
        # Family classification performance
        try:
            family_df = pd.read_csv(results_path / "reports" / "family_classification_report.csv")
            best_model_name = best_model['Model'].lower().replace(' ', '_')
            
            best_model_families = family_df[family_df['Model'] == best_model['Model']]
            if not best_model_families.empty:
                report_lines.append("FAMILY CLASSIFICATION PERFORMANCE (BEST MODEL):")
                report_lines.append("-" * 50)
                report_lines.append("")
                
                for _, row in best_model_families.iterrows():
                    report_lines.append(f"{row['Family']}:")
                    report_lines.append(f"  Precision: {row['Precision']:.3f}")
                    report_lines.append(f"  Recall: {row['Recall']:.3f}")
                    report_lines.append(f"  F1-Score: {row['F1-Score']:.3f}")
                    report_lines.append(f"  Support: {int(row['Support'])}")
                    report_lines.append("")
        except:
            report_lines.append("Family classification details not available")
        
        # Files generated
        report_lines.append("GENERATED FILES:")
        report_lines.append("-" * 15)
        report_lines.append("Models saved in: models/")
        report_lines.append("Visualizations saved in: results/visualizations/plots/")
        report_lines.append("Detailed results saved in: results/benchmarks/")
        report_lines.append("Reports saved in: results/reports/")
        report_lines.append("")
        
        # Save report
        report_content = "\n".join(report_lines)
        
        reports_path = results_path / "reports"
        reports_path.mkdir(parents=True, exist_ok=True)
        
        with open(reports_path / "final_report.txt", 'w') as f:
            f.write(report_content)
        
        logger.info("Final report generated: results/reports/final_report.txt")
        
        # Print summary to console
        logger.info("\n" + "="*60)
        logger.info("PIPELINE EXECUTION COMPLETE!")
        logger.info("="*60)
        logger.info(f"Best Model: {best_model['Model']} (Accuracy: {best_model['Test Accuracy']:.4f})")
        logger.info("Check the following directories for results:")
        logger.info("- models/ : Trained models")
        logger.info("- results/visualizations/plots/ : Charts and graphs")
        logger.info("- results/benchmarks/ : Performance metrics")
        logger.info("- results/reports/ : Final report")
        
        return True
        
    except Exception as e:
        logger.error(f"Error generating final report: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Malware Classification Pipeline')
    
    parser.add_argument('--steps', nargs='+', 
                       choices=['download', 'process', 'train', 'visualize', 'all'],
                       default=['all'],
                       help='Pipeline steps to run')
    
    parser.add_argument('--max-samples', type=int, default=1000,
                       help='Maximum number of samples to download/process')
    
    parser.add_argument('--use-synthetic', action='store_true',
                       help='Use synthetic dataset instead of downloading')
    
    parser.add_argument('--tune-hyperparameters', action='store_true',
                       help='Perform hyperparameter tuning')
    
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip download step (use existing data)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    logger.info("Starting Malware Classification Pipeline")
    logger.info(f"Arguments: {vars(args)}")
    
    start_time = time.time()
    
    # Determine steps to run
    if 'all' in args.steps:
        steps_to_run = ['download', 'process', 'train', 'visualize']
    else:
        steps_to_run = args.steps
    
    # Skip download if requested
    if args.skip_download and 'download' in steps_to_run:
        steps_to_run.remove('download')
        logger.info("Skipping download step as requested")
    
    success = True
    
    try:
        # Step 1: Data Download
        if 'download' in steps_to_run:
            samples_count = run_data_download(args, logger)
            logger.info(f"Data download completed: {samples_count} samples")
        
        # Step 2: Data Processing
        if 'process' in steps_to_run and success:
            success = run_data_processing(args, logger)
            if success:
                logger.info("Data processing completed successfully")
            else:
                logger.error("Data processing failed")
        
        # Step 3: Model Training
        if 'train' in steps_to_run and success:
            success = run_model_training(args, logger)
            if success:
                logger.info("Model training completed successfully")
            else:
                logger.error("Model training failed")
        
        # Step 4: Visualization
        if 'visualize' in steps_to_run and success:
            success = run_visualization(args, logger)
            if success:
                logger.info("Visualization completed successfully")
            else:
                logger.error("Visualization failed")
        
        # Generate final report
        if success:
            generate_final_report(logger)
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        success = False
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        success = False
    
    # Summary
    end_time = time.time()
    execution_time = end_time - start_time
    
    logger.info("="*60)
    if success:
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    else:
        logger.info("PIPELINE FAILED!")
    
    logger.info(f"Total execution time: {execution_time:.2f} seconds")
    logger.info("="*60)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
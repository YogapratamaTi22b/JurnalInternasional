# Malware Classification Project

A comprehensive machine learning project for classifying malware samples using Random Forest, Decision Tree, and SVM algorithms. This project downloads malware samples from MalwareBazaar, extracts features from PE files, and trains multiple models to classify malware into different families.

## 🎯 Project Overview

This project implements a complete pipeline for malware classification:

1. **Data Download**: Automatically downloads malware samples from MalwareBazaar API
2. **Feature Extraction**: Extracts PE (Portable Executable) features from malware samples
3. **Model Training**: Trains Random Forest, Decision Tree, and SVM classifiers
4. **Evaluation & Visualization**: Provides comprehensive model comparison and visualization

## 🏗️ Project Structure

```
malware_classification_project/
│
├── data/                          # Data storage
│   ├── raw/                       # Raw datasets
│   ├── processed/                 # Processed datasets
│   └── external/                  # External datasets
│
├── src/                           # Source code modules
│   ├── data/                      # Data handling modules
│   │   ├── downloader.py          # Data download from MalwareBazaar
│   │   ├── processor.py           # Feature extraction and processing
│   │   └── validator.py           # Data validation utilities
│   │
│   ├── models/                    # Model related modules
│   │   ├── trainer.py             # Model training
│   │   ├── evaluator.py           # Model evaluation
│   │   └── ensemble.py            # Ensemble methods
│   │
│   ├── visualization/             # Visualization modules
│   │   ├── plotter.py             # Visualization and plotting
│   │   └── dashboard.py           # Interactive dashboard
│   │
│   └── utils/                     # Utility modules
│       ├── config.py              # Configuration settings
│       ├── logger.py              # Logging utilities
│       └── helpers.py             # Helper functions
│
├── models/                        # Saved trained models
├── results/                       # Results and outputs
├── notebooks/                     # Jupyter notebooks
├── scripts/                       # Executable scripts
├── config/                        # Configuration files
├── tests/                         # Unit tests
└── docs/                          # Documentation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Windows OS (configured for Windows environment)
- VS Code (recommended IDE)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd malware_classification_project
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements/requirements.txt
   ```

4. **Install the package**:
   ```bash
   pip install -e .
   ```

### Basic Usage

#### Run Complete Pipeline

```bash
python scripts/run_full_pipeline.py --max-samples 1000
```

#### Run with Synthetic Data (for testing)

```bash
python scripts/run_full_pipeline.py --use-synthetic --max-samples 1000
```

#### Run Specific Steps

```bash
# Only download data
python scripts/run_full_pipeline.py --steps download --max-samples 500

# Only train models
python scripts/run_full_pipeline.py --steps train

# Only generate visualizations
python scripts/run_full_pipeline.py --steps visualize
```

#### Run with Hyperparameter Tuning

```bash
python scripts/run_full_pipeline.py --tune-hyperparameters
```

## 📊 Features

### Data Collection
- Downloads malware samples from MalwareBazaar API
- Supports multiple file types (exe, dll, pdf, doc, zip)
- Automatic malware family classification
- Synthetic data generation for testing

### Feature Extraction
- **PE Header Features**: DOS header, NT headers, optional headers
- **Section Analysis**: Entropy, sizes, virtual sizes
- **Import/Export Analysis**: DLL imports, function exports
- **File Properties**: File size, overall entropy
- **Advanced Features**: 50+ engineered features per sample

### Machine Learning Models
- **Random Forest**: Ensemble of decision trees with feature importance
- **Decision Tree**: Interpretable tree-based classifier
- **SVM**: Support Vector Machine with RBF kernel
- **Ensemble Methods**: Voting and Stacking classifiers

### Malware Families Supported
- Trojan
- Ransomware
- Adware
- Spyware
- Backdoor
- Rootkit
- Worm
- Virus
- Botnet
- Stealer

## 📈 Results and Visualization

The project generates comprehensive visualizations:

- **Confusion Matrices**: Model performance per class
- **ROC Curves**: Multi-class classification performance
- **Feature Importance**: Most important features for classification
- **Learning Curves**: Model performance vs training size
- **Class Distribution**: Dataset balance analysis
- **Model Comparison**: Accuracy, training time, and CV scores

## 🔧 Configuration

Edit `config/config.yaml` to customize:

```yaml
# Data Configuration
data:
  malware_bazaar:
    download_limit: 1000
    file_types: ["exe", "dll", "pdf", "doc", "zip"]

# Model Configuration
models:
  test_size: 0.2
  random_state: 42
  cross_validation_folds: 5
  
  random_forest:
    n_estimators: 100
    max_depth: 20
    min_samples_split: 5
```

## 🔍 Individual Module Usage

### Data Download
```python
from src.data.downloader import MalwareBazaarDownloader

downloader = MalwareBazaarDownloader()
samples_df = downloader.get_recent_samples(limit=100)
downloaded = downloader.download_samples_batch(samples_df)
```

### Feature Processing
```python
from src.data.processor import MalwareProcessor

processor = MalwareProcessor()
features_df = processor.process_downloaded_samples()
X_train, X_test, y_train, y_test = processor.prepare_training_data()
```

### Model Training
```python
from src.models.trainer import MalwareTrainer

trainer = MalwareTrainer()
trained_models = trainer.train_all_models()
comparison = trainer.get_model_comparison()
```

### Visualization
```python
from src.visualization.plotter import MalwareVisualizer

visualizer = MalwareVisualizer()
visualizer.generate_all_plots()
```

## 📝 Output Files

After running the pipeline, you'll find:

### Models
- `models/random_forest/rf_model.pkl` - Trained Random Forest
- `models/decision_tree/dt_model.pkl` - Trained Decision Tree  
- `models/svm/svm_model.pkl` - Trained SVM
- `models/ensemble/` - Ensemble models

### Results
- `results/benchmarks/model_comparison.csv` - Model performance comparison
- `results/benchmarks/performance_metrics.json` - Detailed metrics
- `results/visualizations/plots/` - All generated plots
- `results/reports/final_report.txt` - Complete analysis report

## 🧪 Testing

Run unit tests:
```bash
python -m pytest tests/ -v
```

Run specific test:
```bash
python -m pytest tests/test_downloader.py -v
```

## 📚 Jupyter Notebooks

Explore the analysis interactively:

1. `notebooks/01_data_exploration.ipynb` - Data exploration
2. `notebooks/02_feature_engineering.ipynb` - Feature analysis
3. `notebooks/03_model_training.ipynb` - Model training
4. `notebooks/04_model_evaluation.ipynb` - Model evaluation
5. `notebooks/05_visualization_analysis.ipynb` - Visualization
6. `notebooks/06_final_report.ipynb` - Final report

## ⚠️ Important Notes

### Security Considerations
- **Malware Handling**: This project downloads real malware samples
- **Isolation**: Run in isolated environment (VM recommended)
- **Antivirus**: Disable antivirus during analysis (in isolated environment)
- **Cleanup**: Properly dispose of malware samples after analysis

### System Requirements
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: At least 10GB free space
- **CPU**: Multi-core processor recommended for faster training

### API Limitations
- MalwareBazaar has rate limits
- Some samples may not be available
- Use `--use-synthetic` flag for testing without API

## 🔧 Troubleshooting

### Common Issues

1. **API Connection Issues**:
   ```bash
   # Use synthetic data instead
   python scripts/run_full_pipeline.py --use-synthetic
   ```

2. **Memory Issues**:
   ```bash
   # Reduce sample size
   python scripts/run_full_pipeline.py --max-samples 500
   ```

3. **PE Parsing Errors**:
   - Ensure `pefile` library is properly installed
   - Some samples may be corrupted (automatically skipped)

4. **Missing Dependencies**:
   ```bash
   pip install -r requirements/requirements.txt --upgrade
   ```

## 📊 Example Results

Typical performance metrics:

| Model | Test Accuracy | Training Time | CV Score |
|-------|---------------|---------------|----------|
| Random Forest | 0.892 | 15.3s | 0.885 ± 0.012 |
| SVM | 0.875 | 45.2s | 0.871 ± 0.018 |
| Decision Tree | 0.834 | 3.1s | 0.828 ± 0.023 |
| Voting Ensemble | 0.903 | 65.8s | 0.897 ± 0.015 |

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚖️ Disclaimer

This project is for educational and research purposes only. The authors are not responsible for any misuse of the tools or techniques described in this project. Always follow applicable laws and regulations when dealing with malware samples.

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

---

**Happy Malware Hunting! 🔍🛡️**
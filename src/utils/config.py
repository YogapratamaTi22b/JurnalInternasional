import configparser
import os

def load_config(config_path):
    """Muat konfigurasi dari file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config = configparser.ConfigParser()
    config.read(config_path)
    
    # Konversi ke dictionary
    config_dict = {}
    for section in config.sections():
        config_dict.update(dict(config.items(section)))
    
    # Set nilai default
    defaults = {
        'SAMPLE_LIMIT': '500',
        'RANDOM_SEED': '42',
        'TEST_SIZE': '0.2',
        'N_ESTIMATORS': '100',
        'SVM_KERNEL': 'rbf',
        'DATA_DIR': 'data/',
        'MODELS_DIR': 'models/',
        'RESULTS_DIR': 'results/'
    }
    
    for key, default_val in defaults.items():
        config_dict[key] = config_dict.get(key, default_val)
        
        # Konversi tipe data
        if key in ['SAMPLE_LIMIT', 'N_ESTIMATORS']:
            config_dict[key] = int(config_dict[key])
        elif key in ['RANDOM_SEED', 'TEST_SIZE']:
            config_dict[key] = float(config_dict[key])
    
    return config_dict
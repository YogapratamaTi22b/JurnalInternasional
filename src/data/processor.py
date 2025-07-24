import pandas as pd
import numpy as np
import os
import zipfile
import pefile
import hashlib
import math
import re
from pathlib import Path
import logging
import yaml
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

class MalwareProcessor:
    """
    Process malware samples and extract features for machine learning
    """
    
    def __init__(self, config_path="config/config.yaml"):
        """Initialize the processor with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Setup paths
        self.raw_data_path = Path(self.config['data']['raw_data_path'])
        self.processed_data_path = Path(self.config['data']['processed_data_path'])
        self.external_data_path = Path(self.config['data']['external_data_path'])
        
        # Create directories
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        (self.processed_data_path / "train_test_split").mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize label encoder
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def extract_pe_features(self, file_path):
        """
        Extract features from PE (Portable Executable) files
        """
        features = {}
        
        try:
            # Basic file information
            file_size = os.path.getsize(file_path)
            features['file_size'] = file_size
            
            # Calculate file entropy
            features['entropy'] = self._calculate_entropy(file_path)
            
            # PE specific features
            pe = pefile.PE(file_path)
            
            # DOS Header features
            features['dos_header_magic'] = pe.DOS_HEADER.e_magic
            features['dos_header_bytes_on_last_page'] = pe.DOS_HEADER.e_cblp
            features['dos_header_pages_in_file'] = pe.DOS_HEADER.e_cp
            
            # NT Headers features
            features['nt_signature'] = pe.NT_HEADERS.Signature
            features['file_header_machine'] = pe.FILE_HEADER.Machine
            features['file_header_characteristics'] = pe.FILE_HEADER.Characteristics
            features['optional_header_magic'] = pe.OPTIONAL_HEADER.Magic
            features['optional_header_major_linker_version'] = pe.OPTIONAL_HEADER.MajorLinkerVersion
            features['optional_header_minor_linker_version'] = pe.OPTIONAL_HEADER.MinorLinkerVersion
            features['optional_header_size_of_code'] = pe.OPTIONAL_HEADER.SizeOfCode
            features['optional_header_size_of_initialized_data'] = pe.OPTIONAL_HEADER.SizeOfInitializedData
            features['optional_header_size_of_uninitialized_data'] = pe.OPTIONAL_HEADER.SizeOfUninitializedData
            features['optional_header_address_of_entry_point'] = pe.OPTIONAL_HEADER.AddressOfEntryPoint
            features['optional_header_base_of_code'] = pe.OPTIONAL_HEADER.BaseOfCode
            features['optional_header_image_base'] = pe.OPTIONAL_HEADER.ImageBase
            features['optional_header_section_alignment'] = pe.OPTIONAL_HEADER.SectionAlignment
            features['optional_header_file_alignment'] = pe.OPTIONAL_HEADER.FileAlignment
            features['optional_header_major_os_version'] = pe.OPTIONAL_HEADER.MajorOperatingSystemVersion
            features['optional_header_minor_os_version'] = pe.OPTIONAL_HEADER.MinorOperatingSystemVersion
            features['optional_header_major_image_version'] = pe.OPTIONAL_HEADER.MajorImageVersion
            features['optional_header_minor_image_version'] = pe.OPTIONAL_HEADER.MinorImageVersion
            features['optional_header_major_subsystem_version'] = pe.OPTIONAL_HEADER.MajorSubsystemVersion
            features['optional_header_minor_subsystem_version'] = pe.OPTIONAL_HEADER.MinorSubsystemVersion
            features['optional_header_size_of_image'] = pe.OPTIONAL_HEADER.SizeOfImage
            features['optional_header_size_of_headers'] = pe.OPTIONAL_HEADER.SizeOfHeaders
            features['optional_header_checksum'] = pe.OPTIONAL_HEADER.CheckSum
            features['optional_header_subsystem'] = pe.OPTIONAL_HEADER.Subsystem
            features['optional_header_dll_characteristics'] = pe.OPTIONAL_HEADER.DllCharacteristics
            
            # Sections information
            features['number_of_sections'] = len(pe.sections)
            
            # Section features (aggregate)
            section_entropies = []
            section_sizes = []
            section_virtual_sizes = []
            
            for section in pe.sections:
                section_entropies.append(section.get_entropy())
                section_sizes.append(section.SizeOfRawData)
                section_virtual_sizes.append(section.Misc_VirtualSize)
            
            features['sections_min_entropy'] = min(section_entropies) if section_entropies else 0
            features['sections_max_entropy'] = max(section_entropies) if section_entropies else 0
            features['sections_mean_entropy'] = np.mean(section_entropies) if section_entropies else 0
            features['sections_min_rawsize'] = min(section_sizes) if section_sizes else 0
            features['sections_max_rawsize'] = max(section_sizes) if section_sizes else 0
            features['sections_mean_rawsize'] = np.mean(section_sizes) if section_sizes else 0
            features['sections_min_virtualsize'] = min(section_virtual_sizes) if section_virtual_sizes else 0
            features['sections_max_virtualsize'] = max(section_virtual_sizes) if section_virtual_sizes else 0
            features['sections_mean_virtualsize'] = np.mean(section_virtual_sizes) if section_virtual_sizes else 0
            
            # Imports information
            try:
                features['number_of_imports'] = len(pe.DIRECTORY_ENTRY_IMPORT)
                
                import_names = []
                for entry in pe.DIRECTORY_ENTRY_IMPORT:
                    import_names.append(entry.dll.decode('utf-8').lower())
                
                # Common suspicious imports
                suspicious_imports = ['kernel32.dll', 'ntdll.dll', 'advapi32.dll', 'ws2_32.dll']
                for imp in suspicious_imports:
                    features[f'imports_{imp.replace(".", "_")}'] = 1 if imp in import_names else 0
                    
            except AttributeError:
                features['number_of_imports'] = 0
                for imp in ['kernel32_dll', 'ntdll_dll', 'advapi32_dll', 'ws2_32_dll']:
                    features[f'imports_{imp}'] = 0
            
            # Exports information
            try:
                features['number_of_exports'] = len(pe.DIRECTORY_ENTRY_EXPORT.symbols)
            except AttributeError:
                features['number_of_exports'] = 0
            
            pe.close()
            
        except Exception as e:
            self.logger.warning(f"Error extracting PE features from {file_path}: {e}")
            # Return default features on error
            features = self._get_default_features()
            
        return features
    
    def _calculate_entropy(self, file_path):
        """Calculate Shannon entropy of a file"""
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            if len(data) == 0:
                return 0
            
            # Count byte frequencies
            byte_counts = [0] * 256
            for byte in data:
                byte_counts[byte] += 1
            
            # Calculate entropy
            entropy = 0
            data_len = len(data)
            
            for count in byte_counts:
                if count > 0:
                    frequency = count / data_len
                    entropy -= frequency * math.log2(frequency)
            
            return entropy
            
        except Exception as e:
            self.logger.warning(f"Error calculating entropy for {file_path}: {e}")
            return 0
    
    def _get_default_features(self):
        """Return default feature values when extraction fails"""
        default_features = {
            'file_size': 0,
            'entropy': 0,
            'dos_header_magic': 0,
            'dos_header_bytes_on_last_page': 0,
            'dos_header_pages_in_file': 0,
            'nt_signature': 0,
            'file_header_machine': 0,
            'file_header_characteristics': 0,
            'optional_header_magic': 0,
            'optional_header_major_linker_version': 0,
            'optional_header_minor_linker_version': 0,
            'optional_header_size_of_code': 0,
            'optional_header_size_of_initialized_data': 0,
            'optional_header_size_of_uninitialized_data': 0,
            'optional_header_address_of_entry_point': 0,
            'optional_header_base_of_code': 0,
            'optional_header_image_base': 0,
            'optional_header_section_alignment': 0,
            'optional_header_file_alignment': 0,
            'optional_header_major_os_version': 0,
            'optional_header_minor_os_version': 0,
            'optional_header_major_image_version': 0,
            'optional_header_minor_image_version': 0,
            'optional_header_major_subsystem_version': 0,
            'optional_header_minor_subsystem_version': 0,
            'optional_header_size_of_image': 0,
            'optional_header_size_of_headers': 0,
            'optional_header_checksum': 0,
            'optional_header_subsystem': 0,
            'optional_header_dll_characteristics': 0,
            'number_of_sections': 0,
            'sections_min_entropy': 0,
            'sections_max_entropy': 0,
            'sections_mean_entropy': 0,
            'sections_min_rawsize': 0,
            'sections_max_rawsize': 0,
            'sections_mean_rawsize': 0,
            'sections_min_virtualsize': 0,
            'sections_max_virtualsize': 0,
            'sections_mean_virtualsize': 0,
            'number_of_imports': 0,
            'imports_kernel32_dll': 0,
            'imports_ntdll_dll': 0,
            'imports_advapi32_dll': 0,
            'imports_ws2_32_dll': 0,
            'number_of_exports': 0
        }
        return default_features
    
    def process_downloaded_samples(self):
        """Process downloaded malware samples and extract features"""
        self.logger.info("Processing downloaded malware samples")
        
        # Load metadata
        metadata_path = self.raw_data_path / "downloaded_samples_metadata.csv"
        
        if not metadata_path.exists():
            self.logger.warning("No downloaded samples metadata found, creating synthetic dataset")
            return self.create_synthetic_features()
        
        metadata_df = pd.read_csv(metadata_path)
        self.logger.info(f"Found {len(metadata_df)} samples to process")
        
        processed_samples = []
        
        for idx, row in metadata_df.iterrows():
            try:
                file_path = row['file_path']
                
                # Extract zip if needed
                extracted_path = self._extract_sample(file_path)
                
                if extracted_path and os.path.exists(extracted_path):
                    # Extract features
                    features = self.extract_pe_features(extracted_path)
                    
                    # Add metadata
                    features['sha256_hash'] = row['sha256_hash']
                    features['family'] = row['family']
                    features['file_type'] = row['file_type']
                    features['signature'] = row['signature']
                    
                    processed_samples.append(features)
                    
                    # Clean up extracted file
                    if extracted_path != file_path:
                        os.remove(extracted_path)
                        
                else:
                    self.logger.warning(f"Could not extract or find file: {file_path}")
                    
            except Exception as e:
                self.logger.error(f"Error processing sample {row['sha256_hash']}: {e}")
                continue
        
        if processed_samples:
            # Convert to DataFrame
            features_df = pd.DataFrame(processed_samples)
            
            # Save processed features
            processed_path = self.processed_data_path / "engineered_features.csv"
            features_df.to_csv(processed_path, index=False)
            
            self.logger.info(f"Processed {len(processed_samples)} samples, saved to {processed_path}")
            return features_df
        else:
            self.logger.warning("No samples were successfully processed, creating synthetic dataset")
            return self.create_synthetic_features()
    
    def _extract_sample(self, zip_path):
        """Extract malware sample from zip file"""
        try:
            extract_dir = Path(zip_path).parent / "temp_extract"
            extract_dir.mkdir(exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # MalwareBazaar zips are password protected with 'infected'
                zip_ref.setpassword(b'infected')
                
                # Extract first file
                file_list = zip_ref.namelist()
                if file_list:
                    extracted_file = zip_ref.extract(file_list[0], extract_dir)
                    return extracted_file
                    
        except Exception as e:
            self.logger.warning(f"Error extracting {zip_path}: {e}")
            
        return None
    
    def create_synthetic_features(self, size=1000):
        """Create synthetic feature dataset for testing"""
        self.logger.info(f"Creating synthetic feature dataset with {size} samples")
        
        np.random.seed(42)
        
        families = self.config['malware_families']
        synthetic_data = []
        
        for i in range(size):
            # Random family
            family = np.random.choice(families)
            
            # Generate realistic-looking features based on family
            features = self._generate_synthetic_features(family)
            features['family'] = family
            features['sha256_hash'] = f"synthetic_{i:06d}"
            
            synthetic_data.append(features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(synthetic_data)
        
        # Save synthetic features
        synthetic_path = self.processed_data_path / "engineered_features.csv"
        features_df.to_csv(synthetic_path, index=False)
        
        self.logger.info(f"Created synthetic dataset with {len(features_df)} samples")
        return features_df
    
    def _generate_synthetic_features(self, family):
        """Generate synthetic features based on malware family"""
        base_features = self._get_default_features()
        
        # Adjust features based on family characteristics
        if family.lower() == 'trojan':
            base_features['file_size'] = np.random.randint(50000, 2000000)
            base_features['entropy'] = np.random.uniform(6.5, 8.0)
            base_features['number_of_imports'] = np.random.randint(10, 50)
            base_features['imports_kernel32_dll'] = 1
            base_features['imports_advapi32_dll'] = np.random.choice([0, 1])
            
        elif family.lower() == 'ransomware':
            base_features['file_size'] = np.random.randint(100000, 5000000)
            base_features['entropy'] = np.random.uniform(7.0, 8.0)
            base_features['number_of_imports'] = np.random.randint(15, 60)
            base_features['imports_kernel32_dll'] = 1
            base_features['imports_advapi32_dll'] = 1
            base_features['number_of_sections'] = np.random.randint(4, 8)
            
        elif family.lower() == 'adware':
            base_features['file_size'] = np.random.randint(20000, 1000000)
            base_features['entropy'] = np.random.uniform(5.5, 7.0)
            base_features['number_of_imports'] = np.random.randint(5, 30)
            base_features['imports_ws2_32_dll'] = 1
            
        elif family.lower() == 'spyware':
            base_features['file_size'] = np.random.randint(30000, 1500000)
            base_features['entropy'] = np.random.uniform(6.0, 7.5)
            base_features['number_of_imports'] = np.random.randint(8, 40)
            base_features['imports_kernel32_dll'] = 1
            base_features['imports_ws2_32_dll'] = np.random.choice([0, 1])
            
        # Add some random noise to all features
        for key in base_features:
            if isinstance(base_features[key], (int, float)) and key != 'entropy':
                noise = np.random.normal(0, 0.1 * abs(base_features[key]))
                base_features[key] = max(0, base_features[key] + noise)
        
        return base_features
    
    def prepare_training_data(self, test_size=None, random_state=None):
        """Prepare data for training - split, scale, and encode labels"""
        if test_size is None:
            test_size = self.config['models']['test_size']
        if random_state is None:
            random_state = self.config['models']['random_state']
            
        # Load processed features
        features_path = self.processed_data_path / "engineered_features.csv"
        
        if not features_path.exists():
            self.logger.error("No processed features found. Run process_downloaded_samples() first.")
            return None
            
        df = pd.read_csv(features_path)
        self.logger.info(f"Loaded {len(df)} samples for training")
        
        # Separate features and labels
        feature_cols = [col for col in df.columns if col not in ['family', 'sha256_hash', 'signature', 'file_type']]
        X = df[feature_cols]
        y = df['family']
        
        # Handle missing values
        X = X.fillna(0)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Save training data
        train_test_dir = self.processed_data_path / "train_test_split"
        
        # Save as CSV
        pd.DataFrame(X_train_scaled, columns=feature_cols).to_csv(train_test_dir / "X_train.csv", index=False)
        pd.DataFrame(X_test_scaled, columns=feature_cols).to_csv(train_test_dir / "X_test.csv", index=False)
        pd.DataFrame(y_train, columns=['family_encoded']).to_csv(train_test_dir / "y_train.csv", index=False)
        pd.DataFrame(y_test, columns=['family_encoded']).to_csv(train_test_dir / "y_test.csv", index=False)
        
        # Save as numpy arrays
        np.savez(self.processed_data_path / "scaled_features.npz",
                X_train=X_train_scaled, X_test=X_test_scaled,
                y_train=y_train, y_test=y_test,
                feature_names=feature_cols)
        
        # Save preprocessors
        with open(self.processed_data_path / "label_encoder.pkl", 'wb') as f:
            pickle.dump(self.label_encoder, f)
            
        with open(self.processed_data_path / "scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save feature names
        with open(self.processed_data_path / "feature_names.pkl", 'wb') as f:
            pickle.dump(feature_cols, f)
        
        self.logger.info(f"Training data prepared: {X_train_scaled.shape[0]} train, {X_test_scaled.shape[0]} test samples")
        self.logger.info(f"Number of features: {X_train_scaled.shape[1]}")
        self.logger.info(f"Number of classes: {len(self.label_encoder.classes_)}")
        self.logger.info(f"Classes: {list(self.label_encoder.classes_)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def load_training_data(self):
        """Load prepared training data"""
        try:
            # Load numpy arrays
            data = np.load(self.processed_data_path / "scaled_features.npz")
            X_train = data['X_train']
            X_test = data['X_test']
            y_train = data['y_train']
            y_test = data['y_test']
            
            # Load preprocessors
            with open(self.processed_data_path / "label_encoder.pkl", 'rb') as f:
                self.label_encoder = pickle.load(f)
                
            with open(self.processed_data_path / "scaler.pkl", 'rb') as f:
                self.scaler = pickle.load(f)
            
            self.logger.info("Training data loaded successfully")
            return X_train, X_test, y_train, y_test
            
        except FileNotFoundError as e:
            self.logger.error(f"Training data not found: {e}")
            self.logger.info("Run prepare_training_data() first")
            return None
    
    def get_class_names(self):
        """Get class names from label encoder"""
        if hasattr(self.label_encoder, 'classes_'):
            return list(self.label_encoder.classes_)
        else:
            return self.config['malware_families']

def main():
    """Main function for testing the processor"""
    processor = MalwareProcessor()
    
    # Process samples
    features_df = processor.process_downloaded_samples()
    print(f"Extracted features from {len(features_df)} samples")
    
    # Prepare training data
    training_data = processor.prepare_training_data()
    
    if training_data:
        X_train, X_test, y_train, y_test = training_data
        print(f"Training data prepared: {X_train.shape}, {X_test.shape}")
        print(f"Classes: {processor.get_class_names()}")

if __name__ == "__main__":
    main()
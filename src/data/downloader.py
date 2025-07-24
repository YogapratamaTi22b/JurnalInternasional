import requests
import pandas as pd
import os
import time
import hashlib
from pathlib import Path
from tqdm import tqdm
import logging
import yaml

class MalwareBazaarDownloader:
    """
    Download malware samples from MalwareBazaar
    """
    
    def __init__(self, config_path="config/config.yaml"):
        """Initialize the downloader with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.base_url = self.config['data']['malware_bazaar']['base_url']
        self.download_limit = self.config['data']['malware_bazaar']['download_limit']
        self.file_types = self.config['data']['malware_bazaar']['file_types']
        
        # Setup directories
        self.raw_data_path = Path(self.config['data']['raw_data_path'])
        self.external_data_path = Path(self.config['data']['external_data_path'])
        self.malware_bazaar_path = self.external_data_path / "malware_bazaar_samples"
        
        # Create directories if they don't exist
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.external_data_path.mkdir(parents=True, exist_ok=True)
        self.malware_bazaar_path.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def get_recent_samples(self, limit=None):
        """
        Get recent malware samples metadata from MalwareBazaar
        """
        if limit is None:
            limit = self.download_limit
            
        self.logger.info(f"Fetching recent samples metadata (limit: {limit})")
        
        # API endpoint for recent samples
        url = f"{self.base_url}"
        
        # Prepare request data
        data = {
            'query': 'get_recent',
            'selector': str(limit)
        }
        
        try:
            response = requests.post(url, data=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            
            if result['query_status'] == 'ok':
                samples_df = pd.DataFrame(result['data'])
                self.logger.info(f"Successfully fetched {len(samples_df)} samples")
                return samples_df
            else:
                self.logger.error(f"API returned error: {result.get('query_status')}")
                return pd.DataFrame()
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching samples: {e}")
            return pd.DataFrame()
    
    def download_sample(self, sha256_hash, max_retries=3):
        """
        Download a single malware sample by SHA256 hash
        """
        url = self.base_url
        data = {
            'query': 'get_file',
            'sha256_hash': sha256_hash
        }
        
        file_path = self.malware_bazaar_path / f"{sha256_hash}.zip"
        
        # Skip if file already exists
        if file_path.exists():
            return str(file_path)
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, data=data, timeout=60)
                
                if response.status_code == 200:
                    # Check if response contains file data
                    if response.headers.get('content-type') == 'application/zip':
                        with open(file_path, 'wb') as f:
                            f.write(response.content)
                        self.logger.info(f"Downloaded: {sha256_hash}")
                        return str(file_path)
                    else:
                        # Parse JSON response for error
                        try:
                            result = response.json()
                            if result.get('query_status') == 'file_not_found':
                                self.logger.warning(f"File not found: {sha256_hash}")
                                return None
                        except:
                            pass
                            
                time.sleep(2 ** attempt)  # Exponential backoff
                
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for {sha256_hash}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    
        self.logger.error(f"Failed to download after {max_retries} attempts: {sha256_hash}")
        return None
    
    def download_samples_batch(self, samples_df, max_samples=None):
        """
        Download multiple malware samples
        """
        if max_samples:
            samples_df = samples_df.head(max_samples)
            
        self.logger.info(f"Starting download of {len(samples_df)} samples")
        
        downloaded_files = []
        failed_downloads = []
        
        for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc="Downloading samples"):
            sha256_hash = row['sha256_hash']
            
            file_path = self.download_sample(sha256_hash)
            
            if file_path:
                downloaded_files.append({
                    'sha256_hash': sha256_hash,
                    'file_path': file_path,
                    'file_type': row.get('file_type', 'unknown'),
                    'signature': row.get('signature', 'unknown'),
                    'family': self._extract_family(row.get('signature', '')),
                    'first_seen': row.get('first_seen', ''),
                    'file_size': row.get('file_size', 0)
                })
            else:
                failed_downloads.append(sha256_hash)
            
            # Rate limiting - be respectful to the API
            time.sleep(1)
        
        # Save download metadata
        if downloaded_files:
            metadata_df = pd.DataFrame(downloaded_files)
            metadata_path = self.raw_data_path / "downloaded_samples_metadata.csv"
            metadata_df.to_csv(metadata_path, index=False)
            self.logger.info(f"Saved metadata for {len(downloaded_files)} downloaded files")
        
        # Save failed downloads
        if failed_downloads:
            failed_df = pd.DataFrame({'failed_sha256': failed_downloads})
            failed_path = self.raw_data_path / "failed_downloads.csv"
            failed_df.to_csv(failed_path, index=False)
            
        self.logger.info(f"Download complete. Success: {len(downloaded_files)}, Failed: {len(failed_downloads)}")
        
        return downloaded_files
    
    def _extract_family(self, signature):
        """
        Extract malware family from signature string
        """
        if not signature:
            return "Unknown"
            
        signature_lower = signature.lower()
        
        # Common family patterns
        family_patterns = {
            'trojan': ['trojan', 'trj'],
            'ransomware': ['ransom', 'crypt', 'locker'],
            'adware': ['adware', 'ads'],
            'spyware': ['spy', 'keylog', 'stealer'],
            'backdoor': ['backdoor', 'back'],
            'rootkit': ['rootkit', 'root'],
            'worm': ['worm'],
            'virus': ['virus'],
            'botnet': ['bot', 'botnet'],
            'stealer': ['steal', 'grab', 'clip']
        }
        
        for family, patterns in family_patterns.items():
            for pattern in patterns:
                if pattern in signature_lower:
                    return family.capitalize()
        
        return "Unknown"
    
    def create_synthetic_dataset(self, size=1000):
        """
        Create synthetic malware dataset for testing when API is not available
        """
        self.logger.info(f"Creating synthetic dataset with {size} samples")
        
        import random
        import string
        
        families = self.config['malware_families']
        file_types = ['exe', 'dll', 'pdf', 'doc']
        
        synthetic_data = []
        
        for i in range(size):
            # Generate fake hash
            fake_hash = ''.join(random.choices(string.hexdigits.lower(), k=64))
            
            # Random family and file type
            family = random.choice(families)
            file_type = random.choice(file_types)
            file_size = random.randint(1024, 10485760)  # 1KB to 10MB
            
            synthetic_data.append({
                'sha256_hash': fake_hash,
                'family': family,
                'file_type': file_type,
                'file_size': file_size,
                'signature': f"{family}.Generic.{random.randint(1000, 9999)}",
                'first_seen': f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
            })
        
        # Save synthetic dataset
        synthetic_df = pd.DataFrame(synthetic_data)
        synthetic_path = self.raw_data_path / "synthetic_malware_dataset.csv"
        synthetic_df.to_csv(synthetic_path, index=False)
        
        self.logger.info(f"Synthetic dataset saved to {synthetic_path}")
        return synthetic_df

def main():
    """Main function for testing the downloader"""
    downloader = MalwareBazaarDownloader()
    
    # Get recent samples
    samples_df = downloader.get_recent_samples(limit=100)
    
    if not samples_df.empty:
        # Download first 10 samples for testing
        downloaded = downloader.download_samples_batch(samples_df, max_samples=10)
        print(f"Downloaded {len(downloaded)} samples")
    else:
        # Create synthetic dataset if API fails
        print("API unavailable, creating synthetic dataset")
        synthetic_df = downloader.create_synthetic_dataset(1000)
        print(f"Created synthetic dataset with {len(synthetic_df)} samples")

if __name__ == "__main__":
    main()
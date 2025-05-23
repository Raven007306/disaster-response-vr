"""
Module for collecting data from NASA Earth Data portal.
"""

import os
import json
import requests
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class NASAEarthDataCollector:
    """
    Collector for NASA Earth Data portal.
    
    This class handles authentication and data retrieval from
    NASA's Earth Data portal, focusing on disaster-related datasets.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the NASA Earth Data collector.
        
        Args:
            config_path (str, optional): Path to the API keys configuration file.
                Defaults to None, which will use the default config path.
        """
        if config_path is None:
            config_path = Path(__file__).parents[3] / "config" / "api_keys.json"
            
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.api_key = config['nasa_earth_data']['api_key']
                self.api_url = config['nasa_earth_data']['api_url']
                self.username = config['nasa_earth_data']['username']
                self.password = config['nasa_earth_data']['password']
        except (FileNotFoundError, KeyError) as e:
            logger.error(f"Failed to load NASA Earth Data configuration: {e}")
            raise
            
        self.session = requests.Session()
        self.authenticated = False
        
    def authenticate(self):
        """
        Authenticate with the NASA Earth Data API.
        
        Returns:
            bool: True if authentication was successful, False otherwise.
        """
        try:
            # NASA Earth Data typically uses token-based authentication
            auth_url = f"{self.api_url}/token"
            response = self.session.post(
                auth_url,
                auth=(self.username, self.password),
                params={"client_id": self.api_key}
            )
            
            if response.status_code == 200:
                token_data = response.json()
                self.session.headers.update({
                    "Authorization": f"Bearer {token_data.get('access_token')}"
                })
                self.authenticated = True
                logger.info("Successfully authenticated with NASA Earth Data API")
                return True
            else:
                logger.error(f"Authentication failed: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False
    
    def search_datasets(self, keywords=None, temporal=None, spatial=None):
        """
        Search for datasets in NASA Earth Data.
        
        Args:
            keywords (list, optional): List of keywords to search for.
            temporal (tuple, optional): Temporal range (start_date, end_date) in ISO format.
            spatial (tuple, optional): Spatial bounding box (min_lon, min_lat, max_lon, max_lat).
            
        Returns:
            list: List of matching datasets.
        """
        if not self.authenticated and not self.authenticate():
            logger.error("Not authenticated. Cannot search datasets.")
            return []
            
        try:
            search_url = f"{self.api_url}/search"
            params = {}
            
            if keywords:
                params["keywords"] = ",".join(keywords)
                
            if temporal:
                params["temporal"] = f"{temporal[0]},{temporal[1]}"
                
            if spatial:
                params["bbox"] = ",".join(map(str, spatial))
                
            response = self.session.get(search_url, params=params)
            
            if response.status_code == 200:
                return response.json().get("entries", [])
            else:
                logger.error(f"Failed to search datasets: {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Error searching datasets: {e}")
            return []
    
    def download_dataset(self, dataset_id, output_path):
        """
        Download a dataset from NASA Earth Data.
        
        Args:
            dataset_id (str): ID of the dataset to download.
            output_path (str): Path to save the downloaded data.
            
        Returns:
            str: Path to the downloaded file, or None if download failed.
        """
        if not self.authenticated and not self.authenticate():
            logger.error("Not authenticated. Cannot download dataset.")
            return None
            
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            download_url = f"{self.api_url}/datasets/{dataset_id}/download"
            response = self.session.get(download_url, stream=True)
            
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info(f"Successfully downloaded dataset to {output_path}")
                return output_path
            else:
                logger.error(f"Failed to download dataset: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            return None 
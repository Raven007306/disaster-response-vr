"""
Collector for ISRO's Bhuvan portal Ellipsoid to Geoid Conversion API.
"""

import os
import json
import logging
import requests
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BhuvanCollector:
    """Collector for ISRO's Bhuvan portal Ellipsoid to Geoid Conversion API."""
    
    def __init__(self):
        """Initialize the Bhuvan collector."""
        try:
            # Load API keys and data sources
            config_path = Path("config/api_keys.json")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Get Bhuvan configuration
                bhuvan_config = config.get('BHUVAN', {})
                self.api_url = bhuvan_config.get('api_url', 'https://bhuvan-app1.nrsc.gov.in/api')
                self.token = bhuvan_config.get('token', '')
            else:
                # If config file doesn't exist, use default values
                self.api_url = 'https://bhuvan-app1.nrsc.gov.in/api'
                self.token = '7982c269f28b394d5d571558dae211cc2c16b3df'  # Default token for testing
            
            # If token is empty, use the default token
            if not self.token:
                self.token = '7982c269f28b394d5d571558dae211cc2c16b3df'  # Default token for testing
                
            logger.info(f"Initialized Bhuvan collector with API URL: {self.api_url}")
            logger.info(f"Using token: {self.token[:5]}...{self.token[-5:] if len(self.token) > 10 else ''}")
        except Exception as e:
            logger.error(f"Error initializing Bhuvan collector: {e}")
            # Set default values if configuration fails
            self.api_url = 'https://bhuvan-app1.nrsc.gov.in/api'
            self.token = '7982c269f28b394d5d571558dae211cc2c16b3df'  # Default token for testing
    
    def authenticate(self):
        
        try:
            if not self.token:
                logger.error("No access token provided. Please obtain a token from the Bhuvan API website.")
                return False
            
            logger.info(f"Using token: {self.token[:5]}...{self.token[-5:] if len(self.token) > 10 else ''}")
            return True
                
        except Exception as e:
            logger.error(f"Error during authentication: {e}")
            return False
    
    def get_ellipsoid_to_geoid_conversion(self, tile_id, datum="geoid", satellite="CDEM", output_path=None):
        
        try:
            if not self.token:
                if not self.authenticate():
                    logger.error("Authentication required to download CartoDEM data")
                    return None
            
            # Construct the API URL based on the documentation
            url = f"{self.api_url}/geoid/curl_gdal_api.php"
            params = {
                "id": tile_id,
                "datum": datum.lower(),  # Ensure lowercase for consistency
                "se": satellite,
                "key": self.token  # Using 'key' parameter as specified in the API docs
            }
            
            # Print the full URL for debugging
            full_url = f"{url}?{'&'.join([f'{k}={v}' for k, v in params.items()])}"
            logger.info(f"Downloading CartoDEM data with URL: {full_url}")
            
            # Stream the response to handle large files
            response = requests.get(url, params=params, stream=True)
            
            if response.status_code == 200:
                # Check if the response is a zip file
                content_type = response.headers.get('Content-Type', '')
                if 'application/zip' not in content_type and 'application/octet-stream' not in content_type:
                    logger.error(f"Unexpected content type: {content_type}. Response may not be a zip file.")
                    logger.error(f"Response text: {response.text[:500]}...")  # Log first 500 chars
                    return {"error": "unexpected_content", "content_type": content_type, "response_text": response.text[:500]}
                
                # Determine output filename
                if not output_path:
                    # Create a filename based on the parameters
                    filename = f"cartodem_{tile_id}_{datum}_{satellite}.zip"
                    output_dir = Path("data/raw/cartodem")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / filename
                
                # Save the file
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                logger.info(f"Successfully downloaded CartoDEM data to {output_path}")
                return str(output_path)
            else:
                logger.error(f"Failed to download CartoDEM data: {response.status_code} - {response.text}")
                return {"error": "http_error", "error_description": f"HTTP {response.status_code}", "response_text": response.text}
                
        except Exception as e:
            logger.error(f"Error downloading CartoDEM data: {e}")
            return {"error": "exception", "error_description": str(e)}

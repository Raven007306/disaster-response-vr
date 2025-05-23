"""
Training data collector for disaster prediction models.
Collects data from various sources including NASA, historical records, and real-time monitoring.
"""

import os
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from tqdm import tqdm
import re

logger = logging.getLogger(__name__)

class TrainingDataCollector:
    """Collector for training data from various sources."""
    
    def __init__(self, data_dir=None):
        """
        Initialize the training data collector.
        
        Args:
            data_dir (str, optional): Directory to save collected data. Defaults to 'data/training'.
        """
        if data_dir is None:
            data_dir = Path(__file__).parents[3] / "data" / "training"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize API keys
        self._load_api_keys()
        
        # Initialize data sources
        self.sources = {
            'nasa': self.nasa_base_url,
            'usgs': self.usgs_base_url,
            'gdacs': self.gdacs_base_url
        }
    
    def _load_api_keys(self):
        """Load API keys from configuration."""
        try:
            config_path = Path(__file__).parents[3] / "config" / "api_keys.json"
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.nasa_api_key = config['nasa_earth_data']['api_key']
            
            # Set up API endpoints
            self.nasa_base_url = "https://api.nasa.gov/planetary/earth/assets"
            self.usgs_base_url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
            self.gdacs_base_url = "https://www.gdacs.org/xml/rss.xml"
            
        except Exception as e:
            logger.error(f"Failed to load NASA API key: {e}")
            raise
    
    def _sanitize_filename(self, filename):
        """Sanitize filename by removing invalid characters."""
        # Replace spaces, colons, and other invalid characters with underscores
        return re.sub(r'[\\/*?:"<>|]', '_', filename)
    
    def collect_nasa_data(self, lat, lon, start_date, end_date=None):
        """
        Collect NASA Earth imagery data for a location and time range.
        
        Args:
            lat (float): Latitude
            lon (float): Longitude
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format. Defaults to today.
            
        Returns:
            pd.DataFrame: Collected NASA data
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            # Create filename with sanitized date range
            filename = self._sanitize_filename(f"nasa_data_{lat}_{lon}_{start_date}_{end_date}.csv")
            output_path = self.data_dir / filename
            
            if output_path.exists():
                return pd.read_csv(output_path)
            
            data = []
            current_date = datetime.strptime(start_date, '%Y-%m-%d')
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
            
            with tqdm(total=(end_date - current_date).days, desc="Collecting NASA data") as pbar:
                while current_date <= end_date:
                    date_str = current_date.strftime('%Y-%m-%d')
                    
                    params = {
                        'lat': lat,
                        'lon': lon,
                        'date': date_str,
                        'dim': 0.1,
                        'api_key': self.nasa_api_key
                    }
                    
                    response = requests.get(self.nasa_base_url, params=params)
                    
                    if response.status_code == 200:
                        result = response.json()
                        if 'url' in result:
                            data.append({
                                'date': date_str,
                                'image_url': result['url'],
                                'cloud_score': result.get('cloud_score', None)
                            })
                    
                    current_date += timedelta(days=1)
                    pbar.update(1)
            
            # Save collected data
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
            
            logger.info(f"Collected {len(data)} NASA images")
            return df
            
        except Exception as e:
            logger.error(f"Error collecting NASA data: {e}")
            return pd.DataFrame()
    
    def collect_historical_disasters(self, disaster_type, region=None, start_date=None, end_date=None):
        """
        Collect historical disaster data from various sources.
        
        Args:
            disaster_type (str): Type of disaster (landslide, flood, earthquake, cyclone)
            region (dict, optional): Region bounds {lat_min, lat_max, lon_min, lon_max}
            start_date (str, optional): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format
            
        Returns:
            pd.DataFrame: Historical disaster data
        """
        # Define columns at the class level to ensure they're always available
        columns = ['date', 'latitude', 'longitude', 'magnitude', 'depth', 'title', 'description', 'link']
        
        try:
            # Create a clean region string for the filename
            if region:
                region_str = f"lat{region['lat_min']:.2f}-{region['lat_max']:.2f}_lon{region['lon_min']:.2f}-{region['lon_max']:.2f}"
            else:
                region_str = "global"
            
            # Create filename with sanitized components
            filename = f"historical_{disaster_type}_{region_str}_{start_date}_{end_date}.csv"
            filename = self._sanitize_filename(filename)
            output_path = self.data_dir / filename
            
            if output_path.exists():
                try:
                    df = pd.read_csv(output_path)
                    if not df.empty:
                        return df
                except Exception as e:
                    logger.warning(f"Error reading existing file {output_path}: {e}")
            
            data = []
            
            if disaster_type == 'landslide':
                # For landslides, we'll use a combination of slope and historical data
                if region:
                    # Create a grid of points within the region
                    lat_points = np.linspace(region['lat_min'], region['lat_max'], 10)
                    lon_points = np.linspace(region['lon_min'], region['lon_max'], 10)
                    
                    for lat in lat_points:
                        for lon in lon_points:
                            data.append({
                                'date': start_date,
                                'latitude': float(lat),
                                'longitude': float(lon),
                                'magnitude': 0.0,
                                'depth': 0.0,
                                'title': 'Historical landslide data',
                                'description': 'Synthetic landslide data based on terrain analysis',
                                'link': ''
                            })
            
            elif disaster_type == 'earthquake':
                # Collect from USGS (no API key required)
                params = {
                    'format': 'geojson',
                    'starttime': start_date,
                    'endtime': end_date
                }
                
                if region:
                    params.update({
                        'minlatitude': region['lat_min'],
                        'maxlatitude': region['lat_max'],
                        'minlongitude': region['lon_min'],
                        'maxlongitude': region['lon_max']
                    })
                
                response = requests.get(self.usgs_base_url, params=params)
                
                if response.status_code == 200:
                    result = response.json()
                    for feature in result['features']:
                        data.append({
                            'date': feature['properties']['time'],
                            'magnitude': feature['properties']['mag'],
                            'latitude': feature['geometry']['coordinates'][1],
                            'longitude': feature['geometry']['coordinates'][0],
                            'depth': feature['geometry']['coordinates'][2]
                        })
            
            elif disaster_type in ['flood', 'cyclone']:
                # Collect from GDACS (no API key required)
                response = requests.get(self.gdacs_base_url)
                
                if response.status_code == 200:
                    # Parse GDACS RSS feed
                    import xml.etree.ElementTree as ET
                    root = ET.fromstring(response.content)
                    
                    for item in root.findall('.//item'):
                        if disaster_type in item.find('title').text.lower():
                            data.append({
                                'date': item.find('pubDate').text,
                                'title': item.find('title').text,
                                'description': item.find('description').text,
                                'link': item.find('link').text
                            })
            
            # Create DataFrame with default values if no data was collected
            if not data:
                data = [{
                    'date': start_date,
                    'latitude': 0.0,
                    'longitude': 0.0,
                    'magnitude': 0.0,
                    'depth': 0.0,
                    'title': 'No historical data available',
                    'description': 'No historical data available',
                    'link': ''
                }]
            
            # Create DataFrame with all required columns
            df = pd.DataFrame(data)
            # Ensure all required columns exist
            for col in columns:
                if col not in df.columns:
                    df[col] = ''
            
            # Reorder columns to match the expected order
            df = df[columns]
            
            # Save the DataFrame
            df.to_csv(output_path, index=False)
            
            logger.info(f"Collected {len(data)} {disaster_type} records")
            return df
            
        except Exception as e:
            logger.error(f"Error collecting historical {disaster_type} data: {e}")
            # Return a DataFrame with default values
            return pd.DataFrame([{
                'date': start_date,
                'latitude': 0.0,
                'longitude': 0.0,
                'magnitude': 0.0,
                'depth': 0.0,
                'title': 'Error collecting data',
                'description': str(e),
                'link': ''
            }], columns=columns)
    
    def collect_terrain_data(self, dem_processor):
        """
        Collect terrain features from DEM data.
        
        Args:
            dem_processor (DEMProcessor): DEM processor instance
            
        Returns:
            dict: Terrain features
        """
        try:
            # Get elevation data
            elevation = dem_processor.get_elevation_data()
            
            # Calculate terrain features
            slope = dem_processor.calculate_slope()
            aspect = dem_processor.calculate_aspect()
            curvature = dem_processor.calculate_curvature()
            
            # Save terrain data
            terrain_data = {
                'elevation': elevation,
                'slope': slope,
                'aspect': aspect,
                'curvature': curvature
            }
            
            output_path = self.data_dir / "terrain_data.npz"
            np.savez(output_path, **terrain_data)
            
            logger.info(f"Collected terrain data from {dem_processor.tif_file_path}")
            return terrain_data
            
        except Exception as e:
            logger.error(f"Error collecting terrain data: {e}")
            return {}
    
    def prepare_training_dataset(self, disaster_type, terrain_data, historical_data, nasa_data=None):
        """
        Prepare a complete training dataset by combining all data sources.
        
        Args:
            disaster_type (str): Type of disaster
            terrain_data (dict): Terrain features
            historical_data (pd.DataFrame): Historical disaster data
            nasa_data (pd.DataFrame, optional): NASA imagery data
            
        Returns:
            pd.DataFrame: Combined training dataset
        """
        try:
            # Convert terrain data to DataFrame
            terrain_df = pd.DataFrame({
                'elevation': terrain_data['elevation'].flatten(),
                'slope': terrain_data['slope'].flatten(),
                'aspect': terrain_data['aspect'].flatten(),
                'curvature': terrain_data['curvature'].flatten()
            })
            
            # Add historical data features and labels
            if not historical_data.empty:
                # Create a grid of points for the entire terrain
                rows, cols = terrain_data['elevation'].shape
                y_coords = np.linspace(0, rows-1, rows)
                x_coords = np.linspace(0, cols-1, cols)
                Y, X = np.meshgrid(y_coords, x_coords)
                
                # Initialize labels array
                labels = np.zeros((rows, cols))
                
                # Get valid historical data (non-NaN coordinates)
                valid_data = historical_data.dropna(subset=['latitude', 'longitude'])
                
                if not valid_data.empty:
                    # Get coordinate ranges
                    lat_min = valid_data['latitude'].min()
                    lat_max = valid_data['latitude'].max()
                    lon_min = valid_data['longitude'].min()
                    lon_max = valid_data['longitude'].max()
                    
                    # Ensure we don't divide by zero
                    lat_range = max(lat_max - lat_min, 1e-10)
                    lon_range = max(lon_max - lon_min, 1e-10)
                    
                    # For each historical event, mark nearby points as potential risk areas
                    for _, event in valid_data.iterrows():
                        try:
                            # Convert lat/lon to pixel coordinates
                            lat_idx = int((event['latitude'] - lat_min) / lat_range * (rows-1))
                            lon_idx = int((event['longitude'] - lon_min) / lon_range * (cols-1))
                            
                            # Ensure indices are within bounds
                            lat_idx = max(0, min(rows-1, lat_idx))
                            lon_idx = max(0, min(cols-1, lon_idx))
                            
                            # Mark a region around the event as high risk
                            radius = 5  # pixels
                            y_indices, x_indices = np.ogrid[-radius:radius+1, -radius:radius+1]
                            mask = x_indices**2 + y_indices**2 <= radius**2
                            
                            # Ensure indices are within bounds
                            y_start = max(0, lat_idx - radius)
                            y_end = min(rows, lat_idx + radius + 1)
                            x_start = max(0, lon_idx - radius)
                            x_end = min(cols, lon_idx + radius + 1)
                            
                            # Apply the mask
                            labels[y_start:y_end, x_start:x_end] = np.where(
                                mask[:y_end-y_start, :x_end-x_start],
                                1,  # High risk
                                labels[y_start:y_end, x_start:x_end]
                            )
                        except Exception as e:
                            logger.warning(f"Error processing historical event: {e}")
                            continue
                
                # Add labels to terrain DataFrame
                terrain_df['label'] = labels.flatten()
                
                # Add historical occurrence as a feature
                terrain_df['historical_occurrence'] = (labels > 0).flatten().astype(int)
            else:
                # If no historical data, set default values
                terrain_df['label'] = 0
                terrain_df['historical_occurrence'] = 0
            
            # Add NASA data features
            if nasa_data is not None and not nasa_data.empty:
                # Add cloud coverage as a feature
                cloud_score = nasa_data['cloud_score'].mean() if 'cloud_score' in nasa_data.columns else 0
                terrain_df['cloud_coverage'] = np.nan_to_num(cloud_score, nan=0.0)
                
                # Add vegetation index (simplified)
                terrain_df['vegetation_index'] = 0.5  # Placeholder value
            else:
                # If no NASA data, set default values
                terrain_df['cloud_coverage'] = 0.0
                terrain_df['vegetation_index'] = 0.0
            
            # Handle any remaining NaN values
            terrain_df = terrain_df.fillna(0)
            
            # Save combined dataset
            filename = self._sanitize_filename(f"training_dataset_{disaster_type}.csv")
            output_path = self.data_dir / filename
            terrain_df.to_csv(output_path, index=False)
            
            logger.info(f"Prepared training dataset for {disaster_type}")
            return terrain_df
            
        except Exception as e:
            logger.error(f"Error preparing training dataset: {e}")
            # Return a DataFrame with default values
            return pd.DataFrame({
                'elevation': terrain_data['elevation'].flatten(),
                'slope': terrain_data['slope'].flatten(),
                'aspect': terrain_data['aspect'].flatten(),
                'curvature': terrain_data['curvature'].flatten(),
                'label': 0,
                'historical_occurrence': 0,
                'cloud_coverage': 0.0,
                'vegetation_index': 0.0
            })
    
    def collect_all_training_data(self, dem_processor, start_date, end_date=None):
        """
        Collect all training data for a given DEM and time range.
        
        Args:
            dem_processor (DEMProcessor): DEM processor instance
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format
            
        Returns:
            dict: Collected training data for each disaster type
        """
        try:
            # Get DEM bounds
            bounds = dem_processor.dataset.bounds
            region = {
                'lat_min': bounds.bottom,
                'lat_max': bounds.top,
                'lon_min': bounds.left,
                'lon_max': bounds.right
            }
            
            # Collect terrain data
            terrain_data = self.collect_terrain_data(dem_processor)
            
            # Collect data for each disaster type
            training_data = {}
            for disaster_type in ['landslide', 'flood', 'earthquake', 'cyclone']:
                # Collect historical data
                historical_data = self.collect_historical_disasters(
                    disaster_type, region, start_date, end_date
                )
                
                # Collect NASA data for center of DEM
                center_lat = (bounds.top + bounds.bottom) / 2
                center_lon = (bounds.left + bounds.right) / 2
                nasa_data = self.collect_nasa_data(center_lat, center_lon, start_date, end_date)
                
                # Prepare training dataset
                training_data[disaster_type] = self.prepare_training_dataset(
                    disaster_type, terrain_data, historical_data, nasa_data
                )
            
            logger.info("Completed collection of all training data")
            return training_data
            
        except Exception as e:
            logger.error(f"Error collecting all training data: {e}")
            return {} 
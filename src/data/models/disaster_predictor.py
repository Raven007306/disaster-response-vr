"""
Disaster prediction models using machine learning and real-time NASA data.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import requests
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DisasterPredictor:
    """Machine learning models for disaster prediction using terrain and NASA data."""
    
    def __init__(self, model_dir=None):
        """
        Initialize the disaster predictor.
        
        Args:
            model_dir (str, optional): Directory to save/load models. Defaults to 'models' in project root.
        """
        if model_dir is None:
            model_dir = Path(__file__).parents[3] / "models"
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Initialize models dictionary
        self.models = {
            'landslide': None,
            'flood': None,
            'earthquake': None,
            'cyclone': None
        }
        
        # Load existing models if available
        self._load_models()
        
        # Load NASA API key
        self._load_api_keys()
        self.nasa_base_url = "https://api.nasa.gov/planetary/earth/assets"
    
    def _load_api_keys(self):
        """Load API keys from configuration."""
        try:
            config_path = Path(__file__).parents[3] / "config" / "api_keys.json"
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.nasa_api_key = config['nasa_earth_data']['api_key']
        except Exception as e:
            logger.error(f"Failed to load NASA API key: {e}")
            self.nasa_api_key = None
    
    def _load_models(self):
        """Load existing trained models if available."""
        for disaster_type in self.models.keys():
            model_path = self.model_dir / f"{disaster_type}_model.joblib"
            if model_path.exists():
                try:
                    self.models[disaster_type] = joblib.load(model_path)
                    logger.info(f"Loaded {disaster_type} prediction model")
                except Exception as e:
                    logger.error(f"Failed to load {disaster_type} model: {e}")
    
    def prepare_training_data(self, dem_data, historical_data=None):
        """
        Prepare training data from DEM and historical disaster data.
        
        Args:
            dem_data (dict): Dictionary containing DEM data and derived features
            historical_data (dict, optional): Historical disaster occurrence data
            
        Returns:
            dict: Prepared training data for each disaster type
        """
        training_data = {}
        
        for disaster_type in self.models.keys():
            # Extract features from DEM data
            features = {
                'elevation': dem_data['elevation_data'],
                'slope': dem_data['slope_degrees'],
                'aspect': dem_data.get('aspect', np.zeros_like(dem_data['elevation_data'])),
                'curvature': dem_data.get('curvature', np.zeros_like(dem_data['elevation_data']))
            }
            
            # Add historical data if available
            if historical_data and disaster_type in historical_data:
                features['historical_occurrence'] = historical_data[disaster_type]
            
            # Convert to DataFrame
            df = pd.DataFrame(features)
            
            # Add labels if available in historical data
            if historical_data and f"{disaster_type}_labels" in historical_data:
                df['label'] = historical_data[f"{disaster_type}_labels"]
            
            training_data[disaster_type] = df
        
        return training_data
    
    def train_models(self, training_data, test_size=0.2):
        """
        Train prediction models for each disaster type.
        
        Args:
            training_data (dict): Training data for each disaster type
            test_size (float): Proportion of data to use for testing
            
        Returns:
            dict: Training results for each model
        """
        results = {}
        
        for disaster_type, data in training_data.items():
            if 'label' not in data.columns:
                logger.warning(f"No labels available for {disaster_type}, skipping training")
                continue
            
            # Prepare features and labels
            X = data.drop('label', axis=1)
            y = data['label']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            # Save model and scaler
            model_path = self.model_dir / f"{disaster_type}_model.joblib"
            scaler_path = self.model_dir / f"{disaster_type}_scaler.joblib"
            
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            
            self.models[disaster_type] = model
            
            results[disaster_type] = {
                'train_score': train_score,
                'test_score': test_score,
                'feature_importance': dict(zip(X.columns, model.feature_importances_))
            }
            
            logger.info(f"Trained {disaster_type} model - Test score: {test_score:.3f}")
        
        return results
    
    def get_nasa_imagery(self, lat, lon, date=None, dim=0.1):
        """
        Get NASA Earth imagery for a specific location and date.
        
        Args:
            lat (float): Latitude
            lon (float): Longitude
            date (str, optional): Date in YYYY-MM-DD format. Defaults to yesterday.
            dim (float, optional): Image dimension in degrees. Defaults to 0.1.
            
        Returns:
            dict: NASA imagery data
        """
        if not self.nasa_api_key:
            logger.error("NASA API key not available")
            return None
        
        if date is None:
            date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        try:
            params = {
                'lat': lat,
                'lon': lon,
                'date': date,
                'dim': dim,
                'api_key': self.nasa_api_key
            }
            
            response = requests.get(self.nasa_base_url, params=params)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get NASA imagery: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting NASA imagery: {e}")
            return None
    
    def predict_disaster_risks(self, dem_data, nasa_data=None):
        """
        Predict disaster risks using trained models and NASA data.
        
        Args:
            dem_data (dict): DEM data and derived features
            nasa_data (dict, optional): NASA Earth imagery data
            
        Returns:
            dict: Risk predictions for each disaster type
        """
        predictions = {}
        
        for disaster_type, model in self.models.items():
            if model is None:
                logger.warning(f"No model available for {disaster_type}")
                continue
            
            # Prepare features
            features = {
                'elevation': dem_data['elevation_data'],
                'slope': dem_data['slope_degrees'],
                'aspect': dem_data.get('aspect', np.zeros_like(dem_data['elevation_data'])),
                'curvature': dem_data.get('curvature', np.zeros_like(dem_data['elevation_data']))
            }
            
            # Add NASA data features if available
            if nasa_data and disaster_type in nasa_data:
                features.update(nasa_data[disaster_type])
            
            # Convert to DataFrame
            X = pd.DataFrame(features)
            
            # Load and apply scaler
            scaler_path = self.model_dir / f"{disaster_type}_scaler.joblib"
            if scaler_path.exists():
                scaler = joblib.load(scaler_path)
                X_scaled = scaler.transform(X)
            else:
                X_scaled = X
            
            # Make predictions
            predictions[disaster_type] = {
                'risk_level': model.predict(X_scaled),
                'risk_probability': model.predict_proba(X_scaled)
            }
        
        return predictions
    
    def update_models(self, new_data):
        """
        Update models with new training data.
        
        Args:
            new_data (dict): New training data for each disaster type
            
        Returns:
            dict: Updated model performance metrics
        """
        # Prepare new training data
        training_data = self.prepare_training_data(new_data)
        
        # Retrain models
        results = self.train_models(training_data)
        
        return results 
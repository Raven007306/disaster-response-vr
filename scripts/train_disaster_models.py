"""
Script to train and evaluate disaster prediction models using historical data and NASA imagery.
"""

import os
import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.data.processors.dem_processor import DEMProcessor
from src.data.models.disaster_predictor import DisasterPredictor

def load_historical_data(data_dir):
    """
    Load historical disaster data from JSON files.
    
    Args:
        data_dir (str): Directory containing historical data files
        
    Returns:
        dict: Historical disaster data
    """
    historical_data = {}
    data_dir = Path(data_dir)
    
    for disaster_type in ['landslide', 'flood', 'earthquake', 'cyclone']:
        data_file = data_dir / f"{disaster_type}_history.json"
        if data_file.exists():
            try:
                with open(data_file, 'r') as f:
                    historical_data[disaster_type] = json.load(f)
                logger.info(f"Loaded historical data for {disaster_type}")
            except Exception as e:
                logger.error(f"Failed to load historical data for {disaster_type}: {e}")
    
    return historical_data

def prepare_training_data(dem_processor, historical_data):
    """
    Prepare training data from DEM and historical data.
    
    Args:
        dem_processor (DEMProcessor): DEM processor instance
        historical_data (dict): Historical disaster data
        
    Returns:
        dict: Prepared training data
    """
    # Get DEM data and derived features
    dem_data = {
        'elevation_data': dem_processor.elevation_data,
        'slope_degrees': dem_processor.calculate_slope(),
        'aspect': dem_processor.calculate_aspect(),
        'curvature': dem_processor.calculate_curvature()
    }
    
    return dem_data, historical_data

def main():
    """Main function to train and evaluate disaster prediction models."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Train disaster prediction models")
    parser.add_argument("dem_file", help="Path to the DEM file")
    parser.add_argument("--historical-data", help="Path to historical data directory", default="data/historical")
    parser.add_argument("--model-dir", help="Path to save models", default="models")
    args = parser.parse_args()
    
    try:
        # Initialize DEM processor
        logger.info(f"Processing DEM file: {args.dem_file}")
        dem_processor = DEMProcessor(args.dem_file)
        
        # Load historical data
        logger.info("Loading historical disaster data")
        historical_data = load_historical_data(args.historical_data)
        
        # Initialize disaster predictor
        predictor = DisasterPredictor(model_dir=args.model_dir)
        
        # Prepare training data
        logger.info("Preparing training data")
        dem_data, historical_data = prepare_training_data(dem_processor, historical_data)
        
        # Train models
        logger.info("Training disaster prediction models")
        results = predictor.train_models(dem_data, historical_data)
        
        # Print training results
        print("\nTraining Results:")
        print("=" * 50)
        for disaster_type, metrics in results.items():
            print(f"\n{disaster_type.upper()} Model:")
            print(f"Training Score: {metrics['train_score']:.3f}")
            print(f"Testing Score: {metrics['test_score']:.3f}")
            print("\nFeature Importance:")
            for feature, importance in metrics['feature_importance'].items():
                print(f"  {feature}: {importance:.3f}")
        
        # Test predictions with NASA data
        print("\nTesting predictions with NASA imagery...")
        bounds = dem_processor.dataset.bounds
        center_lat = (bounds.top + bounds.bottom) / 2
        center_lon = (bounds.left + bounds.right) / 2
        
        nasa_data = predictor.get_nasa_imagery(center_lat, center_lon)
        if nasa_data:
            predictions = predictor.predict_disaster_risks(dem_data, nasa_data)
            
            print("\nPredictions with NASA Data:")
            print("=" * 50)
            for disaster_type, pred in predictions.items():
                print(f"\n{disaster_type.upper()}:")
                print(f"Risk Level: {pred['risk_level']}")
                print(f"Risk Probability: {pred['risk_probability']}")
        
        logger.info("Model training and evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
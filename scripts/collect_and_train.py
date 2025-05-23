"""
Script to collect training data and train disaster prediction models.
"""

import os
import sys
import logging
from pathlib import Path
import argparse
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.data.processors.dem_processor import DEMProcessor
from src.data.collectors.training_data_collector import TrainingDataCollector
from src.data.models.disaster_predictor import DisasterPredictor

def main():
    """Main function to collect data and train models."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Collect training data and train disaster prediction models")
    parser.add_argument("dem_file", help="Path to the DEM file")
    parser.add_argument("--start-date", help="Start date for data collection (YYYY-MM-DD)", 
                       default=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
    parser.add_argument("--end-date", help="End date for data collection (YYYY-MM-DD)",
                       default=datetime.now().strftime('%Y-%m-%d'))
    parser.add_argument("--data-dir", help="Directory to save collected data", default="data/training")
    parser.add_argument("--model-dir", help="Directory to save trained models", default="models")
    args = parser.parse_args()
    
    try:
        # Initialize DEM processor
        logger.info(f"Processing DEM file: {args.dem_file}")
        dem_processor = DEMProcessor(args.dem_file)
        
        # Initialize data collector
        collector = TrainingDataCollector(data_dir=args.data_dir)
        
        # Collect all training data
        logger.info("Collecting training data...")
        training_data = collector.collect_all_training_data(
            dem_processor,
            args.start_date,
            args.end_date
        )
        
        # Initialize disaster predictor
        predictor = DisasterPredictor(model_dir=args.model_dir)
        
        # Train models
        logger.info("Training disaster prediction models...")
        results = predictor.train_models(training_data)
        
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
        
        logger.info("Data collection and model training completed successfully")
        
    except Exception as e:
        logger.error(f"Error in data collection and training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
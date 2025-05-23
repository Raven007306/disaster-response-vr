"""
Script to run the real-time disaster risk assessment dashboard.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import joblib

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_models(models_dir):
    """Load trained models from the models directory."""
    models = {}
    for model_file in models_dir.glob('*.joblib'):
        disaster_type = model_file.stem.split('_')[0]
        try:
            models[disaster_type] = joblib.load(model_file)
            logger.info(f"Loaded {disaster_type} model from {model_file}")
        except Exception as e:
            logger.error(f"Error loading model {model_file}: {e}")
    return models

def main():
    parser = argparse.ArgumentParser(description="Run the real-time disaster risk assessment dashboard")
    parser.add_argument("dem_file", help="Path to the DEM file")
    parser.add_argument("--models-dir", default="models", help="Directory containing trained models")
    parser.add_argument("--port", type=int, default=8050, help="Port to run the dashboard on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    
    try:
        # Import here to avoid circular imports
        from src.data.processors.dem_processor import DEMProcessor
        from src.visualization.realtime_dashboard import RealtimeDashboard
        
        # Initialize DEM processor
        logger.info(f"Loading DEM file: {args.dem_file}")
        dem_processor = DEMProcessor(args.dem_file)
        
        # Load trained models
        models_dir = Path(args.models_dir)
        if not models_dir.exists():
            logger.error(f"Models directory not found: {models_dir}")
            return
        
        models = load_models(models_dir)
        if not models:
            logger.error("No models found in the models directory")
            return
        
        # Create and run dashboard
        logger.info("Initializing dashboard...")
        dashboard = RealtimeDashboard(dem_processor, models)
        
        logger.info(f"Starting dashboard on port {args.port}")
        dashboard.run_server(debug=args.debug, port=args.port)
        
    except Exception as e:
        logger.error(f"Error running dashboard: {e}")
        return

if __name__ == "__main__":
    main() 
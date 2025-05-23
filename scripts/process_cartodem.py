"""
Script to process CartoDEM data for visualization and Unity integration.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_cartodem(tif_file_path, show_plots=True, export_unity=True, resolution=1024):
    
    try:
        # Try to import rasterio
        try:
            import rasterio
        except ImportError:
            logger.error("rasterio is not installed. Please install it with: pip install rasterio")
            print("\nThis script requires rasterio, which is not installed.")
            print("Please install it with one of these methods:")
            print("1. Using pip: pip install rasterio")
            print("2. Using conda: conda install -c conda-forge rasterio")
            return None
        
        from src.data.processors.dem_processor import DEMProcessor
        
        # Create the processor
        logger.info(f"Processing CartoDEM file: {tif_file_path}")
        processor = DEMProcessor(tif_file_path)
        
        # Get metadata
        metadata = processor.get_metadata()
        if metadata is None:
            logger.error("Failed to get metadata")
            return None
        
        logger.info(f"DEM dimensions: {metadata['dimensions']['width']}x{metadata['dimensions']['height']}")
        logger.info(f"Elevation range: {metadata['elevation']['min']} to {metadata['elevation']['max']} meters")
        
        # Create visualizations
        results = {"metadata": metadata}
        
        # 2D elevation map
        elevation_map = processor.visualize_elevation(show=show_plots)
        results["elevation_map"] = elevation_map
        
        # 3D terrain model
        terrain_model = processor.create_3d_model(show=show_plots, vertical_exaggeration=2.0)
        results["terrain_model"] = terrain_model
        
        # Export for Unity if requested
        if export_unity:
            unity_export = processor.export_for_unity(resolution=resolution)
            results["unity_export"] = unity_export
        
        # Create web visualization
        web_vis_file = processor.create_web_visualization()
        results['web_visualization'] = web_vis_file
        
        # Analyze disaster risks
        risk_analysis = processor.analyze_disaster_risks()
        results.update(risk_analysis)
        
        # Create flood simulation
        flood_sim_file = processor.create_flood_simulation()
        results['flood_simulation'] = flood_sim_file
        
        # Create dashboard
        dashboard_file = processor.create_dashboard(results)
        results['dashboard'] = dashboard_file
        
        logger.info("Processing completed successfully")
        return results
    
    except Exception as e:
        logger.error(f"Error processing CartoDEM data: {e}")
        return None

def main():

    parser = argparse.ArgumentParser(description="Process CartoDEM data for visualization and Unity integration")
    parser.add_argument("tif_file", help="Path to the GeoTIFF file")
    parser.add_argument("--no-plots", action="store_true", help="Don't display plots")
    parser.add_argument("--no-unity", action="store_true", help="Don't export for Unity")
    parser.add_argument("--resolution", type=int, default=1024, help="Resolution for Unity export (default: 1024)")
    
    args = parser.parse_args()
    
    
    if not os.path.exists(args.tif_file):
        print(f"Error: The file '{args.tif_file}' does not exist.")
        return
    
    
    results = process_cartodem(
        args.tif_file,
        show_plots=not args.no_plots,
        export_unity=not args.no_unity,
        resolution=args.resolution
    )
    
    if results:
        print("\nProcessing Results:")
        print(f"Metadata: {results['metadata']['file_name']}")
        print(f"Elevation Map: {results['elevation_map']}")
        print(f"3D Terrain Model: {results['terrain_model']}")
        
        if "unity_export" in results:
            print("\nUnity Export:")
            print(f"Heightmap: {results['unity_export']['heightmap_file']}")
            print(f"Metadata: {results['unity_export']['metadata_file']}")
            print(f"Resolution: {results['unity_export']['resolution']}x{results['unity_export']['resolution']}")
            print(f"Elevation Range: {results['unity_export']['min_elevation']} to {results['unity_export']['max_elevation']} meters")
            
            print("\nTo import into Unity:")
            print("1. Create a new Terrain GameObject")
            print("2. In the Terrain Inspector, go to 'Terrain Settings'")
            print("3. Under 'Import Raw...':")
            print(f"   - Select the heightmap file: {os.path.basename(results['unity_export']['heightmap_file'])}")
            print(f"   - Set Width and Height to {results['unity_export']['resolution']}")
            print("   - Set Depth to 16 bit")
            print("   - Set Byte Order to 'Little Endian'")
            print("4. Click 'Import' to apply the heightmap")

        print("\nProcessing complete!")
        print(f"To view the interactive 3D visualization, open: {results['web_visualization']}")
        print(f"To view the flood simulation, open: {results['flood_simulation']}")
        print(f"To view the complete dashboard, open: {results['dashboard']}")

if __name__ == "__main__":
    main() 
"""
Data Exploration for Disaster Response VR/AR Project - Geoid Conversion API Test
"""

import os
import json
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_and_process_cartodem(zip_file_path):
    """
    Extract and process a CartoDEM zip file.
    
    Args:
        zip_file_path (str): Path to the CartoDEM zip file
        
    Returns:
        str: Path to the extracted GeoTIFF file
    """
    import zipfile
    import os
    
    # Create an extraction directory
    extract_dir = os.path.splitext(zip_file_path)[0]  # Remove .zip extension
    os.makedirs(extract_dir, exist_ok=True)
    
    # Extract the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # List all files in the zip archive
        file_list = zip_ref.namelist()
        print(f"Files in the zip archive: {file_list}")
        
        # Extract all files
        zip_ref.extractall(extract_dir)
    
    # Find the GeoTIFF file (looking for both .tif and .TIF extensions)
    tif_files = []
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            if file.lower().endswith(('.tif', '.tiff')):
                tif_files.append(os.path.join(root, file))
    
    if not tif_files:
        # If no TIF files found, look for any other potentially useful files
        all_files = []
        for root, dirs, files in os.walk(extract_dir):
            for file in files:
                all_files.append(os.path.join(root, file))
        
        print(f"No GeoTIFF files found. All extracted files: {all_files}")
        return None
    
    tif_file = tif_files[0]
    print(f"Extracted GeoTIFF file: {tif_file}")
    
    return tif_file

def test_geoid_conversion():
    """Test the Ellipsoid to Geoid Conversion API."""
    logger.info("Testing Ellipsoid to Geoid Conversion API...")
    
    try:
        # Import the BhuvanCollector class
        from src.data.collectors.bhuvan_collector import BhuvanCollector
        
        # Create an instance of the collector
        bhuvan = BhuvanCollector()
        
        # Print configuration for debugging
        print("Bhuvan Configuration:")
        print(f"  API URL: {bhuvan.api_url}")
        print(f"  API Key/Token: {'Set' if bhuvan.token else 'Not set'}")
        
        # Authenticate with Bhuvan
        print("\nAttempting to authenticate...")
        if bhuvan.authenticate():
            print("Successfully authenticated with Bhuvan")
            
            # Test ellipsoid to geoid conversion
            print("\nTesting Ellipsoid to Geoid Conversion API...")
            output_dir = Path("data/raw/cartodem")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # List of tile IDs to test
            tile_ids = ["cdnc43e"]  # Add more tile IDs as needed
            
            for tile_id in tile_ids:
                print(f"\nDownloading CartoDEM data for tile ID: {tile_id}")
                
                # Test geoid conversion
                output_path = output_dir / f"cartodem_{tile_id}_geoid_CDEM.zip"
                conversion_result = bhuvan.get_ellipsoid_to_geoid_conversion(
                    tile_id=tile_id, 
                    datum="geoid", 
                    satellite="CDEM",
                    output_path=str(output_path)
                )
                
                if conversion_result and not isinstance(conversion_result, dict):
                    print(f"Successfully downloaded CartoDEM data to {conversion_result}")
                    print(f"File size: {os.path.getsize(conversion_result) / (1024*1024):.2f} MB")
                    
                    # Extract and process the downloaded file
                    print("\nExtracting and processing the CartoDEM data...")
                    tif_file = extract_and_process_cartodem(conversion_result)
                    if tif_file:
                        print(f"Successfully extracted GeoTIFF file: {tif_file}")
                    else:
                        print("Failed to extract GeoTIFF file from the zip archive")
                else:
                    print("Failed to download CartoDEM data")
                    if isinstance(conversion_result, dict) and "error" in conversion_result:
                        print(f"Error: {conversion_result['error']} - {conversion_result.get('error_description', '')}")
                        if "response_text" in conversion_result:
                            print(f"Response text: {conversion_result['response_text']}")
        else:
            print("Failed to authenticate with Bhuvan")
            print("\nTo get a Bhuvan API token:")
            print("1. Visit https://bhuvan-app1.nrsc.gov.in/api/#")
            print("2. Click on 'Access Token' in the menu")
            print("3. Copy the token and add it to your config/api_keys.json file")
    except ImportError:
        logger.error("BhuvanCollector module not found or not implemented yet")
        print("BhuvanCollector module not found or not implemented yet")
    except Exception as e:
        logger.error(f"Error testing Geoid conversion: {e}")
        print(f"Error testing Geoid conversion: {e}")

def main():
    """Main function to run the Geoid conversion test."""
    print("\n" + "=" * 50)
    print("Ellipsoid to Geoid Conversion API Test")
    print("=" * 50 + "\n")
    
    # Test Geoid conversion
    test_geoid_conversion()
    
    print("\n" + "=" * 50)
    print("Next Steps:")
    print("1. Process the downloaded CartoDEM data")
    print("2. Extract elevation information")
    print("3. Visualize the terrain data")
    print("=" * 50)

if __name__ == "__main__":
    main()
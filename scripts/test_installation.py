"""
Test script to verify all required packages are installed correctly.
"""

import sys
import importlib

def test_import(module_name):
    try:
        importlib.import_module(module_name)
        print(f"✓ {module_name} imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import {module_name}: {e}")
        return False

def main():
    """Test all required package imports."""
    required_packages = [
        'numpy',
        'pandas',
        'geopandas',
        'rasterio',
        'rioxarray',
        'xarray',
        'cv2',
        'pyproj',
        'shapely',
        'fiona',
        'sklearn',
        'matplotlib',
        'seaborn',
        'fastapi',
        'uvicorn',
        'pydantic',
        'httpx',
        'pytest',
        'yaml',
        'dotenv',
        'tqdm',
        'jupyter'
    ]
    
    print("Testing package imports...")
    print("=" * 50)
    
    all_success = True
    for package in required_packages:
        if not test_import(package):
            all_success = False
    
    print("\nSummary:")
    print("=" * 50)
    if all_success:
        print("✓ All packages installed successfully!")
    else:
        print("✗ Some packages failed to import. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 
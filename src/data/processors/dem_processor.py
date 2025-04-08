"""
Digital Elevation Model (DEM) processor for handling GeoTIFF elevation data.
Uses rasterio instead of GDAL for better compatibility.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DEMProcessor:
    """Processor for Digital Elevation Model data from GeoTIFF files."""
    
    def __init__(self, tif_file_path):
        """
        Initialize the DEM processor with a GeoTIFF file.
        
        Args:
            tif_file_path (str): Path to the GeoTIFF file
        """
        self.tif_file_path = Path(tif_file_path)
        self.dataset = None
        self.elevation_data = None
        self.transform = None
        self.crs = None
        self.width = None
        self.height = None
        self.min_elevation = None
        self.max_elevation = None
        self.output_dir = self.tif_file_path.parent / "processed"
        self.output_dir.mkdir(exist_ok=True)
        
        # Load the dataset
        self._load_dataset()
    
    def _load_dataset(self):
        """Load the GeoTIFF dataset and extract basic information."""
        try:
            import rasterio
            
            # Open the dataset
            self.dataset = rasterio.open(str(self.tif_file_path))
            
            # Get basic information
            self.width = self.dataset.width
            self.height = self.dataset.height
            self.transform = self.dataset.transform
            self.crs = self.dataset.crs
            
            # Get elevation data from band 1
            self.elevation_data = self.dataset.read(1)
            
            # Get elevation range
            self.min_elevation = np.min(self.elevation_data)
            self.max_elevation = np.max(self.elevation_data)
            
            logger.info(f"Loaded DEM with dimensions: {self.width}x{self.height}")
            logger.info(f"Elevation range: {self.min_elevation} to {self.max_elevation} meters")
            
            return True
        except ImportError:
            logger.error("rasterio is not installed. Please install it with: pip install rasterio")
            return False
        except Exception as e:
            logger.error(f"Error loading GeoTIFF dataset: {e}")
            return False
    
    def get_metadata(self):
        """
        Get metadata about the DEM.
        
        Returns:
            dict: Metadata about the DEM
        """
        if self.dataset is None:
            return None
        
        # Calculate additional statistics
        mean_elevation = np.mean(self.elevation_data)
        std_elevation = np.std(self.elevation_data)
        
        # Calculate pixel size
        pixel_width = self.transform[0]
        pixel_height = abs(self.transform[4])
        
        # Calculate bounding box coordinates
        bounds = self.dataset.bounds
        
        return {
            "file_name": self.tif_file_path.name,
            "dimensions": {
                "width": self.width,
                "height": self.height,
                "pixel_width_m": pixel_width,
                "pixel_height_m": pixel_height
            },
            "coordinates": {
                "x_min": bounds.left,
                "y_min": bounds.bottom,
                "x_max": bounds.right,
                "y_max": bounds.top
            },
            "elevation": {
                "min": float(self.min_elevation),
                "max": float(self.max_elevation),
                "mean": float(mean_elevation),
                "std": float(std_elevation)
            },
            "crs": str(self.crs)
        }
    
    def visualize_elevation(self, output_file=None, show=True):
        """
        Create a visualization of the elevation data.
        
        Args:
            output_file (str, optional): Path to save the visualization. If None, uses default path.
            show (bool, optional): Whether to display the plot. Defaults to True.
            
        Returns:
            str: Path to the saved visualization file
        """
        if self.elevation_data is None:
            logger.error("No elevation data available for visualization")
            return None
        
        try:
            # Create figure
            plt.figure(figsize=(12, 10))
            
            # Create a colormap for elevation
            cmap = plt.cm.terrain
            
            # Plot the elevation data
            im = plt.imshow(self.elevation_data, cmap=cmap)
            plt.colorbar(im, label='Elevation (meters)')
            plt.title(f'Elevation Map - {self.tif_file_path.stem}')
            plt.xlabel('East-West (pixels)')
            plt.ylabel('North-South (pixels)')
            
            # Save the figure if output_file is provided
            if output_file is None:
                output_file = self.output_dir / f"{self.tif_file_path.stem}_elevation.png"
            
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved elevation visualization to {output_file}")
            
            if show:
                plt.show()
            else:
                plt.close()
            
            return str(output_file)
        except Exception as e:
            logger.error(f"Error visualizing elevation data: {e}")
            return None
    
    def create_3d_model(self, output_file=None, show=True, vertical_exaggeration=1.0):
        """
        Create a 3D model visualization of the terrain.
        
        Args:
            output_file (str, optional): Path to save the visualization. If None, uses default path.
            show (bool, optional): Whether to display the plot. Defaults to True.
            vertical_exaggeration (float, optional): Factor to exaggerate the vertical scale. Defaults to 1.0.
            
        Returns:
            str: Path to the saved visualization file
        """
        if self.elevation_data is None:
            logger.error("No elevation data available for 3D visualization")
            return None
        
        try:
            from mpl_toolkits.mplot3d import Axes3D
            
            # Create figure
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Create a downsampled grid for better performance
            # For large DEMs, we need to downsample to avoid memory issues
            downsample_factor = max(1, min(self.width, self.height) // 500)
            
            # Create coordinate grids
            y, x = np.mgrid[0:self.height:downsample_factor, 0:self.width:downsample_factor]
            z = self.elevation_data[::downsample_factor, ::downsample_factor] * vertical_exaggeration
            
            # Create a colormap for elevation
            cmap = plt.cm.terrain
            
            # Plot the surface
            surf = ax.plot_surface(x, y, z, cmap=cmap, linewidth=0, antialiased=True, shade=True)
            
            # Add a color bar
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Elevation (meters)')
            
            # Set labels and title
            ax.set_title(f'3D Terrain Model - {self.tif_file_path.stem}')
            ax.set_xlabel('East-West (pixels)')
            ax.set_ylabel('North-South (pixels)')
            ax.set_zlabel('Elevation (meters)')
            
            # Save the figure if output_file is provided
            if output_file is None:
                output_file = self.output_dir / f"{self.tif_file_path.stem}_3d_model.png"
            
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved 3D model visualization to {output_file}")
            
            if show:
                plt.show()
            else:
                plt.close()
            
            return str(output_file)
        except Exception as e:
            logger.error(f"Error creating 3D model: {e}")
            return None
    
    def export_for_unity(self, output_file=None, resolution=1024):
        """
        Export the elevation data as a RAW heightmap for Unity.
        
        Args:
            output_file (str, optional): Path to save the RAW file. If None, uses default path.
            resolution (int, optional): Resolution of the output heightmap. Defaults to 1024.
                                       Unity works best with power-of-two resolutions.
            
        Returns:
            dict: Information about the exported heightmap
        """
        if self.elevation_data is None:
            logger.error("No elevation data available for export")
            return None
        
        try:
            # Determine output file path
            if output_file is None:
                output_file = self.output_dir / f"{self.tif_file_path.stem}_heightmap_{resolution}.raw"
            else:
                output_file = Path(output_file)
            
            # Resample the elevation data to the desired resolution
            from scipy.ndimage import zoom
            
            # Calculate zoom factors
            zoom_x = resolution / self.width
            zoom_y = resolution / self.height
            
            # Resample the data
            resampled_data = zoom(self.elevation_data, (zoom_y, zoom_x), order=1)
            
            # Normalize the data to 0-1 range for Unity
            # Unity heightmaps are typically 0-1 normalized
            normalized_data = (resampled_data - self.min_elevation) / (self.max_elevation - self.min_elevation)
            
            # Convert to 16-bit unsigned integers (Unity format)
            heightmap_data = (normalized_data * 65535).astype(np.uint16)
            
            # Save as RAW file (16-bit, little-endian)
            heightmap_data.tofile(output_file)
            
            # Create a metadata file with information about the heightmap
            metadata_file = output_file.with_suffix('.json')
            import json
            
            metadata = {
                "original_file": str(self.tif_file_path),
                "heightmap_file": str(output_file),
                "resolution": resolution,
                "min_elevation": float(self.min_elevation),
                "max_elevation": float(self.max_elevation),
                "width": self.width,
                "height": self.height,
                "transform": [self.transform[0], self.transform[1], self.transform[2], 
                             self.transform[3], self.transform[4], self.transform[5]],
                "unity_import_settings": {
                    "resolution": resolution,
                    "depth": 16,
                    "byte_order": "little-endian",
                    "width": resolution,
                    "height": resolution
                }
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Exported Unity heightmap to {output_file}")
            logger.info(f"Saved metadata to {metadata_file}")
            
            return {
                "heightmap_file": str(output_file),
                "metadata_file": str(metadata_file),
                "resolution": resolution,
                "min_elevation": float(self.min_elevation),
                "max_elevation": float(self.max_elevation)
            }
        except Exception as e:
            logger.error(f"Error exporting heightmap for Unity: {e}")
            return None 
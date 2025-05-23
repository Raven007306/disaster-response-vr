"""
Digital Elevation Model (DEM) processor for handling GeoTIFF elevation data.
Uses rasterio instead of GDAL for better compatibility.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import rasterio
import json
from src.data.models.disaster_predictor import DisasterPredictor
from rasterio.transform import from_origin
from scipy.ndimage import sobel

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
        self.dataset = rasterio.open(tif_file_path)
        self.elevation_data = self.dataset.read(1)
        logger.info(f"Loaded DEM with dimensions: {self.elevation_data.shape}")
        logger.info(f"Elevation range: {self.elevation_data.min()} to {self.elevation_data.max()} meters")
        self.transform = self.dataset.transform
        self.crs = self.dataset.crs
        self.width = self.dataset.width
        self.height = self.dataset.height
        self.min_elevation = self.elevation_data.min()
        self.max_elevation = self.elevation_data.max()
        self.output_dir = self.tif_file_path.parent / "processed"
        self.output_dir.mkdir(exist_ok=True)
    
    def get_elevation_data(self):
        """
        Get the elevation data array.
        
        Returns:
            numpy.ndarray: 2D array of elevation values
        """
        return self.elevation_data
    
    def _load_dataset(self):
        """Load the GeoTIFF dataset and extract basic information."""
        try:
            # Get basic information
            self.width = self.dataset.width
            self.height = self.dataset.height
            self.transform = self.dataset.transform
            self.crs = self.dataset.crs
            
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
    
    def create_unity_heightmap(self, resolution=1024):
        """Create a heightmap suitable for Unity terrain.
        
        Args:
            resolution: The desired resolution of the heightmap (power of 2)
            
        Returns:
            tuple: Paths to the RAW file and JSON metadata file
        """
        import json
        from scipy.ndimage import zoom
        import numpy as np
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Get elevation range
        min_elevation = self.min_elevation
        max_elevation = self.max_elevation
        
        # Resize to desired resolution
        zoom_factor = resolution / max(self.elevation_data.shape)
        resized_data = zoom(self.elevation_data, zoom_factor, order=1)
        
        # Ensure exact dimensions
        if resized_data.shape[0] > resolution:
            resized_data = resized_data[:resolution, :]
        if resized_data.shape[1] > resolution:
            resized_data = resized_data[:, :resolution]
        
        # Normalize to 0-65535 (16-bit)
        normalized = (resized_data - min_elevation) / (max_elevation - min_elevation)
        heightmap_data = (normalized * 65535).astype(np.uint16)
        
        # Save as RAW file (16-bit, little-endian)
        filename_stem = self.tif_file_path.stem
        raw_file = self.output_dir / f"{filename_stem}_heightmap_{resolution}.raw"
        heightmap_data.tofile(raw_file)
        
        # Save metadata
        metadata = {
            "original_dimensions": self.elevation_data.shape,
            "heightmap_dimensions": heightmap_data.shape,
            "min_elevation": float(min_elevation),
            "max_elevation": float(max_elevation),
            "resolution": resolution,
            "bit_depth": 16,
            "byte_order": "little-endian"
        }
        
        json_file = self.output_dir / f"{filename_stem}_heightmap_{resolution}.json"
        with open(json_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return str(raw_file), str(json_file)
    
    def create_web_visualization(self):
        """Create an interactive web-based visualization of the terrain.
        
        Returns:
            str: Path to the saved HTML file
        """
        try:
            # Install plotly if not already installed
            # pip install plotly
            import plotly.graph_objects as go
            import numpy as np
            
            # Downsample for performance
            sample_rate = 4  # Use every 4th point
            z = self.elevation_data[::sample_rate, ::sample_rate]
            
            # Create x and y coordinates
            y, x = np.mgrid[0:z.shape[0], 0:z.shape[1]]
            
            # Create the 3D surface plot
            fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
            
            # Update layout
            fig.update_layout(
                title=f'3D Terrain Visualization: {self.tif_file_path.stem}',
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title=f'Elevation (min: {self.min_elevation:.1f}m, max: {self.max_elevation:.1f}m)',
                    aspectratio=dict(x=1, y=1, z=0.5)
                )
            )
            
            # Add camera controls info
            fig.add_annotation(
                text="Use mouse to rotate, scroll to zoom",
                xref="paper", yref="paper",
                x=0, y=0,
                showarrow=False
            )
            
            # Save as HTML
            html_file = self.output_dir / f"{self.tif_file_path.stem}_3d_interactive.html"
            fig.write_html(str(html_file))
            
            print(f"Created interactive 3D visualization: {html_file}")
            return str(html_file)
        except Exception as e:
            print(f"Error creating web visualization: {e}")
            return None
    
    def create_dashboard(self, results=None):
        """Create a comprehensive HTML dashboard with all visualizations.
        
        Args:
            results: Dictionary of analysis results
            
        Returns:
            str: Path to the saved HTML file
        """
        try:
            # Create a simple dashboard HTML
            dashboard_file = self.output_dir / f"{self.tif_file_path.stem}_dashboard.html"
            
            # Get the filename stem for referencing other files
            filename_stem = self.tif_file_path.stem
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Terrain Analysis Dashboard: {filename_stem}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                    .container {{ display: flex; flex-wrap: wrap; justify-content: center; }}
                    .card {{ 
                        margin: 15px; 
                        box-shadow: 0 4px 8px rgba(0,0,0,0.1); 
                        background-color: white;
                        border-radius: 8px;
                        overflow: hidden;
                        width: 45%;
                        min-width: 300px;
                    }}
                    .card-header {{ 
                        background-color: #4285f4; 
                        color: white; 
                        padding: 10px 15px;
                    }}
                    .card-body {{ padding: 15px; }}
                    img {{ max-width: 100%; height: auto; display: block; margin: 0 auto; }}
                    h1 {{ color: #333; text-align: center; }}
                    h2 {{ margin-top: 0; }}
                    .button {{
                        display: inline-block;
                        background-color: #4285f4;
                        color: white;
                        padding: 10px 15px;
                        text-decoration: none;
                        border-radius: 4px;
                        margin-top: 10px;
                    }}
                    .button:hover {{ background-color: #3367d6; }}
                    .stats {{ 
                        background-color: #f9f9f9; 
                        padding: 10px; 
                        border-radius: 4px;
                        margin-top: 10px;
                    }}
                </style>
            </head>
            <body>
                <h1>Terrain Analysis Dashboard: {filename_stem}</h1>
                
                <div class="container">
                    <div class="card">
                        <div class="card-header">
                            <h2>Elevation Map</h2>
                        </div>
                        <div class="card-body">
                            <img src="{filename_stem}_elevation.png" alt="Elevation Map">
                            <div class="stats">
                                <p><strong>Elevation Range:</strong> {self.min_elevation:.1f}m to {self.max_elevation:.1f}m</p>
                                <p><strong>Resolution:</strong> {self.width} x {self.height} pixels</p>
                            </div>
                        </div>
                    </div>
                    
                    <div class="card">
                        <div class="card-header">
                            <h2>3D Terrain Model</h2>
                        </div>
                        <div class="card-body">
                            <img src="{filename_stem}_3d_model.png" alt="3D Terrain Model">
                            <a href="{filename_stem}_3d_interactive.html" target="_blank" class="button">
                                Open Interactive 3D Model
                            </a>
                        </div>
                    </div>

                    <div class="card">
                        <div class="card-header">
                            <h2>Landslide Risk Assessment</h2>
                        </div>
                        <div class="card-body">
                            <img src="{filename_stem}_landslide_risk.png" alt="Landslide Risk">
                            <div class="stats">
                                <p><strong>Risk Levels:</strong></p>
                                <p>0 - Low Risk</p>
                                <p>1 - Moderate Risk (slopes > 30°)</p>
                                <p>2 - High Risk (slopes > 45°)</p>
                            </div>
                        </div>
                    </div>

                    <div class="card">
                        <div class="card-header">
                            <h2>Flood Risk Assessment</h2>
                        </div>
                        <div class="card-body">
                            <img src="{filename_stem}_flood_risk.png" alt="Flood Risk">
                            <div class="stats">
                                <p><strong>Risk Levels:</strong></p>
                                <p>0 - Low Risk</p>
                                <p>1 - Moderate Risk (bottom 20% elevation)</p>
                                <p>2 - High Risk (bottom 10% elevation)</p>
                            </div>
                        </div>
                    </div>
                </div>
                
                <footer style="text-align: center; margin-top: 30px; color: #666;">
                    <p>Created with Disaster Response VR/AR Project</p>
                </footer>
            </body>
            </html>
            """
            
            with open(dashboard_file, 'w') as f:
                f.write(html_content)
            
            print(f"Created dashboard: {dashboard_file}")
            return str(dashboard_file)
        except Exception as e:
            print(f"Error creating dashboard: {e}")
            return None
    
    def analyze_disaster_risks(self):
        """Analyze potential disaster risks based on terrain features.
        
        Returns:
            dict: Paths to the saved risk analysis files
        """
        try:
            import numpy as np
            from scipy.ndimage import sobel
            import matplotlib.pyplot as plt
            
            # Calculate slope in degrees
            dx = sobel(self.elevation_data, axis=1)
            dy = sobel(self.elevation_data, axis=0)
            
            # Fix: Handle NaN or negative values before sqrt
            slope_squared = dx**2 + dy**2
            slope_squared = np.maximum(slope_squared, 0)  # Ensure no negative values
            slope_radians = np.arctan(np.sqrt(slope_squared))
            slope_degrees = np.degrees(slope_radians)
            
            # Get coordinates for each pixel
            rows, cols = self.elevation_data.shape
            x_coords = np.linspace(self.dataset.bounds.left, self.dataset.bounds.right, cols)
            y_coords = np.linspace(self.dataset.bounds.bottom, self.dataset.bounds.top, rows)
            lon_grid, lat_grid = np.meshgrid(x_coords, y_coords)
            
            # Landslide risk (high slope areas)
            landslide_risk = np.zeros_like(slope_degrees)
            landslide_risk[slope_degrees > 30] = 1  # Moderate risk
            landslide_risk[slope_degrees > 45] = 2  # High risk
            
            # Flood risk (low elevation areas)
            normalized_elevation = (self.elevation_data - self.min_elevation) / (self.max_elevation - self.min_elevation)
            flood_risk = np.zeros_like(normalized_elevation)
            flood_risk[normalized_elevation < 0.1] = 2  # High risk
            flood_risk[normalized_elevation < 0.2] = 1  # Moderate risk
            
            # Earthquake risk (based on proximity to fault lines and historical data)
            # This is a simplified model - in reality, you would use actual fault line data
            earthquake_risk = np.zeros_like(self.elevation_data)
            # Simulate fault lines as diagonal lines across the terrain
            fault_lines = np.abs(lon_grid - lat_grid) < 0.01  # Simplified fault line model
            earthquake_risk[fault_lines] = 1
            earthquake_risk[fault_lines & (slope_degrees > 20)] = 2  # Higher risk near steep slopes
            
            # Cyclone risk (based on coastal proximity and elevation)
            # This is a simplified model - in reality, you would use actual coastal data
            coastal_proximity = np.minimum(
                np.abs(lon_grid - self.dataset.bounds.left),
                np.abs(lon_grid - self.dataset.bounds.right)
            ) / (self.dataset.bounds.right - self.dataset.bounds.left)
            cyclone_risk = np.zeros_like(self.elevation_data)
            cyclone_risk[coastal_proximity < 0.1] = 2  # High risk near coasts
            cyclone_risk[(coastal_proximity < 0.2) & (normalized_elevation < 0.3)] = 1  # Moderate risk
            
            # Save risk maps
            risk_maps = {
                'landslide': landslide_risk,
                'flood': flood_risk,
                'earthquake': earthquake_risk,
                'cyclone': cyclone_risk
            }
            
            risk_files = {}
            for risk_type, risk_data in risk_maps.items():
                plt.figure(figsize=(10, 8))
                plt.imshow(risk_data, cmap='YlOrRd')
                plt.colorbar(label='Risk Level')
                plt.title(f'{risk_type.capitalize()} Risk Assessment')
                
                # Add coordinate information
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                
                # Save the risk map
                risk_file = self.output_dir / f"{self.tif_file_path.stem}_{risk_type}_risk.png"
                plt.savefig(risk_file)
                plt.close()
                
                risk_files[f'{risk_type}_risk_file'] = str(risk_file)
            
            # Create a combined risk assessment
            combined_risk = np.maximum.reduce([
                landslide_risk,
                flood_risk,
                earthquake_risk,
                cyclone_risk
            ])
            
            plt.figure(figsize=(12, 10))
            plt.imshow(combined_risk, cmap='YlOrRd')
            plt.colorbar(label='Risk Level')
            plt.title('Combined Disaster Risk Assessment')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            
            combined_file = self.output_dir / f"{self.tif_file_path.stem}_combined_risk.png"
            plt.savefig(combined_file)
            plt.close()
            
            risk_files['combined_risk_file'] = str(combined_file)
            
            # Create a risk summary report
            risk_summary = {
                'landslide': {
                    'high_risk_area': np.sum(landslide_risk == 2) / landslide_risk.size * 100,
                    'moderate_risk_area': np.sum(landslide_risk == 1) / landslide_risk.size * 100
                },
                'flood': {
                    'high_risk_area': np.sum(flood_risk == 2) / flood_risk.size * 100,
                    'moderate_risk_area': np.sum(flood_risk == 1) / flood_risk.size * 100
                },
                'earthquake': {
                    'high_risk_area': np.sum(earthquake_risk == 2) / earthquake_risk.size * 100,
                    'moderate_risk_area': np.sum(earthquake_risk == 1) / earthquake_risk.size * 100
                },
                'cyclone': {
                    'high_risk_area': np.sum(cyclone_risk == 2) / cyclone_risk.size * 100,
                    'moderate_risk_area': np.sum(cyclone_risk == 1) / cyclone_risk.size * 100
                }
            }
            
            # Save risk summary as JSON
            summary_file = self.output_dir / f"{self.tif_file_path.stem}_risk_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(risk_summary, f, indent=2)
            
            risk_files['risk_summary_file'] = str(summary_file)
            
            return risk_files
            
        except Exception as e:
            logger.error(f"Error analyzing disaster risks: {e}")
            return {}
    
    def create_flood_simulation(self):
        """Create a simple flood simulation visualization."""
        try:
            # Return a placeholder message instead of creating the simulation
            print("Flood simulation temporarily disabled due to data type issues")
            
            # Create a simple HTML file with a message
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Flood Simulation Placeholder</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; text-align: center; }
                    .message { 
                        padding: 20px; 
                        background-color: #f8f9fa; 
                        border-radius: 10px;
                        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                        margin: 40px auto;
                        max-width: 600px;
                    }
                    h1 { color: #4285f4; }
                </style>
            </head>
            <body>
                <h1>Flood Simulation</h1>
                <div class="message">
                    <h2>Coming Soon</h2>
                    <p>The interactive flood simulation is currently being developed.</p>
                    <p>Please check back later for this feature.</p>
                </div>
            </body>
            </html>
            """
            
            # Save as HTML
            html_file = self.output_dir / f"{self.tif_file_path.stem}_flood_simulation.html"
            with open(html_file, 'w') as f:
                f.write(html_content)
            
            print(f"Created flood simulation placeholder: {html_file}")
            return str(html_file)
            
            """
            # Original code commented out
            import plotly.graph_objects as go
            import numpy as np
            
            # Downsample for performance
            sample_rate = 4
            z = self.elevation_data[::sample_rate, ::sample_rate].copy()
            
            # Replace NaN values with the minimum elevation
            z = np.nan_to_num(z, nan=self.min_elevation)
            
            # Create x and y coordinates
            y, x = np.mgrid[0:z.shape[0], 0:z.shape[1]]
            
            # Create the base figure
            fig = go.Figure()
            
            # Add the terrain surface
            fig.add_trace(go.Surface(
                z=z,
                x=x,
                y=y,
                colorscale='Earth',
                opacity=0.9,
                name='Terrain'
            ))
            
            # Create a simple static flood level at 10%
            water_level_percent = 10
            actual_level = self.min_elevation + (water_level_percent / 100) * (self.max_elevation - self.min_elevation)
            
            # Create a flat surface for water
            water_z = np.full_like(z, actual_level)
            
            # Create a mask for where water should be visible (above terrain)
            water_mask = water_z < z
            water_z_masked = water_z.copy()
            water_z_masked[water_mask] = np.nan  # This line causes the error
            
            # Add water surface
            fig.add_trace(go.Surface(
                z=water_z_masked,
                x=x,
                y=y,
                colorscale=[[0, 'rgba(0,100,200,0.5)'], [1, 'rgba(0,100,200,0.5)']],
                showscale=False,
                name='Water'
            ))
            
            # Update layout
            fig.update_layout(
                title=f'Flood Simulation - Water Level: {actual_level:.1f}m ({water_level_percent}%)',
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Elevation (m)',
                    aspectratio=dict(x=1, y=1, z=0.5)
                )
            )
            
            # Save as HTML
            html_file = self.output_dir / f"{self.tif_file_path.stem}_flood_simulation.html"
            fig.write_html(str(html_file))
            
            print(f"Created interactive flood simulation: {html_file}")
            return str(html_file)
            """
        except Exception as e:
            print(f"Error creating flood simulation: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def calculate_slope(self):
        """
        Calculate slope in degrees for each pixel.
        
        Returns:
            numpy.ndarray: 2D array of slope values in degrees
        """
        try:
            # Calculate gradients using Sobel operator
            dx = sobel(self.elevation_data, axis=1)
            dy = sobel(self.elevation_data, axis=0)
            
            # Calculate slope in degrees with proper handling of invalid values
            slope_squared = dx**2 + dy**2
            slope_squared = np.maximum(slope_squared, 0)  # Ensure no negative values
            slope = np.arctan(np.sqrt(slope_squared)) * (180 / np.pi)
            
            # Replace any NaN values with 0
            slope = np.nan_to_num(slope, nan=0.0)
            return slope
            
        except Exception as e:
            logger.error(f"Error calculating slope: {e}")
            return np.zeros_like(self.elevation_data)
    
    def calculate_aspect(self):
        """
        Calculate aspect (direction of slope) in degrees.
        
        Returns:
            numpy.ndarray: 2D array of aspect values in degrees
        """
        try:
            # Calculate gradients using Sobel operator
            dx = sobel(self.elevation_data, axis=1)
            dy = sobel(self.elevation_data, axis=0)
            
            # Calculate aspect in degrees
            aspect = np.arctan2(dy, dx) * (180 / np.pi)
            
            # Convert to 0-360 degrees
            aspect = (aspect + 360) % 360
            return aspect
            
        except Exception as e:
            logger.error(f"Error calculating aspect: {e}")
            return np.zeros_like(self.elevation_data)
    
    def calculate_curvature(self):
        """
        Calculate terrain curvature.
        
        Returns:
            numpy.ndarray: 2D array of curvature values
        """
        try:
            # Calculate second derivatives
            dxx = sobel(sobel(self.elevation_data, axis=1), axis=1)
            dyy = sobel(sobel(self.elevation_data, axis=0), axis=0)
            
            # Calculate curvature
            curvature = -(dxx + dyy)
            return curvature
            
        except Exception as e:
            logger.error(f"Error calculating curvature: {e}")
            return np.zeros_like(self.elevation_data)
    
    def get_bounds(self):
        """
        Get the geographic bounds of the DEM.
        
        Returns:
            dict: Dictionary containing min/max latitude and longitude
        """
        bounds = self.dataset.bounds
        return {
            'min_lat': bounds.bottom,
            'max_lat': bounds.top,
            'min_lon': bounds.left,
            'max_lon': bounds.right
        }
    
    def close(self):
        """Close the raster dataset."""
        self.dataset.close()
    
    def create_realtime_dashboard(self, training_progress=None):
        """Create a real-time dashboard that updates during training.
        
        Args:
            training_progress (dict, optional): Current training progress data
            
        Returns:
            str: Path to the saved HTML file
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import json
            
            # Create a figure with subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Elevation Map', 'Risk Assessment', 'Training Progress', 'Model Metrics'),
                specs=[[{"type": "surface"}, {"type": "surface"}],
                      [{"type": "indicator"}, {"type": "table"}]]
            )
            
            # Add elevation surface
            sample_rate = 4  # Use every 4th point for performance
            z = self.elevation_data[::sample_rate, ::sample_rate]
            y, x = np.mgrid[0:z.shape[0], 0:z.shape[1]]
            
            fig.add_trace(
                go.Surface(z=z, x=x, y=y, colorscale='terrain', name='Elevation'),
                row=1, col=1
            )
            
            # Add risk assessment surface
            if training_progress and 'risk_maps' in training_progress:
                risk_data = training_progress['risk_maps'].get('combined', np.zeros_like(z))
                fig.add_trace(
                    go.Surface(z=risk_data, x=x, y=y, colorscale='Reds', name='Risk'),
                    row=1, col=2
                )
            
            # Add training progress indicator
            if training_progress and 'metrics' in training_progress:
                metrics = training_progress['metrics']
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number",
                        value=metrics.get('accuracy', 0) * 100,
                        title={'text': "Model Accuracy"},
                        gauge={'axis': {'range': [0, 100]},
                              'bar': {'color': "darkblue"}},
                    ),
                    row=2, col=1
                )
                
                # Add metrics table
                metrics_table = go.Table(
                    header=dict(values=['Metric', 'Value'],
                              fill_color='paleturquoise',
                              align='left'),
                    cells=dict(values=[[k for k in metrics.keys()],
                                     [f"{v:.3f}" if isinstance(v, float) else str(v) for v in metrics.values()]],
                              fill_color='lavender',
                              align='left'),
                    row=2, col=2
                )
                fig.add_trace(metrics_table, row=2, col=2)
            
            # Update layout
            fig.update_layout(
                title_text="Real-time Training Dashboard",
                height=1000,
                showlegend=True,
                template="plotly_white"
            )
            
            # Save as HTML with auto-refresh
            html_file = self.output_dir / f"{self.tif_file_path.stem}_realtime_dashboard.html"
            
            # Add auto-refresh script
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Real-time Training Dashboard</title>
                <script>
                    function refreshPage() {{
                        location.reload();
                    }}
                    // Refresh every 5 seconds
                    setInterval(refreshPage, 5000);
                </script>
            </head>
            <body>
                {fig.to_html(full_html=False)}
            </body>
            </html>
            """
            
            with open(html_file, 'w') as f:
                f.write(html_content)
            
            logger.info(f"Created real-time dashboard: {html_file}")
            return str(html_file)
            
        except Exception as e:
            logger.error(f"Error creating real-time dashboard: {e}")
            return None 
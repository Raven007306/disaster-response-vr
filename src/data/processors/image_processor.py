"""
Module for processing satellite imagery and remote sensing data.
"""

import os
import numpy as np
import cv2
import rasterio
from rasterio.warp import calculate_default_transform, reproject
from rasterio.enums import Resampling
import logging

logger = logging.getLogger(__name__)

class SatelliteImageProcessor:
    """
    Processor for satellite imagery and remote sensing data.
    
    This class provides methods for preprocessing satellite imagery,
    including atmospheric correction, georeferencing, and feature extraction.
    """
    
    def __init__(self):
        """Initialize the satellite image processor."""
        pass
        
    def load_image(self, image_path):
        """
        Load a satellite image using rasterio.
        
        Args:
            image_path (str): Path to the satellite image file.
            
        Returns:
            tuple: (image_data, metadata) where image_data is a numpy array
                  and metadata is a dictionary of image metadata.
        """
        try:
            with rasterio.open(image_path) as src:
                image_data = src.read()
                metadata = {
                    'crs': src.crs,
                    'transform': src.transform,
                    'bounds': src.bounds,
                    'width': src.width,
                    'height': src.height,
                    'count': src.count,
                    'dtype': src.dtypes[0],
                    'nodata': src.nodata
                }
                
            logger.info(f"Successfully loaded image from {image_path}")
            return image_data, metadata
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise
    
    def save_image(self, image_data, metadata, output_path):
        """
        Save processed image data to a file.
        
        Args:
            image_data (numpy.ndarray): Image data to save.
            metadata (dict): Metadata for the image.
            output_path (str): Path to save the image.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=metadata['height'],
                width=metadata['width'],
                count=image_data.shape[0],
                dtype=image_data.dtype,
                crs=metadata['crs'],
                transform=metadata['transform'],
                nodata=metadata.get('nodata')
            ) as dst:
                dst.write(image_data)
                
            logger.info(f"Successfully saved image to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving image to {output_path}: {e}")
            return False
    
    def apply_atmospheric_correction(self, image_data, metadata, method='dos'):
        """
        Apply atmospheric correction to satellite imagery.
        
        Args:
            image_data (numpy.ndarray): Raw satellite image data.
            metadata (dict): Image metadata.
            method (str): Correction method ('dos', 'flaash', etc.).
            
        Returns:
            numpy.ndarray: Atmospherically corrected image.
        """
        try:
            # This is a simplified implementation of Dark Object Subtraction (DOS)
            # A real implementation would be more complex and depend on the specific sensor
            
            if method.lower() == 'dos':
                corrected_data = image_data.copy()
                
                for band in range(image_data.shape[0]):
                    band_data = image_data[band]
                    # Find dark objects (1% percentile)
                    mask = band_data != metadata.get('nodata', 0)
                    if mask.any():
                        dark_value = np.percentile(band_data[mask], 1)
                        # Subtract dark value
                        corrected_band = band_data.astype(np.float32) - dark_value
                        # Clip negative values to 0
                        corrected_band = np.clip(corrected_band, 0, None)
                        # Convert back to original data type
                        corrected_data[band] = corrected_band.astype(image_data.dtype)
                
                logger.info(f"Applied {method} atmospheric correction")
                return corrected_data
            else:
                logger.warning(f"Atmospheric correction method {method} not implemented")
                return image_data
                
        except Exception as e:
            logger.error(f"Error applying atmospheric correction: {e}")
            return image_data
    
    def reproject_image(self, image_data, src_metadata, dst_crs):
        """
        Reproject image to a different coordinate reference system.
        
        Args:
            image_data (numpy.ndarray): Image data to reproject.
            src_metadata (dict): Source image metadata.
            dst_crs (str or dict): Target coordinate reference system.
            
        Returns:
            tuple: (reprojected_data, new_metadata) with the reprojected image
                  and updated metadata.
        """
        try:
            # Calculate the ideal dimensions and transformation parameters
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src_metadata['crs'],
                dst_crs,
                src_metadata['width'],
                src_metadata['height'],
                *src_metadata['bounds']
            )
            
            # Create a new metadata dictionary with updated information
            dst_metadata = src_metadata.copy()
            dst_metadata.update({
                'crs': dst_crs,
                'transform': dst_transform,
                'width': dst_width,
                'height': dst_height
            })
            
            # Create the destination array
            dst_data = np.zeros((image_data.shape[0], dst_height, dst_width), dtype=image_data.dtype)
            
            # Reproject each band
            for i in range(image_data.shape[0]):
                reproject(
                    source=image_data[i],
                    destination=dst_data[i],
                    src_transform=src_metadata['transform'],
                    src_crs=src_metadata['crs'],
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                    src_nodata=src_metadata.get('nodata'),
                    dst_nodata=src_metadata.get('nodata')
                )
            
            logger.info(f"Reprojected image to {dst_crs}")
            return dst_data, dst_metadata
            
        except Exception as e:
            logger.error(f"Error reprojecting image: {e}")
            raise
    
    def extract_ndvi(self, image_data, red_band_index, nir_band_index):
        """
        Calculate Normalized Difference Vegetation Index (NDVI).
        
        Args:
            image_data (numpy.ndarray): Multispectral image data.
            red_band_index (int): Index of the red band.
            nir_band_index (int): Index of the near-infrared band.
            
        Returns:
            numpy.ndarray: NDVI image.
        """
        try:
            # Extract the red and NIR bands
            red = image_data[red_band_index].astype(np.float32)
            nir = image_data[nir_band_index].astype(np.float32)
            
            # Calculate NDVI
            # NDVI = (NIR - Red) / (NIR + Red)
            denominator = nir + red
            ndvi = np.zeros_like(red)
            
            # Avoid division by zero
            valid_mask = denominator > 0
            ndvi[valid_mask] = (nir[valid_mask] - red[valid_mask]) / denominator[valid_mask]
            
            # NDVI ranges from -1 to 1
            ndvi = np.clip(ndvi, -1.0, 1.0)
            
            logger.info("Successfully calculated NDVI")
            return ndvi
            
        except Exception as e:
            logger.error(f"Error calculating NDVI: {e}")
            raise
    
    def extract_water_index(self, image_data, green_band_index, nir_band_index):
        """
        Calculate Normalized Difference Water Index (NDWI).
        
        Args:
            image_data (numpy.ndarray): Multispectral image data.
            green_band_index (int): Index of the green band.
            nir_band_index (int): Index of the near-infrared band.
            
        Returns:
            numpy.ndarray: NDWI image.
        """
        try:
            # Extract the green and NIR bands
            green = image_data[green_band_index].astype(np.float32)
            nir = image_data[nir_band_index].astype(np.float32)
            
            # Calculate NDWI
            # NDWI = (Green - NIR) / (Green + NIR)
            denominator = green + nir
            ndwi = np.zeros_like(green)
            
            # Avoid division by zero
            valid_mask = denominator > 0
            ndwi[valid_mask] = (green[valid_mask] - nir[valid_mask]) / denominator[valid_mask]
            
            # NDWI ranges from -1 to 1
            ndwi = np.clip(ndwi, -1.0, 1.0)
            
            logger.info("Successfully calculated NDWI")
            return ndwi
            
        except Exception as e:
            logger.error(f"Error calculating NDWI: {e}")
            raise 
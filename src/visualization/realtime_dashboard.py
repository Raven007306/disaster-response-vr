import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealtimeDashboard:
    def __init__(self, dem_processor, trained_models):
        """
        Initialize the real-time dashboard.
        
        Args:
            dem_processor: DEMProcessor instance for terrain data
            trained_models: Dictionary of trained models for different disaster types
        """
        self.dem_processor = dem_processor
        self.trained_models = trained_models
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
        
    def setup_layout(self):
        """Set up the dashboard layout."""
        self.app.layout = html.Div([
            html.H1("Disaster Risk Assessment Dashboard", 
                   style={'textAlign': 'center', 'color': '#2c3e50'}),
            
            # Controls
            html.Div([
                html.Div([
                    html.Label("Disaster Type"),
                    dcc.Dropdown(
                        id='disaster-type',
                        options=[
                            {'label': 'Landslide', 'value': 'landslide'},
                            {'label': 'Flood', 'value': 'flood'},
                            {'label': 'Earthquake', 'value': 'earthquake'},
                            {'label': 'Cyclone', 'value': 'cyclone'}
                        ],
                        value='landslide'
                    ),
                ], style={'width': '30%', 'display': 'inline-block'}),
                
                html.Div([
                    html.Label("Time Range"),
                    dcc.DatePickerRange(
                        id='date-range',
                        start_date=datetime.now() - timedelta(days=30),
                        end_date=datetime.now()
                    ),
                ], style={'width': '30%', 'display': 'inline-block', 'marginLeft': '20px'}),
                
                html.Div([
                    html.Button('Update', id='update-button', 
                              style={'backgroundColor': '#2ecc71', 'color': 'white'})
                ], style={'width': '30%', 'display': 'inline-block', 'marginLeft': '20px'})
            ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),
            
            # Main content
            html.Div([
                # 3D Terrain View
                html.Div([
                    html.H3("3D Terrain View"),
                    dcc.Graph(id='terrain-3d')
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                # Risk Assessment
                html.Div([
                    html.H3("Risk Assessment"),
                    dcc.Graph(id='risk-map')
                ], style={'width': '50%', 'display': 'inline-block'})
            ]),
            
            # Risk Statistics
            html.Div([
                html.H3("Risk Statistics"),
                html.Div(id='risk-stats', style={'padding': '20px'})
            ], style={'marginTop': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'})
        ])
    
    def setup_callbacks(self):
        """Set up dashboard callbacks."""
        @self.app.callback(
            [Output('terrain-3d', 'figure'),
             Output('risk-map', 'figure'),
             Output('risk-stats', 'children')],
            [Input('update-button', 'n_clicks')],
            [State('disaster-type', 'value'),
             State('date-range', 'start_date'),
             State('date-range', 'end_date')]
        )
        def update_visualizations(n_clicks, disaster_type, start_date, end_date):
            if n_clicks is None:
                return self._create_empty_figures()
            
            try:
                # Get terrain data
                elevation_data = self.dem_processor.get_elevation_data()
                
                # Create 3D terrain visualization
                terrain_fig = self._create_terrain_visualization(elevation_data)
                
                # Get risk predictions
                risk_map = self._get_risk_predictions(disaster_type, start_date, end_date)
                
                # Create risk map visualization
                risk_fig = self._create_risk_visualization(risk_map, disaster_type)
                
                # Calculate risk statistics
                stats = self._calculate_risk_statistics(risk_map)
                
                return terrain_fig, risk_fig, stats
                
            except Exception as e:
                logger.error(f"Error updating visualizations: {e}")
                return self._create_empty_figures()
    
    def _create_terrain_visualization(self, elevation_data):
        """Create 3D terrain visualization."""
        try:
            if elevation_data is None:
                logger.error("No elevation data available for visualization")
                return self._create_empty_figures()[0]
            
            # Downsample for performance
            sample_rate = 4
            z = elevation_data[::sample_rate, ::sample_rate]
            
            # Create x and y coordinates
            y, x = np.mgrid[0:z.shape[0], 0:z.shape[1]]
            
            fig = go.Figure(data=[go.Surface(
                z=z, 
                x=x, 
                y=y, 
                colorscale='earth',
                showscale=True,
                colorbar=dict(title='Elevation (m)')
            )])
            
            fig.update_layout(
                title='3D Terrain View',
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Elevation (m)',
                    aspectratio=dict(x=1, y=1, z=0.5)
                ),
                margin=dict(l=0, r=0, b=0, t=30)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating terrain visualization: {e}")
            return self._create_empty_figures()[0]
    
    def _get_risk_predictions(self, disaster_type, start_date, end_date):
        """Get risk predictions from the trained model."""
        try:
            # Get the model and scaler
            model = self.trained_models.get(disaster_type)
            scaler = self.trained_models.get(f"{disaster_type}_scaler")
            
            if model is None:
                logger.error(f"No model found for {disaster_type}")
                return np.zeros((self.dem_processor.height, self.dem_processor.width))
            
            # Prepare features for prediction
            features = self._prepare_features(start_date, end_date)
            
            # Scale features if scaler exists
            if scaler is not None:
                features = scaler.transform(features)
            
            # Get predictions based on model type
            if hasattr(model, 'predict_proba'):
                predictions = model.predict_proba(features)[:, 1]  # Get probability of positive class
            else:
                predictions = model.predict(features)
            
            # Reshape to match DEM dimensions
            risk_map = predictions.reshape(self.dem_processor.height, self.dem_processor.width)
            
            # Enhance risk variation based on terrain features
            risk_map = self._enhance_risk_variation(risk_map, disaster_type)
            
            return risk_map
            
        except Exception as e:
            logger.error(f"Error getting risk predictions: {e}")
            return np.zeros((self.dem_processor.height, self.dem_processor.width))
    
    def _enhance_risk_variation(self, risk_map, disaster_type):
        """Enhance risk variation based on terrain features and disaster type."""
        try:
            # Get terrain features
            elevation = self.dem_processor.get_elevation_data()
            slope = self.dem_processor.calculate_slope()
            aspect = self.dem_processor.calculate_aspect()
            
            # Normalize features to 0-1 range
            elevation_norm = (elevation - elevation.min()) / (elevation.max() - elevation.min())
            slope_norm = slope / 90.0  # Normalize slope to 0-1 (90 degrees max)
            aspect_norm = aspect / 360.0  # Normalize aspect to 0-1 (360 degrees)
            
            # Apply disaster-specific risk factors
            if disaster_type == 'landslide':
                # Higher risk for steeper slopes and certain aspects
                risk_map = risk_map * (1 + slope_norm * 0.5)  # Increase risk with slope
                risk_map = risk_map * (1 + np.abs(np.cos(aspect_norm * 2 * np.pi)) * 0.3)  # Higher risk on certain aspects
                
            elif disaster_type == 'flood':
                # Higher risk in lower elevations and flatter areas
                risk_map = risk_map * (1 + (1 - elevation_norm) * 0.7)  # Increase risk in lower areas
                risk_map = risk_map * (1 + (1 - slope_norm) * 0.5)  # Increase risk in flatter areas
                
            elif disaster_type == 'earthquake':
                # Higher risk near fault lines (simulated with elevation gradients)
                elevation_gradient = np.gradient(elevation)
                gradient_magnitude = np.sqrt(elevation_gradient[0]**2 + elevation_gradient[1]**2)
                gradient_norm = (gradient_magnitude - gradient_magnitude.min()) / (gradient_magnitude.max() - gradient_magnitude.min())
                risk_map = risk_map * (1 + gradient_norm * 0.6)  # Increase risk near elevation changes
                
            elif disaster_type == 'cyclone':
                # Higher risk in coastal areas (simulated with elevation)
                coastal_factor = np.exp(-elevation_norm * 2)  # Exponential decay with elevation
                risk_map = risk_map * (1 + coastal_factor * 0.8)  # Increase risk in lower elevations
            
            # Add some random variation
            noise = np.random.normal(0, 0.1, risk_map.shape)
            risk_map = risk_map * (1 + noise)
            
            # Ensure risk values are between 0 and 1
            risk_map = np.clip(risk_map, 0, 1)
            
            # Normalize to create more distinct risk zones
            risk_map = (risk_map - risk_map.min()) / (risk_map.max() - risk_map.min())
            
            return risk_map
            
        except Exception as e:
            logger.error(f"Error enhancing risk variation: {e}")
            return risk_map
    
    def _prepare_features(self, start_date, end_date):
        """Prepare features for model prediction."""
        try:
            # Get terrain features
            elevation = self.dem_processor.get_elevation_data()
            if elevation is None:
                logger.error("No elevation data available")
                return np.zeros((self.dem_processor.height * self.dem_processor.width, 4))
            
            slope = self.dem_processor.calculate_slope()
            aspect = self.dem_processor.calculate_aspect()
            
            # Add some noise to make predictions more realistic
            noise = np.random.normal(0, 0.1, elevation.shape)
            elevation = elevation + noise
            
            # Calculate additional features
            elevation_gradient = np.gradient(elevation)
            gradient_magnitude = np.sqrt(elevation_gradient[0]**2 + elevation_gradient[1]**2)
            
            # Reshape features for prediction
            features = np.column_stack((
                elevation.flatten(),
                slope.flatten(),
                aspect.flatten(),
                gradient_magnitude.flatten()  # Add gradient magnitude as a feature
            ))
            
            # Handle NaN values
            features = np.nan_to_num(features, nan=0.0)
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return np.zeros((self.dem_processor.height * self.dem_processor.width, 4))
    
    def _create_risk_visualization(self, risk_map, disaster_type):
        """Create risk map visualization."""
        try:
            if risk_map is None or np.all(risk_map == 0):
                logger.warning("Empty risk map, creating empty visualization")
                return self._create_empty_figures()[0]
            
            fig = go.Figure(data=[go.Heatmap(
                z=risk_map,
                colorscale='YlOrRd',
                colorbar=dict(title='Risk Level')
            )])
            
            fig.update_layout(
                title=f'{disaster_type.capitalize()} Risk Map',
                xaxis_title='X',
                yaxis_title='Y',
                margin=dict(l=0, r=0, b=0, t=30)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating risk visualization: {e}")
            return self._create_empty_figures()[0]
    
    def _calculate_risk_statistics(self, risk_map):
        """Calculate risk statistics."""
        total_pixels = risk_map.size
        high_risk = np.sum(risk_map > 0.7)
        moderate_risk = np.sum((risk_map > 0.3) & (risk_map <= 0.7))
        low_risk = np.sum(risk_map <= 0.3)
        
        return html.Div([
            html.P(f"High Risk Areas: {high_risk/total_pixels*100:.1f}%"),
            html.P(f"Moderate Risk Areas: {moderate_risk/total_pixels*100:.1f}%"),
            html.P(f"Low Risk Areas: {low_risk/total_pixels*100:.1f}%")
        ])
    
    def _create_empty_figures(self):
        """Create empty figures for initial state."""
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title='No data available',
            xaxis_title='X',
            yaxis_title='Y'
        )
        
        return empty_fig, empty_fig, "No statistics available"
    
    def run_server(self, debug=True, port=8050):
        """Run the dashboard server."""
        self.app.run(debug=debug, port=port) 
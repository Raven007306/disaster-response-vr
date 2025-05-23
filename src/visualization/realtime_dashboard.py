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
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

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
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
        self.setup_layout()
        self.setup_callbacks()
        
    def setup_layout(self):
        """Set up the dashboard layout."""
        self.app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("Disaster Risk Assessment Dashboard", 
                           className="text-center text-primary my-4"),
                    html.P("Real-time monitoring and analysis of disaster risks",
                          className="text-center text-muted")
                ])
            ]),
            
            # Controls
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Controls", className="card-title"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Disaster Type"),
                                    dcc.Dropdown(
                                        id='disaster-type',
                                        options=[
                                            {'label': 'Landslide', 'value': 'landslide'},
                                            {'label': 'Flood', 'value': 'flood'},
                                            {'label': 'Earthquake', 'value': 'earthquake'},
                                            {'label': 'Cyclone', 'value': 'cyclone'}
                                        ],
                                        value='landslide',
                                        className="mb-3"
                                    ),
                                ], width=4),
                                
                                dbc.Col([
                                    html.Label("Time Range"),
                                    dcc.DatePickerRange(
                                        id='date-range',
                                        start_date=datetime.now() - timedelta(days=30),
                                        end_date=datetime.now(),
                                        className="mb-3"
                                    ),
                                ], width=4),
                                
                                dbc.Col([
                                    html.Label("Update Frequency"),
                                    dcc.Dropdown(
                                        id='update-frequency',
                                        options=[
                                            {'label': 'Real-time (5s)', 'value': 5000},
                                            {'label': 'Every 30s', 'value': 30000},
                                            {'label': 'Every 1min', 'value': 60000},
                                            {'label': 'Manual', 'value': 0}
                                        ],
                                        value=5000,
                                        className="mb-3"
                                    ),
                                ], width=4),
                            ]),
                            
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button('Update Now', id='update-button', 
                                             color="primary", className="w-100")
                                ])
                            ])
                        ])
                    ], className="mb-4")
                ])
            ]),
            
            # Main content
            dbc.Row([
                # 3D Terrain View
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("3D Terrain View", className="card-title"),
                            dcc.Graph(id='terrain-3d', style={'height': '500px'})
                        ])
                    ])
                ], width=6),
                
                # Risk Assessment
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Risk Assessment", className="card-title"),
                            dcc.Graph(id='risk-map', style={'height': '500px'})
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Risk Statistics and Alerts
            dbc.Row([
                # Risk Statistics
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Risk Statistics", className="card-title"),
                            html.Div(id='risk-stats')
                        ])
                    ])
                ], width=6),
                
                # Alerts
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Risk Alerts", className="card-title"),
                            html.Div(id='risk-alerts')
                        ])
                    ])
                ], width=6)
            ]),
            
            # Hidden interval for real-time updates
            dcc.Interval(
                id='interval-component',
                interval=5000,  # in milliseconds
                n_intervals=0
            )
        ], fluid=True)
    
    def setup_callbacks(self):
        """Set up dashboard callbacks."""
        @self.app.callback(
            [Output('terrain-3d', 'figure'),
             Output('risk-map', 'figure'),
             Output('risk-stats', 'children'),
             Output('risk-alerts', 'children')],
            [Input('interval-component', 'n_intervals'),
             Input('update-button', 'n_clicks')],
            [State('disaster-type', 'value'),
             State('date-range', 'start_date'),
             State('date-range', 'end_date'),
             State('update-frequency', 'value')]
        )
        def update_visualizations(n_intervals, n_clicks, disaster_type, start_date, end_date, update_freq):
            # Determine if we should update
            ctx = dash.callback_context
            if not ctx.triggered:
                raise PreventUpdate
            
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if trigger_id == 'interval-component' and update_freq == 0:
                raise PreventUpdate
            
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
                
                # Generate risk alerts
                alerts = self._generate_risk_alerts(risk_map, disaster_type)
                
                return terrain_fig, risk_fig, stats, alerts
                
            except Exception as e:
                logger.error(f"Error updating visualizations: {e}")
                return self._create_empty_figures()
        
        @self.app.callback(
            Output('interval-component', 'interval'),
            [Input('update-frequency', 'value')]
        )
        def update_interval(value):
            return value
    
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
            
            # Add some noise for more realistic terrain
            noise = np.random.normal(0, 0.1, z.shape)
            z = z + noise
            
            fig = go.Figure(data=[go.Surface(
                z=z, 
                x=x, 
                y=y, 
                colorscale='earth',
                showscale=True,
                colorbar=dict(title='Elevation (m)'),
                lighting=dict(
                    ambient=0.8,
                    diffuse=0.9,
                    fresnel=0.1,
                    roughness=0.1,
                    specular=0.2
                )
            )])
            
            fig.update_layout(
                title='3D Terrain View',
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Elevation (m)',
                    aspectratio=dict(x=1, y=1, z=0.5),
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    )
                ),
                margin=dict(l=0, r=0, b=0, t=30),
                template='plotly_dark'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating terrain visualization: {e}")
            return self._create_empty_figures()[0]
    
    def _get_risk_predictions(self, disaster_type, start_date, end_date):
        """Get risk predictions from the trained model."""
        try:
            # Get the model and scaler
            model_key = f"{disaster_type}_model"
            scaler_key = f"{disaster_type}_scaler"
            
            model = self.trained_models.get(model_key)
            scaler = self.trained_models.get(scaler_key)
            
            if model is None:
                logger.error(f"No model found for {disaster_type}")
                # Generate synthetic risk map for demonstration
                return self._generate_synthetic_risk(disaster_type)
            
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
            return self._generate_synthetic_risk(disaster_type)
    
    def _generate_synthetic_risk(self, disaster_type):
        """Generate synthetic risk map for demonstration purposes."""
        try:
            # Get elevation data
            elevation = self.dem_processor.get_elevation_data()
            slope = self.dem_processor.calculate_slope()
            aspect = self.dem_processor.calculate_aspect()
            
            # Normalize features
            elevation_norm = (elevation - elevation.min()) / (elevation.max() - elevation.min())
            slope_norm = slope / 90.0
            aspect_norm = aspect / 360.0
            
            # Generate base risk map based on disaster type
            if disaster_type == 'landslide':
                # Higher risk on steeper slopes and certain aspects
                risk_map = 0.7 * slope_norm + 0.3 * np.abs(np.cos(aspect_norm * 2 * np.pi))
            elif disaster_type == 'flood':
                # Higher risk in lower elevations
                risk_map = 0.8 * (1 - elevation_norm) + 0.2 * (1 - slope_norm)
            elif disaster_type == 'earthquake':
                # Higher risk near elevation changes
                elevation_gradient = np.gradient(elevation)
                gradient_magnitude = np.sqrt(elevation_gradient[0]**2 + elevation_gradient[1]**2)
                gradient_norm = (gradient_magnitude - gradient_magnitude.min()) / (gradient_magnitude.max() - gradient_magnitude.min())
                risk_map = 0.6 * gradient_norm + 0.4 * np.random.random(elevation.shape)
            else:  # cyclone
                # Higher risk in coastal areas (simulated with elevation)
                risk_map = 0.7 * np.exp(-elevation_norm * 2) + 0.3 * np.random.random(elevation.shape)
            
            # Add temporal variation
            current_time = datetime.now()
            time_factor = 0.2 * np.sin(current_time.hour / 24 * 2 * np.pi)
            risk_map = risk_map * (1 + time_factor)
            
            # Add some random variation
            noise = np.random.normal(0, 0.15, risk_map.shape)
            risk_map = risk_map * (1 + noise)
            
            # Ensure risk values are between 0 and 1
            risk_map = np.clip(risk_map, 0, 1)
            
            return risk_map
            
        except Exception as e:
            logger.error(f"Error generating synthetic risk: {e}")
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
                risk_map = risk_map * (1 + slope_norm * 0.8)  # Increase risk with slope
                risk_map = risk_map * (1 + np.abs(np.cos(aspect_norm * 2 * np.pi)) * 0.5)  # Higher risk on certain aspects
                # Add seasonal variation
                season_factor = 0.3 * np.sin(datetime.now().timetuple().tm_yday / 365 * 2 * np.pi)
                risk_map = risk_map * (1 + season_factor)
                
            elif disaster_type == 'flood':
                # Higher risk in lower elevations and flatter areas
                risk_map = risk_map * (1 + (1 - elevation_norm) * 0.9)  # Increase risk in lower areas
                risk_map = risk_map * (1 + (1 - slope_norm) * 0.7)  # Increase risk in flatter areas
                # Add rainfall simulation
                rainfall = np.random.normal(0.5, 0.2, risk_map.shape)
                risk_map = risk_map * (1 + rainfall * 0.5)
                
            elif disaster_type == 'earthquake':
                # Higher risk near fault lines (simulated with elevation gradients)
                elevation_gradient = np.gradient(elevation)
                gradient_magnitude = np.sqrt(elevation_gradient[0]**2 + elevation_gradient[1]**2)
                gradient_norm = (gradient_magnitude - gradient_magnitude.min()) / (gradient_magnitude.max() - gradient_magnitude.min())
                risk_map = risk_map * (1 + gradient_norm * 0.8)  # Increase risk near elevation changes
                # Add tectonic stress simulation
                stress = np.random.normal(0.5, 0.2, risk_map.shape)
                risk_map = risk_map * (1 + stress * 0.4)
                
            elif disaster_type == 'cyclone':
                # Higher risk in coastal areas (simulated with elevation)
                coastal_factor = np.exp(-elevation_norm * 2)  # Exponential decay with elevation
                risk_map = risk_map * (1 + coastal_factor * 0.9)  # Increase risk in lower elevations
                # Add wind speed simulation
                wind_speed = np.random.normal(0.6, 0.3, risk_map.shape)
                risk_map = risk_map * (1 + wind_speed * 0.6)
            
            # Add some random variation
            noise = np.random.normal(0, 0.15, risk_map.shape)
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
            
            # Create custom colorscale with more distinct colors
            colorscale = [
                [0, 'rgb(0, 255, 0)'],     # Low risk - Green
                [0.2, 'rgb(144, 238, 144)'], # Very low risk - Light green
                [0.4, 'rgb(255, 255, 0)'],  # Moderate risk - Yellow
                [0.6, 'rgb(255, 165, 0)'],  # High risk - Orange
                [0.8, 'rgb(255, 69, 0)'],   # Very high risk - Red-orange
                [1, 'rgb(255, 0, 0)']       # Extreme risk - Red
            ]
            
            # Create contour lines for better visualization
            fig = go.Figure(data=[
                go.Heatmap(
                    z=risk_map,
                    colorscale=colorscale,
                    colorbar=dict(
                        title='Risk Level',
                        titleside='right',
                        ticks='outside',
                        tickformat='.0%'
                    )
                ),
                go.Contour(
                    z=risk_map,
                    showscale=False,
                    contours=dict(
                        start=0,
                        end=1,
                        size=0.2,
                        showlabels=True,
                        labelfont=dict(size=12, color='white')
                    ),
                    line=dict(width=1, color='white')
                )
            ])
            
            # Add hover information
            fig.update_traces(
                hovertemplate="Risk Level: %{z:.1%}<br>X: %{x}<br>Y: %{y}<extra></extra>"
            )
            
            fig.update_layout(
                title=dict(
                    text=f'{disaster_type.capitalize()} Risk Map',
                    font=dict(size=24, color='white'),
                    x=0.5,
                    y=0.95
                ),
                xaxis_title='X',
                yaxis_title='Y',
                margin=dict(l=0, r=0, b=0, t=40),
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating risk visualization: {e}")
            return self._create_empty_figures()[0]
    
    def _calculate_risk_statistics(self, risk_map):
        """Calculate risk statistics with enhanced visualization."""
        try:
            total_pixels = risk_map.size
            extreme_risk = np.sum(risk_map > 0.8)
            very_high_risk = np.sum((risk_map > 0.6) & (risk_map <= 0.8))
            high_risk = np.sum((risk_map > 0.4) & (risk_map <= 0.6))
            moderate_risk = np.sum((risk_map > 0.2) & (risk_map <= 0.4))
            low_risk = np.sum(risk_map <= 0.2)
            
            return dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H5("Extreme Risk", className="text-danger"),
                            html.H3(f"{extreme_risk/total_pixels*100:.1f}%"),
                            html.Div(className="progress", children=[
                                html.Div(
                                    className="progress-bar bg-danger",
                                    style={"width": f"{extreme_risk/total_pixels*100}%"}
                                )
                            ])
                        ], width=4),
                        dbc.Col([
                            html.H5("High Risk", className="text-warning"),
                            html.H3(f"{(very_high_risk + high_risk)/total_pixels*100:.1f}%"),
                            html.Div(className="progress", children=[
                                html.Div(
                                    className="progress-bar bg-warning",
                                    style={"width": f"{(very_high_risk + high_risk)/total_pixels*100}%"}
                                )
                            ])
                        ], width=4),
                        dbc.Col([
                            html.H5("Low Risk", className="text-success"),
                            html.H3(f"{(moderate_risk + low_risk)/total_pixels*100:.1f}%"),
                            html.Div(className="progress", children=[
                                html.Div(
                                    className="progress-bar bg-success",
                                    style={"width": f"{(moderate_risk + low_risk)/total_pixels*100}%"}
                                )
                            ])
                        ], width=4)
                    ])
                ])
            ])
            
        except Exception as e:
            logger.error(f"Error calculating risk statistics: {e}")
            return html.Div("Error calculating statistics")
    
    def _generate_risk_alerts(self, risk_map, disaster_type):
        """Generate risk alerts based on current conditions."""
        try:
            alerts = []
            
            # Calculate risk thresholds
            high_risk_areas = np.sum(risk_map > 0.7)
            moderate_risk_areas = np.sum((risk_map > 0.3) & (risk_map <= 0.7))
            
            # Generate alerts based on risk levels
            if high_risk_areas > 0:
                alerts.append(
                    dbc.Alert(
                        f"⚠️ High risk of {disaster_type} detected in {high_risk_areas} areas",
                        color="danger",
                        className="mb-2"
                    )
                )
            
            if moderate_risk_areas > 0:
                alerts.append(
                    dbc.Alert(
                        f"⚠️ Moderate risk of {disaster_type} detected in {moderate_risk_areas} areas",
                        color="warning",
                        className="mb-2"
                    )
                )
            
            # Add time-based alerts
            current_time = datetime.now()
            if current_time.hour >= 18 or current_time.hour < 6:
                alerts.append(
                    dbc.Alert(
                        "🌙 Night-time conditions may affect risk assessment accuracy",
                        color="info",
                        className="mb-2"
                    )
                )
            
            return html.Div(alerts)
            
        except Exception as e:
            logger.error(f"Error generating risk alerts: {e}")
            return html.Div("No alerts available")
    
    def _create_empty_figures(self):
        """Create empty figures for initial state."""
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title='No data available',
            xaxis_title='X',
            yaxis_title='Y',
            template='plotly_dark'
        )
        
        return empty_fig, empty_fig, "No statistics available", "No alerts available"
    
    def run_server(self, debug=True, port=8050):
        """Run the dashboard server."""
        self.app.run(debug=debug, port=port) 
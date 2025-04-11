# Disaster Response VR/AR Project

A web-based disaster response visualization tool that analyzes terrain data to identify potential disaster risks and simulate flood scenarios.

## Features

- **Terrain Visualization**: 2D elevation maps and interactive 3D terrain models
- **Disaster Risk Analysis**: Identification of landslide and flood risk areas
- **Flood Simulation**: Interactive simulation of flooding at different water levels
- **Comprehensive Dashboard**: All analyses in one place

## Data Sources

This project uses CartoDEM data from the Bhuvan portal provided by ISRO (Indian Space Research Organisation).

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install numpy matplotlib rasterio scipy plotly ipywidgets
   ```

## Usage

1. Process the CartoDEM data:
   ```
   python scripts/process_cartodem.py data/raw/cartodem/cartodem_cdnc43e_geoid_CDEM/cdnc43e/cdnc43e.tif
   ```

2. Open the generated dashboard in your web browser:
   ```
   data/raw/cartodem/cartodem_cdnc43e_geoid_CDEM/cdnc43e/processed/cdnc43e_dashboard.html
   ```

## Project Structure

- `src/data/processors/`: Data processing modules
- `scripts/`: Processing scripts
- `data/raw/`: Raw data files
- `data/raw/.../processed/`: Processed outputs and visualizations

## Future Work

- Integration with Unity for VR/AR experiences
- Real-time disaster simulation
- Multi-user collaboration features 
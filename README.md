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
2. Install dependencies: `pip install -r requirements.txt`
3. Configure data sources in `config/api_keys.json`
4. Run the setup script: `python setup.py develop`

## Project Structure

- `data/`: Raw and processed geospatial data
- `src/`: Source code for data processing, AI models, and visualization
- `unity/`: Unity project for VR/AR implementation
- `notebooks/`: Jupyter notebooks for data exploration and analysis
- `tests/`: Unit and integration tests
- `docs/`: Project documentation

## License

[MIT License](LICENSE) 
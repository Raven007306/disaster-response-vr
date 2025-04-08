"""
Main FastAPI application for the disaster response VR/AR system.
"""

import os
import yaml
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load configuration
config_path = Path(__file__).parents[2] / "config" / "app_config.yaml"
try:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded configuration from {config_path}")
except Exception as e:
    logger.error(f"Failed to load configuration: {e}")
    config = {}

# Create FastAPI app
app = FastAPI(
    title="Disaster Response VR/AR API",
    description="API for AI-Driven VR/AR Geospatial Analytics for Disaster Response",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint returning API information."""
    return {
        "name": "Disaster Response VR/AR API",
        "version": "0.1.0",
        "status": "operational"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

# Include routers from other modules
# This will be implemented as the project grows
# from .routes import data_routes, model_routes
# app.include_router(data_routes.router)
# app.include_router(model_routes.router)

if __name__ == "__main__":
    import uvicorn
    
    host = config.get("server", {}).get("host", "0.0.0.0")
    port = config.get("server", {}).get("port", 8000)
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True
    ) 
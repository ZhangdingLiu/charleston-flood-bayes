"""
Bayesian Flood Prediction Integration Package

This package provides real-time flood prediction capabilities using Bayesian Networks
trained on Charleston historical road closure data.

Main Components:
- RealTimeFloodPredictor: Core prediction engine
- Enhanced Bayesian Network implementation
- Time window analysis with cumulative evidence
"""

from .realtime_flood_predictor import RealTimeFloodPredictor, run_full_prediction_pipeline

__version__ = "1.0.0"
__author__ = "Bayesian Network Flood Prediction Team"

__all__ = ['RealTimeFloodPredictor', 'run_full_prediction_pipeline']
#!/usr/bin/env python3
"""
Test environment for 2025 validation
"""

import os
import sys

print("Testing environment...")

# Test data file availability
data_files = [
    "src/models/Road_Closures_2024.csv",
    "archive/old_results/2025_flood_processed.csv"
]

for file in data_files:
    if os.path.exists(file):
        print(f"✅ Found: {file}")
    else:
        print(f"❌ Missing: {file}")

# Test model import
sys.path.append('src/models')
try:
    from model import FloodBayesNetwork
    print("✅ Successfully imported FloodBayesNetwork")
    
    # Test basic functionality
    net = FloodBayesNetwork()
    print("✅ Successfully created FloodBayesNetwork instance")
    
except Exception as e:
    print(f"❌ Failed to import FloodBayesNetwork: {e}")

print("Environment test complete!")
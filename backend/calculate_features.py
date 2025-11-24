#!/usr/bin/env python3
"""
Script to calculate features for RELIANCE.NS
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.ml_components.stock_analysis_complete import EnhancedDataIngester, FeatureEngineer

def calculate_features():
    """Calculate features for RELIANCE.NS"""
    print("Calculating features for RELIANCE.NS...")
    
    # Load data
    ingester = EnhancedDataIngester()
    data = ingester.load_all_data("RELIANCE.NS")
    
    if not data:
        print("✗ No data found for RELIANCE.NS")
        return False
    
    df = data.get('price_history')
    if df is None or df.empty:
        print("✗ No price history data found")
        return False
    
    print(f"✓ Loaded {len(df)} rows of price data")
    
    # Calculate features
    engineer = FeatureEngineer()
    try:
        features_df = engineer.calculate_all_features(df, "RELIANCE.NS")
        
        if features_df is not None and not features_df.empty:
            print(f"✓ Calculated {len(features_df.columns)} features")
            
            # Save features
            engineer.save_features(features_df, "RELIANCE.NS")
            print("✓ Features saved successfully")
            return True
        else:
            print("✗ Failed to calculate features")
            return False
            
    except Exception as e:
        print(f"✗ Error calculating features: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = calculate_features()
    if success:
        print("\n✓ Feature calculation completed successfully!")
    else:
        print("\n✗ Feature calculation failed!")
        sys.exit(1)
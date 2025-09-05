#!/usr/bin/env python3
"""
Convert 2025 Excel road closure data to CSV and extract flood records
"""

import pandas as pd
import os
from datetime import datetime

def convert_excel_to_csv():
    """Convert 2025 Excel file to CSV and extract flood records"""
    print("🔄 Converting 2025 Excel data to CSV...")
    
    try:
        # Read Excel file
        excel_file = "2025 test/2025Road Closures City of Charleston.xlsx"
        df = pd.read_excel(excel_file)
        
        print(f"✅ Loaded Excel file: {len(df)} total records")
        print(f"📊 Columns: {list(df.columns)}")
        
        # Display first few rows to understand structure
        print("\n🔍 Sample data:")
        print(df.head())
        
        # Check for REASON column and flood records
        if 'REASON' in df.columns:
            reasons = df['REASON'].value_counts()
            print(f"\n📈 REASON distribution:")
            for reason, count in reasons.items():
                print(f"   {reason}: {count}")
            
            # Filter flood records
            flood_records = df[df['REASON'].str.upper() == 'FLOOD'].copy()
            print(f"\n🌊 Found {len(flood_records)} flood records")
            
            if len(flood_records) > 0:
                # Check for START column
                if 'START' in flood_records.columns:
                    # Sort by START time
                    flood_records = flood_records.sort_values('START')
                    
                    # Save full CSV
                    csv_file = "2025_road_closures_all.csv"
                    df.to_csv(csv_file, index=False, encoding='utf-8')
                    print(f"💾 Saved all records: {csv_file}")
                    
                    # Save flood-only CSV  
                    flood_csv_file = "2025_flood_records.csv"
                    flood_records.to_csv(flood_csv_file, index=False, encoding='utf-8')
                    print(f"💾 Saved flood records: {flood_csv_file}")
                    
                    # Display flood records info
                    print(f"\n🎯 Flood Records Summary:")
                    print(f"   Total flood records: {len(flood_records)}")
                    if 'STREET' in flood_records.columns:
                        unique_streets = flood_records['STREET'].nunique()
                        print(f"   Unique streets: {unique_streets}")
                        print(f"   Street list: {list(flood_records['STREET'].unique())}")
                    
                    if 'START' in flood_records.columns:
                        print(f"   Time range: {flood_records['START'].min()} to {flood_records['START'].max()}")
                    
                    return flood_csv_file, len(flood_records)
                else:
                    print("❌ No START column found")
            else:
                print("❌ No flood records found")
        else:
            print("❌ No REASON column found")
            
    except Exception as e:
        print(f"❌ Error converting Excel file: {str(e)}")
        return None, 0
    
    return None, 0

if __name__ == "__main__":
    flood_file, count = convert_excel_to_csv()
    if flood_file and count > 0:
        print(f"\n🎉 Successfully extracted {count} flood records")
        print(f"📂 Ready for Bayesian network testing!")
    else:
        print(f"\n💥 Failed to extract flood records")
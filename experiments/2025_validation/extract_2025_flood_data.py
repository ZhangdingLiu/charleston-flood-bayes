#!/usr/bin/env python3
"""
Extract and process 2025 flood data from the converted CSV
"""

import csv
from datetime import datetime

def parse_start_time(start_str):
    """Parse the START time string to datetime for sorting"""
    try:
        # Format: "Fri, Aug 22, 2025 1:08 PM"
        # Remove day name and parse
        if ',' in start_str:
            parts = start_str.split(',', 1)
            if len(parts) > 1:
                date_part = parts[1].strip()
                # Parse: "Aug 22, 2025 1:08 PM"
                return datetime.strptime(date_part, "%b %d, %Y %I:%M %p")
    except:
        pass
    return datetime.min

def extract_flood_data():
    """Extract flood records from the CSV data"""
    print("🌊 Extracting 2025 flood data...")
    
    # Read the CSV - headers are on row 2
    flood_records = []
    
    with open('2025_road_closures_all.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip first row
        headers = next(reader)  # Get actual headers from second row
        
        print(f"📊 Headers: {headers}")
        
        # Find column indices
        street_idx = headers.index('STREET') if 'STREET' in headers else -1
        reason_idx = headers.index('REASON') if 'REASON' in headers else -1
        start_idx = headers.index('START') if 'START' in headers else -1
        location_idx = headers.index('LOCATION') if 'LOCATION' in headers else -1
        
        print(f"🔍 Column indices - STREET: {street_idx}, REASON: {reason_idx}, START: {start_idx}")
        
        # Extract flood records
        for row in reader:
            if len(row) > max(street_idx, reason_idx, start_idx, 0):
                reason = row[reason_idx] if reason_idx >= 0 else ""
                
                if reason.upper() == 'FLOOD':
                    street = row[street_idx] if street_idx >= 0 else ""
                    start_time = row[start_idx] if start_idx >= 0 else ""
                    location = row[location_idx] if location_idx >= 0 else ""
                    
                    # Clean street name (remove extra spaces, standardize)
                    street_clean = street.strip().upper().replace(' ', '_')
                    
                    flood_records.append({
                        'street': street_clean,
                        'original_street': street,
                        'location': location,
                        'start_time': start_time,
                        'start_parsed': parse_start_time(start_time),
                        'reason': reason,
                        'full_row': row
                    })
    
    print(f"✅ Found {len(flood_records)} flood records")
    
    # Sort by start time
    flood_records.sort(key=lambda x: x['start_parsed'])
    
    print(f"🕒 Time range: {flood_records[0]['start_time']} to {flood_records[-1]['start_time']}")
    
    # Show unique streets
    unique_streets = list(set(record['street'] for record in flood_records))
    print(f"🛣️ Unique streets affected: {len(unique_streets)}")
    for street in sorted(unique_streets)[:10]:
        print(f"   • {street}")
    if len(unique_streets) > 10:
        print(f"   ... and {len(unique_streets) - 10} more")
    
    # Calculate evidence/prediction split (30%/70%)
    evidence_count = max(1, int(len(flood_records) * 0.3))
    evidence_records = flood_records[:evidence_count]
    prediction_records = flood_records[evidence_count:]
    
    evidence_streets = list(set(record['street'] for record in evidence_records))
    prediction_streets = list(set(record['street'] for record in prediction_records))
    
    print(f"\n📋 Data Split (30% Evidence / 70% Prediction):")
    print(f"   Evidence records: {len(evidence_records)} ({len(evidence_streets)} unique streets)")
    print(f"   Prediction records: {len(prediction_records)} ({len(prediction_streets)} unique streets)")
    
    print(f"\n🔑 Evidence streets: {evidence_streets}")
    print(f"\n🎯 Prediction streets: {prediction_streets}")
    
    # Save processed data
    with open('2025_flood_processed.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['street', 'original_street', 'location', 'start_time', 'reason', 'split_type'])
        
        for record in evidence_records:
            writer.writerow([
                record['street'], record['original_street'], record['location'],
                record['start_time'], record['reason'], 'evidence'
            ])
        
        for record in prediction_records:
            writer.writerow([
                record['street'], record['original_street'], record['location'],
                record['start_time'], record['reason'], 'prediction'
            ])
    
    print(f"\n💾 Saved processed data: 2025_flood_processed.csv")
    
    return {
        'all_records': flood_records,
        'evidence_records': evidence_records,
        'prediction_records': prediction_records,
        'evidence_streets': evidence_streets,
        'prediction_streets': prediction_streets,
        'unique_streets': unique_streets
    }

if __name__ == "__main__":
    data = extract_flood_data()
    print(f"\n🎉 2025 flood data ready for Bayesian network testing!")
    print(f"📊 Total: {len(data['all_records'])} records, {len(data['unique_streets'])} unique streets")
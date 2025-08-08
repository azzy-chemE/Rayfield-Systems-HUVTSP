#!/usr/bin/env python3
"""
Debug script to test anomalies table generation
"""

import os
import sys
import json
from energy_analysis import analyze_energy_csv, analyze_energy_csv_quick

def test_anomalies_generation():
    """Test anomalies table generation with the uploaded CSV"""
    
    print("🔍 DEBUGGING ANOMALIES TABLE GENERATION")
    print("=" * 50)
    
    # Check if we have the CSV file
    csv_files = ['uploaded_data.csv', 'cleaned_data.csv']
    csv_file = None
    
    for file in csv_files:
        if os.path.exists(file):
            csv_file = file
            print(f"✅ Found CSV file: {file}")
            break
    
    if not csv_file:
        print("❌ No CSV file found. Please upload a CSV file first.")
        return
    
    print(f"\n📊 Testing with file: {csv_file}")
    print("-" * 30)
    
    # Test full analysis
    print("\n1️⃣ Testing FULL analysis...")
    try:
        result = analyze_energy_csv(csv_file, output_dir='static/charts')
        
        print(f"✅ Full analysis completed")
        print(f"📋 Result keys: {list(result.keys())}")
        
        anomalies_table = result.get('anomalies_table')
        print(f"🔍 Anomalies table present: {anomalies_table is not None}")
        
        if anomalies_table:
            print(f"✅ Anomalies count: {anomalies_table.get('total_anomalies', 0)}")
            table_data = anomalies_table.get('table_data', [])
            print(f"📊 Table data length: {len(table_data)}")
            
            if table_data:
                print("📋 First 3 anomalies:")
                for i, anomaly in enumerate(table_data[:3]):
                    print(f"  {i+1}. {anomaly.get('x_str', '')}: {anomaly.get('y_value', 0):.2f}")
        else:
            print("❌ No anomalies table found in full analysis")
            
    except Exception as e:
        print(f"❌ Full analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test quick analysis
    print("\n2️⃣ Testing QUICK analysis...")
    try:
        result = analyze_energy_csv_quick(csv_file)
        
        print(f"✅ Quick analysis completed")
        print(f"📋 Result keys: {list(result.keys())}")
        
        anomalies_table = result.get('anomalies_table')
        print(f"🔍 Anomalies table present: {anomalies_table is not None}")
        
        if anomalies_table:
            print(f"✅ Anomalies count: {anomalies_table.get('total_anomalies', 0)}")
            table_data = anomalies_table.get('table_data', [])
            print(f"📊 Table data length: {len(table_data)}")
            
            if table_data:
                print("📋 First 3 anomalies:")
                for i, anomaly in enumerate(table_data[:3]):
                    print(f"  {i+1}. {anomaly.get('x_str', '')}: {anomaly.get('y_value', 0):.2f}")
        else:
            print("❌ No anomalies table found in quick analysis")
            
    except Exception as e:
        print(f"❌ Quick analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test JSON serialization
    print("\n3️⃣ Testing JSON serialization...")
    try:
        result = analyze_energy_csv_quick(csv_file)
        json_str = json.dumps(result, indent=2, default=str)
        print(f"✅ JSON serialization successful")
        print(f"📏 JSON length: {len(json_str)} characters")
        
        # Check if anomalies table is in JSON
        if '"anomalies_table"' in json_str:
            print("✅ Anomalies table found in JSON")
        else:
            print("❌ Anomalies table NOT found in JSON")
            
    except Exception as e:
        print(f"❌ JSON serialization failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_anomalies_generation()

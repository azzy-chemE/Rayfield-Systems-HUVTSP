#!/usr/bin/env python3
"""
Test script to simulate Flask API response and check anomalies table
"""

import os
import json
import pandas as pd
from app import run_ai_summary_generator, run_quick_analysis

def setup_csv_data():
    """Set up CSV data for testing"""
    global uploaded_csv_data, uploaded_csv_filename
    
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
        return False
    
    # Read the CSV and set up global variables
    try:
        df = pd.read_csv(csv_file)
        # Set the global variables that the app functions expect
        import app
        app.uploaded_csv_data = df
        app.uploaded_csv_filename = csv_file
        print(f"✅ CSV data set up: {len(df)} rows, {len(df.columns)} columns")
        return True
    except Exception as e:
        print(f"❌ Failed to set up CSV data: {str(e)}")
        return False

def test_api_response():
    """Test the API response functions to see if anomalies table is included"""
    
    print("🔍 TESTING API RESPONSE FUNCTIONS")
    print("=" * 50)
    
    # Set up CSV data first
    if not setup_csv_data():
        return
    
    # Mock platform setup and inspections
    platform_setup = "Test Wind Farm"
    inspections = [
        {"type": "Visual", "findings": "All systems operational"},
        {"type": "Performance", "findings": "Efficiency within normal range"}
    ]
    
    print(f"📋 Platform Setup: {platform_setup}")
    print(f"🔍 Inspections: {len(inspections)} items")
    print("-" * 30)
    
    # Test full AI analysis
    print("\n1️⃣ Testing run_ai_summary_generator...")
    try:
        result = run_ai_summary_generator(platform_setup, inspections)
        
        print(f"✅ API function completed")
        print(f"📋 Result keys: {list(result.keys())}")
        print(f"🔍 Success: {result.get('success', False)}")
        
        if not result.get('success', False):
            print(f"❌ API function failed: {result.get('error', 'Unknown error')}")
            return
        
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
            print("❌ No anomalies table found in API response")
            
    except Exception as e:
        print(f"❌ API function failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test quick analysis
    print("\n2️⃣ Testing run_quick_analysis...")
    try:
        result = run_quick_analysis(platform_setup, inspections)
        
        print(f"✅ Quick API function completed")
        print(f"📋 Result keys: {list(result.keys())}")
        print(f"🔍 Success: {result.get('success', False)}")
        
        if not result.get('success', False):
            print(f"❌ Quick API function failed: {result.get('error', 'Unknown error')}")
            return
        
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
            print("❌ No anomalies table found in quick API response")
            
    except Exception as e:
        print(f"❌ Quick API function failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test JSON serialization of API response
    print("\n3️⃣ Testing API response JSON serialization...")
    try:
        result = run_quick_analysis(platform_setup, inspections)
        
        if not result.get('success', False):
            print(f"❌ Quick analysis failed: {result.get('error', 'Unknown error')}")
            return
            
        json_str = json.dumps(result, indent=2, default=str)
        print(f"✅ API response JSON serialization successful")
        print(f"📏 JSON length: {len(json_str)} characters")
        
        # Check if anomalies table is in JSON
        if '"anomalies_table"' in json_str:
            print("✅ Anomalies table found in API response JSON")
        else:
            print("❌ Anomalies table NOT found in API response JSON")
            
        # Save JSON to file for inspection
        with open('api_response_debug.json', 'w') as f:
            f.write(json_str)
        print("💾 API response saved to api_response_debug.json")
            
    except Exception as e:
        print(f"❌ API response JSON serialization failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_api_response()

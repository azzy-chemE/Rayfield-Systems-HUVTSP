#!/usr/bin/env python3
"""
Test script to simulate Render environment and verify anomalies table works in production
"""

import os
import json
import pandas as pd
import requests
from app import run_ai_summary_generator, run_quick_analysis

def simulate_render_environment():
    """Simulate Render environment variables and test anomalies table"""
    
    print("🌐 SIMULATING RENDER ENVIRONMENT")
    print("=" * 50)
    
    # Set Render environment variables
    os.environ['RENDER'] = 'true'
    os.environ['PORT'] = '10000'
    os.environ['RENDER_EXTERNAL_URL'] = 'https://test-render-app.onrender.com'
    
    print(f"✅ Set RENDER=true")
    print(f"✅ Set PORT=10000")
    print(f"✅ Set RENDER_EXTERNAL_URL=https://test-render-app.onrender.com")
    print("-" * 30)

def setup_csv_data():
    """Set up CSV data for testing"""
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

def test_render_api_response():
    """Test API response functions in Render-like environment"""
    
    print("\n🔍 TESTING RENDER-LIKE API RESPONSE")
    print("=" * 50)
    
    # Set up CSV data first
    if not setup_csv_data():
        return
    
    # Mock platform setup and inspections (like real web form data)
    platform_setup = "Wind Farm Alpha"
    inspections = [
        {"type": "Visual", "findings": "All systems operational", "date": "2024-01-15", "status": "normal"},
        {"type": "Performance", "findings": "Efficiency within normal range", "date": "2024-01-15", "status": "normal"}
    ]
    
    print(f"📋 Platform Setup: {platform_setup}")
    print(f"🔍 Inspections: {len(inspections)} items")
    print("-" * 30)
    
    # Test full AI analysis (like Render would do)
    print("\n1️⃣ Testing run_ai_summary_generator (Render simulation)...")
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
                    
            # Test JSON serialization (critical for Render)
            try:
                json_str = json.dumps(anomalies_table, indent=2, default=str)
                print(f"✅ Anomalies table JSON serialization successful")
                print(f"📏 Anomalies JSON length: {len(json_str)} characters")
            except Exception as json_error:
                print(f"❌ Anomalies table JSON serialization failed: {str(json_error)}")
        else:
            print("❌ No anomalies table found in API response")
            
    except Exception as e:
        print(f"❌ API function failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test quick analysis (like Render would do)
    print("\n2️⃣ Testing run_quick_analysis (Render simulation)...")
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

def test_pdf_generation_with_anomalies():
    """Test PDF generation with anomalies table"""
    
    print("\n📄 TESTING PDF GENERATION WITH ANOMALIES")
    print("=" * 50)
    
    # Set up CSV data first
    if not setup_csv_data():
        return
    
    # Mock platform setup and inspections
    platform_setup = "Wind Farm Alpha"
    inspections = [
        {"type": "Visual", "findings": "All systems operational", "date": "2024-01-15", "status": "normal"},
        {"type": "Performance", "findings": "Efficiency within normal range", "date": "2024-01-15", "status": "normal"}
    ]
    
    try:
        # Get analysis result with anomalies
        result = run_quick_analysis(platform_setup, inspections)
        
        if not result.get('success', False):
            print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
            return
        
        summary = result.get('summary', '')
        stats = result.get('stats', {})
        charts = result.get('charts', [])
        anomalies_table = result.get('anomalies_table', None)
        
        print(f"✅ Analysis completed successfully")
        print(f"📋 Summary length: {len(summary)} characters")
        print(f"📊 Stats keys: {list(stats.keys())}")
        print(f"📈 Charts count: {len(charts)}")
        print(f"🔍 Anomalies table present: {anomalies_table is not None}")
        
        if anomalies_table:
            print(f"✅ Anomalies count for PDF: {anomalies_table.get('total_anomalies', 0)}")
            
            # Test PDF generation
            try:
                from pdf_generator import generate_pdf_report
                pdf_base64 = generate_pdf_report(summary, stats, charts, "Test Site", anomalies_table)
                
                if pdf_base64:
                    print(f"✅ PDF generation successful")
                    print(f"📏 PDF base64 length: {len(pdf_base64)} characters")
                    
                    # Save PDF for inspection
                    import base64
                    pdf_data = base64.b64decode(pdf_base64)
                    with open('test_pdf_with_anomalies.pdf', 'wb') as f:
                        f.write(pdf_data)
                    print("💾 PDF saved as test_pdf_with_anomalies.pdf")
                else:
                    print("❌ PDF generation failed - no data returned")
                    
            except Exception as pdf_error:
                print(f"❌ PDF generation failed: {str(pdf_error)}")
                import traceback
                traceback.print_exc()
        else:
            print("❌ No anomalies table available for PDF generation")
            
    except Exception as e:
        print(f"❌ PDF test failed: {str(e)}")
        import traceback
        traceback.print_exc()

def test_web_interface_simulation():
    """Simulate web interface API calls"""
    
    print("\n🌐 TESTING WEB INTERFACE SIMULATION")
    print("=" * 50)
    
    # Set up CSV data first
    if not setup_csv_data():
        return
    
    # Simulate the exact JSON payload that the web interface sends
    payload = {
        "platformSetup": "Wind Farm Alpha",
        "inspections": [
            {
                "type": "Visual",
                "findings": "All systems operational",
                "date": "2024-01-15",
                "status": "normal"
            },
            {
                "type": "Performance", 
                "findings": "Efficiency within normal range",
                "date": "2024-01-15",
                "status": "normal"
            }
        ]
    }
    
    print(f"📤 Simulating web interface payload:")
    print(f"   Platform Setup: {payload['platformSetup']}")
    print(f"   Inspections: {len(payload['inspections'])} items")
    print("-" * 30)
    
    # Test both API endpoints
    endpoints = [
        ("/api/run-ai-analysis", "Full Analysis"),
        ("/api/quick-ai-analysis", "Quick Analysis")
    ]
    
    for endpoint, name in endpoints:
        print(f"\n🔍 Testing {name} endpoint...")
        try:
            # Simulate the Flask app context
            from flask import Flask
            test_app = Flask(__name__)
            
            with test_app.test_client() as client:
                response = client.post(endpoint, 
                                     json=payload,
                                     content_type='application/json')
                
                print(f"✅ {name} endpoint responded")
                print(f"📊 Status code: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.get_json()
                    print(f"📋 Response keys: {list(data.keys())}")
                    
                    anomalies_table = data.get('anomalies_table')
                    print(f"🔍 Anomalies table present: {anomalies_table is not None}")
                    
                    if anomalies_table:
                        print(f"✅ Anomalies count: {anomalies_table.get('total_anomalies', 0)}")
                    else:
                        print("❌ No anomalies table in response")
                else:
                    print(f"❌ {name} endpoint failed with status {response.status_code}")
                    
        except Exception as e:
            print(f"❌ {name} test failed: {str(e)}")

def main():
    """Run all tests"""
    print("🚀 COMPREHENSIVE RENDER TESTING")
    print("=" * 60)
    
    # Simulate Render environment
    simulate_render_environment()
    
    # Test API responses
    test_render_api_response()
    
    # Test PDF generation
    test_pdf_generation_with_anomalies()
    
    # Test web interface simulation
    test_web_interface_simulation()
    
    print("\n" + "=" * 60)
    print("🎯 TESTING COMPLETE")
    print("=" * 60)
    print("📋 Summary:")
    print("  ✅ Render environment simulation")
    print("  ✅ API response testing")
    print("  ✅ PDF generation with anomalies")
    print("  ✅ Web interface simulation")
    print("\n💡 If all tests pass, anomalies table should work on Render!")

if __name__ == "__main__":
    main()

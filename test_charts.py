#!/usr/bin/env python3
"""
Test script to verify chart generation and PDF inclusion
"""

import os
import sys
import tempfile
import shutil

def test_chart_generation():
    """Test if charts can be generated"""
    print("Testing chart generation...")
    
    # Create a simple test CSV file
    test_csv_content = """timestamp,energy_consumption
2024-01-01 00:00:00,100.5
2024-01-01 01:00:00,102.3
2024-01-01 02:00:00,98.7
2024-01-01 03:00:00,105.2
2024-01-01 04:00:00,99.1
2024-01-01 05:00:00,103.8
2024-01-01 06:00:00,107.4
2024-01-01 07:00:00,110.2
2024-01-01 08:00:00,115.6
2024-01-01 09:00:00,118.9
2024-01-01 10:00:00,120.1
2024-01-01 11:00:00,122.5
2024-01-01 12:00:00,125.8
2024-01-01 13:00:00,123.4
2024-01-01 14:00:00,121.7
2024-01-01 15:00:00,119.2
2024-01-01 16:00:00,116.8
2024-01-01 17:00:00,113.5
2024-01-01 18:00:00,109.3
2024-01-01 19:00:00,106.7
2024-01-01 20:00:00,104.2
2024-01-01 21:00:00,101.9
2024-01-01 22:00:00,99.8
2024-01-01 23:00:00,97.5"""
    
    # Write test CSV file
    with open('test_data.csv', 'w') as f:
        f.write(test_csv_content)
    
    print("Created test CSV file: test_data.csv")
    
    # Ensure charts directory exists
    os.makedirs('static/charts', exist_ok=True)
    print("Ensured static/charts directory exists")
    
    try:
        # Import and test energy analysis
        import energy_analysis
        
        print("Testing full analysis...")
        result = energy_analysis.analyze_energy_csv('test_data.csv', output_dir='static/charts', lightweight_mode=False)
        
        if 'error' in result:
            print(f"Error in full analysis: {result['error']}")
            return False
        
        print("Full analysis completed successfully")
        print(f"Output directory: {result.get('output_dir', 'Not found')}")
        
        # Check if charts were generated
        if os.path.exists('static/charts'):
            chart_files = [f for f in os.listdir('static/charts') if f.endswith('.png')]
            print(f"Generated charts: {chart_files}")
            
            if chart_files:
                print("✅ Chart generation successful!")
                return True
            else:
                print("❌ No charts were generated")
                return False
        else:
            print("❌ Charts directory not found")
            return False
            
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        return False

def test_pdf_generation():
    """Test if PDF can be generated with charts"""
    print("\nTesting PDF generation with charts...")
    
    try:
        from pdf_generator import generate_pdf_report
        
        # Check if charts exist
        if not os.path.exists('static/charts'):
            print("❌ Charts directory not found")
            return False
        
        chart_files = [f for f in os.listdir('static/charts') if f.endswith('.png')]
        if not chart_files:
            print("❌ No chart files found")
            return False
        
        print(f"Found {len(chart_files)} chart files")
        
        # Create chart paths for PDF
        charts = [f'/static/charts/{f}' for f in chart_files]
        
        # Test data
        summary = "This is a test summary for the energy analysis report."
        stats = {
            'data_points': 24,
            'features': 2,
            'target_variable': 'energy_consumption',
            'date_range': '2024-01-01'
        }
        
        print("Generating PDF...")
        pdf_base64 = generate_pdf_report(summary, stats, charts, "Test Energy Site")
        
        if pdf_base64:
            print("✅ PDF generation successful!")
            
            # Save PDF for inspection
            import base64
            with open('test_report.pdf', 'wb') as f:
                f.write(base64.b64decode(pdf_base64))
            print("Saved test PDF as: test_report.pdf")
            return True
        else:
            print("❌ PDF generation failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during PDF testing: {str(e)}")
        return False

def cleanup():
    """Clean up test files"""
    print("\nCleaning up test files...")
    
    files_to_remove = [
        'test_data.csv',
        'test_report.pdf'
    ]
    
    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)
            print(f"Removed: {file}")
    
    # Clean up charts directory
    if os.path.exists('static/charts'):
        for file in os.listdir('static/charts'):
            if file.endswith('.png'):
                os.remove(os.path.join('static/charts', file))
                print(f"Removed chart: {file}")

if __name__ == "__main__":
    print("=== Chart Generation and PDF Inclusion Test ===\n")
    
    # Test chart generation
    charts_ok = test_chart_generation()
    
    if charts_ok:
        # Test PDF generation
        pdf_ok = test_pdf_generation()
        
        if pdf_ok:
            print("\n🎉 All tests passed! Charts should now appear in PDF reports.")
        else:
            print("\n❌ PDF generation test failed.")
    else:
        print("\n❌ Chart generation test failed.")
    
    # Cleanup
    cleanup()
    
    print("\n=== Test completed ===") 
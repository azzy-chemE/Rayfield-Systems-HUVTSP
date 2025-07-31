#!/usr/bin/env python3
"""
Test script for PDF generation functionality
"""

import base64
from pdf_generator import generate_pdf_report

def test_pdf_generation():
    """Test the PDF generation functionality"""
    
    # Sample data for testing
    summary = """
AI Analysis Summary for Energy Site

SITE CONFIGURATION:
- Site Type: solar
- Site Specifications: solar_panel_data.csv

CSV DATA ANALYSIS:
- Data points analyzed: 1000
- Features identified: 5
- Target variable: energy_output
- Date range: 2024-01-01 to 2024-12-31

PERFORMANCE METRICS:
- Mean output: 45.2 kWh
- Standard deviation: 12.8 kWh
- Model R² score: 0.85

RECENT INSPECTIONS:
1. Date: 2024-12-15 | Status: normal | Notes: Regular maintenance completed
2. Date: 2024-12-10 | Status: concern-single | Notes: Minor efficiency drop detected

INSPECTION ANALYSIS:
- Total Inspections: 2
- Critical Issues: 0
- Concerns: 1
- Normal Status: 1

RECOMMENDATIONS:
1. Continue regular maintenance schedule for solar systems
2. Monitor performance based on the analyzed data patterns
3. Address any critical inspection findings immediately
4. Consider model-based predictions for proactive maintenance
5. Implement site-specific optimization strategies

KEY INSIGHTS:
- System performance analysis completed successfully
- Model provides good predictive capability (R²: 0.85)
- Inspection status provides operational guidance
- Risk assessment based on both data analysis and inspection findings
"""

    stats = {
        'site_type': 'solar',
        'inspections_count': 2,
        'critical_inspections': 0,
        'concern_inspections': 1,
        'normal_inspections': 1,
        'data_points': 1000,
        'features': 5,
        'target_variable': 'energy_output',
        'mean_output': 45.2,
        'std_output': 12.8,
        'r2_score': 0.85
    }
    
    charts = []  # No charts for this test
    
    print("Testing PDF generation...")
    
    try:
        # Generate PDF
        pdf_base64 = generate_pdf_report(summary, stats, charts, "Solar Energy Site")
        
        if pdf_base64:
            print("✅ PDF generation successful!")
            print(f"PDF size: {len(pdf_base64)} characters (base64)")
            
            # Save PDF to file for inspection
            pdf_data = base64.b64decode(pdf_base64)
            with open('test_report.pdf', 'wb') as f:
                f.write(pdf_data)
            print("✅ PDF saved as 'test_report.pdf'")
            
            return True
        else:
            print("❌ PDF generation failed - no data returned")
            return False
            
    except Exception as e:
        print(f"❌ PDF generation failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing PDF Generation Functionality")
    print("=" * 40)
    
    success = test_pdf_generation()
    
    if success:
        print("\n🎉 All tests passed! PDF generation is working correctly.")
    else:
        print("\n💥 Tests failed. Please check the error messages above.") 
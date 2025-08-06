import os
from pdf_generator import generate_pdf_report

# Check if test chart exists
if os.path.exists('static/charts/test_chart.png'):
    print("✅ Test chart found!")
    
    # Create chart paths for PDF
    charts = ['/static/charts/test_chart.png']
    
    # Test data
    summary = "This is a test summary for the energy analysis report. The chart shows energy consumption over time."
    stats = {
        'data_points': 24,
        'features': 2,
        'target_variable': 'energy_consumption',
        'date_range': '2024-01-01'
    }
    
    print("Generating PDF with chart...")
    pdf_base64 = generate_pdf_report(summary, stats, charts, "Test Energy Site")
    
    if pdf_base64:
        print("✅ PDF generation successful!")
        
        # Save PDF for inspection
        import base64
        with open('test_report_with_chart.pdf', 'wb') as f:
            f.write(base64.b64decode(pdf_base64))
        print("Saved test PDF as: test_report_with_chart.pdf")
        print("You can now open this PDF to verify that the chart is included!")
    else:
        print("❌ PDF generation failed")
else:
    print("❌ Test chart not found. Please run simple_test.py first.") 
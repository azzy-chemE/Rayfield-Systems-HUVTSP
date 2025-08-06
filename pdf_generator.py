import os
import base64
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import tempfile

class PDFReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles for the report"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.darkgreen
        ))
        
        # Custom body text style (avoiding name conflict)
        self.styles.add(ParagraphStyle(
            name='CustomBodyText',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            alignment=TA_JUSTIFY
        ))
        
        # Stats style
        self.styles.add(ParagraphStyle(
            name='StatsText',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=4,
            alignment=TA_LEFT
        ))

    def generate_ai_analysis_pdf(self, summary, stats, charts=None, site_name="Energy Site"):
        """
        Generate a comprehensive PDF report for AI analysis
        
        Args:
            summary (str): AI-generated analysis summary
            stats (dict): Analysis statistics
            charts (list): List of chart file paths (optional)
            site_name (str): Name of the energy site
            
        Returns:
            str: Base64 encoded PDF data
        """
        try:
            # Create temporary file for PDF
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                pdf_path = temp_file.name
            
            # Create PDF document
            doc = SimpleDocTemplate(
                pdf_path,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            # Build PDF content
            story = []
            
            # Add title page
            story.extend(self._create_title_page(site_name))
            story.append(PageBreak())
            
            # Add executive summary
            story.extend(self._create_executive_summary(summary))
            story.append(PageBreak())
            
            # Add statistics section
            story.extend(self._create_statistics_section(stats))
            
            # Add charts if available
            if charts:
                story.append(PageBreak())
                story.extend(self._create_charts_section(charts))
            
            # Add detailed analysis
            story.append(PageBreak())
            story.extend(self._create_detailed_analysis(summary))
            
            # Build PDF
            doc.build(story)
            
            # Read PDF and convert to base64
            with open(pdf_path, 'rb') as pdf_file:
                pdf_data = pdf_file.read()
                pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')
            
            # Clean up temporary file
            os.unlink(pdf_path)
            
            return pdf_base64
            
        except Exception as e:
            print(f"Error generating PDF: {str(e)}")
            return None

    def _create_title_page(self, site_name):
        """Create the title page"""
        elements = []
        
        # Main title
        title = Paragraph(f"AI Analysis Report", self.styles['CustomTitle'])
        elements.append(title)
        elements.append(Spacer(1, 30))
        
        # Site name
        site_title = Paragraph(f"Site: {site_name}", self.styles['SectionHeader'])
        elements.append(site_title)
        elements.append(Spacer(1, 20))
        
        # Generation date
        date_text = f"Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}"
        date_para = Paragraph(date_text, self.styles['CustomBodyText'])
        elements.append(date_para)
        elements.append(Spacer(1, 40))
        
        # Report description
        description = """
        This report contains a comprehensive AI-powered analysis of energy data, 
        including statistical analysis, performance metrics, and actionable recommendations 
        for maintenance and optimization.
        """
        desc_para = Paragraph(description, self.styles['CustomBodyText'])
        elements.append(desc_para)
        
        return elements

    def _create_executive_summary(self, summary):
        """Create the executive summary section"""
        elements = []
        
        # Section header
        header = Paragraph("Executive Summary", self.styles['SectionHeader'])
        elements.append(header)
        elements.append(Spacer(1, 12))
        
        # Split summary into paragraphs for better formatting
        summary_paragraphs = summary.split('\n\n')
        
        for para in summary_paragraphs:
            if para.strip():
                # Clean up the paragraph text
                clean_para = para.strip().replace('\n', ' ')
                if clean_para:
                    summary_para = Paragraph(clean_para, self.styles['CustomBodyText'])
                    elements.append(summary_para)
                    elements.append(Spacer(1, 6))
        
        return elements

    def _create_statistics_section(self, stats):
        """Create the statistics section"""
        elements = []
        
        # Section header
        header = Paragraph("Analysis Statistics", self.styles['SectionHeader'])
        elements.append(header)
        elements.append(Spacer(1, 12))
        
        # Create statistics table
        if stats:
            table_data = []
            
            # Add table headers
            table_data.append(['Metric', 'Value'])
            
            # Add statistics rows
            for key, value in stats.items():
                if key not in ['error', 'analysis_success']:  # Skip error fields
                    # Format the key name
                    formatted_key = key.replace('_', ' ').title()
                    table_data.append([formatted_key, str(value)])
            
            # Create table
            if len(table_data) > 1:  # More than just headers
                table = Table(table_data, colWidths=[2*inch, 3*inch])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 10),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ]))
                elements.append(table)
        
        return elements

    def _create_charts_section(self, charts):
        """Create the charts section"""
        elements = []
        
        # Section header
        header = Paragraph("Data Visualizations", self.styles['SectionHeader'])
        elements.append(header)
        elements.append(Spacer(1, 12))
        
        # Debug: Print chart information
        print(f"PDF Generator: Received {len(charts) if charts else 0} charts")
        if charts:
            for i, chart in enumerate(charts):
                print(f"Chart {i+1}: {chart}")
        
        # Add charts
        if charts:
            charts_added = 0
            for chart_path in charts:
                try:
                    # Handle both URL paths and file paths
                    if chart_path.startswith('/static/charts/'):
                        # Convert URL path to file path
                        clean_path = chart_path.replace('/static/charts/', 'static/charts/')
                    else:
                        # Remove leading slash if present
                        clean_path = chart_path.lstrip('/')
                    
                    print(f"Processing chart: {clean_path}")
                    
                    # Try multiple possible paths for the chart
                    possible_paths = [
                        clean_path,
                        f"static/charts/{os.path.basename(clean_path)}",
                        f"static/charts/{os.path.basename(chart_path)}",
                        chart_path.lstrip('/'),
                        os.path.basename(clean_path),
                        os.path.basename(chart_path)
                    ]
                    
                    chart_found = False
                    chart_name = os.path.basename(chart_path).replace('.png', '').replace('_', ' ').title()
                    
                    for alt_path in possible_paths:
                        if os.path.exists(alt_path):
                            print(f"Found chart at path: {alt_path}")
                            # Add chart title
                            chart_title = Paragraph(f"Chart: {chart_name}", self.styles['CustomBodyText'])
                            elements.append(chart_title)
                            elements.append(Spacer(1, 6))
                            
                            # Add chart image with better sizing
                            try:
                                # Try with keepAspectRatio first (newer versions)
                                try:
                                    img = Image(alt_path, width=5*inch, height=3.5*inch, keepAspectRatio=True)
                                except TypeError:
                                    # Fallback for older versions without keepAspectRatio
                                    img = Image(alt_path, width=5*inch, height=3.5*inch)
                                elements.append(img)
                                elements.append(Spacer(1, 12))
                                print(f"Successfully added chart: {chart_name}")
                                chart_found = True
                                charts_added += 1
                                break
                            except Exception as img_error:
                                print(f"Error loading chart image {alt_path}: {str(img_error)}")
                                continue
                    
                    if not chart_found:
                        print(f"Chart file not found for: {chart_path}")
                        print(f"Tried paths: {possible_paths}")
                        # Add a placeholder if image fails
                        elements.append(Paragraph(f"[Chart: {chart_name} - Image could not be loaded]", self.styles['CustomBodyText']))
                        elements.append(Spacer(1, 12))
                        
                except Exception as e:
                    print(f"Error adding chart {chart_path}: {str(e)}")
                    continue
            
            if charts_added == 0:
                # Add a note if no charts could be loaded
                no_charts_note = Paragraph("Charts were generated but could not be loaded into the PDF. Please check the chart files in the static/charts directory.", self.styles['CustomBodyText'])
                elements.append(no_charts_note)
                elements.append(Spacer(1, 12))
        else:
            # Add a note if no charts are provided
            no_charts_note = Paragraph("No charts available for this analysis.", self.styles['CustomBodyText'])
            elements.append(no_charts_note)
            elements.append(Spacer(1, 12))
        
        return elements

    def _create_detailed_analysis(self, summary):
        """Create the detailed analysis section"""
        elements = []
        
        # Section header
        header = Paragraph("Detailed Analysis", self.styles['SectionHeader'])
        elements.append(header)
        elements.append(Spacer(1, 12))
        
        # Add detailed analysis content
        analysis_text = """
        The AI analysis provides comprehensive insights into the energy system performance, 
        including data-driven recommendations for maintenance and optimization. The analysis 
        combines statistical modeling with operational insights to deliver actionable 
        recommendations for energy system management.
        """
        
        analysis_para = Paragraph(analysis_text, self.styles['CustomBodyText'])
        elements.append(analysis_para)
        elements.append(Spacer(1, 12))
        
        # Add key points from summary
        key_points = [
            "• Performance analysis based on comprehensive data review",
            "• Statistical modeling for predictive insights",
            "• Risk assessment and mitigation strategies",
            "• Maintenance optimization recommendations",
            "• Operational efficiency improvements"
        ]
        
        for point in key_points:
            point_para = Paragraph(point, self.styles['CustomBodyText'])
            elements.append(point_para)
            elements.append(Spacer(1, 4))
        
        return elements

def generate_pdf_report(summary, stats, charts=None, site_name="Energy Site"):
    """
    Convenience function to generate PDF report
    
    Args:
        summary (str): AI-generated analysis summary
        stats (dict): Analysis statistics
        charts (list): List of chart file paths (optional)
        site_name (str): Name of the energy site
        
    Returns:
        str: Base64 encoded PDF data
    """
    generator = PDFReportGenerator()
    return generator.generate_ai_analysis_pdf(summary, stats, charts, site_name) 
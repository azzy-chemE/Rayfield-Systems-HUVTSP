import os
import traceback
import gc
import time
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
from dotenv import load_dotenv
from ai_summary_generator import qwen_summary, create_mock_summary_with_csv_analysis
from pdf_generator import generate_pdf_report


# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Check if API key is set in environment
if not os.getenv("OPENROUTER_API_KEY"):
    print("WARNING: OPENROUTER_API_KEY environment variable not set!")
    print("Please set the environment variable for production deployment.")
    print("For local development, create a .env file or set the environment variable.")

# Detect if running on Render (production environment)
IS_RENDER = os.environ.get('RENDER', False) or os.environ.get('PORT', False) or os.environ.get('RENDER_EXTERNAL_URL', False)

# Global variable to store uploaded CSV data
uploaded_csv_data = None
uploaded_csv_filename = None

def ensure_charts_directory():
    """Ensure the static/charts directory exists"""
    try:
        os.makedirs('static/charts', exist_ok=True)
        return True
    except Exception as e:
        print(f"Warning: Could not create charts directory: {e}")
        return False

def get_chart_files(analysis_results):
    """Get chart file paths from analysis results or static/charts directory"""
    chart_files = []
    
    if 'output_dir' in analysis_results:
        charts_dir = analysis_results['output_dir']
        try:
            if os.path.exists(charts_dir):
                files_in_dir = os.listdir(charts_dir)
                for file in files_in_dir:
                    if file.endswith('.png'):
                        chart_files.append(f'/static/charts/{file}')
            elif os.path.exists('static/charts'):
                files_in_dir = os.listdir('static/charts')
                for file in files_in_dir:
                    if file.endswith('.png'):
                        chart_files.append(f'/static/charts/{file}')
        except Exception as e:
            print(f"Warning: Could not access charts directory: {e}")
    
    return chart_files

def clean_nan_values(obj):
    """Clean NaN values from objects for JSON compatibility"""
    if isinstance(obj, dict):
        return {k: clean_nan_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_values(v) for v in obj]
    elif hasattr(obj, 'dtype') and hasattr(obj, 'item'):  # numpy/pandas types
        if pd.isna(obj):
            return None
        else:
            return float(obj) if hasattr(obj, 'item') else obj
    else:
        return obj

# Serve static files from root directory
@app.route('/')
def index():
    try:
        return send_from_directory('.', 'index.html')
    except Exception as e:
        return jsonify({'error': f'Failed to serve index.html: {str(e)}'}), 500

@app.route('/<path:filename>')
def serve_files(filename):
    # Only serve specific static files, not API routes
    if filename in ['index.html', 'script.js', 'style.css', 'logo.png']:
        try:
            return send_from_directory('.', filename)
        except Exception as e:
            return jsonify({'error': f'Failed to serve {filename}: {str(e)}'}), 500
    else:
        return jsonify({'error': 'File not found'}), 404

@app.route('/static/charts/<path:filename>')
def serve_charts(filename):
    """Serve chart files from static/charts directory"""
    try:
        return send_from_directory('static/charts', filename)
    except Exception as e:
        return jsonify({'error': f'Failed to serve chart {filename}: {str(e)}'}), 500

# File upload endpoint
@app.route('/api/upload-csv', methods=['POST'])
def upload_csv():
    global uploaded_csv_data, uploaded_csv_filename
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.csv'):
            return jsonify({'error': 'Please upload a CSV file'}), 400
        
        # Read and validate the CSV file
        try:
            df = pd.read_csv(file)
            uploaded_csv_filename = file.filename
            uploaded_csv_data = df
            df.to_csv('uploaded_data.csv', index=False)
            
            return jsonify({
                'success': True,
                'message': f'CSV file uploaded successfully: {file.filename}',
                'filename': file.filename,
                'rows': len(df),
                'columns': list(df.columns)
            })
            
        except Exception as e:
            return jsonify({'error': f'Invalid CSV file: {str(e)}'}), 400
            
    except Exception as e:
        print(f"Error in upload_csv: {str(e)}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

# API endpoint for AI analysis
@app.route('/api/run-ai-analysis', methods=['POST'])
def run_ai_analysis():
    start_time = time.time()
    
    try:
        # Ensure request has JSON content
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON data'}), 400
        
        platform_setup = data.get('platformSetup')
        inspections = data.get('inspections', [])
        
        if not platform_setup:
            return jsonify({'error': 'Platform setup required'}), 400
        
        if not inspections:
            return jsonify({'error': 'Inspection data required'}), 400
        
        result = run_ai_summary_generator(platform_setup, inspections)
        elapsed_time = time.time() - start_time
        
        if IS_RENDER and elapsed_time > 25 and result.get('success'):
            return jsonify({
                'success': True,
                'summary': result.get('summary', 'Analysis completed with timeout warning'),
                'stats': result.get('stats', {}),
                'charts': result.get('charts', []),
                'analysis_results': result.get('analysis_results', {}),
                'csv_stats': result.get('csv_stats', {}),
                'message': 'AI analysis completed (partial results due to timeout)',
                'note': 'Request was taking too long, returning partial results',
                'elapsed_time': elapsed_time
            })
        
        if result['success']:
            return jsonify({
                'success': True,
                'summary': result['summary'],
                'stats': result['stats'],
                'charts': result.get('charts', []),
                'analysis_results': result.get('analysis_results', {}),
                'csv_stats': result.get('csv_stats', {}),
                'message': 'AI analysis completed successfully',
                'note': result.get('note', ''),
                'elapsed_time': elapsed_time
            })
        else:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 500
            
    except Exception as e:
        print(f"Error in run_ai_analysis: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

def run_ai_summary_generator(platform_setup, inspections):
    """
    Run the AI summary generator and return results with charts
    """
    try:
        # Check if we have uploaded CSV data
        if uploaded_csv_data is not None and uploaded_csv_filename:
            csv_file_path = 'uploaded_data.csv'
        elif os.path.exists('cleaned_data.csv'):
            csv_file_path = 'cleaned_data.csv'
        else:
            return {
                'success': False,
                'error': 'No CSV data available. Please upload a CSV file first.'
            }
        
        ensure_charts_directory()
        import energy_analysis
        analysis_results = energy_analysis.analyze_energy_csv(csv_file_path, output_dir='static/charts')
        analysis_results = clean_nan_values(analysis_results)
        stats = {
            'data_points': analysis_results.get('stats', {}).get('data_points', 0),
            'features': analysis_results.get('stats', {}).get('features', 0),
            'target_variable': analysis_results.get('stats', {}).get('target_variable', 'Unknown'),
            'date_range': analysis_results.get('stats', {}).get('date_range', 'Unknown'),
            'uploaded_filename': uploaded_csv_filename if uploaded_csv_filename else 'cleaned_data.csv'
        }
        
        summary = None
        
        analysis_text = analysis_results.get('analysis_results', '')
        inspection_text = f"Platform Setup: {platform_setup}\nInspections: {inspections}"
        
        prompt = f"""
        Based on the following energy data analysis and inspection information, provide a comprehensive summary:

        ENERGY DATA ANALYSIS:
        {analysis_text}

        INSPECTION INFORMATION:
        {inspection_text}

        Please provide:
        1. Key findings from the energy data
        2. Recommendations based on the inspection data
        3. Overall assessment of the energy system
        """
        
        try:
            summary = qwen_summary(prompt)
        except Exception as e:
            print(f"AI API error: {str(e)}")
            summary = None
        
        if not summary:
            summary = create_mock_summary_with_csv_analysis(
                analysis_results, platform_setup, inspections, "Renewable Energy Site"
            )
        chart_files = get_chart_files(analysis_results)
        
        # Clean up memory
        gc.collect()
        
        if summary:
            return {
                'success': True,
                'summary': summary,
                'stats': stats,
                'charts': chart_files,
                'analysis_results': analysis_results.get('analysis_results', {}),
                'csv_stats': analysis_results.get('stats', {}),
                'anomalies_table': analysis_results.get('anomalies_table', None)
            }
        else:
            return {
                'success': False,
                'error': 'Failed to generate summary - no response from AI model'
            }
            
    except MemoryError as e:
        print(f"Memory error in run_ai_summary_generator: {str(e)}")
        gc.collect()
        return {
            'success': False,
            'error': 'Memory limit exceeded. Try using lightweight mode.',
            'note': 'Memory error occurred during chart generation'
        }
    except Exception as e:
        print(f"Error in run_ai_summary_generator: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return {
            'success': False,
            'error': f'AI summary generation failed: {str(e)}'
        }

# API endpoint for quick AI analysis (fast mode)
@app.route('/api/quick-ai-analysis', methods=['POST'])
def quick_ai_analysis():
    start_time = time.time()
    
    try:
        # Ensure request has JSON content
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON data'}), 400
        
        platform_setup = data.get('platformSetup')
        inspections = data.get('inspections', [])
        
        if not platform_setup:
            return jsonify({'error': 'Platform setup required'}), 400
        
        if not inspections:
            return jsonify({'error': 'Inspection data required'}), 400
        
        # Run quick analysis (no charts, just stats and summary)
        result = run_quick_analysis(platform_setup, inspections)
        
        elapsed_time = time.time() - start_time
        
        if result['success']:
            return jsonify({
                'success': True,
                'summary': result['summary'],
                'stats': result['stats'],
                'csv_stats': result.get('csv_stats', {}),
                'message': 'Quick analysis completed successfully',
                'elapsed_time': elapsed_time,
                'mode': 'quick'
            })
        else:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 500
            
    except Exception as e:
        print(f"Error in quick_ai_analysis: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

def run_quick_analysis(platform_setup, inspections):
    """
    Run quick analysis without heavy chart generation
    """
    try:
        # Check if we have uploaded CSV data
        if uploaded_csv_data is not None and uploaded_csv_filename:
            csv_file_path = 'uploaded_data.csv'
        elif os.path.exists('cleaned_data.csv'):
            csv_file_path = 'cleaned_data.csv'
        else:
            return {
                'success': False,
                'error': 'No CSV data available. Please upload a CSV file first.'
            }
        
        ensure_charts_directory()
        import energy_analysis
        analysis_results = energy_analysis.analyze_energy_csv_quick(csv_file_path)
        
        analysis_results = clean_nan_values(analysis_results)
        stats = {
            'data_points': analysis_results.get('stats', {}).get('data_points', 0),
            'features': analysis_results.get('stats', {}).get('features', 0),
            'target_variable': analysis_results.get('stats', {}).get('target_column', 'Unknown'),
            'date_range': analysis_results.get('stats', {}).get('date_range', 'Unknown'),
            'uploaded_filename': uploaded_csv_filename if uploaded_csv_filename else 'cleaned_data.csv'
        }
        
        summary = None
        
        analysis_text = analysis_results.get('analysis_results', '')
        inspection_text = f"Platform Setup: {platform_setup}\nInspections: {inspections}"
        
        prompt = f"""
        Based on the following energy data analysis and inspection information, provide a comprehensive summary:

        ENERGY DATA ANALYSIS:
        {analysis_text}

        INSPECTION INFORMATION:
        {inspection_text}

        Please provide:
        1. Key findings from the energy data
        2. Recommendations based on the inspection data
        3. Overall assessment of the energy system
        """
        
        try:
            summary = qwen_summary(prompt)
        except Exception as e:
            print(f"AI API error: {str(e)}")
            summary = None
        
        if not summary:
            summary = create_mock_summary_with_csv_analysis(
                analysis_results, platform_setup, inspections, "Renewable Energy Site"
            )

        chart_files = get_chart_files(analysis_results)
        
        # Clean up memory
        gc.collect()
        
        if summary:
            return {
                'success': True,
                'summary': summary,
                'stats': stats,
                'charts': chart_files,
                'analysis_results': analysis_results.get('analysis_results', {}),
                'csv_stats': analysis_results.get('stats', {}),
                'anomalies_table': analysis_results.get('anomalies_table', None),
                'mode': 'quick'
            }
        else:
            return {
                'success': False,
                'error': 'Failed to generate summary - no response from AI model'
            }
            
    except MemoryError as e:
        print(f"Memory error in run_quick_analysis: {str(e)}")
        gc.collect()
        return {
            'success': False,
            'error': 'Memory limit exceeded during quick analysis.'
        }
    except Exception as e:
        print(f"Error in run_quick_analysis: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return {
            'success': False,
            'error': f'Quick analysis failed: {str(e)}'
        }

# API endpoint to get current data status
@app.route('/api/status', methods=['GET'])
def get_status():
    try:
        # Check if data files exist (with error handling for Render)
        data_exists = False
        model_exists = False
        try:
            data_exists = os.path.exists('cleaned_data.csv')
        except:
            pass
        
        try:
            model_exists = os.path.exists('model.pkl')
        except:
            pass
            
        api_key_set = bool(os.getenv("OPENROUTER_API_KEY"))
        
        return jsonify({
            'data_available': data_exists,
            'model_available': model_exists,
            'api_key_set': api_key_set,
            'is_render': IS_RENDER,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# API endpoint for generating and downloading PDF report
@app.route('/api/generate-pdf-report', methods=['POST'])
def generate_pdf_report_endpoint():
    try:
        # Ensure request has JSON content
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON data'}), 400
        
        summary = data.get('summary')
        stats = data.get('stats', {})
        charts = data.get('charts', [])
        site_name = data.get('site_name', 'Energy Site')
        anomalies_table = data.get('anomalies_table', None)
        
        if not summary:
            return jsonify({'error': 'Summary data required'}), 400
        

        
        try:
            pdf_base64 = generate_pdf_report(summary, stats, charts, site_name, anomalies_table)
        except Exception as pdf_error:
            print(f"PDF generation error: {str(pdf_error)}")
            try:
                pdf_base64 = generate_pdf_report(summary, stats, [], site_name, anomalies_table)
            except Exception as retry_error:
                print(f"PDF generation retry failed: {str(retry_error)}")
                return jsonify({
                    'success': False,
                    'error': f'PDF generation failed: {str(retry_error)}'
                }), 500
        
        if pdf_base64:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"AI_Analysis_Report_{site_name.replace(' ', '_')}_{timestamp}.pdf"
            
            return jsonify({
                'success': True,
                'pdf_data': pdf_base64,
                'filename': filename,
                'message': 'PDF report generated successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to generate PDF report'
            }), 500
            
    except Exception as e:
        print(f"Error in generate_pdf_report_endpoint: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': f'PDF generation failed: {str(e)}'}), 500

if __name__ == '__main__':
    gc.collect()
    app.run(
        debug=True, 
        host='0.0.0.0', 
        port=int(os.environ.get('PORT', 5000)),
        threaded=True,
        use_reloader=False
    )

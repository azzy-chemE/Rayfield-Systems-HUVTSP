import os
import traceback
import gc
import time
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
from dotenv import load_dotenv
from ai_summary_generator import generate_comprehensive_analysis, generate_summary_from_user_data_only, qwen_summary, generate_weekly_summary, generate_weekly_summary_with_user_data, create_mock_summary_with_csv_analysis
import werkzeug

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
            print(f"Uploaded CSV: {file.filename}, Shape: {df.shape}")
            
            # Save the uploaded file
            uploaded_csv_filename = file.filename
            uploaded_csv_data = df
            
            # Save to a temporary file for analysis
            temp_filename = 'uploaded_data.csv'
            df.to_csv(temp_filename, index=False)
            
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
        lightweight_mode = data.get('lightweight', IS_RENDER)  # Default to lightweight on Render
        
        if not platform_setup:
            return jsonify({'error': 'Platform setup required'}), 400
        
        if not inspections:
            return jsonify({'error': 'Inspection data required'}), 400
        
        # Force lightweight mode on Render to prevent timeouts
        if IS_RENDER and not lightweight_mode:
            print("Forcing lightweight mode on Render to prevent timeouts")
            lightweight_mode = True
        
        # Run the AI summary generator with user data
        result = run_ai_summary_generator(platform_setup, inspections, lightweight_mode)
        
        # Check if we're approaching timeout (30 seconds for Render)
        elapsed_time = time.time() - start_time
        if IS_RENDER and elapsed_time > 25:
            print(f"Warning: Request taking too long ({elapsed_time:.2f}s)")
        
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
                'lightweight_mode': lightweight_mode,
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

def run_ai_summary_generator(platform_setup, inspections, lightweight_mode=False):
    """
    Run the AI summary generator and return results with charts
    """
    try:
        # Check if we have uploaded CSV data
        csv_file_path = None
        if uploaded_csv_data is not None and uploaded_csv_filename:
            csv_file_path = 'uploaded_data.csv'
            print(f"Using uploaded CSV file: {uploaded_csv_filename}")
        elif os.path.exists('cleaned_data.csv'):
            csv_file_path = 'cleaned_data.csv'
            print("Using fallback cleaned_data.csv")
        else:
            return {
                'success': False,
                'error': 'No CSV data available. Please upload a CSV file first.'
            }
        
        # Create static/charts directory if it doesn't exist (with error handling for Render)
        try:
            os.makedirs('static/charts', exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create charts directory: {e}")
            # Continue without charts directory
        
        # Import energy_analysis here to avoid memory issues
        import energy_analysis
        
        # Perform data analysis
        print("Starting data analysis...")
        analysis_results = energy_analysis.analyze_energy_csv(csv_file_path, output_dir='static/charts', lightweight_mode=lightweight_mode)
        
        # Get basic stats
        stats = {
            'data_points': analysis_results.get('stats', {}).get('data_points', 0),
            'features': analysis_results.get('stats', {}).get('features', 0),
            'target_variable': analysis_results.get('stats', {}).get('target_variable', 'Unknown'),
            'date_range': analysis_results.get('stats', {}).get('date_range', 'Unknown'),
            'uploaded_filename': uploaded_csv_filename if uploaded_csv_filename else 'cleaned_data.csv'
        }
        
        # Generate AI summary
        print("Generating AI summary...")
        summary = None
        
        # Construct prompt for AI analysis
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
        
        # If API fails, use mock summary
        if not summary:
            print("API failed, using mock summary...")
            summary = create_mock_summary_with_csv_analysis(
                analysis_results, platform_setup, inspections, "Renewable Energy Site"
            )
        
        print(f"Summary generated: {bool(summary)}")
        
        # Get chart file paths (only if not in lightweight mode)
        chart_files = []
        if not lightweight_mode and 'output_dir' in analysis_results:
            charts_dir = analysis_results['output_dir']
            try:
                if os.path.exists(charts_dir):
                    for file in os.listdir(charts_dir):
                        if file.endswith('.png'):
                            chart_files.append(f'/static/charts/{file}')
            except Exception as e:
                print(f"Warning: Could not access charts directory: {e}")
                # Continue without charts
        
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
                'lightweight_mode': lightweight_mode
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

# Debug endpoint to show all routes
@app.route('/api/debug', methods=['GET'])
def debug_endpoint():
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            'endpoint': rule.endpoint,
            'methods': list(rule.methods),
            'rule': str(rule)
        })
    return jsonify({
        'status': 'success',
        'routes': routes,
        'timestamp': datetime.now().isoformat()
    })

# Test endpoint
@app.route('/api/test', methods=['GET'])
def test_endpoint():
    return jsonify({
        'status': 'success',
        'message': 'Flask server is running',
        'timestamp': datetime.now().isoformat()
    })

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

if __name__ == '__main__':
    # Memory-efficient configuration
    gc.collect()  # Clean up memory before starting
    
    # Use threaded mode instead of processes for lower memory usage
    app.run(
        debug=True, 
        host='0.0.0.0', 
        port=int(os.environ.get('PORT', 5000)),
        threaded=True,  # Use threads instead of processes
        use_reloader=False  # Disable reloader to reduce memory usage
    ) 

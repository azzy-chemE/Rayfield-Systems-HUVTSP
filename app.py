import os
import traceback
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
from dotenv import load_dotenv
from ai_summary_generator import generate_comprehensive_analysis, generate_summary_from_user_data_only, qwen_summary, generate_weekly_summary, generate_weekly_summary_with_user_data

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Check if API key is set in environment
if not os.getenv("OPENROUTER_API_KEY"):
    print("WARNING: OPENROUTER_API_KEY environment variable not set!")
    print("Please set the environment variable for production deployment.")
    print("For local development, create a .env file or set the environment variable.")

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

# API endpoint for AI analysis
@app.route('/api/run-ai-analysis', methods=['POST'])
def run_ai_analysis():
    try:
        # Ensure request has JSON content
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON data'}), 400
        
        platform_setup = data.get('platformSetup')
        inspections = data.get('inspections', [])
        lightweight_mode = data.get('lightweight', False)  # New parameter for memory-constrained mode
        
        if not platform_setup:
            return jsonify({'error': 'Platform setup required'}), 400
        
        if not inspections:
            return jsonify({'error': 'Inspection data required'}), 400
        
        # Run the AI summary generator with user data
        result = run_ai_summary_generator(platform_setup, inspections, lightweight_mode)
        
        if result['success']:
            return jsonify({
                'success': True,
                'summary': result['summary'],
                'stats': result['stats'],
                'message': 'AI analysis completed successfully',
                'note': result.get('note', '')  # Include note if mock was used
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
        # Check if cleaned_data.csv exists
        if not os.path.exists('cleaned_data.csv'):
            return {
                'success': False,
                'error': 'cleaned_data.csv not found'
            }
        
        # Check if API key is set
        api_key = os.getenv("OPENROUTER_API_KEY")
        print(f"API Key check: {bool(api_key)}")
        if not api_key:
            return {
                'success': False,
                'error': 'OPENROUTER_API_KEY environment variable not set'
            }
        
        # Import and run the AI summary generator
        from ai_summary_generator import generate_weekly_summary, generate_weekly_summary_with_user_data, generate_summary_from_user_data_only, qwen_summary
        from energy_analysis import analyze_energy_csv
        
        # Load data
        df = pd.read_csv('cleaned_data.csv')
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        
        print(f"Data loaded: {len(df)} rows")
        print(f"Platform setup: {platform_setup}")
        print(f"Inspections: {len(inspections)} items")
        
        # Analyze the CSV data and generate charts
        if lightweight_mode:
            print("Running in lightweight mode - skipping chart generation...")
            analysis_results = {'error': 'lightweight_mode'}
        else:
            print("Analyzing CSV data and generating charts...")
            try:
                analysis_results = analyze_energy_csv('cleaned_data.csv', output_dir='static/charts')
            except MemoryError:
                print("Memory error during chart generation, using fallback...")
                # Return basic analysis without charts
                return {
                    'success': True,
                    'summary': "Analysis completed but chart generation failed due to memory constraints. Please try with a smaller dataset or restart the server.",
                    'stats': {
                        'site_type': platform_setup.get('siteType', 'energy'),
                        'inspections_count': len(inspections),
                        'critical_inspections': len([i for i in inspections if i.get('status') == 'critical']),
                        'concern_inspections': len([i for i in inspections if 'concern' in i.get('status', '')]),
                        'normal_inspections': len([i for i in inspections if i.get('status') == 'normal']),
                        'memory_error': True
                    },
                    'charts': [],
                    'analysis_results': {},
                    'csv_stats': {}
                }
        
        if 'error' in analysis_results:
            print(f"Error in CSV analysis: {analysis_results['error']}")
            # Continue with user data only if CSV analysis fails
            summary, stats = generate_summary_from_user_data_only(
                platform_setup, 
                inspections, 
                "Renewable Energy Site"
            )
        else:
            # Generate comprehensive analysis with CSV data
            # Create a comprehensive prompt that includes the analysis results
            csv_stats = analysis_results.get('stats', {})
            model_performance = analysis_results.get('analysis_results', {}).get('linear_regression', {})
            
            # Create a comprehensive prompt
            prompt = f"""
            Analyze the following renewable energy site with comprehensive data analysis:

            SITE CONFIGURATION:
            - Site Type: {platform_setup.get('siteType', 'renewable')}
            - Site Specifications: {platform_setup.get('siteSpecs', 'Standard renewable energy site')}

            CSV DATA ANALYSIS RESULTS:
            - Data points analyzed: {csv_stats.get('data_points', 'N/A')}
            - Features identified: {csv_stats.get('features', 'N/A')}
            - Target variable: {csv_stats.get('target_column', 'N/A')}
            - Date range: {csv_stats.get('date_range', {}).get('start', 'N/A')} to {csv_stats.get('date_range', {}).get('end', 'N/A')}

            MODEL PERFORMANCE:
            - Mean Squared Error: {model_performance.get('mse', 'N/A'):.2f}
            - R² Score: {model_performance.get('r2', 'N/A'):.4f}

            INSPECTION DATA:
            {chr(10).join([f"- Date: {i.get('date', 'Unknown')} | Status: {i.get('status', 'Unknown')} | Notes: {i.get('notes', 'No notes')}" for i in inspections])}

            Please provide a comprehensive analysis that includes:
            1. Overall performance assessment based on the CSV data analysis
            2. Analysis of inspection findings and their correlation with performance data
            3. Model performance evaluation and feature importance insights
            4. Specific recommendations for maintenance or optimization
            5. Key insights for operational decision-making
            6. Risk assessment based on inspection status and performance metrics
            7. Anomaly detection and potential causes
            8. Predictive maintenance recommendations

            Format the response in a clear, professional manner suitable for maintenance teams.
            Focus on actionable insights that maintenance teams can use immediately.
            """
            
            # Generate AI summary
            summary = qwen_summary(prompt)
            
            # Prepare comprehensive stats
            stats = {
                'csv_analysis': csv_stats,
                'model_performance': model_performance,
                'site_type': platform_setup.get('siteType', 'energy'),
                'inspections_count': len(inspections),
                'critical_inspections': len([i for i in inspections if i.get('status') == 'critical']),
                'concern_inspections': len([i for i in inspections if 'concern' in i.get('status', '')]),
                'normal_inspections': len([i for i in inspections if i.get('status') == 'normal']),
                'analysis_success': True
            }
            
            # If API fails, use mock summary
            if not summary:
                print("API failed, using mock summary...")
                summary = create_mock_summary_with_csv_analysis(
                    analysis_results, platform_setup, inspections, "Renewable Energy Site"
                )
        
        print(f"Summary generated: {bool(summary)}")
        
        # Get chart file paths
        chart_files = []
        if 'output_dir' in analysis_results:
            charts_dir = analysis_results['output_dir']
            if os.path.exists(charts_dir):
                for file in os.listdir(charts_dir):
                    if file.endswith('.png'):
                        chart_files.append(f'/static/charts/{file}')
        
        if summary:
            return {
                'success': True,
                'summary': summary,
                'stats': stats,
                'charts': chart_files,
                'analysis_results': analysis_results.get('analysis_results', {}),
                'csv_stats': analysis_results.get('stats', {})
            }
        else:
            return {
                'success': False,
                'error': 'Failed to generate summary - no response from AI model'
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
        # Check if data files exist
        data_exists = os.path.exists('cleaned_data.csv')
        model_exists = os.path.exists('model.pkl')
        api_key_set = bool(os.getenv("OPENROUTER_API_KEY"))
        
        return jsonify({
            'data_available': data_exists,
            'model_available': model_exists,
            'api_key_set': api_key_set,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Memory-efficient configuration
    import gc
    gc.collect()  # Clean up memory before starting
    
    # Use threaded mode instead of processes for lower memory usage
    app.run(
        debug=True, 
        host='0.0.0.0', 
        port=int(os.environ.get('PORT', 5000)),
        threaded=True,  # Use threads instead of processes
        use_reloader=False  # Disable reloader to reduce memory usage
    ) 

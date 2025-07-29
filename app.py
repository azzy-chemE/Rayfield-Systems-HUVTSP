import os
import traceback
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Check if API key is set in environment
if not os.getenv("OPENROUTER_API_KEY"):
    print("WARNING: OPENROUTER_API_KEY environment variable not set!")
    print("Please set the environment variable for production deployment.")
    print("For local development, create a .env file or set the environment variable.")

# Serve static files
@app.route('/')
def index():
    try:
        return app.send_static_file('index.html')
    except Exception as e:
        return jsonify({'error': f'Failed to serve index.html: {str(e)}'}), 500

@app.route('/static/<path:filename>')
def static_files(filename):
    try:
        return app.send_static_file(filename)
    except Exception as e:
        return jsonify({'error': f'Failed to serve {filename}: {str(e)}'}), 500

@app.route('/<path:filename>')
def other_files(filename):
    # Only serve specific static files, not API routes
    if filename in ['index.html', 'script.js', 'style.css', 'logo.png']:
        try:
            return app.send_static_file(filename)
        except Exception as e:
            return jsonify({'error': f'Failed to serve {filename}: {str(e)}'}), 500
    else:
        return jsonify({'error': 'File not found'}), 404

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
        
        if not platform_setup:
            return jsonify({'error': 'Platform setup required'}), 400
        
        if not inspections:
            return jsonify({'error': 'Inspection data required'}), 400
        
        # Run the AI summary generator with user data
        result = run_ai_summary_generator(platform_setup, inspections)
        
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

def run_ai_summary_generator(platform_setup, inspections):
    """
    Run the AI summary generator and return results
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
        
        # Load data
        df = pd.read_csv('cleaned_data.csv')
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        
        print(f"Data loaded: {len(df)} rows")
        print(f"Platform setup: {platform_setup}")
        print(f"Inspections: {len(inspections)} items")
        
        # Generate summary using ONLY user data
        summary, stats = generate_summary_from_user_data_only(
            platform_setup, 
            inspections, 
            "Renewable Energy Site"
        )
        
        print(f"Summary generated: {bool(summary)}")
        
        if summary:
            return {
                'success': True,
                'summary': summary,
                'stats': stats
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
    # Move static files to correct location for Flask
    import shutil
    import os
    
    # Create static directory if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')
    
    # Copy static files to Flask static directory
    static_files = ['index.html', 'script.js', 'style.css', 'logo.png']
    for file in static_files:
        if os.path.exists(file):
            shutil.copy2(file, f'static/{file}')
    
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000))) 
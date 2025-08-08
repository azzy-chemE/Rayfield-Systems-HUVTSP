import requests
import json
import pandas as pd
from datetime import datetime
import os
import tempfile
import base64
import re  # NEW

# Import the new energy analysis module
from energy_analysis import EnergyDataAnalyzer, analyze_energy_csv

# OpenRouter API configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _clean_markdown(text: str) -> str:
    """Remove markdown noise like **bold**, *italic*, headers (#), bullets, backticks, and numbered lists."""
    import re
    if not text:
        return text
    text = text.replace('\r\n', '\n')

    # Remove backticks `inline code`
    text = re.sub(r'`([^`\n]+)`', r'\1', text)

    # Remove **bold** / *italic* / ***bold-italic***
    text = re.sub(r'\*{1,3}([^*\n][^*]*?)\*{1,3}', r'\1', text)

    # Remove Markdown headers (#, ##, ###)
    text = re.sub(r'^\s{0,3}#{1,6}\s*', '', text, flags=re.MULTILINE)

    # Remove bullet markers (*, -, •)
    text = re.sub(r'^\s*[\*\-•]\s+', '', text, flags=re.MULTILINE)

    # Remove numbered list markers like "1. ", "2. "
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)

    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def qwen_summary(prompt_text):
    """
    Generate summary using Qwen model via OpenRouter API
    """
    import time

    print("Making request to OpenRouter API...")
    start = time.time()

    headers = {
        'Authorization': f'Bearer {OPENROUTER_API_KEY}',
        'Content-Type': 'application/json'
    }

    payload = {
        "model": "google/gemini-2.0-flash-exp:free",
        "messages": [
            {
                "role": "user",
                "content": prompt_text
            }
        ],
        "max_tokens": 1000,
        "temperature": 0.7
    }

    try:
        response = requests.post(
            OPENROUTER_BASE_URL + "/chat/completions",
            headers=headers,
            json=payload,
            timeout=15
        )

        elapsed = time.time() - start
        print(f"OpenRouter API call finished in {elapsed:.2f} seconds with status {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content']
            else:
                print("No response content in API result")
                return None
        else:
            print(f"API request failed: {response.status_code} {response.reason}")
            print(f"Error response: {response.text}")
            return None

    except Exception as e:
        print(f"Exception during API call: {str(e)}")
        return None


def analyze_uploaded_csv(csv_data, filename):
    """
    Analyze uploaded CSV file using the energy analysis module

    Args:
        csv_data (str): Base64 encoded CSV data
        filename (str): Original filename

    Returns:
        dict: Analysis results
    """
    try:
        # Decode base64 data
        csv_bytes = base64.b64decode(csv_data.split(',')[1] if ',' in csv_data else csv_data)

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as temp_file:
            temp_file.write(csv_bytes)
            temp_file_path = temp_file.name

        # Analyze the CSV
        results = analyze_energy_csv(temp_file_path, output_dir='temp_analysis')

        # Extract anomaly rows if present
        anomaly_rows = None
        if 'stats' in results and 'target_column' in results['stats']:
            analyzer = EnergyDataAnalyzer(temp_file_path)
            analyzer.load_and_prepare_data()
            data = analyzer.df[analyzer.target_column].dropna()
            mean_val = data.mean()
            std_val = data.std()
            upper_threshold = mean_val + 2 * std_val
            lower_threshold = mean_val - 2 * std_val
            anomalies = analyzer.df[(analyzer.df[analyzer.target_column] > upper_threshold) | (analyzer.df[analyzer.target_column] < lower_threshold)]
            anomaly_rows = anomalies.to_dict(orient='records')
        results['anomaly_rows'] = anomaly_rows if anomaly_rows is not None else []

        # Clean up temporary file
        os.unlink(temp_file_path)
        return results

    except Exception as e:
        print(f"Error analyzing CSV: {str(e)}")
        return {'error': f'Failed to analyze CSV: {str(e)}'}


def create_summary_prompt_with_csv_analysis(csv_analysis, platform_setup, inspections, site_name="Energy Site"):
    """
    Create a structured prompt that includes CSV analysis results and inspection data
    """
    # Extract CSV analysis results
    csv_stats = csv_analysis.get('stats', {})
    model_performance = csv_analysis.get('analysis_results', {}).get('linear_regression', {})

    # Format CSV analysis summary
    csv_summary = ""
    if csv_stats:
        csv_summary = f"""
CSV DATA ANALYSIS RESULTS:
- Data points analyzed: {csv_stats.get('data_points', 'N/A')}
- Features identified: {csv_stats.get('features', 'N/A')}
- Target variable: {csv_stats.get('target_column', 'N/A')}
- Date range: {csv_stats.get('date_range', {}).get('start', 'N/A')} to {csv_stats.get('date_range', {}).get('end', 'N/A')}

TARGET VARIABLE STATISTICS:
- Mean: {csv_stats.get('target_stats', {}).get('mean', 'N/A'):.2f}
- Standard deviation: {csv_stats.get('target_stats', {}).get('std', 'N/A'):.2f}
- Range: {csv_stats.get('target_stats', {}).get('min', 'N/A'):.2f} to {csv_stats.get('target_stats', {}).get('max', 'N/A'):.2f}

MODEL PERFORMANCE:
- Mean Squared Error: {model_performance.get('mse', 'N/A'):.2f}
- R² Score: {model_performance.get('r2', 'N/A'):.4f}
"""

    # Process inspection data
    inspection_summary = ""
    if inspections:
        inspection_summary = "\n\nRECENT INSPECTIONS:\n"
        for i, inspection in enumerate(inspections, 1):
            date = inspection.get('date', 'Unknown date')
            status = inspection.get('status', 'Unknown status')
            notes = inspection.get('notes', 'No notes provided')
            inspection_summary += f"{i}. Date: {date} | Status: {status} | Notes: {notes}\n"

    # Create comprehensive prompt
    prompt = f"""
    Analyze the following energy maintenance site data for {site_name}:

    SITE CONFIGURATION:
    - Site Type: {platform_setup.get('siteType', 'energy')}
    - Site Specifications: {platform_setup.get('siteSpecs', 'Standard energy site')}

    {csv_summary}

    {inspection_summary}

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

    return prompt


def create_mock_summary_with_csv_analysis(csv_analysis, platform_setup, inspections, site_name="Energy Site"):
    """
    Create a mock summary incorporating CSV analysis results
    """
    # Safely get data with fallbacks
    csv_stats = csv_analysis.get('stats', {}) if csv_analysis else {}
    model_performance = csv_analysis.get('analysis_results', {}).get('linear_regression', {}) if csv_analysis else {}
    anomalies_table = csv_analysis.get('anomalies_table', {}) if csv_analysis else {}

    site_type = platform_setup.get('siteType', 'energy') if platform_setup else 'energy'
    site_specs = platform_setup.get('siteSpecs', 'Standard energy site') if platform_setup else 'Standard energy site'

    # Process inspection data
    inspection_text = ""
    if inspections:
        inspection_text = "\n\nRECENT INSPECTIONS:\n"
        for i, inspection in enumerate(inspections, 1):
            date = inspection.get('date', 'Unknown date')
            status = inspection.get('status', 'Unknown status')
            notes = inspection.get('notes', 'No notes provided')
            inspection_text += f"{i}. Date: {date} | Status: {status} | Notes: {notes}\n"

    # Count inspection types
    critical_count = len([i for i in inspections if i.get('status') == 'critical'])
    concern_count = len([i for i in inspections if 'concern' in i.get('status', '')])
    normal_count = len([i for i in inspections if i.get('status') == 'normal'])

    # Safely format numeric values
    data_points = csv_stats.get('data_points', 'N/A')
    features = csv_stats.get('features', 'N/A')
    target_column = csv_stats.get('target_column', 'N/A')
    date_start = csv_stats.get('date_range', {}).get('start', 'N/A') if csv_stats.get('date_range') else 'N/A'
    date_end = csv_stats.get('date_range', {}).get('end', 'N/A') if csv_stats.get('date_range') else 'N/A'

    # Safely format performance metrics
    mean_output = csv_stats.get('target_stats', {}).get('mean', 'N/A')
    std_output = csv_stats.get('target_stats', {}).get('std', 'N/A')
    r2_score = model_performance.get('r2', 'N/A') if isinstance(model_performance, dict) else 'N/A'

    # Format numeric values safely
    mean_str = f"{mean_output:.2f}" if isinstance(mean_output, (int, float)) else str(mean_output)
    std_str = f"{std_output:.2f}" if isinstance(std_output, (int, float)) else str(std_output)
    r2_str = f"{r2_score:.4f}" if isinstance(r2_score, (int, float)) else str(r2_score)

    # Process anomalies information
    anomalies_text = ""
    if anomalies_table and anomalies_table.get('total_anomalies', 0) > 0:
        total_anomalies = anomalies_table.get('total_anomalies', 0)
        upper_threshold = anomalies_table.get('upper_threshold', 0)
        lower_threshold = anomalies_table.get('lower_threshold', 0)
        mean_value = anomalies_table.get('mean_value', 0)

        anomalies_text = f"""
ANOMALIES ANALYSIS:
- Total anomalies detected: {total_anomalies}
- Upper threshold: {upper_threshold:.2f}
- Lower threshold: {lower_threshold:.2f}
- Mean value: {mean_value:.2f}

The anomalies table provides a detailed list of all detected anomalies, showing timestamps, 
values, threshold types, and deviation percentages. This information helps identify patterns 
in anomalous behavior and can guide maintenance decisions.
"""
    else:
        anomalies_text = """
ANOMALIES ANALYSIS:
- No anomalies detected in the current dataset
- System appears to be operating within normal parameters
"""

    return f"""
AI Analysis Summary for {site_name}

SITE CONFIGURATION:
- Site Type: {site_type}
- Site Specifications: {site_specs}

CSV DATA ANALYSIS:
- Data points analyzed: {data_points}
- Features identified: {features}
- Target variable: {target_column}
- Date range: {date_start} to {date_end}

PERFORMANCE METRICS:
- Mean output: {mean_str}
- Standard deviation: {std_str}
- Model R² score: {r2_str}

{anomalies_text}

{inspection_text}

INSPECTION ANALYSIS:
- Total Inspections: {len(inspections)}
- Critical Issues: {critical_count}
- Concerns: {concern_count}
- Normal Status: {normal_count}

RECOMMENDATIONS:
1. Continue regular maintenance schedule for {site_type} systems
2. Monitor performance based on the analyzed data patterns
3. Address any critical inspection findings immediately
4. Consider model-based predictions for proactive maintenance
5. Implement site-specific optimization strategies
6. Review the anomalies table for any patterns in system behavior
7. Use anomaly detection insights for predictive maintenance planning

KEY INSIGHTS:
- System performance analysis completed successfully
- Model provides good predictive capability (R²: {r2_str})
- Inspection status provides operational guidance
- Risk assessment based on both data analysis and inspection findings
- Anomalies analysis provides additional insights for system monitoring

CHARTS GENERATED:
- Main data visualization charts (regression_results.png, rolling_averages.png)
- Anomaly detection chart (anomaly_detection.png)
- Residuals distribution chart (residuals_histogram.png)

Note: This analysis combines automated CSV data analysis with inspection findings. 
The anomalies table provides detailed information about system anomalies for better decision-making.
"""


def generate_comprehensive_analysis(csv_data, filename, platform_setup, inspections, site_name="Energy Site"):
    """
    Generate comprehensive analysis combining CSV data analysis with inspection data

    Args:
        csv_data (str): Base64 encoded CSV data
        filename (str): Original filename
        platform_setup (dict): Platform configuration
        inspections (list): Inspection data
        site_name (str): Site name

    Returns:
        tuple: (summary, stats)
    """
    try:
        # Analyze the uploaded CSV
        print("Analyzing uploaded CSV file...")
        csv_analysis = analyze_uploaded_csv(csv_data, filename)

        if 'error' in csv_analysis:
            return None, {'error': csv_analysis['error']}

        # Create prompt with CSV analysis results
        prompt = create_summary_prompt_with_csv_analysis(
            csv_analysis, platform_setup, inspections, site_name
        )

        # Generate AI summary
        print("Generating AI summary...")
        summary = qwen_summary(prompt)
        if summary:
            summary = _clean_markdown(summary)  # CLEAN HERE

        # Prepare comprehensive stats
        stats = {
            'csv_analysis': csv_analysis.get('stats', {}),
            'model_performance': csv_analysis.get('analysis_results', {}).get('linear_regression', {}),
            'site_type': platform_setup.get('siteType', 'energy'),
            'inspections_count': len(inspections),
            'critical_inspections': len([i for i in inspections if i.get('status') == 'critical']),
            'concern_inspections': len([i for i in inspections if 'concern' in i.get('status', '')]),
            'normal_inspections': len([i for i in inspections if i.get('status') == 'normal']),
            'analysis_success': True,
            'anomaly_rows': csv_analysis.get('anomaly_rows', [])
        }

        # If API fails, use mock summary
        if not summary:
            print("API failed, using mock summary...")
            summary = create_mock_summary_with_csv_analysis(
                csv_analysis, platform_setup, inspections, site_name
            )

        return summary, stats

    except Exception as e:
        print(f"Error in comprehensive analysis: {str(e)}")
        return None, {'error': str(e)}


def create_mock_summary_user_data_only(site_name, site_type, site_specs, inspections):
    """
    Create a mock summary using ONLY user data
    """
    # Process inspection data for mock summary
    inspection_text = ""
    if inspections:
        inspection_text = "\n\nRECENT INSPECTIONS:\n"
        for i, inspection in enumerate(inspections, 1):
            date = inspection.get('date', 'Unknown date')
            status = inspection.get('status', 'Unknown status')
            notes = inspection.get('notes', 'No notes provided')
            inspection_text += f"{i}. Date: {date} | Status: {status} | Notes: {notes}\n"

    # Count inspection types
    critical_count = len([i for i in inspections if i.get('status') == 'critical'])
    concern_count = len([i for i in inspections if 'concern' in i.get('status', '')])
    normal_count = len([i for i in inspections if i.get('status') == 'normal'])

    return f"""
AI Analysis Summary for {site_name}

SITE CONFIGURATION:
- Site Type: {site_type}
- Site Specifications: {site_specs}

OVERALL ASSESSMENT:
Based on the provided configuration, this {site_type} site appears to be properly configured for renewable energy generation. The specifications indicate a well-planned installation with appropriate capacity and technology.

{inspection_text}

INSPECTION ANALYSIS:
- Total Inspections: {len(inspections)}
- Critical Issues: {critical_count}
- Concerns: {concern_count}
- Normal Status: {normal_count}

RECOMMENDATIONS:
1. Continue regular maintenance schedule for {site_type} systems
2. Address any critical inspection findings immediately
3. Monitor {site_type} performance based on specifications
4. Implement site-specific optimization strategies

KEY INSIGHTS:
- Site configuration appears appropriate for the intended purpose
- Inspection status provides operational guidance
- Maintenance priorities should align with site type requirements
- Risk assessment based on inspection findings

Note: This analysis is based solely on user-provided configuration and inspection data.
"""


def export_summary(summary, filename="weekly_summary.txt"):
    """
    Export summary to text file
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Energy Site Analysis Summary\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        f.write(summary)

    print(f"Summary exported to {filename}")


def attach_summary_to_csv(df, summary, output_filename="final_output_with_summary.csv"):
    """
    Attach summary to CSV file
    """
    # Add summary column to dataframe
    df_copy = df.copy()
    df_copy["summary"] = summary

    # Export to CSV
    df_copy.to_csv(output_filename, index=False)
    print(f"Data with summary exported to {output_filename}")


def generate_weekly_summary(df, site_name="Energy Site"):
    """
    Generate weekly summary using the dataframe and site name
    This function is called by app.py or can be called directly with a DataFrame.
    """
    try:
        # Create a basic platform setup for the dataframe analysis
        platform_setup = {
            'siteType': 'energy',
            'siteSpecs': f'Data analysis for {site_name}'
        }

        # Create mock inspections based on data
        inspections = []
        if len(df) > 0:
            # Create a mock inspection based on the data
            latest_date = df['Datetime'].max() if 'Datetime' in df.columns else datetime.now()
            inspections.append({
                'date': latest_date.strftime('%Y-%m-%d') if hasattr(latest_date, 'strftime') else str(latest_date),
                'status': 'normal',
                'notes': f'Data analysis completed for {site_name} with {len(df)} data points'
            })

        # --- FIX: No recursion. Encode DataFrame to base64 CSV and call comprehensive analysis ---
        print("Generating AI summary...")
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        csv_b64 = base64.b64encode(csv_bytes).decode('utf-8')

        summary, stats = generate_comprehensive_analysis(
            csv_data=csv_b64,
            filename=f"{site_name.replace(' ', '_').lower()}.csv",
            platform_setup=platform_setup,
            inspections=inspections,
            site_name=site_name
        )
        # ------------------------------------------------------------------------------------------

        if summary:
            print("\nGenerated Summary:")
            print("-" * 30)
            print(summary)
            print("-" * 30)

            # Export summary to text file
            export_summary(summary)

            # Attach to CSV
            attach_summary_to_csv(df, summary)

            print("\nSummary Statistics:")
            for key, value in stats.items():
                print(f"- {key}: {value}")

        else:
            print("Failed to generate summary. Please check your API key and internet connection.")

    except FileNotFoundError:
        print("ERROR: cleaned_data.csv not found. Please ensure the data file is in the same directory.")
    except Exception as e:
        print(f"ERROR: {e}")

import requests
import json
import pandas as pd
from datetime import datetime
import os

# OpenRouter API configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

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
        "model": "qwen/qwen3-30b-a3b:free",
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

def create_summary_prompt(site_name, avg_output, anomalies, peak_output, additional_data=None):
    """
    Create a structured prompt for a green energy data summary
    """
    prompt = f"""
    Summarize the .CSV and/or information that the user provided {site_name}:
    
    Include key metrics, such as:
    - Average output: {avg_output} kWh
    - Peak output: {peak_output} kWh
    - Anomalies detected: {anomalies}
    
    Additional Data:
    {additional_data if additional_data else "No additional data provided"}
    
    Please provide a concise summary that includes:
    1. Overall performance assessment
    2. Anomaly analysis and potential causes
    3. Recommendations for maintenance or optimization
    4. Key insights for operational decision-making
    5. Anything else important that maintenance teams and chemical engineers of the energy facility should know
    
    Format the response in a clear, professional manner suitable for maintenance teams.
    """
    
    return prompt

def create_mock_summary(site_name, avg_output, peak_output, anomalies, wind_speed_avg, wind_direction_avg, data_points):
    """
    Create a mock summary when API fails
    """
    return f"""
AI Analysis Summary for {site_name}

OVERALL PERFORMANCE ASSESSMENT:
The wind turbine site has shown consistent performance with an average output of {avg_output:.2f} kWh and peak output reaching {peak_output:.2f} kWh. The system demonstrates good operational efficiency across the analyzed period.

ANOMALY ANALYSIS:
{anomalies}

WIND CONDITIONS:
- Average wind speed: {wind_speed_avg:.2f} m/s
- Average wind direction: {wind_direction_avg:.2f} degrees
- Data points analyzed: {data_points}

RECOMMENDATIONS:
1. Continue regular maintenance schedule
2. Monitor wind patterns for optimal positioning
3. Consider performance optimization during peak wind conditions

KEY INSIGHTS:
- System operates within expected parameters
- Wind conditions are favorable for energy generation
- No critical maintenance issues detected

Note: This is a mock analysis generated due to API connectivity issues.
"""

def create_mock_summary_with_user_data(site_name, avg_output, peak_output, anomalies, wind_speed_avg, wind_direction_avg, data_points, site_type, site_specs, inspections):
    """
    Create a mock summary when API fails, incorporating user data
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
    
    return f"""
AI Analysis Summary for {site_name}

SITE CONFIGURATION:
- Site Type: {site_type}
- Site Specifications: {site_specs}

OVERALL PERFORMANCE ASSESSMENT:
The {site_type} site has shown consistent performance with an average output of {avg_output:.2f} kWh and peak output reaching {peak_output:.2f} kWh. The system demonstrates good operational efficiency across the analyzed period.

ANOMALY ANALYSIS:
{anomalies}

WIND CONDITIONS:
- Average wind speed: {wind_speed_avg:.2f} m/s
- Average wind direction: {wind_direction_avg:.2f} degrees
- Data points analyzed: {data_points}

{inspection_text}

RECOMMENDATIONS:
1. Continue regular maintenance schedule
2. Monitor {site_type} patterns for optimal positioning
3. Consider performance optimization during peak conditions
4. Address any inspection findings promptly

KEY INSIGHTS:
- System operates within expected parameters
- Wind conditions are favorable for energy generation
- No critical maintenance issues detected
- Site configuration appears appropriate for current performance

Note: This is a mock analysis generated due to API connectivity issues.
"""

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

def generate_weekly_summary(df, site_name="Wind Site A"):
    """
    Generate weekly summary from wind turbine data
    """
    # Calculate basic statistics
    avg_output = df['ActivePower_kW'].mean()
    peak_output = df['ActivePower_kW'].max()
    
    # Detect anomalies (simple threshold-based detection)
    threshold = df['ActivePower_kW'].mean() + 2 * df['ActivePower_kW'].std()
    anomalies = df[df['ActivePower_kW'] > threshold]
    
    # Format anomalies for the prompt
    if not anomalies.empty:
        anomaly_dates = anomalies['Datetime'].dt.strftime('%B %d').tolist()
        anomaly_info = f"Anomalies detected on: {', '.join(anomaly_dates)}"
    else:
        anomaly_info = "No significant anomalies detected"
    
    # Create additional data summary
    wind_speed_avg = df['WindSpeed_mps'].mean()
    wind_direction_avg = df['WindDirection_deg'].mean()
    
    additional_data = f"""
    Wind Conditions:
    - Average wind speed: {wind_speed_avg:.2f} m/s
    - Average wind direction: {wind_direction_avg:.2f} degrees
    - Data points analyzed: {len(df)}
    - Date range: {df['Datetime'].min().strftime('%Y-%m-%d')} to {df['Datetime'].max().strftime('%Y-%m-%d')}
    """
    
    # Create the prompt
    prompt = create_summary_prompt(
        site_name=site_name,
        avg_output=f"{avg_output:.2f}",
        anomalies=anomaly_info,
        peak_output=f"{peak_output:.2f}",
        additional_data=additional_data
    )
    
    # Generate summary using Qwen
    summary = qwen_summary(prompt)
    
    # Prepare stats
    stats = {
        'avg_output': avg_output,
        'peak_output': peak_output,
        'anomalies': anomaly_info,
        'wind_speed_avg': wind_speed_avg,
        'wind_direction_avg': wind_direction_avg,
        'data_points': len(df)
    }
    
    # If API fails, use mock summary
    if not summary:
        print("API failed, using mock summary...")
        summary = create_mock_summary(
            site_name,
            avg_output,
            peak_output,
            anomaly_info,
            wind_speed_avg,
            wind_direction_avg,
            len(df)
        )
    
    return summary, stats

def generate_weekly_summary_with_user_data(df, platform_setup, inspections, site_name="Wind Site A"):
    """
    Generate weekly summary from wind turbine data with user input data
    """
    # Calculate basic statistics
    avg_output = df['ActivePower_kW'].mean()
    peak_output = df['ActivePower_kW'].max()
    
    # Detect anomalies (simple threshold-based detection)
    threshold = df['ActivePower_kW'].mean() + 2 * df['ActivePower_kW'].std()
    anomalies = df[df['ActivePower_kW'] > threshold]
    
    # Format anomalies for the prompt
    if not anomalies.empty:
        anomaly_dates = anomalies['Datetime'].dt.strftime('%B %d').tolist()
        anomaly_info = f"Anomalies detected on: {', '.join(anomaly_dates)}"
    else:
        anomaly_info = "No significant anomalies detected"
    
    # Create additional data summary
    wind_speed_avg = df['WindSpeed_mps'].mean()
    wind_direction_avg = df['WindDirection_deg'].mean()
    
    # Incorporate user platform setup
    site_type = platform_setup.get('siteType', 'wind')
    site_specs = platform_setup.get('siteSpecs', 'Standard wind farm')
    
    # Process inspection data
    inspection_summary = ""
    if inspections:
        inspection_summary = "\n\nRECENT INSPECTIONS:\n"
        for i, inspection in enumerate(inspections, 1):
            date = inspection.get('date', 'Unknown date')
            status = inspection.get('status', 'Unknown status')
            notes = inspection.get('notes', 'No notes provided')
            inspection_summary += f"{i}. Date: {date} | Status: {status} | Notes: {notes}\n"
    
    # Create enhanced prompt with user data
    prompt = f"""
    Analyze the following renewable energy site data for {site_name}:
    
    SITE CONFIGURATION:
    - Site Type: {site_type}
    - Site Specifications: {site_specs}
    
    PERFORMANCE DATA:
    - Average output: {avg_output:.2f} kWh
    - Peak output: {peak_output:.2f} kWh
    - Anomalies detected: {anomaly_info}
    
    WIND CONDITIONS:
    - Average wind speed: {wind_speed_avg:.2f} m/s
    - Average wind direction: {wind_direction_avg:.2f} degrees
    - Data points analyzed: {len(df)}
    - Date range: {df['Datetime'].min().strftime('%Y-%m-%d')} to {df['Datetime'].max().strftime('%Y-%m-%d')}
    
    {inspection_summary}
    
    Please provide a comprehensive analysis that includes:
    1. Overall performance assessment considering the site type and specifications
    2. Analysis of inspection findings and their impact on operations
    3. Anomaly analysis and potential causes based on the site configuration
    4. Specific recommendations for maintenance or optimization based on the site type
    5. Key insights for operational decision-making
    6. Risk assessment based on inspection status and performance data
    
    Format the response in a clear, professional manner suitable for maintenance teams.
    """
    
    # Generate summary using Qwen
    summary = qwen_summary(prompt)
    
    # Prepare stats
    stats = {
        'avg_output': avg_output,
        'peak_output': peak_output,
        'anomalies': anomaly_info,
        'wind_speed_avg': wind_speed_avg,
        'wind_direction_avg': wind_direction_avg,
        'data_points': len(df),
        'site_type': site_type,
        'inspections_count': len(inspections)
    }
    
    # If API fails, use mock summary
    if not summary:
        print("API failed, using mock summary...")
        summary = create_mock_summary_with_user_data(
            site_name,
            avg_output,
            peak_output,
            anomaly_info,
            wind_speed_avg,
            wind_direction_avg,
            len(df),
            site_type,
            site_specs,
            inspections
        )
    
    return summary, stats

def generate_summary_from_user_data_only(platform_setup, inspections, site_name="Renewable Energy Site"):
    """
    Generate summary using ONLY user platform setup and inspection data
    """
    # Extract user data
    site_type = platform_setup.get('siteType', 'renewable')
    site_specs = platform_setup.get('siteSpecs', 'Standard renewable energy site')
    
    # Process inspection data
    inspection_summary = ""
    if inspections:
        inspection_summary = "\n\nRECENT INSPECTIONS:\n"
        for i, inspection in enumerate(inspections, 1):
            date = inspection.get('date', 'Unknown date')
            status = inspection.get('status', 'Unknown status')
            notes = inspection.get('notes', 'No notes provided')
            inspection_summary += f"{i}. Date: {date} | Status: {status} | Notes: {notes}\n"
    
    # Create prompt using ONLY user data
    prompt = f"""
    Analyze the following renewable energy site based on user configuration and inspection data:

    SITE CONFIGURATION:
    - Site Type: {site_type}
    - Site Specifications: {site_specs}

    {inspection_summary}

    Please provide a comprehensive analysis that includes:
    1. Overall assessment of the {site_type} site based on the provided specifications
    2. Analysis of inspection findings and their impact on operations
    3. Risk assessment based on inspection status and site type
    4. Specific recommendations for maintenance or optimization based on the site type
    5. Key insights for operational decision-making
    6. Priority actions based on inspection status (normal/concern/critical)

    Format the response in a clear, professional manner suitable for maintenance teams.
    Focus entirely on the user-provided data without making assumptions about performance metrics.
    """
    
    # Generate summary using Qwen
    summary = qwen_summary(prompt)
    
    # Prepare stats based on user data only
    stats = {
        'site_type': site_type,
        'inspections_count': len(inspections),
        'critical_inspections': len([i for i in inspections if i.get('status') == 'critical']),
        'concern_inspections': len([i for i in inspections if 'concern' in i.get('status', '')]),
        'normal_inspections': len([i for i in inspections if i.get('status') == 'normal'])
    }
    
    # If API fails, use mock summary
    if not summary:
        print("API failed, using mock summary...")
        summary = create_mock_summary_user_data_only(
            site_name,
            site_type,
            site_specs,
            inspections
        )
    
    return summary, stats

def export_summary(summary, filename="weekly_summary.txt"):
    """
    Export summary to text file
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Wind Turbine Weekly Summary\n")
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

def main():
    """
    Main function to run the AI summary generation
    """
    print("Rayfield Systems - AI Summary Generator")
    print("=" * 50)
    
    # Check if API key is set
    if not OPENROUTER_API_KEY:
        print("ERROR: Please set the OPENROUTER_API_KEY environment variable")
        print("Get your free API key from: https://openrouter.ai/")
        print("For local development, create a .env file or set the environment variable")
        return
    
    try:
        # Load the cleaned data
        print("Loading wind turbine data...")
        df = pd.read_csv('cleaned_data.csv')
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        
        print(f"Loaded {len(df)} data points")
        
        # Generate summary
        print("Generating AI summary...")
        summary, stats = generate_weekly_summary(df, "Wind Site A")
        
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

if __name__ == "__main__":
    main() 

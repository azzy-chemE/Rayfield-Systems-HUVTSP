#!/usr/bin/env python3

import energy_analysis

# Test the anomalies table generation
result = energy_analysis.analyze_energy_csv('uploaded_data.csv', 'static/charts')
anomalies = result.get('anomalies_table')

print('Anomalies table structure:')
print(f'Keys: {list(anomalies.keys()) if anomalies else None}')
print(f'Total anomalies: {anomalies.get("total_anomalies", 0) if anomalies else 0}')

if anomalies and anomalies.get('table_data'):
    sample_data = anomalies.get('table_data', [])[:2]
    print(f'Sample data: {sample_data}')
    
    # Check if the data has the expected structure
    if sample_data:
        first_anomaly = sample_data[0]
        print(f'First anomaly keys: {list(first_anomaly.keys())}')
        print(f'First anomaly: {first_anomaly}')
else:
    print('No anomalies table data found') 
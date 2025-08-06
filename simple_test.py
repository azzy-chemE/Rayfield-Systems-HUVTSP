import os
import pandas as pd
import matplotlib.pyplot as plt

# Create test data
data = {
    'timestamp': pd.date_range('2024-01-01', periods=24, freq='H'),
    'energy_consumption': [100 + i*2 + (i%3)*5 for i in range(24)]
}

df = pd.DataFrame(data)

# Ensure charts directory exists
os.makedirs('static/charts', exist_ok=True)

# Create a simple chart
plt.figure(figsize=(10, 6))
plt.plot(df['timestamp'], df['energy_consumption'])
plt.title('Energy Consumption Over Time')
plt.xlabel('Time')
plt.ylabel('Energy Consumption')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('static/charts/test_chart.png', dpi=120, bbox_inches='tight')
plt.close()

print("Chart created successfully!")
print(f"Chart saved to: static/charts/test_chart.png")
print(f"File exists: {os.path.exists('static/charts/test_chart.png')}") 
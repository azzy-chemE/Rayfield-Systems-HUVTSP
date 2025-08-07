# Anomalies Table Format

## Section Header

**Anomalies Analysis**

## Summary Information

```
Total anomalies detected: 222
Upper threshold: 3586.53
Lower threshold: -799.16
Mean value: 1393.69
Standard deviation: 1096.42
```

## Table Format

| Timestamp/Index     | Value   | Threshold Type | Threshold | Deviation | Deviation % |
| ------------------- | ------- | -------------- | --------- | --------- | ----------- |
| 2018-01-01 21:00:00 | 3604.21 | Above Upper    | 3586.53   | +2210.52  | +158.6%     |
| 2018-01-01 23:10:00 | 3597.60 | Above Upper    | 3586.53   | +2203.91  | +158.1%     |
| 2018-01-02 01:20:00 | 3589.44 | Above Upper    | 3586.53   | +2195.75  | +157.5%     |
| 2018-01-02 07:40:00 | 3589.59 | Above Upper    | 3586.53   | +2195.90  | +157.5%     |
| 2018-01-02 14:40:00 | 3589.43 | Above Upper    | 3586.53   | +2195.74  | +157.5%     |
| 2018-01-03 02:30:00 | 3590.12 | Above Upper    | 3586.53   | +2196.43  | +157.6%     |
| 2018-01-03 10:20:00 | 3588.67 | Above Upper    | 3586.53   | +2194.98  | +157.4%     |
| 2018-01-03 18:15:00 | 3589.34 | Above Upper    | 3586.53   | +2195.65  | +157.5%     |
| 2018-01-04 06:45:00 | 3590.89 | Above Upper    | 3586.53   | +2197.20  | +157.6%     |
| 2018-01-04 14:30:00 | 3588.23 | Above Upper    | 3586.53   | +2194.54  | +157.4%     |

## Table Features

1. **Timestamp/Index Column**: Shows the exact time when the anomaly occurred (e.g., "2018-01-01 21:00:00")
2. **Value Column**: The actual value that was detected as anomalous (e.g., 3604.21)
3. **Threshold Type**: Whether it was "Above Upper" or "Below Lower" threshold
4. **Threshold**: The threshold value that was exceeded (upper or lower)
5. **Deviation**: How far the value deviated from the mean (positive or negative)
6. **Deviation %**: Percentage deviation from the mean

## Table Styling

- **Header Row**: Grey background with white text, bold font
- **Data Rows**: Alternating beige and white backgrounds
- **Grid Lines**: Black borders around all cells
- **Text Alignment**: Centered in all cells
- **Font**: Helvetica, 8pt for data, 10pt for headers

## Notes

- Table shows first 50 anomalies if more than 50 are detected (222 total in this case)
- Anomalies are sorted by timestamp in increasing order
- All anomalies in this example are "Above Upper" threshold
- Values are formatted to 2 decimal places
- Deviation percentages are formatted to 1 decimal place

## Example Data Structure

```python
anomalies_table = {
    'table_data': [
        {
            'x_value': '2018-01-01 21:00:00',
            'x_str': '2018-01-01 21:00:00',
            'y_value': 3604.21,
            'threshold_type': 'Above Upper',
            'threshold_value': 3586.53,
            'deviation': 2210.52,
            'deviation_percent': 158.6
        },
        {
            'x_value': '2018-01-01 23:10:00',
            'x_str': '2018-01-01 23:10:00',
            'y_value': 3597.60,
            'threshold_type': 'Above Upper',
            'threshold_value': 3586.53,
            'deviation': 2203.91,
            'deviation_percent': 158.1
        },
        # ... more anomalies (222 total)
    ],
    'total_anomalies': 222,
    'upper_threshold': 3586.53,
    'lower_threshold': -799.16,
    'mean_value': 1393.69,
    'std_value': 1096.42
}
```

## Key Points

- **Detection Method**: Uses 2-standard deviation threshold from mean
- **Sorting**: Anomalies are sorted chronologically (earliest first)
- **Limitation**: Shows maximum 50 anomalies in PDF for readability
- **Completeness**: All anomalies are included in the data structure
- **Accuracy**: Values are precise to 2 decimal places
- **Context**: Provides both absolute deviation and percentage deviation

# Maintenance Automation System

A comprehensive web application for energy maintenance automation with AI-powered analysis and predictive insights.

## Features

- **Platform Setup**: Configure different types of energy sites (Solar, Wind, Hydro, Thermal)
- **CSV Data Upload**: Upload CSV files with energy data for analysis
- **Inspection Data**: Log maintenance inspections with status tracking
- **AI Analysis**: Advanced AI-powered analysis using OpenRouter API with Qwen model
- **Data Visualization**: Automatic generation of linear regression charts and data analysis
- **Real-time Alerts**: Automated alert system for critical issues
- **Status Overview**: Comprehensive dashboard for system monitoring

## Technology Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript
- **AI**: OpenRouter API with Qwen model
- **Data Analysis**: Pandas, scikit-learn, Matplotlib, Seaborn
- **Deployment**: Render (with memory optimization)

## Setup Instructions

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd rayfield-gang
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file with:
   ```
   OPENROUTER_API_KEY=your_api_key_here
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the application**
   Open `http://localhost:5000` in your browser

### Render Deployment

1. **Set environment variables in Render dashboard:**
   - `OPENROUTER_API_KEY`: Your OpenRouter API key
   - `RENDER`: Set to `true` (enables lightweight mode)

2. **Deploy using the provided `render.yaml` configuration**

## Usage Guide

### 1. Platform Setup
- Select your site type (Solar, Wind, Hydro, Thermal)
- **Upload a CSV file** with your energy data
- The CSV should contain columns like:
  - `Datetime`: Timestamp of measurements
  - `Power_Output`: Energy output values
  - `Panel_Temperature`: Temperature readings
  - `Inverter_Efficiency`: Efficiency metrics
  - Any other relevant energy data columns

### 2. Inspection Data
- Add inspection records with dates, notes, and status
- Status options: Normal, Concern (Single/Multiple), Critical

### 3. AI Analysis
- Click "Run AI Analysis" to process your data
- The system will:
  - Analyze your uploaded CSV data
  - Generate linear regression models
  - Create data visualization charts
  - Provide AI-powered insights
  - Display comprehensive analysis results

### 4. Charts and Analysis
The system automatically generates:
- **Linear Regression Analysis**: Shows relationships between variables
- **Time Series Plots**: Visualizes data trends over time
- **Feature Importance**: Identifies key factors affecting performance
- **Correlation Heatmaps**: Shows relationships between all variables
- **Distribution Plots**: Analyzes data distributions

## Data Format Requirements

Your CSV file should include:
- **Datetime column**: For time series analysis
- **Numeric columns**: For regression analysis
- **Target variable**: Usually power output or energy generation
- **Feature columns**: Temperature, efficiency, weather conditions, etc.

Example CSV structure:
```csv
Datetime,Power_Output,Panel_Temperature,Inverter_Efficiency,Weather_Condition
2024-01-01 08:00:00,150.5,45.2,0.92,Sunny
2024-01-01 09:00:00,180.3,48.1,0.94,Sunny
...
```

## Memory Optimization

The application includes memory optimization features:
- **Lightweight mode**: Automatically enabled on Render to prevent timeouts
- **Chart generation**: Can be disabled to save memory
- **Data sampling**: Large datasets are sampled for plotting
- **Memory cleanup**: Automatic garbage collection after processing

## API Endpoints

- `POST /api/upload-csv`: Upload CSV files for analysis
- `POST /api/run-ai-analysis`: Run AI analysis on uploaded data
- `GET /api/test`: Test server connectivity
- `GET /api/status`: Check system status

## Troubleshooting

### Common Issues

1. **No charts displayed**: 
   - Check if lightweight mode is enabled
   - Verify CSV data format
   - Check browser console for errors

2. **Upload errors**:
   - Ensure file is CSV format
   - Check file size (should be < 10MB for Render)
   - Verify file contains numeric data

3. **Memory/timeout errors on Render**:
   - Set `RENDER=true` environment variable
   - Use smaller CSV files for testing
   - Check Render logs for specific errors

### Performance Tips

- **Local development**: Full functionality with charts
- **Render deployment**: Lightweight mode (no charts) to prevent timeouts
- **Large datasets**: Consider sampling data for faster processing

## Known Issues

- Chart generation may be slow with very large datasets
- Memory usage can be high during analysis (mitigated by lightweight mode)
- Some browsers may have issues with large chart images

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.

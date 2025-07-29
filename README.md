# Rayfield Maintenance Automation

A comprehensive maintenance automation platform for renewable energy plants with AI-powered analysis.

## Features

- **Platform Setup**: Configure different types of renewable energy sites (solar, wind, hydro, thermal)
- **Inspection Data**: Log maintenance inspections with status tracking(upload .csv or .pdf)
- **AI Analysis**: Real-time AI-powered analysis using Qwen model via OpenRouter
- **Dynamic Analysis**: AI responses change based on user inputs and site configuration
- **Professional UI**: Modern web interface with real-time updates

## Quick Start

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
   ```bash
   # For local development, create a .env file:
   echo "OPENROUTER_API_KEY=your_openrouter_api_key_here" > .env
   
   # Or set environment variable directly:
   export OPENROUTER_API_KEY="your_openrouter_api_key_here"
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the web interface**
   Open your browser and go to `http://localhost:5000` or 'https://rayfield-systems-huvtsp.onrender.com'

## How to Use

1. **Setup Platform**: Configure your renewable energy site type and specifications
2. **Add Inspections**: Log maintenance inspections with dates, notes, and status
3. **Run AI Analysis**: Get comprehensive AI-powered analysis based on your inputs. Metrics are AI determined based on site type, summarized at bottom under Ai analysis.
4. **View Results**: See dynamic analysis that changes based on your configuration. Outputs cleaned data and graphs if csv/pdf imported(work in progress).

## Technology Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript
- **AI**: OpenRouter API with Qwen model (free tier)
- **Data**: Pandas, scikit-learn
- **Deployment**: Ready for Render/Heroku

## Environment Setup

See `env_setup.md` for detailed environment configuration instructions.

## Team

Harvard Undergraduate Ventures-Tech Summer Program
- Azzy Xiang
- Akash Arun Kumar Soumya  
- Luis Cruz Mondragon
- Pushkar Kamma
- Tony Sun
- Anjali Vempati

# Rayfield Systems: Maintenance Automation

A comprehensive maintenance automation platform for energy plants, process managers, maintenance teams, and chemical engineers, with AI-powered analysis and trend reports.

## Quick Start

Open your browser and go to 'https://rayfield-systems-huvtsp.onrender.com'

## How to Use

- **Setup Platform**: Configure your energy site type and specifications
- **Add Inspections**: Log maintenance inspections with dates, notes, and status
- **Run AI Analysis:** Get comprehensive AI-powered analysis based on your inputs. Metrics are AI determined based on site type, summarized at bottom under AI analysis
- **View Results**: See dynamic analysis that changes based on your configuration. Outputs cleaned data and graphs if .CSV imported

## Latest Software Updates from August 8th, 2025

- **What we completed**: We completed a fully functional interactive web app prototype using the Render frontend framework. It includes charts with real energy data, an AI-generated summary using Gemini 2.0 experimental (Free) from the OpenRouter API service, and anomaly and classification markers clarified with a table of values. This also includes an embedded Google Form that allows you to automatically trigger an alert pipeline using Zapier, which sends you an email from a teammate's personal email address including a pre-loaded Google Calendar link and .ICS file to schedule the next maintenance visit to our app. Multiple scripts were consolidated for efficiency and memory usage was optimized. Our prototype creates a histogram of residuals, detects time-based anomalies, reports regression results and energy prediction accuracy, and includes rolling averages of values like kW generation. The code also creates a ready-to-email PDF that can be downloaded and sent to senior chemical engineers.
- **What parts didn’t get finished:** We have finished all essential parts of our prototype as instructed by the project lead. There are additional features that could be implemented, but are not necessary for the successful function of our Minimum Viable Product. The prototype is fully functional on the cleaned SCADA dataset with 30,000+ timestamps regarding wind turbines. 
- **What we’ll work on in the following week:** We will be preparing to present our prototype to the founder of Rayfield Systems, who will be receiving our product as part of the HUVTSP program's structured internship. If time allows, we may focus on integrating a login page or upscaling the algorithm to be more adaptable to different data and energy types, rather than being single-use-case and tailored towards one type of energy company's data. However, limitations may exist due to the fact that we are students using 100% free services. 

## About This Project
This platform was designed under the Harvard Undergraduate Ventures-Tech Summer Program (HUVTSP), where we were connected to an internship experience with the startup Rayfield Systems, to automate energy site maintenance through AI-powered inspection analysis and data visualization. It allows energy operators to stay proactive, reduce downtime, and improve long-term system health.

- Upload inspection and specification data in real-time
- Run AI analysis with one click and get downloadable reports
- Receive alerts and schedule check-ins automatically
- Track system status and generate historical summaries
  
## Features
- **Platform Setup**: Configure different types of renewable energy sites (solar, wind, hydro, thermal)
- **Inspection Data**: Log maintenance inspections with status tracking(upload .csv or .pdf)
- **AI Analysis**: Real-time AI-powered analysis using Qwen model via OpenRouter
- **Dynamic Analysis**: AI responses change based on user inputs and site configuration
- **Professional UI**: Modern web interface with real-time updates

## Technology Stack
- **Backend**: Python
- **Frontend**: HTML5, CSS3, JavaScript
- **AI**: OpenRouter API with Gemini 2.0 Experimental model (Free tier)
- **Data**: Pandas, scikit-learn, matplotlib
- **Deployment**: Render

## Important details

- .CSV "date" column must be formatted as such: "1/1/2018 12:00:00 AM"
- For testing, please use "uploaded_data.csv" from our GitHub repository.
- Open Source

## Project implications

This project represents our initial step in developing intelligent, automation-driven tools for the chemical engineering and energy sectors. If possible, we dream of continuing to refine and scale this MVP to support Rayfield Systems in delivering even more efficient, proactive site maintenance and data-driven decision-making. As aspiring engineers and computer scientists, we hope to grow this into a long-term collaboration and earn the opportunity to intern with Rayfield Systems (whether pro-bono or paid), contributing directly to the next generation of energy optimization software.

As part of this process, we have mapped out several features that we are capable of engineering for the release candidate (RC) phase of this web tool. Some of these features require more advanced web hosting services and API keys that cost a small price (such as a couple cents per week), which is why we require institutional backing to continue this project for Rayfield Systems.

- Adapt the web prototype for permitting, interconnection, or data automation, as needed by Rayfield Systems.
- Encrypted login page and database for energy maintenance and development teams to ensure privacy & effective data storage
- Internal dashboard and team access settings to encourage collaboration and discussion on data
- Automatic data-cleaning software, so that uploaded .CSVs and data do not need to be formatted a certain way for algorithm to execute
- An adaptable AI and Matplotlib system that can analyze a much wider variety of datasets from several energy sources
- Upgrading to the paid version of the Render frontend framework we're using to increase processing speed and quality

## Team

Harvard Undergraduate Ventures-Tech Summer Program

- Azzy Xiang (Group Founder & Lead Software Engineer)
- Akash Arun Kumar Soumya (Lead AI/ML Engineer & Software Developer)
- Pushkar Kamma (Lead Data Visualizer & Software/Frontend Developer)
- Tony Sun (Lead Frontend Developer)
- Luis Cruz Mondragon (Lead Research/Data Manager)
- Anjali Vempati (Lead Frontend/Backend Coordinator)
